"""Pairwise dataset for reranker training.

Creates (query, document, label) pairs from authorship data with
BM25-mined hard negatives for binary same-author classification.
"""

import copy
import os
import random
from typing import Any, Dict, List, Optional

import datasets
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from transformers import PreTrainedTokenizer

from authorship.data.utils import tokenize_pair


def _sample(data, n, rng: random.Random):
    if n >= len(data):
        return list(data)
    return [data[i] for i in rng.sample(range(len(data)), n)]


class PairwiseAuthorshipDataset(Dataset):
    """Per-query: creates pos/neg (query, doc, label) pairs for reranker training."""

    def __init__(
        self,
        dataname: str,
        tokenizer: PreTrainedTokenizer,
        candidate_data: datasets.Dataset,
        max_seq_length: int = 1024,
        num_train_examples: int = -1,
        num_pos_per_query: int = 1,
        num_neg_per_query: int = 3,
        seed: int = 777,
        instruction: Optional[str] = None,
    ):
        super().__init__()
        self.data_name = dataname
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.instruction = instruction
        self.num_pos = num_pos_per_query
        self.num_neg = num_neg_per_query
        self.rng = random.Random(seed)

        self.candidate_data = copy.deepcopy(candidate_data)
        data = copy.deepcopy(candidate_data)

        original_len = len(data)
        data = data.filter(
            lambda x: 0 < len(x.get("sameAuthor_docIDs", [])) < 1000,
            num_proc=min(os.cpu_count() or 4, 32),
        )
        if len(data) < original_len:
            print(f"[{dataname}] Filtered {original_len - len(data)} examples (no pos or bot-like)")

        if 0 < num_train_examples < len(data):
            data = data.select(_sample(range(len(data)), num_train_examples, self.rng))
        self.data = data

    def _get_positives(self, example: dict) -> List[dict]:
        # Prefer embedder-mined hard positives (lowest cosine sim); fall back to
        # random same-author sampling for datasets without the new column.
        pos_ids = example.get("hard_positive_docIDs") or example.get("sameAuthor_docIDs", [])
        if not pos_ids:
            fb = copy.deepcopy(example)
            words = fb["fullText"].split()
            fb["fullText"] = " ".join(words[: len(words) // 2])
            return [fb]
        return _sample(self.candidate_data.select(pos_ids).to_list(), self.num_pos, self.rng)

    def _get_negatives(self, example: dict) -> List[dict]:
        author_id = example["authorIDs"]
        # Prefer embedder-mined hard negatives (highest cosine sim); fall back to
        # BM25-mined negatives for datasets without the new column.
        neg_ids = example.get("hard_negative_docIDs") or example.get("BM25_retrieved_docIDs", [])
        if neg_ids:
            cands = self.candidate_data.select(neg_ids[: self.num_neg * 3]).to_list()
            negs = [c for c in cands if c["authorIDs"] != author_id]
        else:
            negs = []
        if len(negs) < self.num_neg:
            extra = _sample(range(len(self.candidate_data)), (self.num_neg - len(negs)) * 2, self.rng)
            negs.extend(c for c in self.candidate_data.select(extra).to_list() if c["authorIDs"] != author_id)
        return negs[: self.num_neg]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> List[Dict[str, Any]]:
        query = self.data[index]
        pairs = []
        for pos in self._get_positives(query):
            pairs.append({"query_text": query["fullText"], "doc_text": pos["fullText"], "label": 1})
        for neg in self._get_negatives(query):
            pairs.append({"query_text": query["fullText"], "doc_text": neg["fullText"], "label": 0})
        self.rng.shuffle(pairs)
        return pairs


class PairwiseCollator:
    """Collates pair lists: formats, tokenizes, pads."""

    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_length: int = 1024, instruction: Optional[str] = None):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.instruction = instruction

    def __call__(self, batch: List[List[Dict[str, Any]]]) -> Dict[str, torch.Tensor]:
        all_pairs = [p for group in batch for p in group]
        tokenized, labels = [], []
        for pair in all_pairs:
            tokenized.append(tokenize_pair(
                pair["query_text"], pair["doc_text"],
                self.tokenizer, self.max_seq_length, self.instruction,
            ))
            labels.append(pair["label"])

        padded = self.tokenizer.pad(tokenized, padding="max_length", max_length=self.max_seq_length, return_tensors="pt")
        return {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.float32),
        }


class RerankerDataModule:
    """Manages pairwise datasets for reranker training."""

    def __init__(self, dataset_names: List[str], languages: List[str] = None, seed: int = 777, num_workers: int = 4):
        self.dataset_names = dataset_names
        self.languages = languages or ["en"]
        self.seed = seed
        self.num_workers = num_workers
        self._datasets: List[PairwiseAuthorshipDataset] = []
        self._tokenizer = None
        self._config: dict = {}

    def connect(
        self, world_size: int, global_rank: int, tokenizer: PreTrainedTokenizer,
        global_batch_size: int = 64, max_seq_length: int = 1024,
        num_train_examples: int = 50000, num_pos_per_query: int = 1,
        num_neg_per_query: int = 3, instruction: Optional[str] = None,
    ):
        self._tokenizer = tokenizer
        self._config = dict(
            world_size=world_size, global_rank=global_rank,
            global_batch_size=global_batch_size, max_seq_length=max_seq_length,
            num_train_examples=num_train_examples, num_pos_per_query=num_pos_per_query,
            num_neg_per_query=num_neg_per_query, instruction=instruction,
        )

    def set_epoch(self, epoch: int):
        self.seed += 1

    def prepare_data(self):
        for name in self.dataset_names:
            datasets.load_dataset(name)

    def setup(self):
        self._datasets = []
        for name in self.dataset_names:
            hf = datasets.load_dataset(name)
            for lang in self.languages:
                if lang not in hf:
                    continue
                cand = hf[lang]
                required = {"authorIDs", "fullText", "sameAuthor_docIDs"}
                if required - set(cand.column_names):
                    continue
                n = self._config.get("num_train_examples", -1)
                if n > len(cand):
                    n = len(cand)
                ds = PairwiseAuthorshipDataset(
                    dataname=name, tokenizer=self._tokenizer, candidate_data=cand,
                    max_seq_length=self._config.get("max_seq_length", 1024),
                    num_train_examples=n,
                    num_pos_per_query=self._config.get("num_pos_per_query", 1),
                    num_neg_per_query=self._config.get("num_neg_per_query", 3),
                    seed=self.seed, instruction=self._config.get("instruction"),
                )
                self._datasets.append(ds)

    def train_dataloader(self) -> DataLoader:
        combined = ConcatDataset(self._datasets)
        ws = self._config.get("world_size", 1)
        bs = max(1, self._config.get("global_batch_size", 64) // ws)
        return DataLoader(
            combined, batch_size=bs, shuffle=True, num_workers=self.num_workers,
            collate_fn=PairwiseCollator(
                self._tokenizer,
                self._config.get("max_seq_length", 1024),
                self._config.get("instruction"),
            ),
            drop_last=True, pin_memory=True,
        )
