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
        num_epochs: int = 3,
        seed: int = 777,
        instruction: Optional[str] = None,
        negative_curriculum: Optional[List[Dict[str, int]]] = None,
    ):
        super().__init__()
        self.data_name = dataname
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.instruction = instruction
        self.num_pos = num_pos_per_query
        self.num_neg = num_neg_per_query
        self.num_epochs = num_epochs
        self.epoch = 0
        self.rng = random.Random(seed)
        self.negative_curriculum = self._normalize_negative_curriculum(negative_curriculum)

        self.candidate_data = copy.deepcopy(candidate_data)
        data = copy.deepcopy(candidate_data)

        original_len = len(data)
        data = data.filter(
            lambda x: 0 < len(x.get("sameAuthor_docIDs", [])) < 1000,
            num_proc=min(os.cpu_count() or 4, 32),
        )
        if len(data) < original_len:
            print(f"[{dataname}] Filtered {original_len - len(data)} examples (no pos or bot-like)")

        self._full_data = data
        self._base_seed = seed
        self.num_train_examples = num_train_examples
        self._resample(epoch=0)

    def _resample(self, epoch: int):
        """Sample a different subset each epoch using a deterministic per-epoch seed."""
        if 0 < self.num_train_examples < len(self._full_data):
            rng = random.Random(self._base_seed + epoch)
            indices = rng.sample(range(len(self._full_data)), self.num_train_examples)
            self.data = self._full_data.select(indices)
        else:
            self.data = self._full_data

    def _normalize_negative_curriculum(
        self,
        curriculum: Optional[List[Dict[str, int]]],
    ) -> Optional[List[Dict[str, int]]]:
        """Convert config entries into per-epoch negative counts."""
        if not curriculum:
            return None

        normalized = []
        for entry in curriculum:
            counts = {
                "easy": int(entry.get("easy", 0)),
                "semihard": int(entry.get("semihard", 0)),
                "hard": int(entry.get("hard", 0)),
            }
            if any(v < 0 for v in counts.values()):
                raise ValueError(f"Negative curriculum counts must be non-negative: {entry}")
            total = sum(counts.values())
            if total != self.num_neg:
                raise ValueError(
                    f"Negative curriculum counts must sum to num_neg_per_query={self.num_neg}; "
                    f"got {total} from {entry}"
                )
            normalized.append(counts)
        return normalized

    def _negative_plan(self) -> Dict[str, int]:
        if self.negative_curriculum:
            idx = min(self.epoch, len(self.negative_curriculum) - 1)
            return self.negative_curriculum[idx]

        # Backward-compatible behavior: one easy calibration negative when the
        # batch has room, with remaining negatives drawn from the hard window.
        n_easy = 1 if self.num_neg >= 2 else 0
        return {"easy": n_easy, "semihard": 0, "hard": self.num_neg - n_easy}

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self._resample(epoch)

    def _curriculum_neg_slice(self, neg_ids: list) -> list:
        """Slide a window from medium-hard toward hardest over training epochs.

        neg_ids is sorted hardest-first (descending cosine sim). Each epoch the
        window moves toward index 0. Final epoch always uses the hardest slice.
        """
        n = len(neg_ids)
        if n == 0:
            return []
        window = min(n, max(self.num_neg * 2, n // (2 ** self.num_epochs)))
        remaining = max(0, self.num_epochs - 1 - self.epoch)
        start = min(window * remaining, n - window)
        return neg_ids[start : start + window]

    def _hard_neg_slice(self, neg_ids: list) -> list:
        """Return the hardest BM25/embedder negatives."""
        window = max(self.num_neg * 3, 1)
        return neg_ids[:window]

    def _get_positives(self, example: dict) -> List[dict]:
        pos_ids = example.get("hard_positive_docIDs") or example.get("sameAuthor_docIDs", [])
        if not pos_ids:
            fb = copy.deepcopy(example)
            words = fb["fullText"].split()
            fb["fullText"] = " ".join(words[: len(words) // 2])
            return [fb]
        # Anchor on (hardest, easiest) plus evenly-spaced middles. pos_ids is
        # sorted hardest-first (low cosine), so index 0 is the calibration
        # challenge and index -1 is the "obvious yes" anchor. The easy anchor
        # gives the model a reference for what high confidence should feel like.
        if self.num_pos >= 2 and len(pos_ids) >= 2:
            picks = self._span_indices(len(pos_ids), self.num_pos)
            return self.candidate_data.select([pos_ids[i] for i in picks]).to_list()
        return self.candidate_data.select(pos_ids[: self.num_pos]).to_list()

    @staticmethod
    def _span_indices(n: int, k: int) -> List[int]:
        """k indices spanning [0, n-1] inclusive, evenly spaced."""
        if k <= 1:
            return [0]
        if k >= n:
            return list(range(n))
        return [round(i * (n - 1) / (k - 1)) for i in range(k)]

    def _select_indexed_negatives(self, neg_ids: list, author_id: Any, count: int, hardness: str) -> List[dict]:
        if count <= 0 or not neg_ids:
            return []
        if hardness == "hard":
            sliced = self._hard_neg_slice(neg_ids)
        elif hardness == "semihard":
            sliced = self._curriculum_neg_slice(neg_ids)
        else:
            raise ValueError(f"Unknown negative hardness: {hardness}")

        cands = self.candidate_data.select(sliced[: count * 4]).to_list()
        return [c for c in cands if c["authorIDs"] != author_id][:count]

    def _sample_easy_negatives(self, author_id: Any, count: int) -> List[dict]:
        if count <= 0:
            return []
        negs: List[dict] = []
        for _ in range(5):
            extra = _sample(range(len(self.candidate_data)), count * 4, self.rng)
            pool = [c for c in self.candidate_data.select(extra).to_list() if c["authorIDs"] != author_id]
            need = count - len(negs)
            negs.extend(pool[:max(0, need)])
            if len(negs) >= count:
                break
        return negs[:count]

    @staticmethod
    def _dedupe_docs(docs: List[dict]) -> List[dict]:
        deduped = []
        seen = set()
        for doc in docs:
            key = doc.get("docID") or doc.get("fullText") or id(doc)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(doc)
        return deduped

    def _backfill_negatives(self, negs: List[dict], author_id: Any) -> List[dict]:
        """Fill rare shortfalls from random different-author docs."""
        for _ in range(5):
            negs = self._dedupe_docs(negs)
            if len(negs) >= self.num_neg:
                return negs[: self.num_neg]
            need = self.num_neg - len(negs)
            extra = _sample(range(len(self.candidate_data)), need * 4, self.rng)
            negs.extend(c for c in self.candidate_data.select(extra).to_list() if c["authorIDs"] != author_id)
        return self._dedupe_docs(negs)[: self.num_neg]

    def _get_negatives(self, example: dict) -> List[dict]:
        """Compose negatives according to the configured hardness curriculum."""
        author_id = example["authorIDs"]
        plan = self._negative_plan()
        neg_ids = example.get("hard_negative_docIDs") or example.get("BM25_retrieved_docIDs", [])

        negs: List[dict] = []
        negs.extend(self._sample_easy_negatives(author_id, plan["easy"]))
        negs.extend(self._select_indexed_negatives(neg_ids, author_id, plan["semihard"], "semihard"))
        negs.extend(self._select_indexed_negatives(neg_ids, author_id, plan["hard"], "hard"))
        negs = self._dedupe_docs(negs)

        return self._backfill_negatives(negs, author_id)

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
        num_neg_per_query: int = 3, num_epochs: int = 3, instruction: Optional[str] = None,
        negative_curriculum: Optional[List[Dict[str, int]]] = None,
    ):
        self._tokenizer = tokenizer
        self._config = dict(
            world_size=world_size, global_rank=global_rank,
            global_batch_size=global_batch_size, max_seq_length=max_seq_length,
            num_train_examples=num_train_examples, num_pos_per_query=num_pos_per_query,
            num_neg_per_query=num_neg_per_query, num_epochs=num_epochs, instruction=instruction,
            negative_curriculum=negative_curriculum,
        )

    def set_epoch(self, epoch: int):
        self.seed += 1
        for ds in self._datasets:
            ds.set_epoch(epoch)

    @staticmethod
    def _load(name: str):
        if os.path.exists(name):
            return datasets.load_from_disk(name)
        return datasets.load_dataset(name)

    def prepare_data(self):
        for name in self.dataset_names:
            self._load(name)

    def setup(self):
        self._datasets = []
        for name in self.dataset_names:
            hf = self._load(name)
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
                    num_neg_per_query=self._config.get("num_neg_per_query", 1),
                    num_epochs=self._config.get("num_epochs", 3),
                    seed=self.seed, instruction=self._config.get("instruction"),
                    negative_curriculum=self._config.get("negative_curriculum"),
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
