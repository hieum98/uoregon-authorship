"""Contrastive learning dataset for embedder training.

Each sample yields a query document plus hard negatives (from BM25) and a
positive (same-author document), which are shuffled and flattened by the collator.
"""

import copy
import json
import logging
import os
import random
from itertools import product
from typing import Any, Dict, List, Optional

import datasets
import lightning as L
import polars as pl
import torch
from datasets import load_dataset
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from transformers import PreTrainedTokenizer

from authorship.data.sampler import ConcatedDataSampler
from authorship.data.utils import tokenize_example

logger = logging.getLogger(__name__)

_NUM_PROC = min(getattr(os, "cpu_count", lambda: 4)() or 4, 64)


def _sample(data, n, rng: random.Random):
    if n >= len(data):
        return list(data)
    return [data[i] for i in rng.sample(range(len(data)), n)]


class HardMiningDataset(Dataset):
    """Contrastive dataset: per sample returns [hard_negatives, positive, query]."""

    def __init__(
        self,
        dataname: str,
        tokenizer: PreTrainedTokenizer,
        candidate_data: datasets.Dataset,
        max_seq_length: int = 512,
        num_train_examples: int = -1,
        num_hard_negatives: int = 256,
        language: str = "en",
        seed: int = 777,
        cluster_docIDs: Optional[List[int]] = None,
    ):
        super().__init__()
        self.data_name = dataname
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.language = language
        self.num_hard_negatives = num_hard_negatives
        self.rng = random.Random(seed)

        self.candidate_data = copy.deepcopy(candidate_data)
        data = candidate_data.select(cluster_docIDs) if cluster_docIDs else copy.deepcopy(candidate_data)

        if num_train_examples > 0 and num_train_examples < len(data):
            data = data.select(_sample(range(len(data)), num_train_examples, self.rng))
        self.data = data

        if len(self.candidate_data) < num_hard_negatives:
            raise ValueError(
                f"{dataname}: candidate pool ({len(self.candidate_data)}) < "
                f"num_hard_negatives ({num_hard_negatives})"
            )

    def _get_hard_negatives(self, example: dict) -> List[dict]:
        neg_ids = example["hard_negative_docIDs"][: self.num_hard_negatives]
        negatives = self.candidate_data.select(neg_ids).to_list()
        negatives = [n for n in negatives if n["authorIDs"] != example["authorIDs"]]
        if len(negatives) < self.num_hard_negatives:
            extra = _sample(range(len(self.candidate_data)), self.num_hard_negatives - len(negatives), self.rng)
            negatives.extend(self.candidate_data.select(extra).to_list())
        return negatives

    def _get_positives(self, example: dict) -> List[dict]:
        pos_ids = example["sameAuthor_docIDs"]
        if not pos_ids:
            fallback = copy.deepcopy(example)
            words = fallback["fullText"].split()
            fallback["fullText"] = " ".join(words[: len(words) // 2])
            return [fallback]

        all_pos = self.candidate_data.select(pos_ids).to_list()
        if self.rng.random() < 0.5 and "cluster" in self.candidate_data.column_names:
            diff_cluster = [p for p in all_pos if p.get("cluster") != example.get("cluster")]
            if diff_cluster:
                return _sample(diff_cluster, 1, self.rng)
        return _sample(all_pos, 1, self.rng)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> List[dict]:
        example = self.data[index]
        items = self._get_hard_negatives(example) + self._get_positives(example) + [example]
        self.rng.shuffle(items)
        return items


class AuthorRepCollator:
    """Collates HardMiningDataset samples: flatten, tokenize, map author IDs."""

    def __init__(self, tokenizer: PreTrainedTokenizer, author_dict: Dict[str, int], max_seq_length: int = 512):
        self.tokenizer = tokenizer
        self.author_dict = author_dict
        self.max_seq_length = max_seq_length

    def __call__(self, batch: List[List[dict]]) -> Dict[str, Any]:
        flat = [ex for group in batch for ex in group]
        author_ids = torch.tensor(
            [self.author_dict[ex["authorIDs"]] for ex in flat], dtype=torch.long,
        )
        tok = tokenize_example(
            [ex["fullText"] for ex in flat],
            self.tokenizer,
            max_seq_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "authorIDs": author_ids,
        }


def _filter_fn(example, max_same_author=1000, min_hard_negatives=0):
    return (
        len(example["sameAuthor_docIDs"]) <= max_same_author
        and len(example.get("hard_negative_docIDs", [])) >= min_hard_negatives
        and example.get("to_train", True)
    )


class EmbedderDataModule(L.LightningDataModule):
    """Loads HuggingFace datasets, builds author_dict, creates per-dataset HardMiningDataset."""

    def __init__(
        self,
        dataset_names: List[str],
        languages: List[str] = None,
        seed: int = 777,
        num_workers: int = 4,
    ):
        super().__init__()
        self.all_data = sorted(dataset_names)
        self.languages = languages or ["en"]
        self.seed = seed
        self.num_workers = num_workers
        self.author_dict: Dict[str, int] = {}
        self.train_ds = None

    def connect(
        self,
        world_size: int = 1,
        global_rank: int = 0,
        tokenizer: PreTrainedTokenizer = None,
        global_batch_size: int = 32,
        max_seq_length: int = 512,
        num_train_examples: int = -1,
        num_hard_negatives: int = 256,
    ):
        self.world_size = world_size
        self.global_rank = global_rank
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.num_train_examples = num_train_examples
        self.num_hard_negatives = num_hard_negatives
        self.global_batch_size = global_batch_size
        self.batch_size = global_batch_size // world_size

    def prepare_data(self, checkpoint_dir: str = "checkpoint"):
        all_ids = []
        for name in self.all_data:
            data = datasets.load_from_disk(name) if os.path.exists(name) else load_dataset(name)
            for lang in self.languages:
                if lang not in data:
                    continue
                ds = data[lang].map(
                    lambda x: {"authorIDs": f"{name}-{x['authorIDs']}"},
                    num_proc=_NUM_PROC,
                )
                all_ids.extend(ds["authorIDs"])

        author_dict = {aid: i for i, aid in enumerate(set(all_ids))}
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, "author_dict.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(author_dict, f)

    def set_epoch(self, epoch: int):
        self.seed = self.seed + epoch

    def setup(self, stage="", checkpoint_dir: str = "checkpoint"):
        path = os.path.join(checkpoint_dir, "author_dict.json")
        with open(path, encoding="utf-8") as f:
            self.author_dict = json.load(f)

        train_ds = []
        for name in self.all_data:
            full = datasets.load_from_disk(name) if os.path.exists(name) else load_dataset(name)
            for lang in self.languages:
                if lang not in full:
                    continue
                data = full[lang]
                if "BM25_retrieved_docIDs" in data.column_names and "hard_negative_docIDs" not in data.column_names:
                    data = data.rename_column("BM25_retrieved_docIDs", "hard_negative_docIDs")

                candidate_data = data.map(
                    lambda x: {"authorIDs": f"{name}-{x['authorIDs']}"},
                    num_proc=_NUM_PROC
                )
                if "docID" not in candidate_data.column_names:
                    raise ValueError(f"{name}: missing docID column")

                raw = datasets.load_from_disk(name)[lang] if os.path.exists(name) else load_dataset(name, split=lang)
                if "BM25_retrieved_docIDs" in raw.column_names and "hard_negative_docIDs" not in raw.column_names:
                    raw = raw.rename_column("BM25_retrieved_docIDs", "hard_negative_docIDs")
                query_data = raw.map(
                    lambda x: {"authorIDs": f"{name}-{x['authorIDs']}"},
                    num_proc=_NUM_PROC
                )
                query_data = query_data.filter(
                    _filter_fn,
                    fn_kwargs={"max_same_author": 1000, "min_hard_negatives": self.num_hard_negatives},
                    num_proc=_NUM_PROC,
                )
                if len(query_data) == 0:
                    continue

                cluster_levels = {}
                level_names = []
                if len(query_data) > 20000:
                    level_names = sorted(c for c in query_data.column_names if c.startswith("cluster"))
                    for lv in level_names:
                        cluster_levels[lv] = query_data.unique(lv)

                if level_names and len(query_data) > 20000:
                    combos = [dict(zip(level_names, c)) for c in product(*[cluster_levels[lv] for lv in level_names])]
                    combos.sort(key=lambda x: tuple(x[lv] for lv in level_names))
                    keep = ["docID"] + level_names
                    qdf = pl.from_pandas(query_data.remove_columns(
                        [c for c in query_data.column_names if c not in keep]
                    ).to_pandas())
                else:
                    combos = [None]
                    qdf = None

                for combo in combos:
                    if combo is not None:
                        filt = None
                        for k, v in combo.items():
                            cond = pl.col(k) == v
                            filt = cond if filt is None else filt & cond
                        cluster_ids = sorted(qdf.filter(filt)["docID"].to_list())
                    else:
                        cluster_ids = sorted(query_data.to_pandas()["docID"].to_list()) if qdf is None else sorted(qdf["docID"].to_list())

                    try:
                        ds = HardMiningDataset(
                            dataname=name, 
                            tokenizer=self.tokenizer,
                            candidate_data=candidate_data,
                            max_seq_length=self.max_seq_length,
                            num_train_examples=self.num_train_examples,
                            num_hard_negatives=self.num_hard_negatives,
                            language=lang, seed=self.seed,
                            cluster_docIDs=cluster_ids,
                        )
                    except Exception as e:
                        logger.warning(f"Skipping {name}/{lang} cluster {combo}: {e}")
                        continue
                    if len(ds) > 0:
                        if self.global_rank == 0:
                            print(f"Loaded {name}/{lang} cluster {combo}: {len(ds)} examples")
                        train_ds.append(ds)

        assert train_ds, "No training data loaded"
        self.train_ds = ConcatDataset(train_ds)

    def train_dataloader(self) -> DataLoader:
        sizes = [len(d) for d in self.train_ds.datasets]
        sampler = ConcatedDataSampler(
            each_data_sizes=sizes,
            global_batch_size=self.global_batch_size,
            shuffle=True,
            num_replicas=self.world_size,
            rank=self.global_rank,
            seed=self.seed,
        )
        collator = AuthorRepCollator(
            tokenizer=self.tokenizer,
            author_dict=self.author_dict,
            max_seq_length=self.max_seq_length,
        )
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=min(self.num_workers, _NUM_PROC),
            collate_fn=collator,
        )



if __name__ == "__main__":

    from authorship.models.embedder import EmbeddingModel

    dataset_names = ["Hieuman/ru_KP", "Hieuman/exorde"]
    languages = ["ru", "en", "ar", 'zh']
    seed = 777
    num_workers = 4
    global_batch_size = 512
    max_seq_length = 256
    num_train_examples = 50000
    num_hard_negatives = 256

    data_module = EmbedderDataModule(
        dataset_names=dataset_names,
        languages=languages,
        seed=seed,
        num_workers=num_workers,
    )
    tokenizer = EmbeddingModel._build_tokenizer("Qwen/Qwen3-8B-Base")
    data_module.connect(
        tokenizer=tokenizer,
        global_batch_size=global_batch_size,
        max_seq_length=max_seq_length,
        num_train_examples=num_train_examples,
        num_hard_negatives=num_hard_negatives,
    )
    data_module.prepare_data()
    data_module.setup()
    train_loader = data_module.train_dataloader()
    for batch in train_loader:
        breakpoint()
