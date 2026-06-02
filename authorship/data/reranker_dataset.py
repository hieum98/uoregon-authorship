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
        # Diagnostics: fraction of queries that receive a genuine high-similarity
        # ("easy") positive vs. falling back to a hard one (see _get_positives).
        self._easy_pos_hits = 0
        self._easy_pos_total = 0

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

        keys = ("random", "hard_q", "pos_med", "pos_hard")
        normalized = []
        for entry in curriculum:
            counts = {k: int(entry.get(k, 0)) for k in keys}
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
        """Per-epoch negative composition.

        Keys: ``random`` = n(r), ``hard_q`` = n(dq) (closest to query),
        ``pos_med`` / ``pos_hard`` = n(dq+) (closest to the med / hard positive).
        """
        if self.negative_curriculum:
            idx = min(self.epoch, len(self.negative_curriculum) - 1)
            return self.negative_curriculum[idx]

        # No curriculum: default to all query-close (hardest) negatives.
        return {"random": 0, "hard_q": self.num_neg, "pos_med": 0, "pos_hard": 0}

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self._resample(epoch)
        self._easy_pos_hits = 0
        self._easy_pos_total = 0

    @property
    def easy_pos_fraction(self) -> float:
        """Share of sampled queries that got a genuine high-similarity positive."""
        return self._easy_pos_hits / self._easy_pos_total if self._easy_pos_total else 0.0

    def _closest_cross_author(self, neg_ids: list, author_id: Any, count: int) -> List[dict]:
        """Return the ``count`` closest different-author docs.

        ``neg_ids`` is a closest-first list (highest embedder cosine first), so the
        head holds the hardest negatives. We take a generous top window, drop any
        same-author docs, and keep the closest ``count``.
        """
        if count <= 0 or not neg_ids:
            return []
        window = neg_ids[: max(count * 6, 32)]
        rows = self.candidate_data.select(window).to_list()
        return [c for c in rows if c["authorIDs"] != author_id][:count]

    def _get_positives(self, example: dict) -> Dict[str, Optional[dict]]:
        """Pick up to three same-author positives by difficulty.

        Roles (ordered): ``random`` — a natural (non-hard) same-author doc, tends
        high-similarity/easy; ``med`` — the middle of the hardest (lowest-cosine)
        same-author band; ``hard`` — the most dissimilar same-author doc.

        ``hard_positive_docIDs`` is the mined hardest same-author subset, hardest-
        first; ``sameAuthor_docIDs`` is the full set (includes the high-similarity
        docs). Each returned value is a full candidate row carrying its own
        ``hard_negative_docIDs`` — used to mine n(dq+) negatives close to that
        positive (see ``_get_negatives``).
        """
        natural_ids = list(example.get("sameAuthor_docIDs", []))
        hard_ids = list(example.get("hard_positive_docIDs") or [])
        q_text = example["fullText"]
        roles: Dict[str, Optional[Any]] = {"random": None, "med": None, "hard": None}

        # No same-author refs: fall back to a half-text self-positive.
        if not (natural_ids or hard_ids):
            fb = copy.deepcopy(example)
            words = fb["fullText"].split()
            fb["fullText"] = " ".join(words[: len(words) // 2])
            roles["random"] = fb
            self._easy_pos_total += 1
            return roles

        used: set = set()

        # HARD: most dissimilar same-author doc (head of hardest-first list).
        for cid in hard_ids:
            if cid not in used:
                roles["hard"] = cid
                used.add(cid)
                break

        # MED: middle of the hard band; scan outward from the centre for an unused id.
        if hard_ids:
            mid = len(hard_ids) // 2
            for k in sorted(range(len(hard_ids)), key=lambda k: abs(k - mid)):
                if hard_ids[k] not in used:
                    roles["med"] = hard_ids[k]
                    used.add(hard_ids[k])
                    break

        # RANDOM: a natural (non-hard) same-author doc; high-similarity/easy anchor.
        hard_set = set(hard_ids)
        easy_pool = [i for i in natural_ids if i not in hard_set and i not in used]
        got_easy = False
        for _ in range(10):
            if not easy_pool:
                break
            cand = self.rng.choice(easy_pool)
            if self.candidate_data[cand]["fullText"] != q_text:
                roles["random"] = cand
                used.add(cand)
                got_easy = True
                break
        # Fallbacks: least-hard hard id, then any unused natural id.
        if roles["random"] is None:
            for cid in reversed(hard_ids):
                if cid not in used:
                    roles["random"] = cid
                    used.add(cid)
                    break
        if roles["random"] is None:
            for cid in natural_ids:
                if cid not in used:
                    roles["random"] = cid
                    used.add(cid)
                    break

        self._easy_pos_hits += int(got_easy)
        self._easy_pos_total += 1

        # Materialize chosen docIDs into full candidate rows (carry hard_negative_docIDs).
        for role, cid in list(roles.items()):
            if cid is not None and not isinstance(cid, dict):
                roles[role] = self.candidate_data[int(cid)]
        return roles

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

    def _get_negatives(self, example: dict, positives: Dict[str, Optional[dict]]) -> List[dict]:
        """Compose negatives per the curriculum stage (paper-style 3-way + close-to-positive).

        Categories: ``random`` = n(r) random different-author docs; ``hard_q`` = n(dq)
        different-author docs closest to the QUERY; ``pos_med`` / ``pos_hard`` = n(dq+)
        different-author docs closest to the MED / HARD positive. Closeness uses the
        finetuned-embedder ``hard_negative_docIDs`` (closest-first) of the query / positive.
        """
        author_id = example["authorIDs"]
        plan = self._negative_plan()
        negs: List[dict] = []

        # n(r): random different-author docs.
        negs.extend(self._sample_easy_negatives(author_id, plan.get("random", 0)))

        # n(dq): different-author docs closest to the query.
        q_neg_ids = example.get("hard_negative_docIDs") or example.get("BM25_retrieved_docIDs", [])
        negs.extend(self._closest_cross_author(q_neg_ids, author_id, plan.get("hard_q", 0)))

        # n(dq+): different-author docs closest to the med / hard positive.
        med = positives.get("med")
        if med is not None:
            negs.extend(self._closest_cross_author(
                med.get("hard_negative_docIDs", []), author_id, plan.get("pos_med", 0)))
        hard = positives.get("hard")
        if hard is not None:
            negs.extend(self._closest_cross_author(
                hard.get("hard_negative_docIDs", []), author_id, plan.get("pos_hard", 0)))

        negs = self._dedupe_docs(negs)
        return self._backfill_negatives(negs, author_id)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> List[Dict[str, Any]]:
        query = self.data[index]
        positives = self._get_positives(query)
        pairs = []
        for pos in positives.values():
            if pos is not None:
                pairs.append({"query_text": query["fullText"], "doc_text": pos["fullText"], "label": 1})
        for neg in self._get_negatives(query, positives):
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
        tokenized, labels, group_ids = [], [], []
        for group_idx, group in enumerate(batch):
            for pair in group:
                tokenized.append(tokenize_pair(
                    pair["query_text"], pair["doc_text"],
                    self.tokenizer, self.max_seq_length, self.instruction,
                ))
                labels.append(pair["label"])
                group_ids.append(group_idx)

        padded = self.tokenizer.pad(tokenized, padding="max_length", max_length=self.max_seq_length, return_tensors="pt")
        return {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.float32),
            "group_ids": torch.tensor(group_ids, dtype=torch.long),
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
