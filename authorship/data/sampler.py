"""Sampler that ensures each batch comes from a single sub-dataset in a ConcatDataset."""

from typing import List

import torch
from torch.utils.data import Sampler


class ConcatedDataSampler(Sampler):
    """Batch sampler for ConcatDataset ensuring same-source batches.

    Each batch draws indices from a single constituent dataset. Incomplete
    batches are shuffled and merged. Supports distributed training via
    num_replicas / rank.
    """

    def __init__(
        self,
        each_data_sizes: List[int],
        global_batch_size: int,
        shuffle: bool = True,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 777,
        drop_last: bool = False,
    ):
        self.each_data_sizes = each_data_sizes
        self.batch_size = global_batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.indices = self._build_indices()
        self.num_samples = len(self.indices) // self.num_replicas

    def _build_indices(self) -> List[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        if self.shuffle:
            indices = [torch.randperm(n, generator=g).tolist() for n in self.each_data_sizes]
        else:
            indices = [list(range(n)) for n in self.each_data_sizes]

        for i in range(len(self.each_data_sizes)):
            offset = sum(self.each_data_sizes[:i])
            indices[i] = [idx + offset for idx in indices[i]]

        batched = []
        for data_indices in indices:
            batched.append(list(torch.split(torch.tensor(data_indices), self.batch_size)))

        incomplete = []
        for b in batched:
            if len(b[-1]) < self.batch_size:
                incomplete.append(b.pop())

        if not self.drop_last and incomplete:
            order = torch.randperm(len(incomplete), generator=g).tolist()
            merged = torch.cat([incomplete[i] for i in order])
            mixed = list(torch.split(merged, self.batch_size))
            if len(mixed[-1]) < self.batch_size:
                mixed.pop()
            all_batches = sum(batched, []) + mixed
        else:
            all_batches = sum(batched, [])

        if self.shuffle:
            order = torch.randperm(len(all_batches), generator=g).tolist()
        else:
            order = list(range(len(all_batches)))

        flat = []
        for batch_idx in order:
            flat.extend(int(i) for i in all_batches[batch_idx])
        return flat

    def __iter__(self):
        subset = self.indices[self.rank : len(self.indices) : self.num_replicas]
        return iter(subset)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self.indices = self._build_indices()
        self.num_samples = len(self.indices) // self.num_replicas
