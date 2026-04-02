"""Learning rate schedulers for training."""

import math

import torch
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    min_reduce_rate: float = 0.0,
    last_epoch: int = -1,
) -> LambdaLR:
    """Linear warmup then cosine decay to min_reduce_rate * lr."""

    def lr_lambda(step):
        if step < num_warmup_steps:
            return min_reduce_rate + (1.0 - min_reduce_rate) * step / max(1, num_warmup_steps)
        progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        cosine = 0.5 * (1.0 + min_reduce_rate + math.cos(math.pi * progress * num_cycles * 2.0) * (1.0 - min_reduce_rate))
        return max(min_reduce_rate, cosine)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_annealing_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int = 1,
    min_reduce_rate: float = 0.0,
    last_epoch: int = -1,
) -> LambdaLR:
    """Cosine annealing with warmup, supporting multiple cycles (one per epoch)."""

    def lr_lambda(step):
        steps_per_cycle = num_training_steps // num_cycles
        if step >= num_training_steps:
            return min_reduce_rate
        cycle_step = step % steps_per_cycle
        if cycle_step < num_warmup_steps:
            return min_reduce_rate + (1.0 - min_reduce_rate) * cycle_step / max(1, num_warmup_steps)
        progress = (cycle_step - num_warmup_steps) / max(1, steps_per_cycle - num_warmup_steps)
        cosine = 0.5 * (1.0 + min_reduce_rate + math.cos(math.pi * progress) * (1.0 - min_reduce_rate))
        return max(min_reduce_rate, cosine)

    return LambdaLR(optimizer, lr_lambda, last_epoch)
