"""GradCache trainer for memory-efficient contrastive learning.

Splits large batches into chunks, computes contrastive loss on concatenated
representations, then replays forward-backward per chunk using cached gradients.
"""

import pathlib
import time
import typing
from collections import UserDict
from contextlib import nullcontext
from itertools import repeat
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import get_device_states, set_device_states
from torch.utils.data import DataLoader
import lightning as L

from authorship.training.losses import compute_contrastive_loss


class RandContext:
    """Save and restore RNG state for deterministic replay in GradCache."""

    def __init__(self, *tensors):
        self.fwd_cpu_state = torch.get_rng_state()
        self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)

    def __enter__(self):
        self._fork = torch.random.fork_rng(devices=self.fwd_gpu_devices, enabled=True)
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None


def split_input(model_input, chunk_size: int) -> List:
    """Split model input (dict of tensors) into chunks along batch dim."""
    if isinstance(model_input, (dict, UserDict)) and all(isinstance(v, torch.Tensor) for v in model_input.values()):
        keys = list(model_input.keys())
        chunked = [model_input[k].split(chunk_size, dim=0) for k in keys]
        return [dict(zip(kk, tt)) for kk, tt in zip(repeat(keys), zip(*chunked))]
    if isinstance(model_input, list) and all(isinstance(x, torch.Tensor) for x in model_input):
        chunked = [t.split(chunk_size, dim=0) for t in model_input]
        return [list(s) for s in zip(*chunked)]
    if isinstance(model_input, torch.Tensor):
        return list(model_input.split(chunk_size, dim=0))
    raise NotImplementedError(f"split_input not implemented for {type(model_input)}")


def _get_input_tensors(model_input) -> List[torch.Tensor]:
    if isinstance(model_input, torch.Tensor):
        return [model_input]
    if isinstance(model_input, (list, tuple)):
        return sum((_get_input_tensors(x) for x in model_input), [])
    if isinstance(model_input, (dict, UserDict)):
        return sum((_get_input_tensors(x) for x in model_input.values()), [])
    return []


class GradCacheTrainer:
    """GradCache training loop for contrastive embedding learning."""

    def __init__(self, fabric: L.Fabric, chunk_size: int = 1):
        self.fabric = fabric
        self.chunk_size = chunk_size

    def forward_no_grad(self, model, model_inputs):
        with torch.no_grad():
            rnd_state = RandContext(*_get_input_tensors(model_inputs))
            outputs = model(input_ids=model_inputs["input_ids"], attention_mask=model_inputs["attention_mask"])
            projection = outputs["projection"]
        return projection, rnd_state

    @typing.no_type_check
    def build_cache(self, projection, labels, temperature, use_miner, normalize, use_cross_device=True):
        projection = projection.detach().requires_grad_(True)
        with self.fabric.autocast():
            loss = compute_contrastive_loss(
                projection, labels, use_miner=use_miner,
                temperature=temperature, normalize=normalize,
                use_cross_device=use_cross_device,
            )
        self.fabric.backward(loss)
        cache = projection.grad
        loss_val = loss.detach() / (self.fabric.world_size if use_cross_device else 1)
        return cache, loss_val

    def forward_backward(self, model, model_inputs, state, reps_gradcache):
        with state:
            projection = model(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
            )["projection"]
            surrogate = torch.dot(projection.flatten(), reps_gradcache.flatten())
            self.fabric.backward(surrogate)

    def train_step(self, model, batch, temperature=0.05, use_miner=False, normalize=True):
        chunks = split_input(batch, self.chunk_size)

        rnd_states, all_proj = [], []
        for chunk in chunks:
            proj, rnd = self.forward_no_grad(model, chunk)
            all_proj.append(proj)
            rnd_states.append(rnd)
        all_proj = torch.cat(all_proj, dim=0)

        cache, loss = self.build_cache(all_proj, batch["authorIDs"], temperature, use_miner, normalize)
        self.fabric.barrier()
        cache_chunks = cache.split(self.chunk_size, dim=0)

        for i, (chunk, rnd, grad_cache) in enumerate(zip(chunks, rnd_states, cache_chunks)):
            is_last = i == len(chunks) - 1
            with self.fabric.no_backward_sync(model, enabled=not is_last):
                self.forward_backward(model, chunk, rnd, grad_cache)

        return loss

    def fit_epoch(
        self,
        model,
        train_loader: DataLoader,
        state: Dict[str, Any],
        temperature: float = 0.05,
        use_miner: bool = False,
        normalize: bool = True,
        lr_max_steps: int = 1000,
        grad_norm_clip: Optional[float] = None,
        log_interval: int = 1,
        checkpoint_interval: int = 10000,
        checkpoint_dir: str = "./checkpoints/",
        checkpoint_filter: Optional[Callable] = None,
    ):
        optimizer = state["optimizer"]
        scheduler = state.get("scheduler")
        iter_num = state.get("iter_num", 0)
        epoch_num = state.get("epoch_num", 0)

        self.fabric.print(f"Starting epoch {epoch_num} ({len(train_loader)} iters)")
        model.train()

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx < iter_num:
                continue
            total_steps = epoch_num * len(train_loader) + batch_idx
            if total_steps > lr_max_steps:
                break

            t0 = time.perf_counter()
            loss = self.train_step(model, batch, temperature, use_miner, normalize)

            if grad_norm_clip:
                self.fabric.clip_gradients(model, optimizer, max_norm=grad_norm_clip)

            optimizer.step()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()

            if batch_idx % log_interval == 0:
                lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]["lr"]
                self.fabric.log_dict(
                    {"con_loss": loss.item(), "lr": lr, "iter_time": time.perf_counter() - t0},
                    step=total_steps,
                )
                self.fabric.print(
                    f"Epoch {epoch_num} | Iter {batch_idx} | Loss: {loss.item():.4f} | LR: {lr:.2e}"
                )

            if batch_idx % checkpoint_interval == 0 or batch_idx == len(train_loader) - 1:
                ckpt_path = pathlib.Path(checkpoint_dir) / f"checkpoint_step{total_steps}.ckpt"
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                save_state = {
                    "model": model, "optimizer": optimizer, "scheduler": scheduler,
                    "iter_num": batch_idx + 1 if batch_idx < len(train_loader) - 1 else 0,
                    "epoch_num": epoch_num if batch_idx < len(train_loader) - 1 else epoch_num + 1,
                }
                if checkpoint_filter:
                    self.fabric.save(ckpt_path, save_state, filter={"model": checkpoint_filter})
                else:
                    self.fabric.save(ckpt_path, save_state)
                self.fabric.barrier()
                if self.fabric.global_rank == 0:
                    pathlib.Path(str(ckpt_path) + ".done").touch()
                self.fabric.print(f"Saved checkpoint: {ckpt_path}")

        return ckpt_path if "ckpt_path" in dir() else None
