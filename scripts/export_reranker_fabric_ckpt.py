#!/usr/bin/env python3
"""Export a Fabric/FSDP reranker checkpoint to a flat model-only .pt for inference.

Training saves sharded checkpoints under e.g. outputs/reranker-v1/checkpoint_step2000.ckpt/
(distcp shards + meta.pt). This script consolidates them on CPU (no GPUs required) using
Lightning's distributed-checkpoint loader, then writes {"model": state_dict} for eval_hrs.

Usage:
  conda activate hiatus-phase3
  python scripts/export_reranker_fabric_ckpt.py \\
    --checkpoint outputs/reranker-v1/checkpoint_step2000.ckpt \\
    --output outputs/reranker-v1/exported/checkpoint_step2000.pt

Alternative CLI (writes .ckpt.consolidated, then extract model manually):
  fabric consolidate outputs/reranker-v1/checkpoint_step2000.ckpt
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_3
    from lightning.fabric.utilities.load import _METADATA_FILENAME, _load_distributed_checkpoint
except ImportError as exc:
    raise SystemExit(
        "lightning is required for FSDP checkpoint consolidation.\n"
        "  conda activate hiatus-phase3 && pip install lightning"
    ) from exc


def _strip_fabric_model_prefix(state_dict: dict) -> dict:
    """``model.model.*`` -> ``model.*`` for RerankerModel.load_state_dict."""
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if keys and all(k.startswith("model.model.") for k in keys):
        return {k[len("model.") :]: v for k, v in state_dict.items()}
    return state_dict


def _extract_model_state_dict(model_obj) -> dict:
    """Normalize checkpoint 'model' entry to a flat state_dict."""
    if isinstance(model_obj, dict):
        return model_obj
    if hasattr(model_obj, "state_dict"):
        return model_obj.state_dict()
    raise TypeError(f"Unsupported model entry type: {type(model_obj)}")


def export_checkpoint(checkpoint_path: str, output_path: str) -> None:
    ckpt_dir = pathlib.Path(checkpoint_path)
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Expected checkpoint directory: {checkpoint_path}")
    if not (ckpt_dir / _METADATA_FILENAME).is_file():
        raise FileNotFoundError(
            f"Not a Fabric FSDP checkpoint (missing {_METADATA_FILENAME}): {checkpoint_path}"
        )
    if not _TORCH_GREATER_EQUAL_2_3:
        raise RuntimeError("PyTorch >= 2.3 is required to consolidate distributed checkpoints")

    print(f"Consolidating {ckpt_dir} on CPU (may take a few minutes)...")
    state = _load_distributed_checkpoint(ckpt_dir)
    if "model" not in state:
        raise KeyError(f"Checkpoint has no 'model' key; found keys: {sorted(state.keys())}")

    model_state = _extract_model_state_dict(state["model"])
    model_state = _strip_fabric_model_prefix(model_state)
    out = pathlib.Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model_state}, out)
    print(f"Exported model weights to {out} ({len(model_state)} tensors)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Fabric checkpoint directory (checkpoint_step{N}.ckpt/)",
    )
    parser.add_argument("--output", required=True, help="Output .pt path for inference")
    parser.add_argument(
        "--config",
        default=None,
        help="Deprecated, ignored (kept for compatibility with evaluate_reranker_v1_all.sh)",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=None,
        help="Deprecated, ignored (consolidation is CPU-only)",
    )
    args = parser.parse_args()
    export_checkpoint(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
