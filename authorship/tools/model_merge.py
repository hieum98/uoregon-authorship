"""Model merge helpers using raw PyTorch checkpoints with mergekit-pytorch."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml


def _extract_state_dict(ckpt_obj) -> Dict[str, torch.Tensor]:
    """Extract a flat tensor state_dict from common checkpoint layouts."""
    if isinstance(ckpt_obj, dict):
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
        if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            return ckpt_obj["model"]

        # Raw state_dict case: tensor-valued dict.
        if ckpt_obj and all(torch.is_tensor(v) for v in ckpt_obj.values()):
            return ckpt_obj

    raise ValueError("Unsupported checkpoint format. Expected raw state_dict or keys: state_dict/model")


def _parse_list_arg(raw: str) -> List:
    """Parse list args from JSON string."""
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON list: {raw}") from exc
    if not isinstance(value, list):
        raise ValueError(f"Expected JSON list, got: {type(value).__name__}")
    return value


def build_mergekit_pytorch_command(config_path: str, output_dir: str, cuda: bool = True) -> List[str]:
    """Build mergekit-pytorch CLI command."""
    cmd = ["mergekit-pytorch", config_path, output_dir]
    if cuda:
        cmd.append("--cuda")
    return cmd


def build_raw_pytorch_config(
    model_paths: List[str],
    weights: List[float],
    output_yaml: str,
    base_model: Optional[str] = None,
    merge_method: str = "linear",
    density: Optional[List[float]] = None,
    normalize: bool = True,
    dtype: str = "float32",
) -> str:
    """Create a raw PyTorch merge config file for mergekit-pytorch."""
    if len(model_paths) != len(weights):
        raise ValueError("model_paths and weights must have same length")
    if not model_paths:
        raise ValueError("At least one model path is required")

    models = []
    for path, weight in zip(model_paths, weights):
        params = {"weight": float(weight)}
        if density is not None:
            params["density"] = density
        models.append({"model": str(path), "parameters": params})

    cfg = {
        "merge_method": merge_method,
        "models": models,
        "parameters": {"normalize": bool(normalize)},
        "dtype": dtype,
    }
    if base_model:
        cfg["base_model"] = str(base_model)

    output = Path(output_yaml)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return str(output)


def _validate_raw_model_paths(model_paths: List[str]) -> None:
    allowed_suffixes = {".pt", ".ckpt", ".bin", ".safetensors"}
    for path in model_paths:
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Model path does not exist: {path}")
        if not path_obj.is_file():
            raise ValueError(f"Raw-PyTorch merge expects files, got: {path}")
        if path_obj.suffix.lower() not in allowed_suffixes:
            raise ValueError(
                f"Unsupported model format: {path}. Expected one of {sorted(allowed_suffixes)}"
            )


def _save_model_only_checkpoint(src_path: str, dst_path: str) -> str:
    """Write a model-only checkpoint (flat tensor state_dict) for mergekit-pytorch."""
    ckpt_obj = torch.load(src_path, map_location="cpu")
    state = _extract_state_dict(ckpt_obj)
    model_only_state = {k: v.detach().cpu().contiguous() for k, v in state.items()}
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_only_state, dst_path)
    return dst_path


def _prepare_model_paths(model_paths: List[str], prepared_dir: str) -> List[str]:
    """Normalize checkpoints into model-only files mergekit-pytorch can ingest."""
    prepared = []
    for path in model_paths:
        suffix = Path(path).suffix.lower()
        if suffix == ".safetensors":
            prepared.append(path)
            continue

        out_name = f"{Path(path).stem}.model_only.pt"
        out_path = str(Path(prepared_dir) / out_name)
        prepared.append(_save_model_only_checkpoint(path, out_path))
    return prepared


def main():
    parser = argparse.ArgumentParser(description="Build/run mergekit-pytorch merge from raw checkpoints.")
    parser.add_argument("--models", required=True, help='JSON list of model paths, e.g. \'["a.pt","b.pt"]\'')
    parser.add_argument("--weights", required=True, help='JSON list of weights, e.g. "[0.6,0.4]"')
    parser.add_argument("--output-dir", required=True, help="Output directory for merged model")
    parser.add_argument("--config-path", required=True, help="Path to write mergekit YAML")
    parser.add_argument("--base-model", default=None, help="Optional base model checkpoint path")
    parser.add_argument("--merge-method", default="linear", help="mergekit merge_method (default: linear)")
    parser.add_argument(
        "--density",
        default=None,
        help='Optional JSON list for density gradient, e.g. "[1,0.7,0.1]"',
    )
    parser.add_argument("--dtype", default="float32", help="Output dtype in mergekit config")
    parser.add_argument("--no-normalize", action="store_true", help="Disable weight normalization")
    parser.add_argument(
        "--prepared-dir",
        default=None,
        help="Directory to write model-only checkpoints (default: <output-dir>/prepared)",
    )
    parser.add_argument(
        "--run-merge",
        action="store_true",
        help="Actually execute mergekit-pytorch command. Default: only print command.",
    )
    parser.add_argument("--no-cuda", action="store_true", help="Do not pass --cuda to mergekit-pytorch")
    args = parser.parse_args()

    models = _parse_list_arg(args.models)
    weights = _parse_list_arg(args.weights)
    density = _parse_list_arg(args.density) if args.density is not None else None
    _validate_raw_model_paths(models)

    prepared_dir = args.prepared_dir or str(Path(args.output_dir) / "prepared")
    prepared_models = _prepare_model_paths(models, prepared_dir)
    prepared_base_model = None
    if args.base_model:
        _validate_raw_model_paths([args.base_model])
        prepared_base_model = _prepare_model_paths([args.base_model], prepared_dir)[0]

    config_path = build_raw_pytorch_config(
        model_paths=prepared_models,
        weights=weights,
        output_yaml=args.config_path,
        base_model=prepared_base_model,
        merge_method=args.merge_method,
        density=density,
        normalize=not args.no_normalize,
        dtype=args.dtype,
    )
    cmd = build_mergekit_pytorch_command(config_path, args.output_dir, cuda=not args.no_cuda)

    print("Prepared mergekit config:", config_path)
    print("Prepared mergekit command:")
    print(" ".join(cmd))

    if args.run_merge:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
