"""Offline HRS evaluation script with checkpoint watcher.

Supports two evaluation modes and two execution modes:

Evaluation modes:
    embedder  -- retriever-only (TA1 S@8, TA2 EER via cosine similarity)
    reranker  -- full system (embedder retrieve + reranker rerank)

Execution modes:
    eval   -- evaluate a single checkpoint
    watch  -- poll a checkpoint directory and evaluate new checkpoints as they appear

Usage:
    # Single checkpoint (embedder)
    python -m authorship.evaluation.eval_hrs eval \\
        --mode embedder --config_dir outputs/embedder-v0 \\
        --checkpoint_path outputs/embedder-v0/checkpoint_0_1000.ckpt

    # Watcher (embedder, log to a dedicated new wandb eval run)
    python -m authorship.evaluation.eval_hrs watch \\
        --mode embedder --config_dir outputs/embedder-v0 \\
        --checkpoint_dir outputs/embedder-v0 \\
        --wandb_project authorship-embedder --wandb_log_to_new_run

    # Single checkpoint (reranker, full system)
    python -m authorship.evaluation.eval_hrs eval \\
        --mode reranker --config_dir outputs/reranker-v0 \\
        --checkpoint_path outputs/reranker-v0/checkpoint_0_2000.ckpt \\
        --embedder_config_dir outputs/embedder-v0 \\
        --embedder_checkpoint_path outputs/embedder-v0/best.ckpt
"""

import argparse
import json
import os
import re
import signal
import time
from glob import glob
from pathlib import Path
from typing import Callable, Dict, List, Optional
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from authorship.evaluation.constants import ALL_HRS_PATHS, GENRE_GROUPS
from authorship.evaluation.evaluator import Evaluator
from authorship.evaluation.hrs_loader import load_ta1, load_ta2
from authorship.model import AuthorshipModel


# Interrupt handling: set by SIGINT/SIGTERM; checked in genre loops for responsive Ctrl-C.
_interrupt_requested = False
_sigint_count = 0


# ---------------------------------------------------------------------------
# Model loading helper
# ---------------------------------------------------------------------------

def _build_model(
    mode: str,
    config_dir: str,
    checkpoint_path: str,
    embedder_config_dir: Optional[str] = None,
    embedder_checkpoint_path: Optional[str] = None,
    batch_size: int = 32,
    max_length: int = 512,
) -> AuthorshipModel:
    if mode == "embedder":
        return AuthorshipModel(
            embedder_config_path=os.path.join(config_dir, "config.yaml"),
            embedder_checkpoint_path=checkpoint_path,
            batch_size=batch_size,
            embedder_max_length=max_length,
        )
    return AuthorshipModel(
        embedder_config_path=os.path.join(embedder_config_dir, "config.yaml"),
        embedder_checkpoint_path=embedder_checkpoint_path,
        reranker_config_path=os.path.join(config_dir, "config.yaml"),
        reranker_checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        embedder_max_length=max_length,
    )


def _print_eval_setup(
    mode: str,
    *,
    config_dir: str,
    checkpoint_path: str,
    embedder_config_dir: Optional[str] = None,
    embedder_checkpoint_path: Optional[str] = None,
    genres: List[str],
    batch_size: int,
    max_length: int,
    top_k: Optional[int] = None,
    reranker_weight: Optional[float] = None,
) -> None:
    """Print the resolved evaluation setup before model loading."""
    lines = [
        "=" * 60,
        "Evaluation setup",
        f"  mode:                {mode}",
        f"  batch_size:          {batch_size}",
        f"  max_length:          {max_length}",
        f"  genres ({len(genres)}):" + (" " * (12 - len(str(len(genres))))) + ", ".join(genres),
    ]
    if mode == "embedder":
        lines += [
            f"  embedder_config:     {os.path.join(config_dir, 'config.yaml')}",
            f"  embedder_checkpoint: {checkpoint_path}",
        ]
    else:
        lines += [
            f"  embedder_config:     {os.path.join(embedder_config_dir, 'config.yaml')}",
            f"  embedder_checkpoint: {embedder_checkpoint_path}",
            f"  reranker_config:     {os.path.join(config_dir, 'config.yaml')}",
            f"  reranker_checkpoint: {checkpoint_path}",
            f"  top_k:               {top_k}",
            f"  reranker_weight:     {reranker_weight}",
        ]
    lines.append("=" * 60)
    print("\n" + "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Genre-level evaluation
# ---------------------------------------------------------------------------

def _get_ta2_data_paths(genre_paths: dict) -> dict:
    """Resolve TA2 file paths from the genre's TA2 data directory."""
    ta2 = genre_paths["TA2"]
    data_dir = ta2["data"]
    q_files = glob(os.path.join(data_dir, "*queries*"))
    c_files = glob(os.path.join(data_dir, "*candidates*"))
    if not q_files or not c_files:
        raise FileNotFoundError(f"TA2 data not found in {data_dir}")
    return {
        "query_path": q_files[0],
        "candidate_path": c_files[0],
        "ground_truth_path": ta2["ground_truth"],
        "ground_truth_query_labels_path": ta2["ground_truth_query_labels"],
        "ground_truth_candidate_labels_path": ta2["ground_truth_candidate_labels"],
    }


def evaluate_genre_embedder(
    model: AuthorshipModel,
    genre: str,
) -> Dict[str, float]:
    """Evaluate a single genre with the embedder (retriever-only)."""
    paths = ALL_HRS_PATHS[genre]
    evaluator = Evaluator()
    results: Dict[str, float] = {}

    # -- TA1: document-level retrieval --
    ta1 = paths["TA1"]
    queries_ta1, candidates_ta1, gt_positions = load_ta1(
        ta1["resample_queries"], ta1["resample_candidates"], ta1["ground_truth"],
    )
    scores_ta1 = model.retrieve(queries_ta1["text"], candidates_ta1["text"])["scores"]
    ta1_metrics = evaluator.evaluate_ta1(scores_ta1, gt_positions)
    s_at_8 = ta1_metrics.get("Average Success at 8", ta1_metrics.get("S@8", 0.0))
    results[f"S@8/{genre}"] = float(s_at_8)

    # -- TA2: author-level attribution --
    try:
        ta2_paths = _get_ta2_data_paths(paths)
        queries_ta2, candidates_ta2, gt_matrix = load_ta2(**ta2_paths)
        scores_ta2 = model.score_author_matrix(queries_ta2["text"], candidates_ta2["text"])
        ta2_metrics = evaluator.evaluate_ta2(scores_ta2, gt_matrix)
        eer = ta2_metrics.get("Equal Error Rate", ta2_metrics.get("EER", 1.0))
        results[f"EER/{genre}"] = float(eer)
    except (FileNotFoundError, KeyError) as e:
        print(f"  Skipping TA2 for {genre}: {e}")
    return results


def evaluate_genre_full_system(
    model: AuthorshipModel,
    genre: str,
    top_k: int = 16,
    reranker_weight: float = 0.5,
    compare_embedder: bool = True,
) -> Dict[str, float]:
    """Evaluate one genre: embedder-only and (if loaded) embedder + reranker."""
    paths = ALL_HRS_PATHS[genre]
    evaluator = Evaluator()
    results: Dict[str, float] = {}

    ta1 = paths["TA1"]
    queries_ta1, candidates_ta1, gt_positions = load_ta1(
        ta1["resample_queries"], ta1["resample_candidates"], ta1["ground_truth"],
    )
    n_q, n_c = len(queries_ta1), len(candidates_ta1)
    query_texts = list(queries_ta1["text"])
    candidate_texts = list(candidates_ta1["text"])

    if compare_embedder:
        print(f"    TA1 embedder: {n_q} queries × {n_c} candidates")
        scores_emb_ta1 = model.retrieve(query_texts, candidate_texts)["scores"]
        emb_ta1 = evaluator.evaluate_ta1(scores_emb_ta1, gt_positions)
        results[f"embedder_S@8/{genre}"] = float(
            emb_ta1.get("Average Success at 8", emb_ta1.get("S@8", 0.0))
        )

    if model._reranker is not None:
        print(f"    TA1 system: {n_q} queries × {n_c} candidates (top_k={top_k})")
        scores_sys_ta1 = model.reranker(
            query_texts, candidate_texts,
            top_k=top_k, reranker_weight=reranker_weight,
            progress_desc=f"{genre} TA1 rerank",
        )["scores"]
        sys_ta1 = evaluator.evaluate_ta1(scores_sys_ta1, gt_positions)
        results[f"system_S@8/{genre}"] = float(
            sys_ta1.get("Average Success at 8", sys_ta1.get("S@8", 0.0))
        )
    elif not compare_embedder:
        scores_ta1 = model.retrieve(query_texts, candidate_texts)["scores"]
        ta1_metrics = evaluator.evaluate_ta1(scores_ta1, gt_positions)
        results[f"ta1_S@8/{genre}"] = float(
            ta1_metrics.get("Average Success at 8", ta1_metrics.get("S@8", 0.0))
        )

    try:
        ta2_paths = _get_ta2_data_paths(paths)
        queries_ta2, candidates_ta2, gt_matrix = load_ta2(**ta2_paths)
        q_authors = queries_ta2["text"]
        c_authors = candidates_ta2["text"]
        print(
            f"    TA2: {len(queries_ta2)} query authors × {len(candidates_ta2)} candidates"
            f" (top_k={top_k})"
        )

        if compare_embedder:
            scores_emb_ta2 = model.score_author_matrix(
                q_authors, c_authors, use_reranker=False,
            )
            emb_ta2 = evaluator.evaluate_ta2(scores_emb_ta2, gt_matrix)
            results[f"embedder_EER/{genre}"] = float(
                emb_ta2.get("Equal Error Rate", emb_ta2.get("EER", 1.0))
            )

        if model._reranker is not None:
            scores_sys_ta2 = model.score_author_matrix(
                q_authors, c_authors,
                top_k=top_k, reranker_weight=reranker_weight, use_reranker=True,
            )
            sys_ta2 = evaluator.evaluate_ta2(scores_sys_ta2, gt_matrix)
            results[f"system_EER/{genre}"] = float(
                sys_ta2.get("Equal Error Rate", sys_ta2.get("EER", 1.0))
            )
    except (FileNotFoundError, KeyError) as e:
        print(f"  Skipping TA2 for {genre}: {e}")

    return results


def _print_embedder_vs_system_summary(results: Dict[str, float], genres: List[str]) -> None:
    """Print side-by-side embedder vs embedder+reranker metrics."""
    evaluated = [g for g in genres if f"embedder_S@8/{g}" in results or f"system_S@8/{g}" in results]
    if not evaluated:
        return

    print("\n" + "=" * 72)
    print("Embedder-only vs Embedder + Reranker")
    print("=" * 72)
    print(f"{'Genre':<28} {'Emb S@8':>10} {'Sys S@8':>10} {'Δ S@8':>8} {'Emb EER':>10} {'Sys EER':>10} {'Δ EER':>8}")
    print("-" * 72)

    for genre in evaluated:
        emb_s = results.get(f"embedder_S@8/{genre}")
        sys_s = results.get(f"system_S@8/{genre}")
        emb_e = results.get(f"embedder_EER/{genre}")
        sys_e = results.get(f"system_EER/{genre}")

        def _cell(v: Optional[float]) -> str:
            return f"{v:>10.4f}" if v is not None else f"{'—':>10}"

        delta_s = (sys_s - emb_s) if emb_s is not None and sys_s is not None else None
        delta_e = (sys_e - emb_e) if emb_e is not None and sys_e is not None else None

        ds = f"{delta_s:>+8.4f}" if delta_s is not None else f"{'—':>8}"
        de = f"{delta_e:>+8.4f}" if delta_e is not None else f"{'—':>8}"
        print(
            f"{genre:<28} {_cell(emb_s)} {_cell(sys_s)} {ds} "
            f"{_cell(emb_e)} {_cell(sys_e)} {de}"
        )

    def _avg(key_tpl: str) -> Optional[float]:
        vals = [results[f"{key_tpl}/{g}"] for g in evaluated if f"{key_tpl}/{g}" in results]
        return float(np.mean(vals)) if vals else None

    emb_s, sys_s = _avg("embedder_S@8"), _avg("system_S@8")
    emb_e, sys_e = _avg("embedder_EER"), _avg("system_EER")
    print("-" * 72)
    if emb_s is not None or sys_s is not None:
        ds = f"{(sys_s - emb_s):>+8.4f}" if emb_s is not None and sys_s is not None else f"{'—':>8}"
        de = f"{(sys_e - emb_e):>+8.4f}" if emb_e is not None and sys_e is not None else f"{'—':>8}"
        print(
            f"{'MEAN':<28} {_cell(emb_s)} {_cell(sys_s)} {ds} {_cell(emb_e)} {_cell(sys_e)} {de}"
        )
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Top-level evaluation drivers
# ---------------------------------------------------------------------------

def _compute_averages(results: Dict[str, float], genres: List[str], prefix: str = "") -> Dict[str, float]:
    """Add group averages for each GENRE_GROUP that overlaps with evaluated genres."""
    avgs: Dict[str, float] = {}
    s_key = f"{prefix}S@8" if prefix else "S@8"
    e_key = f"{prefix}EER" if prefix else "EER"

    for group_name, group_genres in GENRE_GROUPS.items():
        overlap = [g for g in group_genres if g in genres]
        if not overlap:
            continue
        s8_vals = [results[f"{s_key}/{g}"] for g in overlap if f"{s_key}/{g}" in results]
        eer_vals = [results[f"{e_key}/{g}"] for g in overlap if f"{e_key}/{g}" in results]
        if s8_vals:
            avgs[f"Avg/{group_name}_{s_key}"] = float(np.mean(s8_vals))
        if eer_vals:
            avgs[f"Avg/{group_name}_{e_key}"] = float(np.mean(eer_vals))
    return avgs


def evaluate_embedder(
    config_dir: str,
    checkpoint_path: str,
    genres: List[str],
    batch_size: int = 32,
    max_length: int = 512,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Dict[str, float]:
    """Run embedder-only evaluation across all requested genres."""
    _print_eval_setup(
        "embedder", config_dir=config_dir, checkpoint_path=checkpoint_path,
        genres=genres, batch_size=batch_size, max_length=max_length,
    )
    print(f"Loading embedder from {checkpoint_path}...")
    model = _build_model("embedder", config_dir, checkpoint_path, batch_size=batch_size, max_length=max_length)
    results: Dict[str, float] = {}
    evaluated_genres: List[str] = []

    for genre in genres:
        if cancel_check and cancel_check():
            print("  Interrupt requested, stopping evaluation.")
            break
        if genre not in ALL_HRS_PATHS:
            print(f"  Skipping unknown genre: {genre}")
            continue
        print(f"  Evaluating {genre}...")
        genre_results = evaluate_genre_embedder(model, genre)
        results.update(genre_results)
        evaluated_genres.append(genre)
        for k, v in sorted(genre_results.items()):
            print(f"    {k}: {v:.4f}")

    results.update(_compute_averages(results, evaluated_genres))
    del model
    torch.cuda.empty_cache()
    return results


def evaluate_full_system(
    config_dir: str,
    checkpoint_path: str,
    embedder_config_dir: str,
    embedder_checkpoint_path: str,
    genres: List[str],
    top_k: int = 16,
    reranker_weight: float = 0.5,
    batch_size: int = 32,
    max_length: int = 512,
    cancel_check: Optional[Callable[[], bool]] = None,
    compare_embedder: bool = True,
) -> Dict[str, float]:
    """Run embedder-only and embedder+reranker evaluation (single model load)."""
    _print_eval_setup(
        "reranker", config_dir=config_dir, checkpoint_path=checkpoint_path,
        embedder_config_dir=embedder_config_dir,
        embedder_checkpoint_path=embedder_checkpoint_path,
        genres=genres, batch_size=batch_size, max_length=max_length,
        top_k=top_k, reranker_weight=reranker_weight,
    )
    print(f"Loading embedder from {embedder_checkpoint_path}...")
    print(f"Loading reranker from {checkpoint_path}...")
    model = _build_model(
        "reranker", config_dir, checkpoint_path,
        embedder_config_dir=embedder_config_dir,
        embedder_checkpoint_path=embedder_checkpoint_path,
        batch_size=batch_size, max_length=max_length,
    )
    results: Dict[str, float] = {}
    evaluated_genres: List[str] = []

    genre_bar = tqdm(genres, desc="HRS genres", unit="genre")
    for genre in genre_bar:
        if cancel_check and cancel_check():
            print("  Interrupt requested, stopping evaluation.")
            break
        if genre not in ALL_HRS_PATHS:
            print(f"  Skipping unknown genre: {genre}")
            continue
        genre_bar.set_postfix_str(genre)
        print(f"  Evaluating {genre}...")
        genre_results = evaluate_genre_full_system(
            model, genre, top_k, reranker_weight, compare_embedder=compare_embedder,
        )
        results.update(genre_results)
        evaluated_genres.append(genre)
        for k, v in sorted(genre_results.items()):
            print(f"    {k}: {v:.4f}")

    if compare_embedder:
        results.update(_compute_averages(results, evaluated_genres, prefix="embedder_"))
        results.update(_compute_averages(results, evaluated_genres, prefix="system_"))
        _print_embedder_vs_system_summary(results, evaluated_genres)
    else:
        results.update(_compute_averages(results, evaluated_genres, prefix="ta1_"))
        results.update(_compute_averages(results, evaluated_genres, prefix="ta2_"))
    del model
    torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def _save_results(results: Dict, output_dir: str, name: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {path}")


# ---------------------------------------------------------------------------
# wandb helpers
# ---------------------------------------------------------------------------

def _init_wandb(
    wandb_run_id: Optional[str],
    wandb_project: Optional[str],
    *,
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_group: Optional[str] = None,
    wandb_log_to_new_run: bool = False,
):
    if not wandb_run_id and not wandb_project:
        return None
    try:
        import wandb
    except ImportError:
        print("wandb not installed, skipping wandb logging.")
        return None
    kwargs = {}
    if wandb_project:
        kwargs["project"] = wandb_project
    if wandb_entity:
        kwargs["entity"] = wandb_entity
    if wandb_run_name:
        kwargs["name"] = wandb_run_name
    if wandb_group:
        kwargs["group"] = wandb_group

    if wandb_run_id and not wandb_log_to_new_run:
        kwargs["id"] = wandb_run_id
        kwargs["resume"] = "must"
    else:
        kwargs["job_type"] = "eval"
    run = wandb.init(**kwargs)
    wandb.define_metric("eval_step")
    wandb.define_metric("S@8/*", step_metric="eval_step")
    wandb.define_metric("EER/*", step_metric="eval_step")
    wandb.define_metric("ta1_S@8/*", step_metric="eval_step")
    wandb.define_metric("ta2_EER/*", step_metric="eval_step")
    wandb.define_metric("Avg/*", step_metric="eval_step")
    return run


def _wandb_log(results: Dict[str, float], step: Optional[int]):
    """Log eval results to wandb using a dedicated eval_step axis."""
    import wandb
    data = dict(results)
    if step is not None:
        data["eval_step"] = step
    wandb.log(data)


# ---------------------------------------------------------------------------
# Checkpoint step parsing
# ---------------------------------------------------------------------------

_CKPT_STEP_PATTERN = re.compile(r"checkpoint_step(\d+)")


def _parse_step_from_filename(filename: str) -> Optional[int]:
    """Extract the global training step from the checkpoint filename.

    Supports ``checkpoint_step{N}.ckpt`` format.
    """
    m = _CKPT_STEP_PATTERN.search(filename)
    if m:
        return int(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Single eval dispatcher
# ---------------------------------------------------------------------------

def _run_single_eval(
    args,
    checkpoint_path: str,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Dict[str, float]:
    if args.mode == "embedder":
        results = evaluate_embedder(
            config_dir=args.config_dir,
            checkpoint_path=checkpoint_path,
            genres=args.genres,
            batch_size=args.batch_size,
            max_length=args.max_length,
            cancel_check=cancel_check,
        )
    else:
        results = evaluate_full_system(
            config_dir=args.config_dir,
            checkpoint_path=checkpoint_path,
            embedder_config_dir=args.embedder_config_dir,
            embedder_checkpoint_path=args.embedder_checkpoint_path,
            genres=args.genres,
            top_k=getattr(args, "top_k", 16),
            reranker_weight=getattr(args, "reranker_weight", 0.5),
            batch_size=args.batch_size,
            max_length=args.max_length,
            cancel_check=cancel_check,
            compare_embedder=not getattr(args, "no_compare_embedder", False),
        )
    return results


# ---------------------------------------------------------------------------
# Watch mode
# ---------------------------------------------------------------------------

def _is_checkpoint_ready(ckpt_path: Path) -> bool:
    """A checkpoint is ready when its ``.done`` sentinel file exists.
    """
    done_path = Path(str(ckpt_path) + ".done")
    return done_path.exists()


def watch_checkpoints(args):
    """Poll checkpoint_dir for new .ckpt files and evaluate them.
    """
    ckpt_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tracker_path = out_dir / "evaluated_checkpoints.json"
    if tracker_path.exists():
        evaluated = set(json.loads(tracker_path.read_text()))
    else:
        evaluated = set()

    wb_run = _init_wandb(
        getattr(args, "wandb_run_id", None),
        getattr(args, "wandb_project", None),
        wandb_entity=getattr(args, "wandb_entity", None),
        wandb_run_name=getattr(args, "wandb_run_name", None),
        wandb_group=getattr(args, "wandb_group", None),
        wandb_log_to_new_run=getattr(args, "wandb_log_to_new_run", False),
    )

    stop = False

    def _handle_signal(signum, frame):
        nonlocal stop
        global _sigint_count, _interrupt_requested
        _sigint_count += 1
        _interrupt_requested = True
        stop = True
        if _sigint_count >= 2:
            print("\nSecond interrupt, exiting immediately.")
            os._exit(1)
        print("\nInterrupt requested, finishing current genre then stopping...")

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    print(f"Watching {ckpt_dir} for new checkpoints (poll every {args.poll_interval}s)...")
    try:
        while not stop:
            ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
            new_ckpts = [c for c in ckpts if str(c) not in evaluated]

            for ckpt in new_ckpts:
                if stop:
                    break
                if not _is_checkpoint_ready(ckpt):
                    continue

                step = _parse_step_from_filename(ckpt.name)
                print(f"\n{'='*60}")
                print(f"New checkpoint: {ckpt.name} (step={step})")
                print(f"{'='*60}")

                results = _run_single_eval(args, str(ckpt), cancel_check=lambda: stop)
                _save_results(results, args.output_dir, ckpt.stem)

                if wb_run:
                    _wandb_log(results, step)

                evaluated.add(str(ckpt))
                tracker_path.write_text(json.dumps(sorted(evaluated), indent=2))

                print("\nSummary:")
                for k, v in sorted(results.items()):
                    print(f"  {k}: {v:.4f}")

            if not stop:
                time.sleep(args.poll_interval)
    finally:
        if wb_run:
            import wandb
            wandb.finish()
        print("Watcher stopped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument("--mode", choices=["embedder", "reranker"], required=True)
    parser.add_argument("--config_dir", required=True, help="Directory containing config.yaml")
    parser.add_argument("--output_dir", default="eval_results")
    parser.add_argument("--genres", nargs="+", default=None,
                        help="HRS genres to evaluate (default: read from config or HRS3 en+zh)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument(
        "--wandb_run_id",
        default=None,
        help="Existing wandb run id to resume for logging (ignored with --wandb_log_to_new_run)",
    )
    parser.add_argument("--wandb_project", default=None, help="wandb project name")
    parser.add_argument("--wandb_entity", default=None, help="wandb entity/user or org")
    parser.add_argument("--wandb_run_name", default=None, help="Set wandb run name (useful for eval runs)")
    parser.add_argument("--wandb_group", default=None, help="Set wandb group (e.g., training run id)")
    parser.add_argument(
        "--wandb_log_to_new_run",
        action="store_true",
        help="Log eval results to a NEW wandb run instead of resuming --wandb_run_id",
    )
    # reranker-specific
    parser.add_argument("--embedder_config_dir", default=None)
    parser.add_argument("--embedder_checkpoint_path", default=None)
    parser.add_argument("--top_k", type=int, default=16)
    parser.add_argument("--reranker_weight", type=float, default=0.5)
    parser.add_argument(
        "--no_compare_embedder",
        action="store_true",
        help="Reranker mode only: skip embedder-only baseline (system metrics only)",
    )


def _resolve_genres(args):
    """Resolve genre list from CLI args or config file."""
    if args.genres:
        return args.genres
    cfg_path = os.path.join(args.config_dir, "config.yaml")
    if os.path.exists(cfg_path):
        cfg = OmegaConf.load(cfg_path)
        eval_cfg = cfg.get("evaluation", {})
        if eval_cfg and "genres" in eval_cfg:
            return list(eval_cfg["genres"])
    return [
        "HRS3.1", "HRS3.2", "HRS3.3", "HRS3.4", "HRS3.5", "en_cross_genre_short",
        "HRS3.301", "HRS3.302", "HRS3.303", "HRS3.304", "HRS3.305",
        "zh_cross_genre_medium", "zh_cross_genre_short",
    ]


def main():
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser(description="HRS evaluation for authorship models")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- eval subcommand --
    eval_parser = subparsers.add_parser("eval", help="Evaluate a single checkpoint")
    _add_common_args(eval_parser)
    eval_parser.add_argument("--checkpoint_path", required=True)

    # -- watch subcommand --
    watch_parser = subparsers.add_parser("watch", help="Watch checkpoint dir and auto-evaluate")
    _add_common_args(watch_parser)
    watch_parser.add_argument("--checkpoint_dir", required=True, help="Directory to watch for .ckpt files")
    watch_parser.add_argument("--poll_interval", type=float, default=60, help="Seconds between polls")

    args = parser.parse_args()
    args.genres = _resolve_genres(args)

    if args.mode == "reranker":
        if not args.embedder_config_dir or not args.embedder_checkpoint_path:
            parser.error("--embedder_config_dir and --embedder_checkpoint_path required for reranker mode")

    print(f"Mode: {args.mode} | Genres: {args.genres}")

    if args.command == "eval":
        global _interrupt_requested, _sigint_count
        _interrupt_requested = False
        _sigint_count = 0

        def _eval_signal_handler(signum, frame):
            global _sigint_count, _interrupt_requested
            _sigint_count += 1
            _interrupt_requested = True
            if _sigint_count >= 2:
                print("\nSecond interrupt, exiting immediately.")
                os._exit(1)
            print("\nInterrupt requested, finishing current genre then stopping...")

        signal.signal(signal.SIGINT, _eval_signal_handler)
        signal.signal(signal.SIGTERM, _eval_signal_handler)

        wb_run = _init_wandb(
            getattr(args, "wandb_run_id", None),
            getattr(args, "wandb_project", None),
            wandb_entity=getattr(args, "wandb_entity", None),
            wandb_run_name=getattr(args, "wandb_run_name", None),
            wandb_group=getattr(args, "wandb_group", None),
            wandb_log_to_new_run=getattr(args, "wandb_log_to_new_run", False),
        )
        try:
            results = _run_single_eval(
                args, args.checkpoint_path, cancel_check=lambda: _interrupt_requested
            )
            _save_results(results, args.output_dir, Path(args.checkpoint_path).stem)
            if wb_run:
                import wandb
                step = _parse_step_from_filename(Path(args.checkpoint_path).name)
                _wandb_log(results, step)
        finally:
            if wb_run:
                import wandb
                wandb.finish()
        print("\nFinal results:")
        for k, v in sorted(results.items()):
            print(f"  {k}: {v:.4f}")

    elif args.command == "watch":
        watch_checkpoints(args)


if __name__ == "__main__":
    main()
