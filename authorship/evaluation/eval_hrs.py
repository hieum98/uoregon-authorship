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

    # Watcher (embedder, logs to same wandb run as training)
    python -m authorship.evaluation.eval_hrs watch \\
        --mode embedder --config_dir outputs/embedder-v0 \\
        --checkpoint_dir outputs/embedder-v0 \\
        --wandb_run_id <run_id> --wandb_project authorship-embedder

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
from tqdm import tqdm
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics.pairwise import cosine_similarity

from authorship.evaluation.constants import ALL_HRS_PATHS, GENRE_GROUPS
from authorship.evaluation.evaluator import Evaluator
from authorship.evaluation.hrs_loader import load_ta1, load_ta2
from authorship.models.embedder import WrappedEmbeddingModel
from authorship.models.reranker import WrappedRerankerModel


# Interrupt handling: set by SIGINT/SIGTERM; checked in genre loops for responsive Ctrl-C.
_interrupt_requested = False
_sigint_count = 0


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def _load_embedder(config_dir: str, checkpoint_path: str) -> WrappedEmbeddingModel:
    cfg = OmegaConf.load(os.path.join(config_dir, "config.yaml"))
    return WrappedEmbeddingModel(
        model_name_or_path=cfg.model.model_name_or_path,
        use_lora=cfg.model.lora.use_lora,
        dropout_prob=cfg.model.get("dropout", 0.1),
        lora_r=cfg.model.lora.get("r", 16),
        lora_alpha=cfg.model.lora.get("alpha", 32),
        lora_dropout=cfg.model.lora.get("dropout", 0.1),
        target_modules=list(cfg.model.lora.get("target_modules", ["all"])),
        adapter_name=cfg.model.lora.get("name"),
        quantization=cfg.model.get("quantization", False),
        attn_implementation=cfg.model.get("attn_implementation"),
        pooling_method=cfg.model.get("pooling", "mean"),
        is_bidirectional=cfg.model.get("is_bidirectional", False),
        model_checkpoint=checkpoint_path,
    )


def _load_reranker(config_dir: str, checkpoint_path: str) -> WrappedRerankerModel:
    cfg = OmegaConf.load(os.path.join(config_dir, "config.yaml"))
    return WrappedRerankerModel(
        model_name_or_path=cfg.model.model_name_or_path,
        use_lora=cfg.model.lora.use_lora,
        lora_r=cfg.model.lora.get("r", 16),
        lora_alpha=cfg.model.lora.get("alpha", 32),
        lora_dropout=cfg.model.lora.get("dropout", 0.1),
        target_modules=list(cfg.model.lora.get("target_modules", ["all"])),
        adapter_name=cfg.model.lora.get("name"),
        quantization=cfg.model.get("quantization", False),
        attn_implementation=cfg.model.get("attn_implementation"),
        model_checkpoint=checkpoint_path,
        max_length=cfg.data.get("max_seq_length", 1024),
        batch_size=cfg.data.get("global_batch_size", 8),
        instruction=cfg.data.get("instruction"),
    )


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def _encode_texts(model: WrappedEmbeddingModel, texts: List[str], batch_size: int, max_length: int) -> np.ndarray:
    embs = model.batch_encode(texts, max_length=max_length, batch_size=batch_size)
    out = embs.cpu().float().numpy()
    if np.isnan(out).any():
        print("Warning: NaN in embeddings, replacing with zeros.")
        out = np.nan_to_num(out)
    return out


def _encode_authors(model: WrappedEmbeddingModel, author_texts: List[List[str]], batch_size: int, max_length: int) -> np.ndarray:
    """Mean-pool per-author document embeddings."""
    author_embs = []
    for docs in tqdm(author_texts, desc="Encoding authors", disable=len(author_texts) < 256):
        embs = model.batch_encode(docs, max_length=max_length, batch_size=batch_size)
        author_embs.append(embs.cpu().float().mean(dim=0).numpy())
    out = np.stack(author_embs)
    if np.isnan(out).any():
        print("Warning: NaN in author embeddings, replacing with zeros.")
        out = np.nan_to_num(out)
    return out


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
    model: WrappedEmbeddingModel,
    genre: str,
    batch_size: int = 32,
    max_length: int = 512,
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
    with torch.autocast("cuda", dtype=torch.bfloat16):
        q_embs = _encode_texts(model, queries_ta1["text"], batch_size, max_length)
        c_embs = _encode_texts(model, candidates_ta1["text"], batch_size, max_length)
    scores_ta1 = cosine_similarity(q_embs, c_embs)
    ta1_metrics = evaluator.evaluate_ta1(scores_ta1, gt_positions)
    s_at_8 = ta1_metrics.get("Average Success at 8", ta1_metrics.get("S@8", 0.0))
    results[f"S@8/{genre}"] = float(s_at_8)

    # -- TA2: author-level attribution --
    try:
        ta2_paths = _get_ta2_data_paths(paths)
        queries_ta2, candidates_ta2, gt_matrix = load_ta2(**ta2_paths)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            q_author_embs = _encode_authors(model, queries_ta2["text"], batch_size, max_length)
            c_author_embs = _encode_authors(model, candidates_ta2["text"], batch_size, max_length)
        scores_ta2 = cosine_similarity(q_author_embs, c_author_embs)
        ta2_metrics = evaluator.evaluate_ta2(scores_ta2, gt_matrix)
        eer = ta2_metrics.get("Equal Error Rate", ta2_metrics.get("EER", 1.0))
        results[f"EER/{genre}"] = float(eer)
    except (FileNotFoundError, KeyError) as e:
        print(f"  Skipping TA2 for {genre}: {e}")
    return results


def evaluate_genre_full_system(
    embedder: WrappedEmbeddingModel,
    reranker: WrappedRerankerModel,
    genre: str,
    top_k: int = 16,
    reranker_weight: float = 0.5,
    batch_size: int = 32,
    max_length: int = 512,
) -> Dict[str, float]:
    """Evaluate a single genre with the full retriever + reranker pipeline."""
    paths = ALL_HRS_PATHS[genre]
    evaluator = Evaluator()
    results: Dict[str, float] = {}
    w = reranker_weight

    # -- TA1: retrieve then rerank --
    ta1 = paths["TA1"]
    queries_ta1, candidates_ta1, gt_positions = load_ta1(
        ta1["resample_queries"], ta1["resample_candidates"], ta1["ground_truth"],
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        q_embs = _encode_texts(embedder, queries_ta1["text"], batch_size, max_length)
        c_embs = _encode_texts(embedder, candidates_ta1["text"], batch_size, max_length)
    retriever_scores = cosine_similarity(q_embs, c_embs).astype(np.float32)

    reranked = np.copy(retriever_scores)
    n_q = retriever_scores.shape[0]
    for i in range(n_q):
        top_indices = np.argsort(-retriever_scores[i])[:top_k]
        q_texts = [queries_ta1["text"][i]] * len(top_indices)
        c_texts = [candidates_ta1["text"][int(j)] for j in top_indices]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            rr_scores = reranker.score_pairs(q_texts, c_texts)
        for local_idx, global_idx in enumerate(top_indices):
            reranked[i, global_idx] = w * rr_scores[local_idx] + (1.0 - w) * retriever_scores[i, global_idx]

    ta1_metrics = evaluator.evaluate_ta1(reranked, gt_positions)
    s_at_8 = ta1_metrics.get("Average Success at 8", ta1_metrics.get("S@8", 0.0))
    results[f"ta1_S@8/{genre}"] = float(s_at_8)

    # -- TA2: author-level retrieve then rerank --
    try:
        ta2_paths = _get_ta2_data_paths(paths)
        queries_ta2, candidates_ta2, gt_matrix = load_ta2(**ta2_paths)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            q_author_embs = _encode_authors(embedder, queries_ta2["text"], batch_size, max_length)
            c_author_embs = _encode_authors(embedder, candidates_ta2["text"], batch_size, max_length)
        author_retriever = cosine_similarity(q_author_embs, c_author_embs).astype(np.float32)

        author_reranked = np.copy(author_retriever)
        n_qa = author_retriever.shape[0]
        for i in range(n_qa):
            top_indices = np.argsort(-author_retriever[i])[:top_k]
            for j in top_indices:
                q_docs = queries_ta2["text"][i]
                c_docs = candidates_ta2["text"][int(j)]
                pair_q = [qd for qd in q_docs for _ in c_docs]
                pair_c = [cd for _ in q_docs for cd in c_docs]
                if pair_q:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        pair_scores = reranker.score_pairs(pair_q, pair_c)
                    rr = float(pair_scores.max())
                else:
                    rr = 0.0
                author_reranked[i, int(j)] = w * rr + (1.0 - w) * author_retriever[i, int(j)]

        ta2_metrics = evaluator.evaluate_ta2(author_reranked, gt_matrix)
        eer = ta2_metrics.get("Equal Error Rate", ta2_metrics.get("EER", 1.0))
        results[f"ta2_EER/{genre}"] = float(eer)
    except (FileNotFoundError, KeyError) as e:
        print(f"  Skipping TA2 for {genre}: {e}")

    return results


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
    print(f"Loading embedder from {checkpoint_path}...")
    model = _load_embedder(config_dir, checkpoint_path)
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
        genre_results = evaluate_genre_embedder(model, genre, batch_size, max_length)
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
) -> Dict[str, float]:
    """Run full-system (retriever + reranker) evaluation across all requested genres."""
    print(f"Loading embedder from {embedder_checkpoint_path}...")
    embedder = _load_embedder(embedder_config_dir, embedder_checkpoint_path)
    print(f"Loading reranker from {checkpoint_path}...")
    reranker_model = _load_reranker(config_dir, checkpoint_path)
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
        genre_results = evaluate_genre_full_system(
            embedder, reranker_model, genre, top_k, reranker_weight, batch_size, max_length,
        )
        results.update(genre_results)
        evaluated_genres.append(genre)
        for k, v in sorted(genre_results.items()):
            print(f"    {k}: {v:.4f}")

    results.update(_compute_averages(results, evaluated_genres, prefix="ta1_"))
    results.update(_compute_averages(results, evaluated_genres, prefix="ta2_"))
    del embedder, reranker_model
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
    parser.add_argument("--wandb_run_id", default=None, help="Resume this wandb run for logging")
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
