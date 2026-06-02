"""Dump the reranker P(yes) distribution on one HRS genre, split by label.

Confirms (or rules out) "collapse / domain-transfer failure": if same-author and
different-author P(yes) distributions overlap heavily (low AUC) or both saturate
near 1.0, the reranker is not discriminating on this genre -> transfer failure,
not a wiring bug. If they separate cleanly here but the system still loses to the
embedder in eval, the problem is downstream (aggregation / interpolation).

Usage (reranker-only; no embedder needed):
    conda run -n hiatus-phase3 python scripts/diagnose_reranker_pyes.py \
        --reranker_config outputs/reranker-v8/config.yaml \
        --reranker_checkpoint outputs/reranker-v8/checkpoint_step17314.ckpt \
        --genre HRS2.101 --n_pairs 1000
"""

import argparse
import os
import random
import sys

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from authorship.evaluation.constants import ALL_HRS_PATHS
from authorship.evaluation.hrs_loader import _normalize_author_ids, load_ta1
from authorship.models.reranker import WrappedRerankerModel


def _build_reranker(config_path: str, checkpoint_path: str) -> WrappedRerankerModel:
    cfg = OmegaConf.load(config_path)
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
        max_length=cfg.data.get("max_seq_length", 512),
        batch_size=cfg.data.get("global_batch_size", 8),
        instruction=cfg.data.get("instruction"),
    )


def _build_labeled_pairs(genre: str, n_pairs: int, seed: int):
    """Build n_pairs same-author and n_pairs different-author doc-doc pairs."""
    paths = ALL_HRS_PATHS[genre]["TA1"]
    queries, candidates, _ = load_ta1(
        paths["resample_queries"], paths["resample_candidates"], paths["ground_truth"],
    )
    # Pool all docs; group document indices by author.
    texts = list(queries["text"]) + list(candidates["text"])
    authors = list(queries["author_id"]) + list(candidates["author_id"])
    by_author: dict = {}
    for i, a in enumerate(authors):
        by_author.setdefault(_normalize_author_ids(a), []).append(i)

    rng = random.Random(seed)
    multi = [a for a, idx in by_author.items() if len(idx) >= 2]
    all_authors = list(by_author.keys())

    pos_q, pos_d, neg_q, neg_d = [], [], [], []
    # Positives: two distinct docs from the same author.
    for _ in range(n_pairs):
        a = rng.choice(multi)
        i, j = rng.sample(by_author[a], 2)
        pos_q.append(texts[i]); pos_d.append(texts[j])
    # Negatives: docs from two different authors.
    for _ in range(n_pairs):
        a, b = rng.sample(all_authors, 2)
        pos_i = rng.choice(by_author[a]); neg_j = rng.choice(by_author[b])
        neg_q.append(texts[pos_i]); neg_d.append(texts[neg_j])
    return pos_q, pos_d, neg_q, neg_d


def _build_genre_split_pairs(genre: str, n_pairs: int, seed: int):
    """Use the query/candidate split as a genre proxy (valid for cross_genre evals).

    Returns three buckets of (q_texts, d_texts):
      same_same  -- same author, both docs on the SAME side (same genre)
      same_cross -- same author, one doc per side    (cross genre)
      diff_cross -- different authors, one doc per side (cross genre)
    If same_same ~ high but same_cross ~ low, the register/genre gap (not the
    topic-matched negatives) is what destroys the same-author signal.
    """
    paths = ALL_HRS_PATHS[genre]["TA1"]
    queries, candidates, _ = load_ta1(
        paths["resample_queries"], paths["resample_candidates"], paths["ground_truth"],
    )
    q_text, c_text = list(queries["text"]), list(candidates["text"])
    # author -> {"q": [idx...], "c": [idx...]}
    sides: dict = {}
    for side, auth_col in (("q", queries["author_id"]), ("c", candidates["author_id"])):
        for i, a in enumerate(auth_col):
            sides.setdefault(_normalize_author_ids(a), {"q": [], "c": []})[side].append(i)

    rng = random.Random(seed)
    txt = {"q": q_text, "c": c_text}
    same_side_auth = [a for a, s in sides.items() if len(s["q"]) >= 2 or len(s["c"]) >= 2]
    cross_side_auth = [a for a, s in sides.items() if s["q"] and s["c"]]

    ss_q, ss_d, sc_q, sc_d, dc_q, dc_d = ([] for _ in range(6))
    for _ in range(n_pairs):
        a = rng.choice(same_side_auth)
        side = "q" if len(sides[a]["q"]) >= 2 else "c"
        i, j = rng.sample(sides[a][side], 2)
        ss_q.append(txt[side][i]); ss_d.append(txt[side][j])
    for _ in range(n_pairs):
        a = rng.choice(cross_side_auth)
        i = rng.choice(sides[a]["q"]); j = rng.choice(sides[a]["c"])
        sc_q.append(q_text[i]); sc_d.append(c_text[j])
    for _ in range(n_pairs):
        a = rng.choice([x for x in sides if sides[x]["q"]])
        b = rng.choice([x for x in sides if sides[x]["c"] and x != a])
        i = rng.choice(sides[a]["q"]); j = rng.choice(sides[b]["c"])
        dc_q.append(q_text[i]); dc_d.append(c_text[j])
    return (ss_q, ss_d), (sc_q, sc_d), (dc_q, dc_d)


def _toks(s: str) -> set:
    return set(s.lower().split())


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _build_similarity_probe(genre: str, n_pairs: int, seed: int, pool_k: int = 200):
    """Cross same/different author with high/low surface overlap.

    For each anchor builds 4 counterparts:
      pos_hi -- same author,  Jaccard-nearest of that author's other docs
      pos_lo -- same author,  random other doc
      neg_hi -- diff author,  Jaccard-nearest among a sampled pool
      neg_lo -- diff author,  random
    If P(yes) tracks overlap (hi<lo) regardless of author label, the reranker is
    a surface-similarity detector, not an authorship model.
    """
    paths = ALL_HRS_PATHS[genre]["TA1"]
    queries, candidates, _ = load_ta1(
        paths["resample_queries"], paths["resample_candidates"], paths["ground_truth"],
    )
    texts = list(queries["text"]) + list(candidates["text"])
    authors = [_normalize_author_ids(a)
               for a in list(queries["author_id"]) + list(candidates["author_id"])]
    tok = [_toks(t) for t in texts]
    by_author: dict = {}
    for i, a in enumerate(authors):
        by_author.setdefault(a, []).append(i)

    rng = random.Random(seed)
    multi = [a for a, idx in by_author.items() if len(idx) >= 2]
    n_docs = len(texts)
    buckets = {k: ([], []) for k in ("pos_hi", "pos_lo", "neg_hi", "neg_lo")}

    def _add(key, i, j):
        buckets[key][0].append(texts[i]); buckets[key][1].append(texts[j])

    for _ in range(n_pairs):
        a = rng.choice(multi)
        anchor = rng.choice(by_author[a])
        others = [x for x in by_author[a] if x != anchor]
        # same author hi/lo
        hi = max(others, key=lambda x: _jaccard(tok[anchor], tok[x]))
        _add("pos_hi", anchor, hi)
        _add("pos_lo", anchor, rng.choice(others))
        # diff author hi/lo from a sampled pool
        pool = [rng.randrange(n_docs) for _ in range(pool_k)]
        pool = [x for x in pool if authors[x] != a] or [
            x for x in range(n_docs) if authors[x] != a][:pool_k]
        hi_n = max(pool, key=lambda x: _jaccard(tok[anchor], tok[x]))
        _add("neg_hi", anchor, hi_n)
        _add("neg_lo", anchor, rng.choice(pool))
    return buckets


def _summary(name: str, scores: np.ndarray) -> None:
    qs = np.percentile(scores, [0, 10, 25, 50, 75, 90, 100])
    print(
        f"  {name:>9}: n={len(scores):4d}  mean={scores.mean():.3f}  std={scores.std():.3f}  "
        f"min/p10/p25/med/p75/p90/max="
        f"{qs[0]:.2f}/{qs[1]:.2f}/{qs[2]:.2f}/{qs[3]:.2f}/{qs[4]:.2f}/{qs[5]:.2f}/{qs[6]:.2f}  "
        f">0.9={np.mean(scores > 0.9):.2f}  <0.1={np.mean(scores < 0.1):.2f}"
    )


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """ROC-AUC = P(pos score > neg score), computed via rank statistic."""
    all_s = np.concatenate([pos, neg])
    order = all_s.argsort()
    ranks = np.empty(len(all_s)); ranks[order] = np.arange(1, len(all_s) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(all_s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    r_pos = ranks[: len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reranker_config", required=True)
    ap.add_argument("--reranker_checkpoint", required=True)
    ap.add_argument("--genre", default="HRS2.101")
    ap.add_argument("--n_pairs", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--genre_split", action="store_true",
        help="Cross-genre evals only: break same-author pairs into same-genre "
             "(same query/cand side) vs cross-genre (across sides) to test whether "
             "the register gap -- not topic-matched negatives -- kills the signal.",
    )
    ap.add_argument(
        "--similarity_probe", action="store_true",
        help="Cross author label (same/diff) with surface overlap (hi/lo) to test "
             "whether P(yes) tracks lexical similarity instead of authorship.",
    )
    args = ap.parse_args()

    print(f"Loading reranker from {args.reranker_checkpoint}...")

    if args.similarity_probe:
        print(f"Genre: {args.genre} | similarity probe, {args.n_pairs} pairs/bucket")
        b = _build_similarity_probe(args.genre, args.n_pairs, args.seed)
        reranker = _build_reranker(args.reranker_config, args.reranker_checkpoint)
        sc = {k: reranker.score_pairs(q, d, desc=k) for k, (q, d) in b.items()}
        print(f"\nP(yes) by (author label x surface overlap) on {args.genre}:")
        for k in ("pos_hi", "pos_lo", "neg_hi", "neg_lo"):
            _summary(k, sc[k])
        same_eff = (sc["pos_hi"].mean() + sc["pos_lo"].mean()) / 2
        diff_eff = (sc["neg_hi"].mean() + sc["neg_lo"].mean()) / 2
        hi_eff = (sc["pos_hi"].mean() + sc["neg_hi"].mean()) / 2
        lo_eff = (sc["pos_lo"].mean() + sc["neg_lo"].mean()) / 2
        print(f"\n  author effect  (same - diff): {same_eff - diff_eff:+.3f}")
        print(f"  overlap effect (hi  - lo)  : {hi_eff - lo_eff:+.3f}")
        print("\nVerdict:")
        if abs(hi_eff - lo_eff) > abs(same_eff - diff_eff) and (hi_eff - lo_eff) < 0:
            print("  -> SURFACE-SIMILARITY DETECTOR: P(yes) is driven by lexical overlap")
            print("     (higher overlap -> LOWER score), not by true authorship. The hard-")
            print("     negative-only sampling taught 'similar => different author'. Fix the")
            print("     negative distribution (keep easy negs; add dissimilar same-author pos).")
        else:
            print("  -> Author label dominates overlap; surface confound is not the main driver.")
        return

    if args.genre_split:
        print(f"Genre: {args.genre} | genre-split test, {args.n_pairs} pairs/bucket")
        (ss_q, ss_d), (sc_q, sc_d), (dc_q, dc_d) = _build_genre_split_pairs(
            args.genre, args.n_pairs, args.seed,
        )
        reranker = _build_reranker(args.reranker_config, args.reranker_checkpoint)
        ss = reranker.score_pairs(ss_q, ss_d, desc="same-author same-genre")
        sc = reranker.score_pairs(sc_q, sc_d, desc="same-author cross-genre")
        dc = reranker.score_pairs(dc_q, dc_d, desc="diff-author cross-genre")
        print(f"\nP(yes) by genre side on {args.genre}:")
        _summary("same/same", ss)
        _summary("same/cross", sc)
        _summary("diff/cross", dc)
        print(f"\n  AUC same-author: same-genre vs cross-genre : {_auc(ss, sc):.3f}")
        print(f"  mean drop same-author when genre differs   : {ss.mean() - sc.mean():+.3f}")
        print("\nVerdict:")
        if ss.mean() - sc.mean() > 0.1 and ss.mean() > dc.mean():
            print("  -> REGISTER GAP confirmed: same-author signal holds within a genre but")
            print("     collapses across genres. Fix = same-author/different-register positives,")
            print("     NOT just an aggregation change.")
        else:
            print("  -> Register gap NOT the dominant factor here; look elsewhere.")
        return

    print(f"Genre: {args.genre} | building {args.n_pairs} pos + {args.n_pairs} neg pairs")
    pos_q, pos_d, neg_q, neg_d = _build_labeled_pairs(args.genre, args.n_pairs, args.seed)
    reranker = _build_reranker(args.reranker_config, args.reranker_checkpoint)

    pos = reranker.score_pairs(pos_q, pos_d, desc="pos pairs")
    neg = reranker.score_pairs(neg_q, neg_d, desc="neg pairs")

    print(f"\nP(yes) distribution on {args.genre}:")
    _summary("same", pos)
    _summary("diff", neg)
    auc = _auc(pos, neg)
    print(f"\n  AUC (same vs diff)        : {auc:.3f}")
    print(f"  mean(same) - mean(diff)   : {pos.mean() - neg.mean():+.3f}")
    print(f"  frac BOTH saturated >0.9  : same={np.mean(pos > 0.9):.2f}  diff={np.mean(neg > 0.9):.2f}")
    print("\nVerdict:")
    if auc < 0.4:
        print(f"  -> INVERTED signal (AUC={auc:.2f}): reranker confidently scores DIFFERENT-author")
        print("     pairs higher than SAME-author. It learned topic/domain similarity, which")
        print("     anti-correlates with authorship on this (cross-genre) genre. Data problem.")
    elif auc < 0.6:
        print("  -> COLLAPSE / transfer failure: reranker barely separates same vs diff here.")
    elif np.mean(pos > 0.9) > 0.8 and np.mean(neg > 0.9) > 0.8:
        print("  -> SATURATED near 1.0 for both labels: scores carry little usable signal.")
    else:
        print("  -> Reranker DOES separate pairwise. Look downstream (aggregation/interpolation).")


if __name__ == "__main__":
    main()
