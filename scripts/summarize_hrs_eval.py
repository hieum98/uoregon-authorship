#!/usr/bin/env python3
"""Print embedder vs embedder+reranker comparison from HRS eval JSON files.

Usage:
  python scripts/summarize_hrs_eval.py outputs/reranker-v1/hrs_eval/checkpoint_step8000.json
  python scripts/summarize_hrs_eval.py outputs/reranker-v1/hrs_eval/*.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from authorship.evaluation.eval_hrs import _print_embedder_vs_system_summary

_GENRE_RE = re.compile(r"^(?:embedder_|system_|ta1_|ta2_)(?:S@8|EER)/(.+)$")


def _genres_from_results(results: dict) -> list[str]:
    genres = set()
    for key in results:
        m = _GENRE_RE.match(key)
        if m:
            genres.add(m.group(1))
    return sorted(genres)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_paths", nargs="+", help="Eval result JSON file(s)")
    args = parser.parse_args()

    for path_str in args.json_paths:
        path = Path(path_str)
        if not path.is_file():
            print(f"Skip missing file: {path}")
            continue
        results = json.loads(path.read_text())
        genres = _genres_from_results(results)
        print(f"\n### {path.name} ###")
        if "embedder_S@8" in "".join(results.keys()) or "system_S@8" in "".join(results.keys()):
            _print_embedder_vs_system_summary(results, genres)
        else:
            print("(Legacy format: system-only keys ta1_S@8 / ta2_EER — re-run eval for dual metrics)")
            for k, v in sorted(results.items()):
                if "/" in k and not k.startswith("Avg/"):
                    print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
