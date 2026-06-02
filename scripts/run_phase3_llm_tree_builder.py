"""Phase 3 runner: build GPT-assisted semantic trees."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from indexing.llm_tree_builder import build_llm_tree_dataset, save_llm_tree_outputs


def parse_problem_ids(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--problem-ids", default="", help="Comma-separated global_problem_id list.")
    parser.add_argument("--model", default=None, help="Override OPENAI_MODEL.")
    parser.add_argument("--force-fallback", action="store_true", help="Do not call GPT; use deterministic local fallback.")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    problems_path = processed_dir / "cp_problems_dataset.csv"
    if not problems_path.exists():
        raise SystemExit(f"Missing {problems_path}")

    problems = pd.read_csv(problems_path)
    tree_df, edge_df, analyses, report = build_llm_tree_dataset(
        problems,
        limit=args.limit,
        problem_ids=parse_problem_ids(args.problem_ids),
        force_fallback=args.force_fallback,
        model=args.model,
    )
    paths = save_llm_tree_outputs(tree_df, edge_df, analyses, report, processed_dir)
    report["paths"] = paths

    report_path = processed_dir / "llm_tree_build_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("Phase 3 GPT semantic tree builder")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    if tree_df.empty:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

