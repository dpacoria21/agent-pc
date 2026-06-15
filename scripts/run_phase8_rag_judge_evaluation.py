"""Phase 8: evaluate the RAG tutor outputs and create the final paper table."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation.rag_judge import (
    build_summary_table,
    evaluate_rag_interactions,
    save_judge_outputs,
)


PROCESSED = ROOT / "data" / "processed"
ASSETS = ROOT / "comparison_assets"


def require_csv(path: Path, message: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(message)
    return pd.read_csv(path)


def main() -> None:
    idea_analysis = require_csv(
        PROCESSED / "student_idea_analysis.csv",
        "Falta student_idea_analysis.csv. Ejecuta scripts/run_phase6_student_idea_analysis.py",
    )
    page_nodes = require_csv(PROCESSED / "cp_page_nodes_dataset.csv", "Falta cp_page_nodes_dataset.csv.")
    recommendations = require_csv(
        PROCESSED / "adaptive_problem_recommendations.csv",
        "Falta adaptive_problem_recommendations.csv. Ejecuta scripts/run_phase7_adaptive_recommendations.py",
    )

    scores = evaluate_rag_interactions(idea_analysis, page_nodes, recommendations)
    summary = build_summary_table(scores)
    paths = save_judge_outputs(scores, summary, PROCESSED, ASSETS)

    report = {
        "phase": "phase8_rag_judge_evaluation",
        "status": "ok",
        "interaction_count": int(len(scores)),
        "summary_rows": int(len(summary)),
        "outputs": paths,
    }
    (PROCESSED / "phase8_rag_judge_evaluation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nTABLE I")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
