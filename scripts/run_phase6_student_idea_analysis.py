"""Phase 6: analyze simulated student ideas."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from student_model.idea_analyzer import (
    analyze_idea_dataset,
    load_or_create_idea_rows,
    save_student_idea_analysis,
)


PROCESSED = ROOT / "data" / "processed"


def main() -> None:
    idea_rows = load_or_create_idea_rows(PROCESSED)
    analysis = analyze_idea_dataset(idea_rows)
    paths = save_student_idea_analysis(analysis, PROCESSED)

    report = {
        "phase": "phase6_student_idea_analysis",
        "status": "ok",
        "input_idea_rows": int(len(idea_rows)),
        "analysis_rows": int(len(analysis)),
        "students": sorted(analysis["student_id"].dropna().astype(str).unique().tolist()),
        "approach_distribution": analysis["detected_approach"].value_counts().to_dict(),
        "reasoning_stage_distribution": analysis["reasoning_stage"].value_counts().to_dict(),
        "strategy_match_rate": float(analysis["strategy_match"].dropna().astype(bool).mean())
        if not analysis["strategy_match"].dropna().empty
        else None,
        "outputs": paths,
    }
    (PROCESSED / "phase6_student_idea_analysis_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not analysis.empty:
        print(
            analysis[
                [
                    "student_id",
                    "problem_title",
                    "detected_approach",
                    "reasoning_stage",
                    "primary_risk_type",
                    "idea_quality",
                    "strategy_match",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
