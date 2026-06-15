"""Phase 7: build student profiles and adaptive problem recommendations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from student_model.profile_recommender import (
    build_student_profiles,
    recommend_for_profiles,
    save_profiles_and_recommendations,
)


PROCESSED = ROOT / "data" / "processed"


def main() -> None:
    idea_path = PROCESSED / "student_idea_analysis.csv"
    if not idea_path.exists():
        raise SystemExit("Falta student_idea_analysis.csv. Ejecuta primero scripts/run_phase6_student_idea_analysis.py")
    problems_path = PROCESSED / "cp_problems_dataset.csv"
    if not problems_path.exists():
        raise SystemExit("Falta cp_problems_dataset.csv.")

    idea_analysis = pd.read_csv(idea_path)
    problems = pd.read_csv(problems_path)
    profiles = build_student_profiles(idea_analysis)
    recommendations = recommend_for_profiles(profiles, problems, top_k=5)
    paths = save_profiles_and_recommendations(profiles, recommendations, PROCESSED)

    report = {
        "phase": "phase7_adaptive_recommendations",
        "status": "ok",
        "profile_rows": int(len(profiles)),
        "recommendation_rows": int(len(recommendations)),
        "students": sorted(profiles["student_id"].dropna().astype(str).tolist()) if not profiles.empty else [],
        "top_recommendations": recommendations.groupby("student_id").head(1)[
            ["student_id", "global_problem_id", "title", "recommendation_score", "reason"]
        ].to_dict("records")
        if not recommendations.empty
        else [],
        "outputs": paths,
    }
    (PROCESSED / "phase7_adaptive_recommendations_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
