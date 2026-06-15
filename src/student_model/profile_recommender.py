"""Phase 7 helpers: student profile and adaptive recommendations."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from student_model.idea_analyzer import normalize_text


def parse_listish(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    for parser in (json.loads,):
        try:
            parsed = parser(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item).strip()]
        except Exception:
            pass
    if text.startswith("[") and text.endswith("]"):
        return [part.strip(" '\"") for part in text.strip("[]").split(",") if part.strip(" '\"")]
    return [text]


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def build_student_profiles(idea_analysis: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for student_id, group in idea_analysis.groupby("student_id"):
        approach_counts = Counter(group["detected_approach"].fillna("UNKNOWN"))
        risk_counts = Counter()
        concept_counts = Counter()
        for _, row in group.iterrows():
            risk_counts.update(parse_listish(row.get("risk_types")))
            concept_counts.update(parse_listish(row.get("math_concepts")))

        match_values = group["strategy_match"].dropna().astype(bool)
        strategy_accuracy = float(match_values.mean()) if len(match_values) else 0.0
        avg_quality = float(group["idea_quality_score"].mean()) if not group.empty else 0.0
        unknown_rate = float((group["detected_approach"] == "UNKNOWN").mean()) if not group.empty else 0.0
        risk_pressure = min(1.0, sum(v for k, v in risk_counts.items() if k not in {"UNKNOWN", "NONE"}) / max(1, len(group) * 2))

        autonomy_score = clamp(78 + 12 * avg_quality - 35 * unknown_rate - 15 * risk_pressure)
        proof_skill_score = clamp(66 + 15 * float((group["reasoning_stage"] == "PROOF").mean()) - 12 * risk_counts.get("WRONG_PROOF", 0))
        implementation_skill_score = clamp(70 - 10 * risk_counts.get("OVERFLOW", 0) - 8 * risk_counts.get("FLOAT_PRECISION", 0))
        strategy_skill_score = clamp(45 + 55 * strategy_accuracy)

        weak_skills = []
        if unknown_rate > 0.25:
            weak_skills.append("strategy_identification")
        if risk_counts.get("WRONG_PROOF", 0):
            weak_skills.append("proof")
        if risk_counts.get("FLOAT_PRECISION", 0) or risk_counts.get("OVERFLOW", 0):
            weak_skills.append("numeric_implementation")
        if strategy_accuracy < 0.75:
            weak_skills.append("fine_grained_strategy")

        rows.append(
            {
                "student_id": student_id,
                "total_ideas": int(len(group)),
                "dominant_approach": approach_counts.most_common(1)[0][0] if approach_counts else "UNKNOWN",
                "approach_distribution": dict(approach_counts),
                "risk_distribution": dict(risk_counts),
                "math_concepts": dict(concept_counts),
                "strategy_accuracy": round(strategy_accuracy, 3),
                "avg_idea_quality_score": round(avg_quality, 3),
                "autonomy_score": round(autonomy_score, 1),
                "proof_skill_score": round(proof_skill_score, 1),
                "implementation_skill_score": round(implementation_skill_score, 1),
                "strategy_skill_score": round(strategy_skill_score, 1),
                "weak_skills": weak_skills,
            }
        )
    return pd.DataFrame(rows)


def _problem_strategy_terms(problem: pd.Series) -> set[str]:
    text = normalize_text(
        " ".join(
            [
                str(problem.get("title", "")),
                str(problem.get("statement", "")),
                str(problem.get("notes", "")),
                str(problem.get("official_editorial", "")),
                " ".join(parse_listish(problem.get("original_tags"))),
                " ".join(parse_listish(problem.get("normalized_tags"))),
            ]
        )
    )
    strategies = set()
    if any(term in text for term in ["binary search", "monotonic", "predicate"]):
        strategies.add("BINARY_SEARCH")
    if any(term in text for term in ["formula", "quadratic", "discriminant", "sqrt", "perfect square"]):
        strategies.add("MATH_FORMULA")
    if "dp" in text or "transition" in text:
        strategies.add("DP")
    if "greedy" in text or "exchange" in text:
        strategies.add("GREEDY")
    return strategies or {"GENERAL"}


def _profile_weak_strategy(profile: pd.Series) -> str:
    distribution = profile.get("approach_distribution", {})
    if isinstance(distribution, str):
        try:
            distribution = json.loads(distribution)
        except Exception:
            distribution = {}
    if profile.get("strategy_accuracy", 0) < 0.75:
        if distribution.get("BINARY_SEARCH", 0) < distribution.get("MATH_FORMULA", 0):
            return "BINARY_SEARCH"
        return "MATH_FORMULA"
    weak_skills = parse_listish(profile.get("weak_skills"))
    if "numeric_implementation" in weak_skills:
        return "MATH_FORMULA"
    return profile.get("dominant_approach", "GENERAL")


def recommend_for_profiles(profiles: pd.DataFrame, problems: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    rows = []
    if profiles.empty or problems.empty:
        return pd.DataFrame()

    for _, profile in profiles.iterrows():
        target_strategy = _profile_weak_strategy(profile)
        target_difficulty = 1000 if profile.get("strategy_skill_score", 0) < 75 else 1300
        for _, problem in problems.iterrows():
            strategies = _problem_strategy_terms(problem)
            strategy_match = 1.0 if target_strategy in strategies else 0.45 if target_strategy == "GENERAL" else 0.0
            difficulty = problem.get("normalized_difficulty")
            try:
                difficulty = float(difficulty)
            except Exception:
                difficulty = target_difficulty
            rating_fit = math.exp(-abs(difficulty - target_difficulty) / 500)
            popularity = float(problem.get("solved_count", 0) or 0)
            popularity_score = min(1.0, math.log1p(popularity) / 12.0) if popularity else 0.3
            topic_diversity = 0.8 if target_strategy in {"BINARY_SEARCH", "MATH_FORMULA"} else 0.55
            score = 0.38 * strategy_match + 0.27 * rating_fit + 0.2 * popularity_score + 0.15 * topic_diversity
            if strategy_match:
                reason = (
                    f"Recomendado porque trabaja {target_strategy}, con dificultad cercana a {int(target_difficulty)} "
                    f"y evidencia en el statement/editorial."
                )
            else:
                reason = (
                    f"Recomendado como practica de transferencia; no coincide directamente con {target_strategy}, "
                    "pero mantiene dificultad razonable."
                )
            rows.append(
                {
                    "student_id": profile["student_id"],
                    "global_problem_id": problem["global_problem_id"],
                    "title": problem.get("title", ""),
                    "normalized_difficulty": problem.get("normalized_difficulty"),
                    "detected_problem_strategies": sorted(strategies),
                    "target_strategy": target_strategy,
                    "recommendation_score": round(score, 4),
                    "reason": reason,
                    "url": problem.get("url", ""),
                }
            )

    recs = pd.DataFrame(rows)
    if recs.empty:
        return recs
    return (
        recs.sort_values(["student_id", "recommendation_score"], ascending=[True, False])
        .groupby("student_id")
        .head(top_k)
        .reset_index(drop=True)
    )


def save_profiles_and_recommendations(
    profiles: pd.DataFrame,
    recommendations: pd.DataFrame,
    processed_dir: str | Path,
) -> dict[str, str]:
    processed = Path(processed_dir)
    processed.mkdir(parents=True, exist_ok=True)
    paths = {}
    for stem, df in {
        "student_profiles": profiles,
        "adaptive_problem_recommendations": recommendations,
    }.items():
        serial = df.copy()
        for column in serial.columns:
            if serial[column].dtype == "object":
                serial[column] = serial[column].apply(
                    lambda value: json.dumps(value, ensure_ascii=False, default=str)
                    if isinstance(value, (list, dict))
                    else value
                )
        csv_path = processed / f"{stem}.csv"
        json_path = processed / f"{stem}.json"
        serial.to_csv(csv_path, index=False)
        df.to_json(json_path, orient="records", force_ascii=False, indent=2, default_handler=str)
        paths[f"{stem}_csv"] = str(csv_path)
        paths[f"{stem}_json"] = str(json_path)
    return paths
