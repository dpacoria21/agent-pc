"""Phase 6 helpers: analyze student ideas without requiring an LLM.

The fallback analyzer is intentionally explicit. It lets the thesis prototype
show what the current agent can infer from a student idea, and which labels a
future GPT/PageIndex agent should improve.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


APPROACH_KEYWORDS = {
    "BINARY_SEARCH": ["binary search", "monotonic", "predicate", "lower_bound", "upper_bound", "bisect"],
    "MATH_FORMULA": ["formula", "quadratic", "discriminant", "sqrt", "square root", "integer root", "perfect square"],
    "DP": ["dp", "dynamic programming", "state", "transition", "memo"],
    "GRAPH": ["graph", "tree", "dfs", "bfs", "component"],
    "GREEDY": ["greedy", "sort", "choose", "always", "exchange"],
    "DATA_STRUCTURES": ["segment tree", "fenwick", "dsu", "priority queue", "ordered set"],
    "BRUTE_FORCE": ["brute force", "try all", "enumerate all"],
    "IMPLEMENTATION": ["implementation", "implement", "code", "parse", "branch"],
}


RISK_KEYWORDS = {
    "TLE": ["too slow", "tle", "o(n^2)", "brute force"],
    "WA": ["wrong answer", "wa", "fails", "incorrect"],
    "EDGE_CASES": ["edge", "corner", "n=1", "zero", "empty"],
    "BAD_COMPLEXITY": ["bad complexity", "complexity too high", "o(n^2)", "too slow"],
    "WRONG_PROOF": ["prove", "proof", "always", "invariant", "exchange"],
    "FLOAT_PRECISION": ["floating", "precision", "sqrt", "double"],
    "OVERFLOW": ["overflow", "long long", "r*r", "x*x"],
}


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def detect_approach(text: str) -> str:
    for label, keywords in APPROACH_KEYWORDS.items():
        if contains_any(text, keywords):
            return label
    return "UNKNOWN"


def detect_reasoning_stage(text: str) -> str:
    if contains_any(text, ["debug", "wa", "tle", "runtime", "wrong answer"]):
        return "DEBUGGING"
    if contains_any(text, ["implement", "code", "branch", "parse"]):
        return "IMPLEMENTATION"
    if contains_any(text, ["prove", "proof", "invariant", "exchange", "why"]):
        return "PROOF"
    if contains_any(text, ["algorithm", "binary search", "transition", "dfs", "bfs", "compute"]):
        return "ALGORITHM"
    if contains_any(text, ["maybe", "i think", "hypothesis", "could"]):
        return "HYPOTHESIS"
    if contains_any(text, ["observe", "notice", "property", "condition"]):
        return "OBSERVATION"
    return "UNDERSTANDING"


def detect_risks(text: str) -> list[str]:
    risks = [label for label, keywords in RISK_KEYWORDS.items() if contains_any(text, keywords)]
    return list(dict.fromkeys(risks or ["UNKNOWN"]))


def detect_math_concepts(text: str) -> list[str]:
    concepts = []
    checks = {
        "triangular_numbers": ["triangular", "x*(x+1)/2", "x(x+1)/2"],
        "quadratic_formula": ["quadratic", "discriminant", "x^2", "sqrt(1+8"],
        "integer_sqrt": ["sqrt", "square root", "integer root"],
        "perfect_square": ["perfect square", "r*r", "root*root"],
        "monotonic_predicate": ["monotonic", "predicate", "binary search", "maximum x"],
    }
    for label, keywords in checks.items():
        if contains_any(text, keywords):
            concepts.append(label)
    return concepts


def estimate_idea_quality(text: str, approach: str, stage: str, risks: list[str]) -> tuple[str, float]:
    quality = 0.48
    if approach != "UNKNOWN":
        quality += 0.16
    if stage in {"ALGORITHM", "PROOF", "IMPLEMENTATION"}:
        quality += 0.1
    if contains_any(text, ["check", "condition", "<=", "==", "cap", "integer"]):
        quality += 0.1
    if any(risk in risks for risk in ["WA", "TLE", "BAD_COMPLEXITY"]):
        quality -= 0.08
    if any(risk in risks for risk in ["FLOAT_PRECISION", "OVERFLOW", "EDGE_CASES"]):
        quality += 0.04
    quality = max(0.0, min(1.0, quality))
    if quality >= 0.78:
        label = "GOOD"
    elif quality >= 0.58:
        label = "PARTIAL"
    else:
        label = "WEAK"
    return label, round(quality, 3)


def analyze_student_idea(idea_text: str, expected_strategy: str | None = None) -> dict[str, Any]:
    text = normalize_text(idea_text)
    approach = detect_approach(text)
    stage = detect_reasoning_stage(text)
    risks = detect_risks(text)
    concepts = detect_math_concepts(text)
    quality_label, quality_score = estimate_idea_quality(text, approach, stage, risks)
    expected = expected_strategy or ""
    strategy_match = bool(expected and approach == expected)
    confidence = 0.55 + (0.18 if approach != "UNKNOWN" else 0.0) + (0.12 if concepts else 0.0)
    if expected:
        confidence += 0.08 if strategy_match else -0.05
    confidence = round(max(0.05, min(0.98, confidence)), 3)

    if approach == "BINARY_SEARCH":
        feedback = "La idea identifica una busqueda sobre respuesta; conviene explicitar la monotonicidad del predicado."
    elif approach == "MATH_FORMULA":
        feedback = "La idea usa una ruta matematica; conviene justificar redondeo, raiz entera y limites."
    elif approach == "UNKNOWN":
        feedback = "La idea aun no separa estrategia, prueba y condiciones de borde."
    else:
        feedback = "La idea tiene una estrategia general; falta conectar evidencia del statement/editorial."

    return {
        "detected_approach": approach,
        "reasoning_stage": stage,
        "risk_types": risks,
        "primary_risk_type": risks[0] if risks else "UNKNOWN",
        "math_concepts": concepts,
        "idea_quality": quality_label,
        "idea_quality_score": quality_score,
        "expected_strategy": expected,
        "strategy_match": strategy_match if expected else None,
        "confidence": confidence,
        "feedback": feedback,
        "analyzer": "heuristic_student_model_v1",
    }


def load_or_create_idea_rows(processed_dir: str | Path) -> pd.DataFrame:
    processed = Path(processed_dir)
    report_path = processed / "math_binary_classification_report.csv"
    if report_path.exists():
        df = pd.read_csv(report_path)
        if "fine_strategy" in df.columns:
            return df.rename(columns={"fine_strategy": "expected_strategy"})

    return pd.DataFrame(
        [
            {
                "student_id": "student_binary",
                "global_problem_id": "codeforces_750_A",
                "problem_title": "New Year and Hurry",
                "idea_text": "I will binary search the maximum x and check if 5*x*(x+1)/2 <= 240-k.",
                "expected_strategy": "BINARY_SEARCH",
            },
            {
                "student_id": "student_formula",
                "global_problem_id": "codeforces_750_A",
                "problem_title": "New Year and Hurry",
                "idea_text": "I can solve the quadratic formula with sqrt(1+8*T) and then floor the root.",
                "expected_strategy": "MATH_FORMULA",
            },
        ]
    )


def analyze_idea_dataset(idea_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, idea in idea_rows.iterrows():
        expected = str(idea.get("expected_strategy", idea.get("fine_strategy", "")) or "")
        analysis = analyze_student_idea(str(idea.get("idea_text", "")), expected_strategy=expected)
        rows.append(
            {
                "idea_id": idea.get("idea_id", f"idea_{index + 1}"),
                "student_id": idea.get("student_id", idea.get("user_id", "")),
                "global_problem_id": idea.get("global_problem_id", ""),
                "problem_title": idea.get("problem_title", ""),
                "idea_text": idea.get("idea_text", ""),
                **analysis,
            }
        )
    return pd.DataFrame(rows)


def save_student_idea_analysis(analysis_df: pd.DataFrame, processed_dir: str | Path) -> dict[str, str]:
    processed = Path(processed_dir)
    processed.mkdir(parents=True, exist_ok=True)
    csv_path = processed / "student_idea_analysis.csv"
    json_path = processed / "student_idea_analysis.json"
    serial = analysis_df.copy()
    for column in serial.columns:
        if serial[column].dtype == "object":
            serial[column] = serial[column].apply(
                lambda value: json.dumps(value, ensure_ascii=False, default=str)
                if isinstance(value, (list, dict))
                else value
            )
    serial.to_csv(csv_path, index=False)
    analysis_df.to_json(json_path, orient="records", force_ascii=False, indent=2, default_handler=str)
    return {"csv": str(csv_path), "json": str(json_path)}
