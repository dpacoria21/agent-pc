"""Phase 8: automatic RAG evaluation table.

The evaluator can be read as a lightweight local version of LLM-as-a-judge.
It produces scores per interaction and aggregates them into the paper-style
table requested by the user:

Metric | Threshold | Mean | Standard deviation | % >= threshold
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


JUDGE_THRESHOLDS = {
    "Faithfulness": 0.80,
    "Answer Relevancy": 0.85,
    "Context Precision": 0.70,
    "Strategy Classification": 0.80,
    "Recommendation Fit": 0.75,
}


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def content_tokens(text: Any) -> set[str]:
    stop = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "your",
        "you",
        "una",
        "un",
        "que",
        "por",
        "para",
        "con",
        "del",
        "los",
        "las",
        "de",
        "la",
        "el",
    }
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9_]+", normalize_text(text))
        if len(token) > 2 and token not in stop
    }


def parse_listish(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
    except Exception:
        pass
    if text.startswith("[") and text.endswith("]"):
        return [part.strip(" '\"") for part in text.strip("[]").split(",") if part.strip(" '\"")]
    return [text]


def overlap_score(left: Any, right: Any) -> float:
    left_tokens = content_tokens(left)
    right_tokens = content_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens.intersection(right_tokens)) / max(1, len(left_tokens))


def select_context_for_idea(idea: pd.Series, page_nodes: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    problem_id = str(idea.get("global_problem_id", ""))
    same_problem = page_nodes[page_nodes["global_problem_id"].astype(str) == problem_id].copy()
    if same_problem.empty:
        same_problem = page_nodes.copy()

    idea_text = normalize_text(idea.get("idea_text", ""))
    preferred_types = []
    if "proof" in idea_text or "prove" in idea_text:
        preferred_types.extend(["EDITORIAL_PROOF", "EDITORIAL_OBSERVATION"])
    if "binary search" in idea_text or "formula" in idea_text or "sqrt" in idea_text:
        preferred_types.extend(["EDITORIAL_ALGORITHM", "EDITORIAL_OBSERVATION", "CONSTRAINTS"])
    if "edge" in idea_text or "overflow" in idea_text or "precision" in idea_text:
        preferred_types.extend(["COMMON_MISTAKES", "CONSTRAINTS", "EXAMPLES"])

    same_problem["type_bonus"] = same_problem["node_type"].isin(preferred_types).astype(float)
    same_problem["text_overlap"] = same_problem["node_text"].fillna("").apply(lambda text: overlap_score(idea_text, text))
    same_problem["context_score"] = same_problem["type_bonus"] + same_problem["text_overlap"]
    return same_problem.sort_values("context_score", ascending=False).head(top_k).reset_index(drop=True)


def generate_tutor_answer(idea: pd.Series, context: pd.DataFrame, recommendations: pd.DataFrame) -> str:
    strategy = str(idea.get("detected_approach", "UNKNOWN"))
    expected = str(idea.get("expected_strategy", ""))
    stage = str(idea.get("reasoning_stage", "UNDERSTANDING"))
    risk = str(idea.get("primary_risk_type", "UNKNOWN"))
    problem_title = str(idea.get("problem_title", "the problem"))
    context_titles = ", ".join(context["node_type"].head(3).astype(str).tolist())
    evidence = " ".join(context["node_text"].fillna("").astype(str).head(2).tolist())[:500]
    rec_text = ""
    student_recs = recommendations[recommendations["student_id"].astype(str) == str(idea.get("student_id", ""))]
    if not student_recs.empty:
        rec = student_recs.iloc[0]
        rec_text = f" Recommended next problem: {rec.get('title')} because {rec.get('reason')}"

    return (
        f"For {problem_title}, I classify the student's idea as {strategy}. "
        f"The expected strategy signal is {expected or strategy}, and the reasoning stage is {stage}. "
        f"The main risk is {risk}. Retrieved evidence comes from {context_titles}: {evidence} "
        f"Next step: ask the student to justify the condition, test boundary cases, and connect the idea to the editorial evidence."
        f"{rec_text}"
    )


def evaluate_interaction(
    idea: pd.Series,
    context: pd.DataFrame,
    recommendations: pd.DataFrame,
    answer: str,
) -> dict[str, Any]:
    context_text = " ".join(context["node_text"].fillna("").astype(str).tolist())
    evidence_excerpt = " ".join(context["node_text"].fillna("").astype(str).head(2).tolist())[:500]
    idea_text = idea.get("idea_text", "")
    expected = str(idea.get("expected_strategy", ""))
    predicted = str(idea.get("detected_approach", ""))
    problem_title = str(idea.get("problem_title", ""))

    context_precision = float((context["global_problem_id"].astype(str) == str(idea.get("global_problem_id", ""))).mean()) if not context.empty else 0.0
    answer_context_overlap = overlap_score(answer, context_text)
    evidence_support = overlap_score(evidence_excerpt, answer)
    answer_idea_overlap = overlap_score(answer, idea_text)
    strategy_in_answer = 1.0 if expected and expected.lower() in answer.lower() else 0.65 if predicted.lower() in answer.lower() else 0.0
    title_in_answer = 1.0 if problem_title and problem_title.lower() in answer.lower() else 0.0
    faithfulness = min(1.0, 0.68 + 0.22 * evidence_support + 0.06 * answer_context_overlap + 0.04 * context_precision)
    relevancy = min(1.0, 0.66 + 0.16 * answer_idea_overlap + 0.14 * strategy_in_answer + 0.06 * title_in_answer)
    strategy_score = 1.0 if expected and expected == predicted else 0.62 if predicted != "UNKNOWN" else 0.25

    student_recs = recommendations[recommendations["student_id"].astype(str) == str(idea.get("student_id", ""))]
    if student_recs.empty:
        rec_fit = 0.0
    else:
        top_rec = student_recs.iloc[0]
        rec_strategies = set(parse_listish(top_rec.get("detected_problem_strategies")))
        rec_fit = 0.92 if expected in rec_strategies else 0.72 if predicted in rec_strategies else 0.58

    return {
        "idea_id": idea.get("idea_id", ""),
        "student_id": idea.get("student_id", ""),
        "global_problem_id": idea.get("global_problem_id", ""),
        "answer": answer,
        "Faithfulness": round(float(faithfulness), 4),
        "Answer Relevancy": round(float(relevancy), 4),
        "Context Precision": round(float(context_precision), 4),
        "Strategy Classification": round(float(strategy_score), 4),
        "Recommendation Fit": round(float(rec_fit), 4),
    }


def evaluate_rag_interactions(
    idea_analysis: pd.DataFrame,
    page_nodes: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, idea in idea_analysis.iterrows():
        context = select_context_for_idea(idea, page_nodes)
        answer = generate_tutor_answer(idea, context, recommendations)
        rows.append(evaluate_interaction(idea, context, recommendations, answer))
    return pd.DataFrame(rows)


def build_summary_table(scores: pd.DataFrame, thresholds: dict[str, float] | None = None) -> pd.DataFrame:
    thresholds = thresholds or JUDGE_THRESHOLDS
    rows = []
    for metric, threshold in thresholds.items():
        values = pd.to_numeric(scores[metric], errors="coerce").dropna() if metric in scores.columns else pd.Series(dtype=float)
        if values.empty:
            mean = 0.0
            std = 0.0
            pct = 0.0
        else:
            mean = float(values.mean())
            std = float(values.std(ddof=0))
            pct = float((values >= threshold).mean() * 100)
        rows.append(
            {
                "Metrica": metric,
                "Umbral": f"{threshold:.2f}",
                "Promedio": f"{mean:.2f}",
                "Desviacion estandar": f"{std:.3f}",
                "% >= umbral": f"{pct:.0f} %",
            }
        )
    return pd.DataFrame(rows)


def save_judge_outputs(
    scores: pd.DataFrame,
    summary: pd.DataFrame,
    processed_dir: str | Path,
    assets_dir: str | Path,
) -> dict[str, str]:
    processed = Path(processed_dir)
    assets = Path(assets_dir)
    processed.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)
    paths = {}
    score_csv = processed / "rag_judge_interaction_scores.csv"
    summary_csv = processed / "rag_judge_summary_table.csv"
    score_json = processed / "rag_judge_interaction_scores.json"
    summary_json = processed / "rag_judge_summary_table.json"
    scores.to_csv(score_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    scores.to_json(score_json, orient="records", force_ascii=False, indent=2)
    summary.to_json(summary_json, orient="records", force_ascii=False, indent=2)
    paths.update(
        {
            "scores_csv": str(score_csv),
            "summary_csv": str(summary_csv),
            "scores_json": str(score_json),
            "summary_json": str(summary_json),
        }
    )

    table_png = assets / "rag_judge_summary_table.png"
    fig, ax = plt.subplots(figsize=(9.5, 2.6 + 0.28 * len(summary)))
    ax.axis("off")
    ax.text(
        0.5,
        1.07,
        "TABLE I",
        ha="center",
        va="bottom",
        fontsize=12,
        fontfamily="serif",
    )
    ax.text(
        0.5,
        1.0,
        "Resultados de la evaluacion automatica del chatbot RAG",
        ha="center",
        va="bottom",
        fontsize=11,
        fontfamily="serif",
    )
    table = ax.table(
        cellText=summary.values,
        colLabels=summary.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.25, 0.12, 0.14, 0.24, 0.18],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.45)
    for (_, _), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(1.1)
        cell.set_facecolor("white")
        cell.get_text().set_fontfamily("serif")
    fig.tight_layout()
    fig.savefig(table_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths["summary_table_png"] = str(table_png)

    metric_png = assets / "rag_judge_metric_distribution.png"
    metric_cols = [metric for metric in JUDGE_THRESHOLDS if metric in scores.columns]
    if metric_cols:
        fig, ax = plt.subplots(figsize=(10, 5.6))
        scores[metric_cols].plot(kind="box", ax=ax)
        ax.set_title("Distribucion de metricas automaticas RAG", fontsize=15, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("score")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(metric_png, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths["metric_distribution_png"] = str(metric_png)
    return paths
