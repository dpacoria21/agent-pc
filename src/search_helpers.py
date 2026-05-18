"""Shared utilities for comparing vector-search backends.

These helpers keep dataset loading, embedding generation, metric calculation,
and plotting separate from backend-specific search engines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


RANDOM_SEED = 42


BACKEND_QUERIES = [
    {
        "query_id": "binary_quadratic_time",
        "query": "binary search maximum x quadratic inequality contest time",
        "expected_problem_ids": ["codeforces_750_A"],
        "preferred_node_types": ["STATEMENT", "EDITORIAL_ALGORITHM", "EDITORIAL_OBSERVATION", "EDITORIAL_PROOF"],
    },
    {
        "query_id": "integer_square_root",
        "query": "integer square root perfect square sum",
        "expected_problem_ids": ["codeforces_1915_C"],
        "preferred_node_types": ["STATEMENT", "EDITORIAL_ALGORITHM", "COMMON_MISTAKES"],
    },
    {
        "query_id": "triangular_discriminant",
        "query": "triangular numbers discriminant formula binary search",
        "expected_problem_ids": ["codeforces_192_A"],
        "preferred_node_types": ["EDITORIAL_ALGORITHM", "EDITORIAL_OBSERVATION", "EDITORIAL_PROOF"],
    },
    {
        "query_id": "dp_transition",
        "query": "dp state transition editorial algorithm",
        "expected_problem_ids": [],
        "preferred_node_types": ["EDITORIAL_ALGORITHM", "EDITORIAL_OBSERVATION"],
        "tag_hints": ["dynamic_programming", "dp"],
    },
    {
        "query_id": "greedy_proof_edges",
        "query": "greedy proof edge cases common mistakes",
        "expected_problem_ids": [],
        "preferred_node_types": ["EDITORIAL_PROOF", "COMMON_MISTAKES", "EDITORIAL_OBSERVATION"],
        "tag_hints": ["greedy"],
    },
]


def parse_listish(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if pd.isna(value) or value == "":
        return []
    text = str(value)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except Exception:
        pass
    if text.startswith("[") and text.endswith("]"):
        return [part.strip(" '\"") for part in text.strip("[]").split(",") if part.strip()]
    return [text]


def load_nodes(nodes_path: Path) -> pd.DataFrame:
    nodes = pd.read_csv(nodes_path)
    nodes["node_text"] = nodes["node_text"].fillna("").astype(str)
    nodes["node_title"] = nodes["node_title"].fillna("").astype(str)
    nodes["normalized_tags_list"] = nodes["normalized_tags"].apply(parse_listish)
    nodes["topic_group_list"] = nodes["topic_group"].apply(parse_listish)
    nodes = nodes[nodes["node_text"].str.len() > 0].copy()
    return nodes.reset_index(drop=True)


def build_documents(nodes: pd.DataFrame) -> list[str]:
    documents = []
    for _, row in nodes.iterrows():
        tags = " ".join(row["normalized_tags_list"])
        topics = " ".join(row["topic_group_list"])
        documents.append(
            " ".join(
                [
                    str(row["node_title"]),
                    str(row["node_type"]),
                    str(row["node_text"]),
                    tags,
                    topics,
                ]
            )
        )
    return documents


def build_tfidf_svd_embeddings(
    documents: list[str],
    queries: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )
    tfidf = vectorizer.fit_transform(documents)
    n_components = min(128, max(2, min(tfidf.shape) - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)
    node_embeddings = normalize(svd.fit_transform(tfidf)).astype("float32")
    query_embeddings = normalize(svd.transform(vectorizer.transform(queries))).astype("float32")
    metadata = {
        "embedding_backend": "tfidf_svd_normalized",
        "tfidf_shape": list(tfidf.shape),
        "embedding_dim": int(node_embeddings.shape[1]),
        "n_components": int(n_components),
        "normalization": "l2",
    }
    return node_embeddings, query_embeddings, metadata


def relevant_node_ids(nodes: pd.DataFrame, spec: dict[str, Any]) -> set[str]:
    mask = pd.Series(False, index=nodes.index)
    expected_problem_ids = set(spec.get("expected_problem_ids", []))
    if expected_problem_ids:
        mask |= nodes["global_problem_id"].isin(expected_problem_ids)

    tag_hints = set(spec.get("tag_hints", []))
    if tag_hints:
        mask |= nodes["normalized_tags_list"].apply(lambda tags: bool(set(tags).intersection(tag_hints)))

    preferred = set(spec.get("preferred_node_types", []))
    if preferred:
        mask &= nodes["node_type"].isin(preferred)

    selected = nodes.loc[mask, "node_id"].tolist()
    if not selected and expected_problem_ids:
        selected = nodes.loc[nodes["global_problem_id"].isin(expected_problem_ids), "node_id"].tolist()
    return set(selected)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not retrieved:
        return 0.0
    return len(set(retrieved[:k]).intersection(relevant)) / len(retrieved[:k])


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]).intersection(relevant)) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for index, node_id in enumerate(retrieved, start=1):
        if node_id in relevant:
            return 1.0 / index
    return 0.0


def evaluate_backend(
    backend_name: str,
    outputs: list[tuple[np.ndarray, np.ndarray]],
    nodes: pd.DataFrame,
    specs: list[dict[str, Any]],
    elapsed_seconds: float,
    top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    detail_rows = []
    per_query_latency_ms = elapsed_seconds * 1000 / max(1, len(specs))
    for spec, (indices, scores) in zip(specs, outputs):
        retrieved = nodes.iloc[indices]["node_id"].tolist()
        relevant = relevant_node_ids(nodes, spec)
        rows.append(
            {
                "backend": backend_name,
                "query_id": spec["query_id"],
                "query": spec["query"],
                "precision_at_k": round(precision_at_k(retrieved, relevant, top_k), 4),
                "recall_at_k": round(recall_at_k(retrieved, relevant, top_k), 4),
                "reciprocal_rank": round(reciprocal_rank(retrieved, relevant), 4),
                "avg_score": round(float(np.mean(scores)) if len(scores) else 0.0, 4),
                "latency_ms_per_query": round(per_query_latency_ms, 3),
                "relevant_count": int(len(relevant)),
            }
        )
        for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
            row = nodes.iloc[int(idx)]
            detail_rows.append(
                {
                    "backend": backend_name,
                    "query_id": spec["query_id"],
                    "rank": rank,
                    "node_id": row["node_id"],
                    "global_problem_id": row["global_problem_id"],
                    "node_type": row["node_type"],
                    "score": round(float(score), 6),
                    "is_relevant": row["node_id"] in relevant,
                }
            )
    return rows, detail_rows


def plot_backend_metrics(metrics: pd.DataFrame, out_dir: Path) -> None:
    grouped = metrics.groupby("backend")[["precision_at_k", "recall_at_k", "reciprocal_rank"]].mean()
    fig, ax = plt.subplots(figsize=(10, 5.6))
    grouped.plot(kind="bar", ax=ax, color=["#2563eb", "#14b8a6", "#f59e0b"])
    ax.set_title("Comparacion de backends vectoriales", fontsize=15, fontweight="bold")
    ax.set_xlabel("backend")
    ax.set_ylabel("score promedio")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "vector_backend_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    latency = metrics.groupby("backend")["latency_ms_per_query"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    latency.plot(kind="bar", ax=ax, color="#8b5cf6")
    ax.set_title("Latencia promedio por backend", fontsize=15, fontweight="bold")
    ax.set_xlabel("backend")
    ax.set_ylabel("ms/query")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "vector_backend_latency.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_backend_overlap(details: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    backends = sorted(details["backend"].unique())
    for query_id, group in details.groupby("query_id"):
        sets = {backend: set(group[group["backend"] == backend]["node_id"]) for backend in backends}
        for left in backends:
            for right in backends:
                denom = len(sets[left].union(sets[right])) or 1
                rows.append(
                    {
                        "query_id": query_id,
                        "left": left,
                        "right": right,
                        "jaccard": len(sets[left].intersection(sets[right])) / denom,
                    }
                )

    overlap_df = pd.DataFrame(rows)
    matrix = overlap_df.groupby(["left", "right"])["jaccard"].mean().unstack().loc[backends, backends]
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(backends)), backends, rotation=25, ha="right")
    ax.set_yticks(range(len(backends)), backends)
    ax.set_title("Overlap top-k promedio entre backends", fontsize=14, fontweight="bold")
    for i in range(len(backends)):
        for j in range(len(backends)):
            ax.text(j, i, f"{matrix.values[i, j]:.2f}", ha="center", va="center", color="#0f172a")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "vector_backend_overlap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def classify_strategy_from_context(
    idea_text: str,
    retrieved_texts: list[str],
) -> tuple[str, int, int]:
    """Classify an idea as binary-search or formula reasoning.

    The idea text is treated as the strongest signal. Retrieved context is used
    as supporting evidence, which lets different retrieval backends influence
    the final label when the idea itself is ambiguous.
    """
    idea = str(idea_text).lower()
    context = " ".join([idea, *[str(text).lower() for text in retrieved_texts]])

    binary_terms = [
        "binary search",
        "monotonic",
        "predicate",
        "lower_bound",
        "upper_bound",
        "maximum x",
        "search the maximum",
        "search if",
    ]
    formula_terms = [
        "formula",
        "quadratic",
        "discriminant",
        "sqrt",
        "square root",
        "integer root",
        "perfect square",
        "root*root",
        "1+8",
        "x^2",
    ]

    binary_score = sum(context.count(term) for term in binary_terms)
    formula_score = sum(context.count(term) for term in formula_terms)

    if "binary search" in idea or "monotonic" in idea or "predicate" in idea:
        return "BINARY_SEARCH", binary_score, formula_score
    if any(term in idea for term in ["formula", "quadratic", "discriminant"]):
        return "MATH_FORMULA", binary_score, formula_score
    if any(term in idea for term in ["sqrt", "square root", "integer root", "perfect square", "root*root"]):
        return "MATH_FORMULA", binary_score, formula_score

    if binary_score > formula_score:
        return "BINARY_SEARCH", binary_score, formula_score
    if formula_score > binary_score:
        return "MATH_FORMULA", binary_score, formula_score
    return "UNKNOWN", binary_score, formula_score


def evaluate_strategy_classification(
    backend_name: str,
    outputs: list[tuple[np.ndarray, np.ndarray]],
    nodes: pd.DataFrame,
    idea_rows: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows = []
    for (_, idea), (indices, scores) in zip(idea_rows.iterrows(), outputs):
        retrieved = nodes.iloc[indices].copy()
        predicted, binary_score, formula_score = classify_strategy_from_context(
            idea_text=idea["idea_text"],
            retrieved_texts=retrieved["node_text"].head(5).tolist(),
        )
        expected = str(idea["fine_strategy"])
        rows.append(
            {
                "model": backend_name,
                "student_id": idea["student_id"],
                "problem_title": idea["problem_title"],
                "global_problem_id": idea["global_problem_id"],
                "idea_text": idea["idea_text"],
                "expected_strategy": expected,
                "predicted_strategy": predicted,
                "is_correct": predicted == expected,
                "binary_score": binary_score,
                "formula_score": formula_score,
                "top_problem_ids": json.dumps(retrieved["global_problem_id"].head(3).tolist()),
                "top_node_types": json.dumps(retrieved["node_type"].head(3).tolist()),
                "top_scores": json.dumps([round(float(score), 6) for score in scores[:3]]),
            }
        )
    return rows


def current_heuristic_strategy_rows(idea_rows: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for _, idea in idea_rows.iterrows():
        predicted = str(idea["current_approach"])
        expected = str(idea["fine_strategy"])
        rows.append(
            {
                "model": "current_heuristic",
                "student_id": idea["student_id"],
                "problem_title": idea["problem_title"],
                "global_problem_id": idea["global_problem_id"],
                "idea_text": idea["idea_text"],
                "expected_strategy": expected,
                "predicted_strategy": predicted,
                "is_correct": predicted == expected,
                "binary_score": "",
                "formula_score": "",
                "top_problem_ids": "[]",
                "top_node_types": "[]",
                "top_scores": "[]",
            }
        )
    return rows


def plot_strategy_classification(strategy_df: pd.DataFrame, out_dir: Path) -> None:
    counts = (
        strategy_df.groupby(["model", "predicted_strategy"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(10, 5.8))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Clasificacion binary vs formula por modelo", fontsize=15, fontweight="bold")
    ax.set_xlabel("modelo")
    ax.set_ylabel("ideas clasificadas")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "strategy_prediction_by_model.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    accuracy = (
        strategy_df.groupby(["model", "expected_strategy"])["is_correct"]
        .mean()
        .unstack(fill_value=0)
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(10, 5.8))
    accuracy.plot(kind="bar", ax=ax, color=["#14b8a6", "#8b5cf6"])
    ax.set_title("Exactitud por modelo y estrategia esperada", fontsize=15, fontweight="bold")
    ax.set_xlabel("modelo")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "strategy_accuracy_by_model.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
