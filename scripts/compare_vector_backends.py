"""Compare local matrix search, FAISS, and ChromaDB on the same Page Nodes."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Callable

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from search_chromadb import search_chromadb_ephemeral
from search_faiss import search_faiss_index_flat_ip
from search_helpers import (
    BACKEND_QUERIES,
    build_documents,
    build_tfidf_svd_embeddings,
    current_heuristic_strategy_rows,
    evaluate_backend,
    evaluate_strategy_classification,
    load_nodes,
    plot_backend_metrics,
    plot_backend_overlap,
    plot_strategy_classification,
)
from search_local import search_local_matrix


PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "comparison_assets"
OUT.mkdir(exist_ok=True)

NODES_PATH = PROCESSED / "cp_page_nodes_dataset.csv"
IDEAS_PATH = PROCESSED / "math_binary_classification_report.csv"
TOP_K = 8


def run_backend(
    backend_name: str,
    search_fn: Callable[[], list],
    nodes: pd.DataFrame,
) -> tuple[list[dict], list[dict], dict]:
    try:
        start = time.perf_counter()
        outputs = search_fn()
        elapsed = time.perf_counter() - start
        rows, details = evaluate_backend(
            backend_name=backend_name,
            outputs=outputs,
            nodes=nodes,
            specs=BACKEND_QUERIES,
            elapsed_seconds=elapsed,
            top_k=TOP_K,
        )
        status = {"status": "ok", "elapsed_seconds": elapsed}
        return rows, details, status
    except Exception as exc:
        status = {"status": "unavailable", "error": repr(exc)}
        return [], [], status


def main() -> None:
    nodes = load_nodes(NODES_PATH)
    idea_rows = pd.read_csv(IDEAS_PATH) if IDEAS_PATH.exists() else pd.DataFrame()
    documents = build_documents(nodes)
    backend_query_texts = [item["query"] for item in BACKEND_QUERIES]
    idea_query_texts = idea_rows["idea_text"].tolist() if not idea_rows.empty else []
    all_query_texts = backend_query_texts + idea_query_texts
    node_embeddings, all_query_embeddings, embedding_meta = build_tfidf_svd_embeddings(
        documents=documents,
        queries=all_query_texts,
    )
    backend_query_embeddings = all_query_embeddings[: len(backend_query_texts)]
    idea_query_embeddings = all_query_embeddings[len(backend_query_texts) :]

    backend_specs = [
        (
            "local_matrix",
            lambda: search_local_matrix(
                node_embeddings=node_embeddings,
                query_embeddings=backend_query_embeddings,
                top_k=TOP_K,
            ),
        ),
        (
            "faiss",
            lambda: search_faiss_index_flat_ip(
                node_embeddings=node_embeddings,
                query_embeddings=backend_query_embeddings,
                top_k=TOP_K,
            ),
        ),
        (
            "chromadb",
            lambda: search_chromadb_ephemeral(
                nodes=nodes,
                documents=documents,
                node_embeddings=node_embeddings,
                query_embeddings=backend_query_embeddings,
                top_k=TOP_K,
            ),
        ),
    ]

    all_metrics = []
    all_details = []
    all_strategy_rows = current_heuristic_strategy_rows(idea_rows) if not idea_rows.empty else []
    statuses = {}
    for backend_name, search_fn in backend_specs:
        rows, details, status = run_backend(backend_name, search_fn, nodes)
        all_metrics.extend(rows)
        all_details.extend(details)
        statuses[backend_name] = status

        if status.get("status") == "ok" and not idea_rows.empty:
            if backend_name == "local_matrix":
                idea_outputs = search_local_matrix(node_embeddings, idea_query_embeddings, TOP_K)
            elif backend_name == "faiss":
                idea_outputs = search_faiss_index_flat_ip(node_embeddings, idea_query_embeddings, TOP_K)
            elif backend_name == "chromadb":
                idea_outputs = search_chromadb_ephemeral(nodes, documents, node_embeddings, idea_query_embeddings, TOP_K)
            else:
                idea_outputs = []
            all_strategy_rows.extend(
                evaluate_strategy_classification(backend_name, idea_outputs, nodes, idea_rows)
            )

    metrics_df = pd.DataFrame(all_metrics)
    details_df = pd.DataFrame(all_details)
    strategy_df = pd.DataFrame(all_strategy_rows)
    metrics_df.to_csv(OUT / "vector_backend_metrics.csv", index=False)
    details_df.to_csv(OUT / "vector_backend_results.csv", index=False)
    strategy_df.to_csv(OUT / "strategy_classification_by_model.csv", index=False)
    metrics_df.to_json(OUT / "vector_backend_metrics.json", orient="records", force_ascii=False, indent=2)
    details_df.to_json(OUT / "vector_backend_results.json", orient="records", force_ascii=False, indent=2)
    strategy_df.to_json(OUT / "strategy_classification_by_model.json", orient="records", force_ascii=False, indent=2)

    if not metrics_df.empty:
        plot_backend_metrics(metrics_df, OUT)
    if not details_df.empty and details_df["backend"].nunique() > 1:
        plot_backend_overlap(details_df, OUT)
    if not strategy_df.empty:
        plot_strategy_classification(strategy_df, OUT)

    summary = {
        "nodes_indexed": int(len(nodes)),
        "queries": BACKEND_QUERIES,
        "top_k": TOP_K,
        "embedding": embedding_meta,
        "backend_status": statuses,
        "why_results_match": (
            "All backends receive the same L2-normalized TF-IDF+SVD vectors and use cosine-equivalent "
            "similarity. Local matrix and FAISS IndexFlatIP are exact under this setup; ChromaDB also "
            "returns the same top-k on this small dataset. Differences should mainly appear with larger "
            "datasets, approximate indexes, different embedding models, filtering, persistence settings, "
            "or backend-specific indexing parameters."
        ),
        "strategy_classification": strategy_df.groupby("model")["is_correct"].mean().round(4).to_dict()
        if not strategy_df.empty
        else {},
        "mean_metrics": metrics_df.groupby("backend")[
            ["precision_at_k", "recall_at_k", "reciprocal_rank", "avg_score", "latency_ms_per_query"]
        ].mean().round(4).to_dict("index")
        if not metrics_df.empty
        else {},
    }
    (OUT / "vector_backend_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
