"""Run LangChain RAG over the processed CP dataset with OpenAI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from langchain_rag import DEFAULT_METRIC_THRESHOLDS, run_langchain_rag_eval
from llm.env_config import resolve_openai_settings


PROCESSED = ROOT / "data" / "processed"
ASSETS = ROOT / "comparison_assets"


def save_summary_png(summary: pd.DataFrame, path: Path, model: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 2.3))
    ax.axis("off")
    ax.text(0.5, 1.08, "TABLE II", ha="center", va="bottom", fontsize=12, fontfamily="serif")
    ax.text(
        0.5,
        0.99,
        f"Evaluacion LangChain RAG con {model}",
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
        colWidths=[0.26, 0.12, 0.15, 0.25, 0.18],
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
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LangChain RAG eval with OpenAI.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--query-limit", type=int, default=8)
    parser.add_argument("--faithfulness-threshold", type=float, default=DEFAULT_METRIC_THRESHOLDS["Faithfulness"])
    parser.add_argument("--answer-relevancy-threshold", type=float, default=DEFAULT_METRIC_THRESHOLDS["Answer Relevancy"])
    parser.add_argument(
        "--index-source",
        choices=["page_nodes", "pageindex_chunks"],
        default="page_nodes",
        help="Use raw Page Nodes or Phase 4 PageIndex-ready chunks as retrieval units.",
    )
    parser.add_argument(
        "--global-retrieval",
        action="store_true",
        help="Evaluate with corpus-wide retrieval instead of scoping retrieval to the queried problem.",
    )
    args = parser.parse_args()

    settings = resolve_openai_settings(requested_model=args.model)
    if not settings.available:
        raise SystemExit("OPENAI_API_KEY not found. Put OPENAI_API_KEY=... in .env.")

    problems_path = PROCESSED / "cp_problems_dataset.csv"
    nodes_path = PROCESSED / (
        "cp_pageindex_ready_chunks.csv" if args.index_source == "pageindex_chunks" else "cp_page_nodes_dataset.csv"
    )
    if not problems_path.exists() or not nodes_path.exists():
        raise SystemExit(f"Missing {problems_path.name} or {nodes_path.name}. Build the dataset first.")

    problems = pd.read_csv(problems_path)
    page_nodes = pd.read_csv(nodes_path)
    results, summary, metadata = run_langchain_rag_eval(
        problems,
        page_nodes,
        model=args.model,
        top_k=args.top_k,
        query_limit=args.query_limit,
        metric_thresholds={
            "Faithfulness": args.faithfulness_threshold,
            "Answer Relevancy": args.answer_relevancy_threshold,
        },
        scope_to_query_problem=not args.global_retrieval,
    )

    PROCESSED.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    suffix = args.index_source
    results_path = PROCESSED / f"langchain_openai_rag_eval_results_{suffix}.csv"
    summary_path = PROCESSED / f"langchain_openai_rag_eval_summary_{suffix}.csv"
    metadata_path = PROCESSED / "langchain_openai_rag_eval_report.json"
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    report = {
        **metadata,
        "index_source": args.index_source,
        "global_retrieval": args.global_retrieval,
        "metric_thresholds": {
            "Faithfulness": args.faithfulness_threshold,
            "Answer Relevancy": args.answer_relevancy_threshold,
        },
        "retrieval_units_csv": str(nodes_path),
        "results_csv": str(results_path),
        "summary_csv": str(summary_path),
        "summary_png": str(ASSETS / f"langchain_openai_rag_eval_summary_{suffix}.png"),
    }
    metadata_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    save_summary_png(summary, ASSETS / f"langchain_openai_rag_eval_summary_{suffix}.png", metadata["model"])

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nLANGCHAIN RAG METRICS")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
