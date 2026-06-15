"""Rebuild LangChain RAG summary tables from existing per-query results.

This script does not call OpenAI. It recalculates the summary rows from the
stored real scores and applies transparent thresholds, defaulting to 0.75.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from run_langchain_openai_rag_eval import save_summary_png


ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
ASSETS = ROOT / "comparison_assets"
MODES = ["page_nodes", "pageindex_chunks"]


def build_summary(results: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    rows = []
    for metric, threshold in thresholds.items():
        values = pd.to_numeric(results[metric], errors="coerce")
        rows.append(
            {
                "Metrica": metric,
                "Umbral": f"{threshold:.2f}",
                "Promedio": f"{values.mean():.2f}",
                "Desviacion estandar": f"{values.std(ddof=0):.3f}",
                "% >= umbral": f"{(values >= threshold).mean() * 100:.0f} %",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--faithfulness-threshold", type=float, default=0.75)
    parser.add_argument("--answer-relevancy-threshold", type=float, default=0.75)
    args = parser.parse_args()

    thresholds = {
        "Faithfulness": args.faithfulness_threshold,
        "Answer Relevancy": args.answer_relevancy_threshold,
    }
    outputs = []
    for mode in MODES:
        results_path = PROCESSED / f"langchain_openai_rag_eval_results_{mode}.csv"
        if not results_path.exists():
            raise FileNotFoundError(f"Missing {results_path}")
        results = pd.read_csv(results_path)
        summary = build_summary(results, thresholds)
        summary_path = PROCESSED / f"langchain_openai_rag_eval_summary_{mode}.csv"
        png_path = ASSETS / f"langchain_openai_rag_eval_summary_{mode}.png"
        summary.to_csv(summary_path, index=False)
        save_summary_png(summary, png_path)
        outputs.append({"mode": mode, "summary": str(summary_path), "png": str(png_path)})
        print(f"\n{mode}")
        print(summary.to_string(index=False))
    print("\nUpdated summaries:")
    for item in outputs:
        print(item)


if __name__ == "__main__":
    main()
