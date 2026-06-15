"""Phase 5: run retrieval and backend comparison experiments."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
PROCESSED = ROOT / "data" / "processed"
ASSETS = ROOT / "comparison_assets"


def run_script(name: str, timeout: int = 600) -> None:
    subprocess.run([sys.executable, str(SCRIPTS / name)], cwd=ROOT, check=True, timeout=timeout)


def ensure_math_binary_demo() -> None:
    if not (PROCESSED / "math_binary_classification_report.csv").exists():
        run_script("add_math_binary_demo.py", timeout=240)


def safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main() -> None:
    ensure_math_binary_demo()
    run_script("run_hybrid_tree_prototype.py", timeout=600)
    run_script("compare_vector_backends.py", timeout=600)

    hybrid_results = safe_read_csv(PROCESSED / "hybrid_tree_search_results.csv")
    backend_metrics = safe_read_csv(ASSETS / "vector_backend_metrics.csv")
    strategy = safe_read_csv(ASSETS / "strategy_classification_by_model.csv")

    report = {
        "phase": "phase5_retrieval_evaluation",
        "status": "ok",
        "hybrid_result_rows": int(len(hybrid_results)),
        "backend_metric_rows": int(len(backend_metrics)),
        "strategy_classification_rows": int(len(strategy)),
        "mean_backend_metrics": backend_metrics.groupby("backend")[
            ["precision_at_k", "recall_at_k", "reciprocal_rank", "latency_ms_per_query"]
        ].mean().round(4).to_dict("index")
        if not backend_metrics.empty
        else {},
        "strategy_accuracy": strategy.groupby("model")["is_correct"].mean().round(4).to_dict()
        if not strategy.empty and "is_correct" in strategy.columns
        else {},
        "outputs": {
            "hybrid_results": str(PROCESSED / "hybrid_tree_search_results.csv"),
            "hybrid_recommendations": str(PROCESSED / "hybrid_tree_recommendations.csv"),
            "backend_metrics": str(ASSETS / "vector_backend_metrics.csv"),
            "backend_results": str(ASSETS / "vector_backend_results.csv"),
            "strategy_classification": str(ASSETS / "strategy_classification_by_model.csv"),
        },
    }
    (PROCESSED / "phase5_retrieval_evaluation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
