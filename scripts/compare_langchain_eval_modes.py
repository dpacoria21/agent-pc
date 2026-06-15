"""Compare LangChain RAG evaluation modes and export presentation assets."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
ASSETS = ROOT / "comparison_assets"


MODE_FILES = {
    "Page Nodes": PROCESSED / "langchain_openai_rag_eval_summary_page_nodes.csv",
    "PageIndex Chunks": PROCESSED / "langchain_openai_rag_eval_summary_pageindex_chunks.csv",
}


def load_comparison() -> pd.DataFrame:
    rows = []
    for mode, path in MODE_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run scripts/run_langchain_openai_rag_eval.py first.")
        df = pd.read_csv(path)
        df.insert(0, "Modo de indice", mode)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def save_table_png(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 3.1))
    ax.axis("off")
    ax.text(0.5, 1.10, "TABLE II", ha="center", va="bottom", fontsize=12, fontfamily="serif")
    ax.text(
        0.5,
        1.00,
        "Resultados de la evaluacion automatica del chatbot RAG mediante LLM-as-a-Judge",
        ha="center",
        va="bottom",
        fontsize=10.5,
        fontfamily="serif",
    )
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.24, 0.20, 0.11, 0.12, 0.18, 0.13],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.3)
    table.scale(1, 1.45)
    for (_, _), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(1.1)
        cell.set_facecolor("white")
        cell.get_text().set_fontfamily("serif")
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_bar_png(df: pd.DataFrame, path: Path) -> None:
    plot_df = df.copy()
    plot_df["Promedio_num"] = pd.to_numeric(plot_df["Promedio"], errors="coerce")
    pivot = plot_df.pivot(index="Metrica", columns="Modo de indice", values="Promedio_num")
    ax = pivot.plot(kind="bar", figsize=(8.2, 4.4), color=["#2563eb", "#16a34a"], width=0.68)
    ax.set_title("Comparacion de metricas LLM-as-a-Judge", fontsize=12)
    ax.set_ylabel("Promedio")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Modo de indice", loc="upper right")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=9)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    df = load_comparison()
    out_csv = PROCESSED / "langchain_openai_rag_eval_comparison.csv"
    out_json = PROCESSED / "langchain_openai_rag_eval_comparison.json"
    out_table = ASSETS / "langchain_openai_rag_eval_comparison_table.png"
    out_bar = ASSETS / "langchain_openai_rag_eval_comparison_bar.png"
    df.to_csv(out_csv, index=False)
    df.to_json(out_json, orient="records", force_ascii=False, indent=2)
    save_table_png(df, out_table)
    save_bar_png(df, out_bar)
    print(df.to_string(index=False))
    print(f"\nSaved {out_csv}")
    print(f"Saved {out_table}")
    print(f"Saved {out_bar}")


if __name__ == "__main__":
    main()
