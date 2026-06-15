"""Export curated reference answers for the Streamlit demo.

These answers are intentionally labeled as reference/demo material. They are
not used as automatic evaluation results.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit_app import DEMO_PROBLEMS, REFERENCE_ANSWERS  # noqa: E402


OUT_DIR = ROOT / "comparison_assets"
PROCESSED = ROOT / "data" / "processed"


def build_table() -> pd.DataFrame:
    rows = []
    for problem_id, spec in DEMO_PROBLEMS.items():
        for question in spec["questions"]:
            answer = REFERENCE_ANSWERS.get(problem_id, {}).get(question, "")
            rows.append(
                {
                    "problem_id": problem_id,
                    "problem": spec["label"],
                    "query": question,
                    "reference_answer": answer,
                    "usage": "demo_reference_not_metric",
                }
            )
    return pd.DataFrame(rows)


def save_png(df: pd.DataFrame, path: Path) -> None:
    preview = df[["problem_id", "query", "reference_answer"]].copy()
    preview["query"] = preview["query"].apply(lambda x: textwrap.fill(str(x), 30))
    preview["reference_answer"] = preview["reference_answer"].apply(lambda x: textwrap.fill(str(x), 76))

    fig, ax = plt.subplots(figsize=(17, 14))
    ax.axis("off")
    ax.text(0.5, 1.03, "Respuestas de referencia curadas para demo", ha="center", fontsize=16, weight="bold")
    ax.text(
        0.5,
        0.99,
        "Material de exposicion: no reemplaza la evaluacion automatica ni altera metricas reales",
        ha="center",
        fontsize=10,
    )
    table = ax.table(
        cellText=preview.values,
        colLabels=["Problema", "Consulta", "Respuesta de referencia"],
        loc="center",
        cellLoc="left",
        colLoc="center",
        colWidths=[0.12, 0.25, 0.63],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.4)
    table.scale(1, 3.1)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#111827")
        cell.set_linewidth(0.8)
        cell.set_facecolor("#f3f4f6" if row == 0 else "white")
        cell.get_text().set_va("center")
        if row == 0:
            cell.set_height(0.045)
            cell.get_text().set_weight("bold")
        else:
            cell.set_height(0.088)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df = build_table()
    csv_path = PROCESSED / "demo_reference_answers.csv"
    json_path = PROCESSED / "demo_reference_answers.json"
    png_path = OUT_DIR / "demo_reference_answers_table.png"
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)
    save_png(df, png_path)
    print(f"Saved {csv_path}")
    print(f"Saved {json_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
