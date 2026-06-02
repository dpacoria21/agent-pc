"""Run a local PageIndex-like Hybrid Tree Search prototype.

The script consumes the processed CP datasets and creates:

- cp_tree_nodes_dataset.*
- cp_tree_chunks_dataset.*
- hybrid_tree_search_results.*
- hybrid_tree_recommendations.*
- comparison_assets/hybrid_tree_architecture.png
- hybrid_tree_dashboard.html
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hybrid_tree_search import HybridTreeSearcher, build_tree_chunks, build_tree_nodes


DATA = ROOT / "data" / "processed"
ASSETS = ROOT / "comparison_assets"
ASSETS.mkdir(exist_ok=True)


DEMO_QUERIES = [
    {
        "query_id": "grid_l_proof",
        "query": "I understand the grid equation but I cannot prove why the construction condition is sufficient",
        "filters": {"tags": ["math"], "min_difficulty": 800, "max_difficulty": 2200},
    },
    {
        "query_id": "unique_values_formula",
        "query": "I need a formula or binary search style strategy for the Unique Values interactive math problem",
        "filters": {"tags": ["math", "techniques", "binary search"], "min_difficulty": 1200, "max_difficulty": 2400},
    },
    {
        "query_id": "tree_mex_implementation",
        "query": "I get wrong answer on tree implementation, maybe edge cases or data structure update is wrong",
        "filters": {"tags": ["graphs", "trees", "data_structures"], "min_difficulty": 1800, "max_difficulty": 3200},
    },
    {
        "query_id": "weird_chessboard_proof",
        "query": "I need the proof idea for a constructive math pattern on a chessboard",
        "filters": {"tags": ["math"], "min_difficulty": 1200, "max_difficulty": 3600},
    },
]


def serializable_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in out.columns:
        if out[column].dtype == "object":
            out[column] = out[column].apply(
                lambda value: json.dumps(value, ensure_ascii=False, default=str)
                if isinstance(value, (dict, list))
                else value
            )
    return out


def save_df(df: pd.DataFrame, stem: str) -> dict[str, str]:
    paths = {}
    csv_path = DATA / f"{stem}.csv"
    json_path = DATA / f"{stem}.json"
    parquet_path = DATA / f"{stem}.parquet"
    serial = serializable_df(df)
    serial.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", force_ascii=False, indent=2, default_handler=str)
    try:
        serial.to_parquet(parquet_path, index=False)
        paths["parquet"] = str(parquet_path)
    except Exception:
        pass
    paths["csv"] = str(csv_path)
    paths["json"] = str(json_path)
    return paths


def box(ax, xy, w, h, label, fc="#ffffff", ec="#111827", size=9, weight="normal"):
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.2,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, label, ha="center", va="center", fontsize=size, fontweight=weight)
    return patch


def arrow(ax, start, end, color="#111827", lw=1.1):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=lw, color=color))


def generate_architecture_image(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 7.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(Rectangle((0.02, 0.05), 0.96, 0.88, fill=False, edgecolor="#111111", linewidth=2.4))
    ax.text(0.50, 0.01, "Fig. Arquitectura del prototipo RAG con Hybrid Tree Search local.", ha="center", fontsize=11)

    box(ax, (0.06, 0.58), 0.13, 0.12, "Usuario\nestudiante", "#fed7aa", "#fb923c", 10, "bold")
    box(ax, (0.05, 0.36), 0.15, 0.10, "Query\nidea + duda", "#f8fafc")
    box(ax, (0.04, 0.19), 0.07, 0.09, "PC", "#e5e7eb")
    box(ax, (0.13, 0.19), 0.07, 0.09, "Web", "#e5e7eb")

    box(ax, (0.30, 0.75), 0.17, 0.08, "Codeforces\nTutorials + API", "#dbeafe", "#2563eb", 9, "bold")
    box(ax, (0.30, 0.62), 0.17, 0.08, "AtCoder\nmetadata", "#ccfbf1", "#14b8a6", 9, "bold")
    box(ax, (0.30, 0.49), 0.17, 0.08, "Dataset builder\nscraping responsable", "#fef3c7", "#f59e0b", 9)

    box(ax, (0.53, 0.73), 0.16, 0.10, "cp_problems\nstatements + editorials", "#f8fafc", "#334155", 8)
    box(ax, (0.53, 0.58), 0.16, 0.10, "Tree Index\nproblem -> sections", "#ede9fe", "#8b5cf6", 9, "bold")
    box(ax, (0.53, 0.42), 0.16, 0.10, "Chunk Store\nTF-IDF + SVD", "#e0f2fe", "#0284c7", 9)

    box(ax, (0.76, 0.72), 0.15, 0.10, "Value Search\ncosine chunks", "#dcfce7", "#16a34a", 9)
    box(ax, (0.76, 0.56), 0.15, 0.10, "Guided Search\nintent + node type", "#fee2e2", "#ef4444", 9)
    box(ax, (0.76, 0.39), 0.15, 0.10, "Hybrid Queue\ndedup + rerank", "#fef9c3", "#ca8a04", 9, "bold")
    box(ax, (0.76, 0.22), 0.15, 0.10, "Respuesta\nrecomendacion", "#dbeafe", "#2563eb", 9, "bold")

    box(ax, (0.52, 0.20), 0.17, 0.09, "LLM opcional\nPageIndex real / agente", "#f3f4f6", "#111827", 8)
    box(ax, (0.30, 0.22), 0.17, 0.09, "Perfil estudiante\nskills + riesgos", "#ecfccb", "#65a30d", 8)

    for start, end in [
        ((0.20, 0.41), (0.30, 0.53)),
        ((0.47, 0.79), (0.53, 0.78)),
        ((0.47, 0.66), (0.53, 0.78)),
        ((0.47, 0.53), (0.53, 0.78)),
        ((0.61, 0.73), (0.61, 0.68)),
        ((0.61, 0.58), (0.61, 0.52)),
        ((0.69, 0.47), (0.76, 0.77)),
        ((0.69, 0.63), (0.76, 0.61)),
        ((0.84, 0.72), (0.84, 0.66)),
        ((0.84, 0.56), (0.84, 0.49)),
        ((0.84, 0.39), (0.84, 0.32)),
        ((0.30, 0.27), (0.76, 0.44)),
        ((0.20, 0.63), (0.76, 0.44)),
        ((0.61, 0.42), (0.61, 0.29)),
    ]:
        arrow(ax, start, end)

    ax.text(0.055, 0.86, "Prototipo local", fontsize=13, fontweight="bold")
    ax.text(0.535, 0.86, "PageIndex-like Hybrid Tree Search", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_tree_structure_image(tree_nodes: pd.DataFrame, path: Path) -> None:
    counts = tree_nodes.groupby(["depth", "node_role"]).size().reset_index(name="count")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for depth, frame in counts.groupby("depth"):
        total = int(frame["count"].sum())
        ax.barh([depth], [total], color="#2563eb", alpha=0.78)
        label = ", ".join(f"{row.node_role}: {row['count']}" for _, row in frame.iterrows())
        ax.text(total + 0.3, depth, label, va="center", fontsize=9)
    ax.set_title("Estructura del arbol generado")
    ax.set_xlabel("Cantidad de nodos")
    ax.set_ylabel("Profundidad")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def generate_score_image(results: pd.DataFrame, path: Path) -> None:
    top = results.head(10).copy()
    if top.empty:
        return
    labels = top["tree_node_id"].astype(str).str.replace("content::", "", regex=False).str[:38]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(labels, top["value_score"], label="value/chunks", color="#2563eb")
    ax.barh(labels, top["guided_score"], left=top["value_score"], label="guided", color="#14b8a6")
    ax.barh(
        labels,
        top["metadata_bonus"],
        left=top["value_score"] + top["guided_score"],
        label="metadata",
        color="#f59e0b",
    )
    ax.set_title("Componentes del ranking Hybrid Tree Search")
    ax.set_xlabel("Score")
    ax.invert_yaxis()
    ax.legend()
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def generate_recommendation_image(recs: pd.DataFrame, path: Path) -> None:
    if recs.empty:
        return
    top = recs.head(8).copy()
    labels = top["global_problem_id"] + "\n" + top["title"].astype(str).str[:28]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(labels, top["recommendation_score"], color="#8b5cf6")
    ax.set_title("Problemas recomendados por score agregado")
    ax.set_ylabel("Recommendation score")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def table_html(df: pd.DataFrame, columns: list[str], max_rows: int = 10) -> str:
    if df.empty:
        return "<p>No hay resultados.</p>"
    rows = []
    for _, row in df.head(max_rows).iterrows():
        cells = "".join(f"<td>{html.escape(str(row.get(col, ''))[:420])}</td>" for col in columns)
        rows.append(f"<tr>{cells}</tr>")
    headers = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def write_dashboard(summary: dict[str, Any], result_df: pd.DataFrame, rec_df: pd.DataFrame) -> Path:
    html_path = ROOT / "hybrid_tree_dashboard.html"
    style = """
    <style>
    body{font-family:Arial, sans-serif; margin:28px; color:#111827; background:#f8fafc;}
    h1,h2{margin-bottom:8px}
    .card{background:white; border:1px solid #e5e7eb; border-radius:8px; padding:18px; margin:16px 0;}
    img{max-width:100%; border:1px solid #e5e7eb; background:white}
    table{border-collapse:collapse; width:100%; font-size:13px}
    th,td{border:1px solid #e5e7eb; padding:8px; vertical-align:top}
    th{background:#eef2ff; text-align:left}
    code{background:#e5e7eb; padding:2px 4px; border-radius:4px}
    </style>
    """
    content = f"""
    <!doctype html><html><head><meta charset="utf-8"><title>Hybrid Tree Search CP RAG</title>{style}</head>
    <body>
    <h1>Prototipo CP RAG: Hybrid Tree Search local</h1>
    <div class="card">
      <h2>Resumen</h2>
      <pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2, default=str))}</pre>
    </div>
    <div class="card">
      <h2>Arquitectura</h2>
      <img src="comparison_assets/hybrid_tree_architecture.png" alt="Arquitectura Hybrid Tree Search">
    </div>
    <div class="card">
      <h2>Estructura del arbol</h2>
      <img src="comparison_assets/hybrid_tree_structure.png" alt="Estructura del arbol">
    </div>
    <div class="card">
      <h2>Componentes del score</h2>
      <img src="comparison_assets/hybrid_tree_score_components.png" alt="Score components">
    </div>
    <div class="card">
      <h2>Recomendaciones</h2>
      <img src="comparison_assets/hybrid_tree_recommendations.png" alt="Recomendaciones">
      {table_html(rec_df, ["query_id", "global_problem_id", "title", "recommendation_score", "reason"], 12)}
    </div>
    <div class="card">
      <h2>Top nodos recuperados</h2>
      {table_html(result_df, ["query_id", "rank", "node_type", "title", "hybrid_score", "value_score", "guided_score", "metadata_bonus", "reason"], 24)}
    </div>
    </body></html>
    """
    html_path.write_text(content, encoding="utf-8")
    return html_path


def main() -> None:
    problems_path = DATA / "cp_problems_dataset.csv"
    page_nodes_path = DATA / "cp_page_nodes_dataset.csv"
    if not problems_path.exists() or not page_nodes_path.exists():
        raise SystemExit("Faltan cp_problems_dataset.csv o cp_page_nodes_dataset.csv en data/processed.")

    problems = pd.read_csv(problems_path)
    page_nodes = pd.read_csv(page_nodes_path)
    tree_nodes = build_tree_nodes(problems, page_nodes)
    chunks = build_tree_chunks(tree_nodes)

    tree_paths = save_df(tree_nodes, "cp_tree_nodes_dataset")
    chunk_paths = save_df(chunks, "cp_tree_chunks_dataset")

    searcher = HybridTreeSearcher(tree_nodes=tree_nodes, chunks=chunks)
    embedding_meta = searcher.fit()

    all_results = []
    all_recs = []
    run_rows = []
    for spec in DEMO_QUERIES:
        output = searcher.hybrid_search(spec["query"], filters=spec["filters"], top_k_nodes=12, top_k_chunks=24)
        results = output["results"].copy()
        recs = output["recommendations"].copy()
        if not results.empty:
            results.insert(0, "rank", range(1, len(results) + 1))
            results.insert(0, "query_id", spec["query_id"])
            results.insert(1, "query", spec["query"])
            all_results.append(results)
        if not recs.empty:
            recs.insert(0, "query_id", spec["query_id"])
            recs.insert(1, "query", spec["query"])
            all_recs.append(recs)
        run_rows.append(
            {
                "query_id": spec["query_id"],
                "query": spec["query"],
                "stage": output["intent"]["stage"],
                "approaches": output["intent"]["approaches"],
                "enough_information": output["enough_information"],
                "elapsed_seconds": round(output["elapsed_seconds"], 4),
                "top_result": results.iloc[0]["tree_node_id"] if not results.empty else "",
            }
        )

    result_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    rec_df = pd.concat(all_recs, ignore_index=True) if all_recs else pd.DataFrame()
    run_df = pd.DataFrame(run_rows)

    result_paths = save_df(result_df, "hybrid_tree_search_results")
    rec_paths = save_df(rec_df, "hybrid_tree_recommendations")
    run_paths = save_df(run_df, "hybrid_tree_demo_runs")

    architecture_path = ASSETS / "hybrid_tree_architecture.png"
    tree_structure_path = ASSETS / "hybrid_tree_structure.png"
    score_path = ASSETS / "hybrid_tree_score_components.png"
    rec_path = ASSETS / "hybrid_tree_recommendations.png"
    generate_architecture_image(architecture_path)
    generate_tree_structure_image(tree_nodes, tree_structure_path)
    first_query_results = result_df[result_df["query_id"] == DEMO_QUERIES[0]["query_id"]] if not result_df.empty else pd.DataFrame()
    generate_score_image(first_query_results, score_path)
    generate_recommendation_image(rec_df, rec_path)

    summary = {
        "problems": int(len(problems)),
        "page_nodes": int(len(page_nodes)),
        "tree_nodes": int(len(tree_nodes)),
        "chunks": int(len(chunks)),
        "queries": len(DEMO_QUERIES),
        "embedding": embedding_meta,
        "tree_paths": tree_paths,
        "chunk_paths": chunk_paths,
        "result_paths": result_paths,
        "recommendation_paths": rec_paths,
        "run_paths": run_paths,
        "architecture_image": str(architecture_path),
    }
    (DATA / "hybrid_tree_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    dashboard_path = write_dashboard(summary, result_df, rec_df)

    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print("\nDemo runs")
    print(run_df.to_string(index=False))
    print(f"\nDashboard: {dashboard_path}")
    print(f"Architecture image: {architecture_path}")


if __name__ == "__main__":
    main()
