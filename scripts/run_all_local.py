"""Local end-to-end runner for the CP RAG thesis prototype.

This script builds a small processed dataset, regenerates analysis assets,
extracts RAG metrics from the local Python prototype, and writes a
presentation-ready report.
"""

from __future__ import annotations

import argparse
import copy
import json
import html
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = ROOT / "scripts"
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cp_dataset_scraper as cpds

DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ASSETS_DIR = ROOT / "comparison_assets"
COMPARISON_PNGS = {
    "page_index_comparison.png",
    "rag_retrieval_metrics.png",
    "retrieval_strategy_scores.png",
    "retrieval_method_overlap.png",
    "retrieval_average_scores.png",
    "math_binary_strategy_classification.png",
    "math_binary_problem_strategy_map.png",
    "vector_backend_metrics.png",
    "vector_backend_latency.png",
    "vector_backend_overlap.png",
    "strategy_prediction_by_model.png",
    "strategy_accuracy_by_model.png",
}


def run_command(args: list[str], timeout: int = 600) -> None:
    print("\n> " + " ".join(args))
    subprocess.run(args, cwd=ROOT, check=True, timeout=timeout)


def build_local_config(max_cf: int, max_atcoder: int, with_content: bool, request_delay: float) -> dict:
    cfg = copy.deepcopy(cpds.CONFIG)
    cfg["platforms"] = ["codeforces", "atcoder"]

    cfg["codeforces"]["max_problems"] = max_cf
    cfg["codeforces"]["tags"] = ["dp", "greedy", "graphs"]
    cfg["codeforces"]["min_rating"] = 800
    cfg["codeforces"]["max_rating"] = 1800
    cfg["codeforces"]["tag_match_mode"] = "any"
    cfg["codeforces"]["sort_by"] = "rating"
    cfg["codeforces"]["sort_order"] = "asc"
    cfg["codeforces"]["download_statements"] = with_content
    cfg["codeforces"]["download_editorials"] = with_content

    cfg["atcoder"]["max_problems"] = max_atcoder
    cfg["atcoder"]["contest_prefixes"] = ["abc", "arc", "dp"]
    cfg["atcoder"]["min_difficulty"] = 0
    cfg["atcoder"]["max_difficulty"] = 1800
    cfg["atcoder"]["sort_by"] = "difficulty"
    cfg["atcoder"]["sort_order"] = "asc"
    cfg["atcoder"]["download_statements"] = with_content
    cfg["atcoder"]["download_editorials"] = with_content

    cfg["scraping"]["use_cache"] = True
    cfg["scraping"]["request_delay_seconds"] = request_delay
    cfg["scraping"]["cache_dir"] = str(DATA_DIR / "cache")

    cfg["output"]["raw_dir"] = str(DATA_DIR / "raw")
    cfg["output"]["processed_dir"] = str(PROCESSED_DIR)
    cfg["output"]["save_raw"] = True
    cfg["output"]["save_csv"] = True
    cfg["output"]["save_json"] = True
    cfg["output"]["save_parquet"] = True
    cfg["output"]["build_page_nodes"] = True
    return cfg


def write_lightweight_quality_files(result: dict, mode: str) -> None:
    problems = result["problems_dataset"]
    nodes = result["page_nodes_dataset"]
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if problems.empty:
        quality = {
            "total_problems": 0,
            "total_page_nodes": 0,
            "status": "empty_dataset",
        }
    else:
        quality = {
            "total_problems": int(len(problems)),
            "total_by_platform": problems["source"].value_counts().to_dict(),
            "difficulty_min": None
            if problems["normalized_difficulty"].dropna().empty
            else float(problems["normalized_difficulty"].min()),
            "difficulty_max": None
            if problems["normalized_difficulty"].dropna().empty
            else float(problems["normalized_difficulty"].max()),
            "statement_status": problems["statement_status"].value_counts().to_dict(),
            "editorial_status": problems["editorial_status"].value_counts().to_dict(),
            "total_page_nodes": int(len(nodes)),
            "node_type_distribution": nodes["node_type"].value_counts().to_dict()
            if not nodes.empty
            else {},
            "processed_files": sorted(path.name for path in PROCESSED_DIR.glob("*") if path.is_file()),
        }

    requested = (
        int(result["config"]["codeforces"].get("max_problems") or 0)
        + int(result["config"]["atcoder"].get("max_problems") or 0)
    )
    summary = {
        "status": "ready_for_comparison_demo",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "platforms_used": sorted(problems["source"].dropna().unique().tolist())
        if not problems.empty
        else [],
        "requested_problem_count": requested,
        "obtained_problem_count": int(len(problems)),
        "total_statements_downloaded": int((problems["statement_status"] == "downloaded").sum())
        if not problems.empty
        else 0,
        "total_editorials_downloaded": int((problems["editorial_status"] == "downloaded").sum())
        if not problems.empty
        else 0,
        "total_page_nodes": int(len(nodes)),
        "paths": result.get("paths", {}),
    }

    (PROCESSED_DIR / "first_run_quality_report.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (PROCESSED_DIR / "first_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def build_dataset(max_cf: int, max_atcoder: int, with_content: bool, request_delay: float) -> None:
    mode = "content_enabled" if with_content else "metadata_only"
    print(f"\n[1/7] Building dataset ({mode})...")
    cfg = build_local_config(max_cf, max_atcoder, with_content, request_delay)
    result = cpds.build_cp_dataset(cfg)
    write_lightweight_quality_files(result, mode)
    print(
        json.dumps(
            {
                "problems": len(result["problems_dataset"]),
                "page_nodes": len(result["page_nodes_dataset"]),
                "paths": result["paths"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def add_math_binary_demo() -> None:
    print("\n[2/7] Adding math-vs-binary-search demo problems...")
    run_command([sys.executable, str(SCRIPT_DIR / "add_math_binary_demo.py")], timeout=180)


def regenerate_assets() -> None:
    print("\n[3/7] Generating dataset evidence charts...")
    run_command([sys.executable, str(SCRIPT_DIR / "generate_presentation_evidence.py")], timeout=240)

    print("\n[4/7] Extracting RAG metrics from the local Python prototype...")
    run_command([sys.executable, str(SCRIPT_DIR / "extract_rag_metrics.py")], timeout=600)

    print("\n[5/7] Comparing local, FAISS, and ChromaDB vector backends...")
    run_command([sys.executable, str(SCRIPT_DIR / "compare_vector_backends.py")], timeout=600)

    print("\n[6/7] Generating diagrams...")
    run_command([sys.executable, str(SCRIPT_DIR / "generate_deck_diagrams.py")], timeout=240)
    prune_non_comparison_assets()


def prune_non_comparison_assets() -> None:
    """Keep the visual output focused on comparison evidence."""
    for path in ASSETS_DIR.glob("*.png"):
        if path.name not in COMPARISON_PNGS:
            path.unlink()
    for path in ASSETS_DIR.glob("*.md"):
        path.unlink()


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def esc(value) -> str:
    return html.escape(str(value))


def short_cell(value, limit: int = 120) -> str:
    text = str(value)
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return esc(text)


def build_comparison_dashboard() -> None:
    print("\n[7/7] Writing comparison dashboard...")
    ASSETS_DIR.mkdir(exist_ok=True)
    problems = pd.read_csv(PROCESSED_DIR / "cp_problems_dataset.csv")
    nodes = pd.read_csv(PROCESSED_DIR / "cp_page_nodes_dataset.csv")
    rag_summary = read_json(ASSETS_DIR / "rag_runtime_summary.json")
    vector_summary = read_json(ASSETS_DIR / "vector_backend_summary.json")
    math_demo_summary = read_json(PROCESSED_DIR / "math_binary_demo_summary.json")
    math_demo_report_path = PROCESSED_DIR / "math_binary_classification_report.csv"
    math_demo_report = (
        pd.read_csv(math_demo_report_path) if math_demo_report_path.exists() else pd.DataFrame()
    )
    math_demo_problems_path = PROCESSED_DIR / "math_binary_demo_problems.csv"
    math_demo_problems = (
        pd.read_csv(math_demo_problems_path) if math_demo_problems_path.exists() else pd.DataFrame()
    )
    vector_results_path = ASSETS_DIR / "vector_backend_results.csv"
    vector_results = (
        pd.read_csv(vector_results_path) if vector_results_path.exists() else pd.DataFrame()
    )
    strategy_results_path = ASSETS_DIR / "strategy_classification_by_model.csv"
    strategy_results = (
        pd.read_csv(strategy_results_path) if strategy_results_path.exists() else pd.DataFrame()
    )

    metrics_rows = rag_summary.get("metrics", [])
    metrics_table = "".join(
        f"<tr><td>{row['query']}</td><td>{row['precision_at_k']}</td><td>{row['recall_at_k']}</td><td>{row['reciprocal_rank']}</td><td>{row['average_similarity_score']}</td></tr>"
        for row in metrics_rows
    )
    if math_demo_report.empty:
        math_summary = "<p>No se genero la demo binary/math.</p>"
    else:
        math_summary = (
            f"<p><strong>Problemas demo:</strong> {math_demo_summary.get('added_problem_ids', [])}</p>"
            f"<p><strong>Clasificador actual:</strong> {math_demo_report['current_approach'].value_counts().to_dict()}</p>"
            f"<p><strong>Estrategia fina:</strong> {math_demo_report['fine_strategy'].value_counts().to_dict()}</p>"
        )

    backend_rows = "".join(
        f"<tr><td>{backend}</td><td>{values.get('precision_at_k', '')}</td><td>{values.get('recall_at_k', '')}</td><td>{values.get('reciprocal_rank', '')}</td><td>{values.get('latency_ms_per_query', '')}</td></tr>"
        for backend, values in vector_summary.get("mean_metrics", {}).items()
    )
    backend_status = vector_summary.get("backend_status", {})
    backend_status_html = "".join(
        f"<li><code>{backend}</code>: {status.get('status')} {status.get('error', '')}</li>"
        for backend, status in backend_status.items()
    )

    if vector_results.empty:
        search_results_table = "<p>No hay resultados vectoriales detallados.</p>"
    else:
        binary_query_ids = [
            "binary_quadratic_time",
            "integer_square_root",
            "triangular_discriminant",
        ]
        search_rows_df = vector_results[
            (vector_results["query_id"].isin(binary_query_ids))
            & (vector_results["rank"] <= 3)
        ].copy()
        search_results_table = "".join(
            "<tr>"
            f"<td>{esc(row['query_id'])}</td>"
            f"<td>{esc(row['backend'])}</td>"
            f"<td>{esc(row['rank'])}</td>"
            f"<td>{esc(row['global_problem_id'])}</td>"
            f"<td>{esc(row['node_type'])}</td>"
            f"<td>{esc(row['score'])}</td>"
            f"<td>{esc(row['is_relevant'])}</td>"
            "</tr>"
            for _, row in search_rows_df.iterrows()
        )

    if math_demo_problems.empty:
        math_problem_rows = "<tr><td colspan='5'>No hay problemas demo.</td></tr>"
    else:
        accepted_by_problem = {}
        if not math_demo_report.empty and "accepted_strategies" in math_demo_report.columns:
            accepted_by_problem = (
                math_demo_report.groupby("global_problem_id")["accepted_strategies"]
                .first()
                .to_dict()
            )
        math_problem_rows = "".join(
            "<tr>"
            f"<td>{esc(row['global_problem_id'])}</td>"
            f"<td>{esc(row['title'])}</td>"
            f"<td>{esc(row['normalized_difficulty'])}</td>"
            f"<td>{short_cell(accepted_by_problem.get(row['global_problem_id'], ''), 100)}</td>"
            f"<td>{short_cell(row.get('notes', ''), 160)}</td>"
            "</tr>"
            for _, row in math_demo_problems.iterrows()
        )

    if math_demo_report.empty:
        math_idea_rows = "<tr><td colspan='6'>No hay ideas simuladas.</td></tr>"
    else:
        math_idea_rows = "".join(
            "<tr>"
            f"<td>{esc(row['student_id'])}</td>"
            f"<td>{esc(row['problem_title'])}</td>"
            f"<td>{short_cell(row['idea_text'], 170)}</td>"
            f"<td>{esc(row['current_approach'])}</td>"
            f"<td>{esc(row['fine_strategy'])}</td>"
            f"<td>{esc(row['agent_gap'])}</td>"
            "</tr>"
            for _, row in math_demo_report.iterrows()
        )

    if strategy_results.empty:
        strategy_rows = "<tr><td colspan='7'>No hay clasificacion por modelo.</td></tr>"
    else:
        strategy_rows = "".join(
            "<tr>"
            f"<td>{esc(row['model'])}</td>"
            f"<td>{esc(row['student_id'])}</td>"
            f"<td>{esc(row['problem_title'])}</td>"
            f"<td>{short_cell(row['idea_text'], 150)}</td>"
            f"<td>{esc(row['expected_strategy'])}</td>"
            f"<td>{esc(row['predicted_strategy'])}</td>"
            f"<td>{esc(row['is_correct'])}</td>"
            "</tr>"
            for _, row in strategy_results.iterrows()
        )

    html_cards = "\n".join(
        f'<section><h2>{title}</h2><img src="comparison_assets/{img}" alt="{title}"></section>'
        for title, img in [
            ("Comparacion conceptual: RAG plano vs Page Index local", "page_index_comparison.png"),
            ("Metricas de retrieval sobre ground truth simulado", "rag_retrieval_metrics.png"),
            ("Score por ranking y metodo", "retrieval_strategy_scores.png"),
            ("Overlap top-k entre metodos", "retrieval_method_overlap.png"),
            ("Score promedio por metodo", "retrieval_average_scores.png"),
            ("FAISS vs ChromaDB vs matriz local", "vector_backend_metrics.png"),
            ("Latencia por backend vectorial", "vector_backend_latency.png"),
            ("Overlap top-k entre backends vectoriales", "vector_backend_overlap.png"),
            ("Clasificacion binary/formula por modelo", "strategy_prediction_by_model.png"),
            ("Exactitud binary/formula por modelo", "strategy_accuracy_by_model.png"),
            ("Clasificacion actual vs estrategia fina", "math_binary_strategy_classification.png"),
            ("Estrategias detectadas por problema demo", "math_binary_problem_strategy_map.png"),
        ]
        if (ASSETS_DIR / img).exists()
    )
    generated_at = datetime.now().isoformat(timespec="seconds")
    html = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Dashboard comparativo - CP RAG local</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #0f172a; background: #f8fafc; line-height: 1.45; }}
    h1 {{ font-size: 34px; margin-bottom: 4px; }}
    h2 {{ margin-top: 0; }}
    .summary, section {{ background: white; border: 1px solid #e2e8f0; border-radius: 10px; padding: 20px; margin: 20px 0; }}
    img {{ max-width: 100%; border-radius: 8px; }}
    code {{ background: #e2e8f0; padding: 2px 5px; border-radius: 4px; }}
    table {{ border-collapse: collapse; width: 100%; background: white; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 8px; text-align: left; }}
    th {{ background: #e2e8f0; }}
  </style>
</head>
<body>
  <h1>Dashboard comparativo - CP RAG local</h1>
  <p>Generado: {generated_at}</p>
  <div class="summary">
    <h2>Resumen verificable</h2>
    <p><strong>Problemas:</strong> {len(problems)} | <strong>Page Nodes:</strong> {len(nodes)} | <strong>Backend semantico:</strong> {rag_summary.get('semantic_backend', 'unknown')}</p>
    <p>Este dashboard compara metodos de recuperacion, backends vectoriales y clasificacion textual de ideas. No implementa PageIndex tree-search ni ejecuta soluciones contra tests.</p>
    <ul>{backend_status_html}</ul>
    {math_summary}
  </div>
  <div class="summary">
    <h2>Metricas base</h2>
    <table>
      <thead><tr><th>Query</th><th>Precision@k</th><th>Recall@k</th><th>Reciprocal rank</th><th>Avg similarity</th></tr></thead>
      <tbody>{metrics_table}</tbody>
    </table>
  </div>
  <div class="summary">
    <h2>Backends vectoriales sobre el mismo dataset</h2>
    <table>
      <thead><tr><th>Backend</th><th>Precision@k</th><th>Recall@k</th><th>Reciprocal rank</th><th>Latency ms/query</th></tr></thead>
      <tbody>{backend_rows}</tbody>
    </table>
  </div>
  <div class="summary">
    <h2>Comparacion explicita de resultados recuperados</h2>
    <p>Esta tabla muestra el top-3 por backend para las queries de la demo binary/math. El grafico equivalente es <code>vector_backend_overlap.png</code>; para metodos del RAG base es <code>retrieval_method_overlap.png</code>.</p>
    <table>
      <thead><tr><th>Query demo</th><th>Backend</th><th>Rank</th><th>Problema recuperado</th><th>Node type</th><th>Score</th><th>Relevante</th></tr></thead>
      <tbody>{search_results_table}</tbody>
    </table>
  </div>
  <div class="summary">
    <h2>Problemas demo: busqueda binaria vs formula directa</h2>
    <p>Estos son los problemas usados para mostrar dos rutas de solucion posibles: una algoritmica por busqueda binaria y otra matematica por formula, raiz entera o discriminante.</p>
    <table>
      <thead><tr><th>ID</th><th>Problema</th><th>Dificultad</th><th>Estrategias aceptadas</th><th>Idea matematica</th></tr></thead>
      <tbody>{math_problem_rows}</tbody>
    </table>
  </div>
  <div class="summary">
    <h2>Ideas simuladas y respuesta del clasificador</h2>
    <p>Aqui se ve que el clasificador actual responde <code>UNKNOWN</code> o <code>MATH</code>, mientras que la capa fina esperada separa <code>BINARY_SEARCH</code> y <code>MATH_FORMULA</code>.</p>
    <table>
      <thead><tr><th>Estudiante</th><th>Problema</th><th>Idea recibida</th><th>Respuesta actual</th><th>Estrategia fina</th><th>Brecha detectada</th></tr></thead>
      <tbody>{math_idea_rows}</tbody>
    </table>
  </div>
  <div class="summary">
    <h2>Clasificacion binary/formula por cada modelo</h2>
    <p>Esta tabla y los graficos <code>strategy_prediction_by_model.png</code> y <code>strategy_accuracy_by_model.png</code> muestran si cada modelo/backend diferencia <code>BINARY_SEARCH</code> de <code>MATH_FORMULA</code>.</p>
    <table>
      <thead><tr><th>Modelo</th><th>Estudiante</th><th>Problema</th><th>Idea</th><th>Esperado</th><th>Predicho</th><th>Correcto</th></tr></thead>
      <tbody>{strategy_rows}</tbody>
    </table>
  </div>
  {html_cards}
</body>
</html>
"""
    (ROOT / "comparison_dashboard.html").write_text(html, encoding="utf-8")
    print("Wrote comparison_dashboard.html")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local CP RAG thesis pipeline.")
    parser.add_argument("--skip-dataset", action="store_true", help="Reuse existing data/processed files.")
    parser.add_argument("--max-cf", type=int, default=20, help="Max Codeforces problems.")
    parser.add_argument("--max-atcoder", type=int, default=20, help="Max AtCoder problems.")
    parser.add_argument("--with-content", action="store_true", help="Download statements/editorials when available.")
    parser.add_argument("--request-delay", type=float, default=1.0, help="Delay between external requests.")
    parser.add_argument(
        "--skip-math-binary-demo",
        action="store_true",
        help="Do not append the curated math-vs-binary-search demo slice.",
    )
    args = parser.parse_args()

    if not args.skip_dataset:
        build_dataset(args.max_cf, args.max_atcoder, args.with_content, args.request_delay)
    else:
        print("[1/7] Skipping dataset build; reusing data/processed.")

    if args.skip_math_binary_demo:
        print("[2/7] Skipping math-vs-binary-search demo.")
    else:
        add_math_binary_demo()

    regenerate_assets()
    build_comparison_dashboard()

    print("\nDONE. Open:")
    print(f"- {ROOT / 'PROJECT_GUIDE.md'}")
    print(f"- {ROOT / 'comparison_dashboard.html'}")
    print(f"- {ROOT / 'comparison_assets'}")


if __name__ == "__main__":
    main()
