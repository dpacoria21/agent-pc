import ast
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "comparison_assets"
OUT.mkdir(exist_ok=True)


def parse_list(value):
    if pd.isna(value) or value == "":
        return []
    if isinstance(value, list):
        return value
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    return []


def save_bar(data, title, xlabel, ylabel, path, color="#2563eb", rotate=0):
    fig, ax = plt.subplots(figsize=(10, 5.6))
    labels = list(data.keys())
    values = list(data.values())
    ax.bar(labels, values, color=color)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotate)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_pie(data, title, path):
    fig, ax = plt.subplots(figsize=(8, 5.6))
    labels = list(data.keys())
    values = list(data.values())
    colors = ["#2563eb", "#14b8a6", "#f59e0b", "#64748b", "#8b5cf6"]
    ax.pie(values, labels=labels, autopct="%1.0f%%", startangle=90, colors=colors[: len(labels)])
    ax.set_title(title, fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def inspect_python(path):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = []
    classes = []
    assignments = []
    strings = []
    imports = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions.append({"name": node.name, "line": node.lineno})
        elif isinstance(node, ast.ClassDef):
            classes.append({"name": node.name, "line": node.lineno})
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
            else:
                imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    name = target.id
                    if name.isupper() or name.endswith("_df") or name in {"ground_truth", "demo_output"}:
                        assignments.append({"name": name, "line": node.lineno})
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value.strip()
            if any(key in text.lower() for key in ["query", "prove", "greedy", "dp", "llm", "prompt", "transition", "cosine"]):
                strings.append({"text": text[:240]})

    return {
        "path": str(path),
        "lines": len(source.splitlines()),
        "functions": functions,
        "classes": classes,
        "important_assignments": assignments,
        "interesting_strings": strings[:30],
        "imports": sorted(set(imports)),
    }


def workspace_files():
    ignored_parts = {".venv", "__pycache__", "comparison_assets", "presentation_assets", "data/cache"}
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        rel_text = rel.as_posix()
        if any(part in rel.parts for part in [".venv", "__pycache__"]):
            continue
        if rel_text.startswith("comparison_assets/") or rel_text.startswith("presentation_assets/") or rel_text.startswith("data/cache/"):
            continue
        files.append(rel_text)
    return sorted(files)


problems = pd.read_csv(ROOT / "data/processed/cp_problems_dataset.csv")
nodes = pd.read_csv(ROOT / "data/processed/cp_page_nodes_dataset.csv")
quality = json.loads((ROOT / "data/processed/first_run_quality_report.json").read_text(encoding="utf-8"))
summary = json.loads((ROOT / "data/processed/first_run_summary.json").read_text(encoding="utf-8"))

normalized_tag_counter = Counter()
original_tag_counter = Counter()
for value in problems["normalized_tags"]:
    normalized_tag_counter.update(parse_list(value))
for value in problems["original_tags"]:
    original_tag_counter.update(parse_list(value))

node_text_lengths = nodes["node_text"].fillna("").astype(str).str.len()

platform_counts = problems["source"].value_counts().to_dict()
node_type_counts = nodes["node_type"].value_counts().to_dict()
statement_counts = problems["statement_status"].value_counts().to_dict()
editorial_counts = problems["editorial_status"].value_counts().to_dict()

save_pie(platform_counts, "Dataset por plataforma", OUT / "platform_distribution.png")
save_bar(dict(normalized_tag_counter.most_common(10)), "Top tags normalizados", "tag", "problemas", OUT / "top_tags.png", color="#14b8a6", rotate=35)
save_bar(node_type_counts, "Distribucion de Page Nodes", "node_type", "nodos", OUT / "node_type_distribution.png", color="#2563eb", rotate=65)
save_bar(statement_counts, "Estado de statements", "statement_status", "problemas", OUT / "statement_status.png", color="#f59e0b")
save_bar(editorial_counts, "Estado de editoriales", "editorial_status", "problemas", OUT / "editorial_status.png", color="#8b5cf6")

fig, ax = plt.subplots(figsize=(10, 5.6))
problems["normalized_difficulty"].dropna().plot(kind="hist", bins=12, color="#2563eb", edgecolor="white", ax=ax)
ax.set_title("Distribucion de dificultad normalizada", fontsize=16, fontweight="bold")
ax.set_xlabel("normalized_difficulty")
ax.set_ylabel("problemas")
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "difficulty_histogram.png", dpi=180, bbox_inches="tight")
plt.close(fig)

implemented = {
    "Page Index": True,
    "Semantic Search": True,
    "TF-IDF Search": True,
    "Hybrid Search": True,
    "Student Simulation": True,
    "Student Profile": True,
    "Recommendation": True,
    "LLM API": False,
    "FAISS": False,
    "ChromaDB": False,
}
save_bar(
    {key: int(value) for key, value in implemented.items()},
    "Componentes detectados en el proyecto",
    "componente",
    "implementado",
    OUT / "component_detection.png",
    color="#0f766e",
    rotate=55,
)

python_files = [
    ROOT / "src" / "cp_dataset_scraper.py",
    ROOT / "src" / "rag_cp_student_profile.py",
    ROOT / "scripts" / "run_all_local.py",
    ROOT / "scripts" / "extract_rag_metrics.py",
    ROOT / "scripts" / "generate_presentation_evidence.py",
    ROOT / "scripts" / "generate_deck_diagrams.py",
]

analysis = {
    "workspace_files": workspace_files(),
    "python_modules": [inspect_python(path) for path in python_files if path.exists()],
    "datasets": {
        "problems_shape": list(problems.shape),
        "nodes_shape": list(nodes.shape),
        "source_counts": platform_counts,
        "node_type_counts": node_type_counts,
        "normalized_tag_counts": dict(normalized_tag_counter.most_common(20)),
        "original_tag_counts": dict(original_tag_counter.most_common(20)),
        "difficulty_min": None if problems["normalized_difficulty"].dropna().empty else float(problems["normalized_difficulty"].min()),
        "difficulty_max": None if problems["normalized_difficulty"].dropna().empty else float(problems["normalized_difficulty"].max()),
        "statement_status": statement_counts,
        "editorial_status": editorial_counts,
        "non_empty_node_text": int((node_text_lengths > 0).sum()),
        "empty_node_text": int((node_text_lengths == 0).sum()),
        "example_problems": problems[
            ["global_problem_id", "source", "title", "normalized_difficulty", "original_tags", "normalized_tags"]
        ].head(10).to_dict("records"),
    },
    "implemented_components": implemented,
    "quality_report": quality,
    "first_run_summary": summary,
    "limits": [
        "El dataset procesado actual contiene metadatos y Page Nodes, pero statements/editoriales estan en estado skipped.",
        "No se detectaron FAISS ni ChromaDB en archivos actuales.",
        "No se detecto una API real de LLM conectada; existe placeholder analyze_student_idea_with_llm.",
        "Las metricas de retrieval se calculan con ground truth simulado.",
    ],
}

(OUT / "project_analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

print(json.dumps({
    "analysis_path": str((OUT / "project_analysis.json").resolve()),
    "charts": sorted(path.name for path in OUT.glob("*.png")),
    "problems": int(problems.shape[0]),
    "nodes": int(nodes.shape[0]),
}, ensure_ascii=False, indent=2))
