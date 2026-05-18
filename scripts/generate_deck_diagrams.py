from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "comparison_assets"
OUT.mkdir(exist_ok=True)


COLORS = {
    "blue": "#2563eb",
    "teal": "#14b8a6",
    "amber": "#f59e0b",
    "slate": "#334155",
    "gray": "#e2e8f0",
    "violet": "#8b5cf6",
    "red": "#ef4444",
}


def box(ax, xy, w, h, text, fc="#ffffff", ec="#334155", color="#0f172a", size=10):
    patch = FancyBboxPatch(
        xy, w, h,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.4,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=size, color=color, wrap=True)
    return patch


def arrow(ax, start, end, color="#334155"):
    arr = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.5, color=color)
    ax.add_patch(arr)


def setup(title, path):
    fig, ax = plt.subplots(figsize=(13.3, 7.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.04, 0.94, title, fontsize=19, fontweight="bold", color="#0f172a")
    return fig, ax, path


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


fig, ax, path = setup("Arquitectura general del prototipo", OUT / "architecture_diagram.png")
box(ax, (0.05, 0.70), 0.18, 0.11, "Fuentes públicas\nCodeforces + AtCoder", COLORS["gray"])
box(ax, (0.30, 0.70), 0.18, 0.11, "Dataset builder\ncp_dataset_scraper.py", "#dbeafe", COLORS["blue"])
box(ax, (0.55, 0.70), 0.18, 0.11, "Dataset procesado\nproblemas + page nodes", "#ccfbf1", COLORS["teal"])
box(ax, (0.80, 0.70), 0.15, 0.11, "RAG pipeline\nPython local", "#fef3c7", COLORS["amber"])
box(ax, (0.20, 0.42), 0.19, 0.12, "Embeddings\nall-MiniLM-L6-v2\nfallback TF-IDF+SVD", "#ede9fe", COLORS["violet"], size=9)
box(ax, (0.43, 0.42), 0.18, 0.12, "Retrieval híbrido\nsemantic + keyword\n+ metadata bonus", "#fef3c7", COLORS["amber"], size=9)
box(ax, (0.66, 0.42), 0.18, 0.12, "Perfil estudiante\nsesiones + ideas\n+ intentos", "#dcfce7", COLORS["teal"], size=9)
box(ax, (0.43, 0.18), 0.20, 0.12, "Recomendación\npersonalizada e interpretable", "#dbeafe", COLORS["blue"], size=9)
for s, e in [((0.23,0.755),(0.30,0.755)),((0.48,0.755),(0.55,0.755)),((0.73,0.755),(0.80,0.755)),((0.70,0.70),(0.52,0.54)),((0.39,0.48),(0.43,0.48)),((0.61,0.48),(0.66,0.48)),((0.75,0.42),(0.61,0.30)),((0.52,0.42),(0.53,0.30))]:
    arrow(ax, s, e)
save(fig, path)

fig, ax, path = setup("Traditional RAG vs Page Index Retrieval", OUT / "page_index_comparison.png")
box(ax, (0.06, 0.67), 0.34, 0.11, "Traditional RAG\nDocumento -> chunks planos", "#f8fafc")
box(ax, (0.06, 0.47), 0.15, 0.09, "Chunk 1", "#e2e8f0")
box(ax, (0.24, 0.47), 0.15, 0.09, "Chunk 2", "#e2e8f0")
box(ax, (0.15, 0.28), 0.18, 0.09, "Top-k por similitud", "#fee2e2", COLORS["red"])
arrow(ax, (0.23,0.67), (0.14,0.56))
arrow(ax, (0.23,0.67), (0.32,0.56))
arrow(ax, (0.14,0.47), (0.22,0.37))
arrow(ax, (0.32,0.47), (0.26,0.37))
box(ax, (0.58, 0.67), 0.34, 0.11, "Page Index Retrieval\nProblema -> nodos semánticos", "#eff6ff", COLORS["blue"])
for x, label, col in [(0.52,"STATEMENT","#dbeafe"),(0.66,"CONSTRAINTS","#ccfbf1"),(0.80,"EDITORIAL\nPROOF","#fef3c7")]:
    box(ax, (x,0.47), 0.12, 0.10, label, col, size=8)
box(ax, (0.62, 0.28), 0.20, 0.09, "Ranking por nodo\n+ filtros + etapa", "#dcfce7", COLORS["teal"], size=9)
for s in [(0.58,0.47),(0.72,0.47),(0.86,0.47)]:
    arrow(ax, s, (0.72,0.37))
ax.text(0.04, 0.10, "Diferencia clave: el prototipo recupera secciones con rol pedagógico, no fragmentos arbitrarios.", fontsize=12, color="#334155")
save(fig, path)

fig, ax, path = setup("Pipeline de datos y recuperación", OUT / "pipeline_diagram.png")
steps = [
    ("Config + presets", "#dbeafe"),
    ("Metadatos públicos", "#ccfbf1"),
    ("Normalización\ndifficulty/tags", "#fef3c7"),
    ("Page Nodes", "#ede9fe"),
    ("Embeddings\n+ TF-IDF", "#dcfce7"),
    ("Hybrid Search", "#fee2e2"),
    ("Perfil + recomendación", "#e0f2fe"),
]
x0 = 0.04
for i, (label, col) in enumerate(steps):
    x = x0 + i * 0.135
    box(ax, (x, 0.50), 0.11, 0.14, label, col, size=8)
    if i < len(steps) - 1:
        arrow(ax, (x + 0.11, 0.57), (x + 0.135, 0.57))
ax.text(0.05, 0.34, "Cada etapa conserva metadatos interpretables: source, rating/difficulty, tags, node_type y estados de parseo.", fontsize=12, color="#334155")
save(fig, path)

fig, ax, path = setup("Scoring del retrieval híbrido", OUT / "hybrid_scoring_diagram.png")
box(ax, (0.08, 0.60), 0.22, 0.12, "semantic_score\ncosine similarity", "#dbeafe", COLORS["blue"])
box(ax, (0.39, 0.60), 0.22, 0.12, "keyword_score\nTF-IDF", "#ccfbf1", COLORS["teal"])
box(ax, (0.70, 0.60), 0.22, 0.12, "metadata_bonus\ntags + rating + node_type", "#fef3c7", COLORS["amber"], size=9)
box(ax, (0.33, 0.30), 0.34, 0.12, "hybrid_score = α·semantic + (1-α)·keyword + metadata_bonus", "#ede9fe", COLORS["violet"], size=10)
for s in [(0.19,0.60),(0.50,0.60),(0.81,0.60)]:
    arrow(ax, s, (0.50,0.42))
ax.text(0.06, 0.16, "Diseño: combina similitud semántica, coincidencia literal y señales pedagógicas del perfil/contexto.", fontsize=12, color="#334155")
save(fig, path)

fig, ax, path = setup("Modelo de perfil del estudiante", OUT / "student_profile_diagram.png")
box(ax, (0.06, 0.66), 0.20, 0.11, "sessions_df\ntiempos, intentos,\nveredicto", "#dbeafe", COLORS["blue"], size=9)
box(ax, (0.40, 0.66), 0.20, 0.11, "ideas_df\napproach, stage,\nrisk, quality", "#ccfbf1", COLORS["teal"], size=9)
box(ax, (0.74, 0.66), 0.20, 0.11, "attempts_df\nverdict, error,\ncomplexity", "#fef3c7", COLORS["amber"], size=9)
box(ax, (0.34, 0.39), 0.32, 0.12, "calculate_student_profile(user_id)", "#ede9fe", COLORS["violet"])
box(ax, (0.18, 0.15), 0.20, 0.10, "habilidades\nautonomy/proof/impl/debug", "#dcfce7", COLORS["teal"], size=8)
box(ax, (0.62, 0.15), 0.20, 0.10, "recomendaciones\nweakness + rating + similarity", "#fee2e2", COLORS["red"], size=8)
for s in [(0.16,0.66),(0.50,0.66),(0.84,0.66)]:
    arrow(ax, s, (0.50,0.51))
arrow(ax, (0.44,0.39), (0.28,0.25))
arrow(ax, (0.56,0.39), (0.72,0.25))
save(fig, path)

print("Generated diagrams:", sorted(path.name for path in OUT.glob("*diagram.png")) + ["page_index_comparison.png"])
