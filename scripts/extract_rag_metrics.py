import importlib
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

OUT = ROOT / "comparison_assets"
OUT.mkdir(exist_ok=True)

# Keep the live extraction deterministic and offline-friendly. The prototype
# module can use SentenceTransformer when available, but the presentation run
# forces the local fallback so it does not depend on model downloads.
os.environ["RAG_FORCE_SEMANTIC_FALLBACK"] = "1"
rag = importlib.import_module("rag_cp_student_profile")

metrics = rag.retrieval_evaluation_df.copy()
metrics.to_csv(OUT / "rag_retrieval_metrics.csv", index=False)
metrics.to_json(OUT / "rag_retrieval_metrics.json", orient="records", force_ascii=False, indent=2)

fig, ax = plt.subplots(figsize=(10, 5.6))
plot_df = metrics.set_index("query")[["precision_at_k", "recall_at_k", "reciprocal_rank"]]
plot_df.plot(kind="bar", ax=ax, color=["#2563eb", "#14b8a6", "#f59e0b"])
ax.set_title("Metricas de retrieval en ground truth simulado", fontsize=15, fontweight="bold")
ax.set_xlabel("query")
ax.set_ylabel("score")
ax.set_ylim(0, 1.05)
ax.tick_params(axis="x", rotation=20)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "rag_retrieval_metrics.png", dpi=180, bbox_inches="tight")
plt.close(fig)

comparison = rag.comparison_outputs["comparison"].copy()
comparison.to_csv(OUT / "retrieval_method_comparison.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 5.6))
for method, group in comparison.groupby("method"):
    group = group.sort_values("rank")
    ax.plot(group["rank"], group["score"], marker="o", linewidth=2, label=method)
ax.set_title("Comparacion de estrategias de retrieval por ranking", fontsize=15, fontweight="bold")
ax.set_xlabel("rank")
ax.set_ylabel("score")
ax.set_ylim(bottom=0)
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "retrieval_strategy_scores.png", dpi=180, bbox_inches="tight")
plt.close(fig)

method_sets = {
    method: set(group["node_id"].tolist())
    for method, group in comparison.groupby("method")
}
methods = list(method_sets.keys())
overlap = []
for left in methods:
    row = []
    for right in methods:
        row.append(len(method_sets[left].intersection(method_sets[right])))
    overlap.append(row)

fig, ax = plt.subplots(figsize=(7.5, 6.2))
image = ax.imshow(overlap, cmap="Blues")
ax.set_xticks(range(len(methods)), methods, rotation=35, ha="right")
ax.set_yticks(range(len(methods)), methods)
ax.set_title("Overlap top-k entre metodos de retrieval", fontsize=14, fontweight="bold")
for i in range(len(methods)):
    for j in range(len(methods)):
        ax.text(j, i, overlap[i][j], ha="center", va="center", color="#0f172a")
fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(OUT / "retrieval_method_overlap.png", dpi=180, bbox_inches="tight")
plt.close(fig)

avg_scores = comparison.groupby("method")["score"].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 5.2))
avg_scores.plot(kind="bar", ax=ax, color=["#2563eb", "#14b8a6", "#f59e0b", "#8b5cf6"])
ax.set_title("Score promedio por estrategia de retrieval", fontsize=15, fontweight="bold")
ax.set_xlabel("method")
ax.set_ylabel("average score")
ax.tick_params(axis="x", rotation=30)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "retrieval_average_scores.png", dpi=180, bbox_inches="tight")
plt.close(fig)

summary = {
    "semantic_backend": rag.SEMANTIC_BACKEND,
    "mock_problems": int(len(rag.problems_df)),
    "mock_page_nodes": int(len(rag.page_nodes_df)),
    "simulated_sessions": int(len(rag.sessions_df)),
    "simulated_ideas": int(len(rag.ideas_df)),
    "simulated_attempts": int(len(rag.attempts_df)),
    "metrics": metrics.to_dict("records"),
    "comparison_methods": sorted(comparison["method"].unique().tolist()),
    "comparison_rows": int(len(comparison)),
    "demo_output_keys": sorted(rag.demo_output.keys()),
}
(OUT / "rag_runtime_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

print(json.dumps(summary, ensure_ascii=False, indent=2))
