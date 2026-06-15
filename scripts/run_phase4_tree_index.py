"""Phase 4: build PageIndex-ready nodes, edges, and chunks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from indexing.pageindex_adapter import (
    build_pageindex_chunks,
    build_pageindex_edges,
    build_pageindex_ready_nodes,
    save_phase4_outputs,
)


PROCESSED = ROOT / "data" / "processed"


def main() -> None:
    problems = pd.read_csv(PROCESSED / "cp_problems_dataset.csv")
    page_nodes = pd.read_csv(PROCESSED / "cp_page_nodes_dataset.csv")
    llm_nodes_path = PROCESSED / "cp_llm_tree_nodes_dataset.csv"
    llm_edges_path = PROCESSED / "cp_llm_tree_edges_dataset.csv"
    llm_nodes = pd.read_csv(llm_nodes_path) if llm_nodes_path.exists() else pd.DataFrame()
    llm_edges = pd.read_csv(llm_edges_path) if llm_edges_path.exists() else pd.DataFrame()

    nodes = build_pageindex_ready_nodes(problems, page_nodes, llm_nodes=llm_nodes)
    edges = build_pageindex_edges(nodes, llm_edges=llm_edges)
    chunks = build_pageindex_chunks(nodes)
    paths = save_phase4_outputs(nodes, edges, chunks, PROCESSED)

    report = {
        "phase": "phase4_tree_index",
        "status": "ok",
        "problem_count": int(len(problems)),
        "page_node_count": int(len(page_nodes)),
        "llm_node_count": int(len(llm_nodes)),
        "pageindex_ready_node_count": int(len(nodes)),
        "pageindex_ready_edge_count": int(len(edges)),
        "pageindex_ready_chunk_count": int(len(chunks)),
        "outputs": paths,
    }
    (PROCESSED / "phase4_tree_index_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
