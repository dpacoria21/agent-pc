"""Phase 4 helpers: export the local trees to a PageIndex-ready shape.

This module does not call PageIndex. It creates explicit node, edge, and chunk
tables that mirror the information a tree-search system needs:

- stable ids;
- parent-child links;
- node text and metadata;
- small chunks attached to nodes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from text_formatting import normalize_math_text


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = normalize_math_text(value).replace("\xa0", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def stable_id(*parts: Any) -> str:
    return "::".join(
        re.sub(r"[^A-Za-z0-9_.-]+", "_", str(part)).strip("_")
        for part in parts
        if str(part).strip() != ""
    )


def parse_listish(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    for parser in (json.loads,):
        try:
            parsed = parser(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item).strip()]
        except Exception:
            pass
    if text.startswith("[") and text.endswith("]"):
        return [part.strip(" '\"") for part in text.strip("[]").split(",") if part.strip(" '\"")]
    return [text]


def _metadata(row: pd.Series, extra: dict[str, Any] | None = None) -> str:
    data = {
        "source": row.get("source", ""),
        "url": row.get("url", ""),
        "normalized_tags": parse_listish(row.get("normalized_tags")),
        "topic_group": parse_listish(row.get("topic_group")),
        "difficulty": row.get("normalized_difficulty", None),
    }
    if extra:
        data.update(extra)
    return json.dumps(data, ensure_ascii=False, default=str)


def build_pageindex_ready_nodes(
    problems: pd.DataFrame,
    page_nodes: pd.DataFrame,
    llm_nodes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(row: dict[str, Any]) -> None:
        if row["node_id"] in seen:
            return
        seen.add(row["node_id"])
        rows.append(row)

    root_id = "pi::root::competitive_programming"
    add(
        {
            "node_id": root_id,
            "parent_node_id": "",
            "global_problem_id": "",
            "node_type": "ROOT",
            "title": "Competitive Programming RAG Corpus",
            "text": "Root node for Codeforces and AtCoder problem knowledge.",
            "depth": 0,
            "order": 0,
            "source": "",
            "normalized_difficulty": None,
            "normalized_tags": "[]",
            "topic_group": "[]",
            "url": "",
            "metadata": json.dumps({"phase": "pageindex_ready_export"}, ensure_ascii=False),
        }
    )

    problem_lookup = problems.set_index("global_problem_id")
    for order, (_, problem) in enumerate(problems.iterrows(), start=1):
        global_problem_id = str(problem["global_problem_id"])
        platform_id = stable_id("pi", "platform", problem.get("source", "unknown"))
        problem_id = stable_id("pi", "problem", global_problem_id)
        topic_values = parse_listish(problem.get("topic_group")) or parse_listish(problem.get("normalized_tags")) or ["uncategorized"]
        topic_id = stable_id("pi", "topic", problem.get("source", "unknown"), topic_values[0])

        add(
            {
                "node_id": platform_id,
                "parent_node_id": root_id,
                "global_problem_id": "",
                "node_type": "PLATFORM",
                "title": str(problem.get("source", "unknown")).title(),
                "text": f"Problems from {problem.get('source', 'unknown')}.",
                "depth": 1,
                "order": order,
                "source": problem.get("source", ""),
                "normalized_difficulty": None,
                "normalized_tags": "[]",
                "topic_group": "[]",
                "url": "",
                "metadata": _metadata(problem),
            }
        )
        add(
            {
                "node_id": topic_id,
                "parent_node_id": platform_id,
                "global_problem_id": "",
                "node_type": "TOPIC",
                "title": topic_values[0].replace("_", " ").title(),
                "text": " ".join(topic_values),
                "depth": 2,
                "order": order,
                "source": problem.get("source", ""),
                "normalized_difficulty": None,
                "normalized_tags": json.dumps(parse_listish(problem.get("normalized_tags")), ensure_ascii=False),
                "topic_group": json.dumps(topic_values, ensure_ascii=False),
                "url": "",
                "metadata": _metadata(problem, {"topic": topic_values[0]}),
            }
        )
        add(
            {
                "node_id": problem_id,
                "parent_node_id": topic_id,
                "global_problem_id": global_problem_id,
                "node_type": "PROBLEM",
                "title": str(problem.get("title", global_problem_id)),
                "text": clean_text(
                    "\n\n".join(
                        [
                            str(problem.get("title", "")),
                            str(problem.get("statement", "")),
                            str(problem.get("official_editorial", "")),
                        ]
                    )
                ),
                "depth": 3,
                "order": order,
                "source": problem.get("source", ""),
                "normalized_difficulty": problem.get("normalized_difficulty"),
                "normalized_tags": json.dumps(parse_listish(problem.get("normalized_tags")), ensure_ascii=False),
                "topic_group": json.dumps(topic_values, ensure_ascii=False),
                "url": problem.get("url", ""),
                "metadata": _metadata(problem, {"rating": problem.get("rating"), "problem_index": problem.get("problem_index")}),
            }
        )

    if llm_nodes is not None and not llm_nodes.empty:
        llm_id_map = {
            row["tree_node_id"]: stable_id("pi", "llm", row["tree_node_id"])
            for _, row in llm_nodes.iterrows()
        }
        for _, row in llm_nodes.iterrows():
            global_problem_id = str(row.get("global_problem_id", ""))
            parent = str(row.get("parent_tree_node_id", ""))
            problem_parent = stable_id("pi", "problem", global_problem_id)
            parent_id = llm_id_map.get(parent) or problem_parent
            problem = problem_lookup.loc[global_problem_id] if global_problem_id in problem_lookup.index else row
            add(
                {
                    "node_id": llm_id_map[row["tree_node_id"]],
                    "parent_node_id": parent_id,
                    "global_problem_id": global_problem_id,
                    "node_type": row.get("node_type", "LLM_NODE"),
                    "title": row.get("title", ""),
                    "text": clean_text(row.get("node_text", "")),
                    "depth": int(row.get("depth", 0)) + 4,
                    "order": int(row.get("order", 0)) if pd.notna(row.get("order", None)) else 0,
                    "source": row.get("source", problem.get("source", "")),
                    "normalized_difficulty": row.get("normalized_difficulty", problem.get("normalized_difficulty", None)),
                    "normalized_tags": json.dumps(parse_listish(row.get("normalized_tags", problem.get("normalized_tags", []))), ensure_ascii=False),
                    "topic_group": json.dumps(parse_listish(row.get("topic_group", problem.get("topic_group", []))), ensure_ascii=False),
                    "url": row.get("url", problem.get("url", "")),
                    "metadata": _metadata(
                        row,
                        {
                            "source_tree_node_id": row.get("tree_node_id", ""),
                            "generation_status": row.get("generation_status", ""),
                            "skills": parse_listish(row.get("skills")),
                            "strategies": parse_listish(row.get("strategies")),
                        },
                    ),
                }
            )

    page_problem_parent = {
        str(row["global_problem_id"]): stable_id("pi", "problem", row["global_problem_id"])
        for _, row in problems.iterrows()
    }
    for _, row in page_nodes.iterrows():
        global_problem_id = str(row.get("global_problem_id", ""))
        problem = problem_lookup.loc[global_problem_id] if global_problem_id in problem_lookup.index else row
        text = clean_text(row.get("node_text", ""))
        add(
            {
                "node_id": stable_id("pi", "page", row.get("node_id", "")),
                "parent_node_id": page_problem_parent.get(global_problem_id, root_id),
                "global_problem_id": global_problem_id,
                "node_type": row.get("node_type", "PAGE_NODE"),
                "title": row.get("node_title", row.get("node_type", "")),
                "text": text,
                "depth": 4,
                "order": int(row.get("order", 0)) if pd.notna(row.get("order", None)) else 0,
                "source": row.get("source", problem.get("source", "")),
                "normalized_difficulty": row.get("normalized_difficulty", problem.get("normalized_difficulty", None)),
                "normalized_tags": json.dumps(parse_listish(row.get("normalized_tags", problem.get("normalized_tags", []))), ensure_ascii=False),
                "topic_group": json.dumps(parse_listish(row.get("topic_group", problem.get("topic_group", []))), ensure_ascii=False),
                "url": row.get("url", problem.get("url", "")),
                "metadata": _metadata(row, {"source_page_node_id": row.get("node_id", "")}),
            }
        )

    return pd.DataFrame(rows).sort_values(["depth", "order", "node_id"]).reset_index(drop=True)


def build_pageindex_edges(nodes: pd.DataFrame, llm_edges: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for _, row in nodes.iterrows():
        parent = str(row.get("parent_node_id", ""))
        if not parent:
            continue
        rows.append(
            {
                "edge_id": stable_id("pi_edge", parent, row["node_id"]),
                "source_node_id": parent,
                "target_node_id": row["node_id"],
                "edge_type": "PARENT_OF",
                "reason": "Tree parent-child relationship.",
                "confidence": 1.0,
            }
        )
    if llm_edges is not None and not llm_edges.empty:
        for _, row in llm_edges.iterrows():
            source = stable_id("pi", "llm", row.get("source_tree_node_id", ""))
            target = stable_id("pi", "llm", row.get("target_tree_node_id", ""))
            rows.append(
                {
                    "edge_id": stable_id("pi_llm_edge", row.get("edge_id", "")),
                    "source_node_id": source,
                    "target_node_id": target,
                    "edge_type": row.get("edge_type", "SUPPORTS"),
                    "reason": row.get("reason", ""),
                    "confidence": row.get("confidence", 0.0),
                }
            )
    return pd.DataFrame(rows)


def build_pageindex_chunks(nodes: pd.DataFrame, max_words: int = 90, overlap: int = 18) -> pd.DataFrame:
    rows = []
    for _, node in nodes.iterrows():
        words = clean_text(node.get("text", "")).split()
        if not words:
            continue
        step = max(1, max_words - overlap)
        for start in range(0, len(words), step):
            piece_words = words[start : start + max_words]
            if len(piece_words) < 8:
                continue
            rows.append(
                {
                    "chunk_id": stable_id("pi_chunk", node["node_id"], start),
                    "node_id": node["node_id"],
                    "global_problem_id": node.get("global_problem_id", ""),
                    "node_type": node.get("node_type", ""),
                    "chunk_text": " ".join(piece_words),
                    "chunk_word_count": len(piece_words),
                    "metadata": node.get("metadata", "{}"),
                }
            )
            if start + max_words >= len(words):
                break
    return pd.DataFrame(rows)


def save_phase4_outputs(nodes: pd.DataFrame, edges: pd.DataFrame, chunks: pd.DataFrame, processed_dir: str | Path) -> dict[str, str]:
    processed = Path(processed_dir)
    processed.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for stem, df in {
        "cp_pageindex_ready_nodes": nodes,
        "cp_pageindex_ready_edges": edges,
        "cp_pageindex_ready_chunks": chunks,
    }.items():
        csv_path = processed / f"{stem}.csv"
        json_path = processed / f"{stem}.json"
        df.to_csv(csv_path, index=False)
        df.to_json(json_path, orient="records", force_ascii=False, indent=2)
        paths[f"{stem}_csv"] = str(csv_path)
        paths[f"{stem}_json"] = str(json_path)
        try:
            parquet_path = processed / f"{stem}.parquet"
            df.to_parquet(parquet_path, index=False)
            paths[f"{stem}_parquet"] = str(parquet_path)
        except Exception:
            pass
    return paths
