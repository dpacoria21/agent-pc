"""Validation and normalization helpers for LLM tree outputs."""

from __future__ import annotations

from typing import Any


REQUIRED_ANALYSIS_KEYS = {
    "problem_id",
    "title",
    "main_topic",
    "difficulty_comment",
    "strategies",
    "student_skills",
    "nodes",
    "edges",
    "warnings",
}


REQUIRED_NODE_KEYS = {
    "node_key",
    "parent_node_key",
    "node_type",
    "title",
    "summary",
    "evidence_text",
    "source_section",
    "skills",
    "strategies",
    "confidence",
}


REQUIRED_EDGE_KEYS = {
    "source_node_key",
    "target_node_key",
    "edge_type",
    "reason",
    "confidence",
}


def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def clamp_confidence(value: Any) -> float:
    try:
        number = float(value)
    except Exception:
        number = 0.0
    return max(0.0, min(1.0, number))


def normalize_llm_tree_output(data: dict[str, Any], fallback_problem_id: str = "", fallback_title: str = "") -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    if not isinstance(data, dict):
        data = {}
        warnings.append("LLM output was not an object")

    normalized = {
        "problem_id": str(data.get("problem_id") or fallback_problem_id),
        "title": str(data.get("title") or fallback_title),
        "main_topic": str(data.get("main_topic") or "unknown"),
        "difficulty_comment": str(data.get("difficulty_comment") or ""),
        "strategies": [str(item) for item in as_list(data.get("strategies")) if str(item).strip()],
        "student_skills": [str(item) for item in as_list(data.get("student_skills")) if str(item).strip()],
        "nodes": [],
        "edges": [],
        "warnings": [str(item) for item in as_list(data.get("warnings")) if str(item).strip()],
    }

    missing = REQUIRED_ANALYSIS_KEYS.difference(data.keys())
    if missing:
        warnings.append(f"Missing top-level keys: {sorted(missing)}")

    node_keys: set[str] = set()
    for index, node in enumerate(as_list(data.get("nodes"))):
        if not isinstance(node, dict):
            warnings.append(f"Node at index {index} was not an object")
            continue
        missing_node = REQUIRED_NODE_KEYS.difference(node.keys())
        if missing_node:
            warnings.append(f"Node {index} missing keys: {sorted(missing_node)}")
        node_key = str(node.get("node_key") or f"node_{index}")
        node_keys.add(node_key)
        normalized["nodes"].append(
            {
                "node_key": node_key,
                "parent_node_key": str(node.get("parent_node_key") or ""),
                "node_type": str(node.get("node_type") or "UNKNOWN"),
                "title": str(node.get("title") or node_key),
                "summary": str(node.get("summary") or ""),
                "evidence_text": str(node.get("evidence_text") or ""),
                "source_section": str(node.get("source_section") or "unknown"),
                "skills": [str(item) for item in as_list(node.get("skills")) if str(item).strip()],
                "strategies": [str(item) for item in as_list(node.get("strategies")) if str(item).strip()],
                "confidence": clamp_confidence(node.get("confidence")),
            }
        )

    if not any(node["parent_node_key"] == "" for node in normalized["nodes"]):
        warnings.append("No root node was present; inserting a PROBLEM root")
        normalized["nodes"].insert(
            0,
            {
                "node_key": "problem",
                "parent_node_key": "",
                "node_type": "PROBLEM",
                "title": normalized["title"],
                "summary": normalized["title"],
                "evidence_text": normalized["title"],
                "source_section": "metadata",
                "skills": [],
                "strategies": normalized["strategies"],
                "confidence": 0.6,
            },
        )
        node_keys.add("problem")

    for index, edge in enumerate(as_list(data.get("edges"))):
        if not isinstance(edge, dict):
            warnings.append(f"Edge at index {index} was not an object")
            continue
        missing_edge = REQUIRED_EDGE_KEYS.difference(edge.keys())
        if missing_edge:
            warnings.append(f"Edge {index} missing keys: {sorted(missing_edge)}")
        source = str(edge.get("source_node_key") or "")
        target = str(edge.get("target_node_key") or "")
        if source not in node_keys or target not in node_keys:
            warnings.append(f"Edge {index} references unknown node(s): {source}->{target}")
        normalized["edges"].append(
            {
                "source_node_key": source,
                "target_node_key": target,
                "edge_type": str(edge.get("edge_type") or "SUPPORTS"),
                "reason": str(edge.get("reason") or ""),
                "confidence": clamp_confidence(edge.get("confidence")),
            }
        )

    normalized["warnings"].extend(warnings)
    return normalized, warnings

