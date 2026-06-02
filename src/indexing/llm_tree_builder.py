"""Phase 3: build semantic problem trees with GPT or a local fallback."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from dataset.schema import parse_listish
from llm.gpt_client import GPTClient
from llm.prompts import LLM_TREE_SCHEMA, SYSTEM_PROMPT, build_problem_tree_prompt
from llm.structured_outputs import normalize_llm_tree_output


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).replace("\xa0", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def stable_id(*parts: Any) -> str:
    return "::".join(re.sub(r"[^A-Za-z0-9_.-]+", "_", str(part)).strip("_") for part in parts if str(part) != "")


def first_topic(problem: pd.Series) -> str:
    for column in ["topic_group", "normalized_tags", "original_tags"]:
        values = parse_listish(problem.get(column))
        if values:
            return values[0].replace(" ", "_").lower()
    return "unknown"


def detect_strategies(text: str) -> list[str]:
    lower = text.lower()
    strategies = []
    keyword_map = {
        "binary_search": ["binary search", "search", "iterate", "bounded"],
        "direct_formula": ["formula", "equation", "quadratic", "sqrt", "discriminant"],
        "constructive": ["construct", "construction", "build", "pattern"],
        "proof": ["proof", "claim", "sufficiency", "necessity"],
        "dynamic_programming": ["dp", "transition", "state"],
        "tree_algorithm": ["tree", "dfs", "subtree"],
        "greedy": ["greedy", "sort", "choose"],
        "data_structure": ["data structure", "segment tree", "fenwick", "set"],
    }
    for label, keywords in keyword_map.items():
        if any(keyword in lower for keyword in keywords):
            strategies.append(label)
    return list(dict.fromkeys(strategies or ["unknown"]))


def detect_skills(text: str) -> list[str]:
    lower = text.lower()
    skills = []
    checks = {
        "problem_understanding": ["given", "determine", "print", "input"],
        "mathematical_modeling": ["equation", "formula", "claim", "sqrt", "mod"],
        "proof": ["proof", "necessity", "sufficiency", "invariant"],
        "algorithm_design": ["algorithm", "iterate", "check", "compute"],
        "complexity_analysis": ["o(", "complexity", "sqrt"],
        "implementation": ["implement", "print", "array", "index", "integer"],
        "edge_cases": ["edge", "corner", "-1", "positive integer"],
    }
    for label, keywords in checks.items():
        if any(keyword in lower for keyword in keywords):
            skills.append(label)
    return list(dict.fromkeys(skills or ["problem_understanding"]))


def evidence_sentence(text: str, keywords: list[str], fallback: str = "") -> str:
    cleaned = clean_text(text)
    sentences = re.split(r"(?<=[.!?])\s+|\n+", cleaned)
    for keyword in keywords:
        for sentence in sentences:
            if keyword.lower() in sentence.lower() and len(sentence.strip()) > 20:
                return sentence.strip()[:420]
    return fallback[:420] if fallback else summarize(cleaned, 420)


def summarize(text: str, max_chars: int = 240) -> str:
    cleaned = clean_text(text)
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rsplit(" ", 1)[0] + "..."


def heuristic_problem_analysis(problem: pd.Series, status: str = "heuristic_fallback") -> dict[str, Any]:
    problem_id = str(problem.get("global_problem_id"))
    title = str(problem.get("title", problem_id))
    statement = clean_text(problem.get("statement", ""))
    editorial = clean_text(problem.get("official_editorial", ""))
    combined = "\n\n".join([statement, editorial])
    strategies = detect_strategies(combined)
    skills = detect_skills(combined)
    main_topic = first_topic(problem)

    nodes = [
        {
            "node_key": "problem",
            "parent_node_key": "",
            "node_type": "PROBLEM",
            "title": title,
            "summary": summarize(statement or editorial, 260),
            "evidence_text": title,
            "source_section": "metadata",
            "skills": skills,
            "strategies": strategies,
            "confidence": 0.72,
        },
        {
            "node_key": "understanding",
            "parent_node_key": "problem",
            "node_type": "UNDERSTANDING",
            "title": "Problem understanding",
            "summary": "Captures what is given, what must be produced, and the main input/output goal.",
            "evidence_text": evidence_sentence(statement, ["given", "determine", "print"], summarize(statement)),
            "source_section": "statement",
            "skills": ["problem_understanding"],
            "strategies": [],
            "confidence": 0.68,
        },
    ]

    if any(word in combined.lower() for word in ["equation", "formula", "quadratic", "sqrt", "mod", "claim"]):
        nodes.append(
            {
                "node_key": "math_model",
                "parent_node_key": "problem",
                "node_type": "MATHEMATICAL_MODEL",
                "title": "Mathematical model",
                "summary": "Represents the mathematical condition or formula behind the solution.",
                "evidence_text": evidence_sentence(combined, ["equation", "formula", "quadratic", "claim", "sqrt"], summarize(editorial or statement)),
                "source_section": "official_editorial",
                "skills": ["mathematical_modeling"],
                "strategies": [item for item in strategies if item in {"direct_formula", "constructive", "binary_search"}],
                "confidence": 0.66,
            }
        )

    if any(word in combined.lower() for word in ["observation", "claim", "key"]):
        nodes.append(
            {
                "node_key": "observation",
                "parent_node_key": "problem",
                "node_type": "OBSERVATION",
                "title": "Key observation",
                "summary": "Core observation used to reduce the problem.",
                "evidence_text": evidence_sentence(editorial, ["observation", "claim", "key"], summarize(editorial)),
                "source_section": "official_editorial",
                "skills": ["mathematical_modeling", "proof"],
                "strategies": strategies,
                "confidence": 0.64,
            }
        )

    if "proof" in " ".join(strategies) or any(word in editorial.lower() for word in ["proof", "necessity", "sufficiency"]):
        nodes.append(
            {
                "node_key": "proof",
                "parent_node_key": "problem",
                "node_type": "PROOF",
                "title": "Correctness proof",
                "summary": "Explains why the condition or algorithm is correct.",
                "evidence_text": evidence_sentence(editorial, ["proof", "necessity", "sufficiency"], summarize(editorial)),
                "source_section": "official_editorial",
                "skills": ["proof"],
                "strategies": ["proof"],
                "confidence": 0.63,
            }
        )

    nodes.append(
        {
            "node_key": "algorithm",
            "parent_node_key": "problem",
            "node_type": "ALGORITHM",
            "title": "Algorithmic plan",
            "summary": "Steps needed to solve or check candidates.",
            "evidence_text": evidence_sentence(editorial or statement, ["iterate", "check", "compute", "algorithm", "print"], summarize(editorial or statement)),
            "source_section": "official_editorial" if editorial else "statement",
            "skills": ["algorithm_design"],
            "strategies": strategies,
            "confidence": 0.62,
        }
    )

    if any(word in combined.lower() for word in ["o(", "sqrt", "complexity"]):
        nodes.append(
            {
                "node_key": "complexity",
                "parent_node_key": "algorithm",
                "node_type": "COMPLEXITY",
                "title": "Complexity",
                "summary": "Estimated time complexity or bound used by the algorithm.",
                "evidence_text": evidence_sentence(combined, ["O(", "sqrt", "complexity"], ""),
                "source_section": "official_editorial",
                "skills": ["complexity_analysis"],
                "strategies": strategies,
                "confidence": 0.6,
            }
        )

    nodes.append(
        {
            "node_key": "implementation",
            "parent_node_key": "algorithm",
            "node_type": "IMPLEMENTATION",
            "title": "Implementation hints",
            "summary": "Practical details needed to code the approach.",
            "evidence_text": evidence_sentence(combined, ["integer", "print", "check", "array", "mod"], summarize(editorial or statement)),
            "source_section": "official_editorial" if editorial else "statement",
            "skills": ["implementation"],
            "strategies": strategies,
            "confidence": 0.55,
        }
    )

    edges = []
    node_keys = {node["node_key"] for node in nodes}
    if "math_model" in node_keys and "algorithm" in node_keys:
        edges.append(
            {
                "source_node_key": "math_model",
                "target_node_key": "algorithm",
                "edge_type": "PREREQUISITE_OF",
                "reason": "The algorithm depends on the mathematical condition extracted from the editorial.",
                "confidence": 0.62,
            }
        )
    if "proof" in node_keys and "algorithm" in node_keys:
        edges.append(
            {
                "source_node_key": "proof",
                "target_node_key": "algorithm",
                "edge_type": "SUPPORTS",
                "reason": "The proof supports why the algorithmic plan is valid.",
                "confidence": 0.58,
            }
        )
    edges.append(
        {
            "source_node_key": "algorithm",
            "target_node_key": "implementation",
            "edge_type": "IMPLEMENTATION_DEPENDS_ON",
            "reason": "Implementation follows the algorithmic plan.",
            "confidence": 0.56,
        }
    )

    return {
        "problem_id": problem_id,
        "title": title,
        "main_topic": main_topic,
        "difficulty_comment": f"normalized_difficulty={problem.get('normalized_difficulty', '')}",
        "strategies": strategies,
        "student_skills": skills,
        "nodes": nodes,
        "edges": edges,
        "warnings": [f"Generated with {status}; review before using as ground truth."],
    }


def build_problem_analysis(problem: pd.Series, client: GPTClient, force_fallback: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    problem_dict = problem.to_dict()
    problem_id = str(problem.get("global_problem_id"))
    title = str(problem.get("title", problem_id))

    if force_fallback:
        raw = heuristic_problem_analysis(problem, status="forced_heuristic_fallback")
        normalized, warnings = normalize_llm_tree_output(raw, problem_id, title)
        return normalized, {"status": "forced_heuristic_fallback", "model": "", "warnings": warnings}

    result = client.generate_structured_json(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=build_problem_tree_prompt(problem_dict),
        json_schema=LLM_TREE_SCHEMA,
        schema_name="cp_problem_tree",
        temperature=0.1,
    )
    if result.ok and result.data:
        normalized, warnings = normalize_llm_tree_output(result.data, problem_id, title)
        return normalized, {"status": result.status, "model": result.model, "warnings": warnings}

    raw = heuristic_problem_analysis(problem, status=result.status)
    normalized, warnings = normalize_llm_tree_output(raw, problem_id, title)
    warnings.append(result.error or result.status)
    return normalized, {"status": f"fallback_after_{result.status}", "model": result.model, "error": result.error, "warnings": warnings}


def analysis_to_tree_rows(analysis: dict[str, Any], problem: pd.Series, generation_meta: dict[str, Any]) -> list[dict[str, Any]]:
    problem_id = analysis["problem_id"]
    node_key_to_id = {
        node["node_key"]: stable_id("llm", problem_id, node["node_key"])
        for node in analysis.get("nodes", [])
    }
    rows = []
    for order, node in enumerate(analysis.get("nodes", []), start=1):
        parent_key = node.get("parent_node_key", "")
        rows.append(
            {
                "tree_node_id": node_key_to_id[node["node_key"]],
                "parent_tree_node_id": node_key_to_id.get(parent_key, ""),
                "global_problem_id": problem_id,
                "node_key": node["node_key"],
                "parent_node_key": parent_key,
                "node_type": node["node_type"],
                "title": node["title"],
                "summary": node["summary"],
                "evidence_text": node["evidence_text"],
                "node_text": clean_text("\n\n".join([node["summary"], node["evidence_text"]])),
                "source_section": node["source_section"],
                "skills": node["skills"],
                "strategies": node["strategies"],
                "confidence": node["confidence"],
                "depth": estimate_depth(node["node_key"], node.get("parent_node_key", ""), analysis.get("nodes", [])),
                "order": order,
                "source": problem.get("source", ""),
                "normalized_difficulty": problem.get("normalized_difficulty"),
                "normalized_tags": parse_listish(problem.get("normalized_tags")),
                "topic_group": parse_listish(problem.get("topic_group")),
                "url": problem.get("url", ""),
                "generation_status": generation_meta.get("status", ""),
                "generation_model": generation_meta.get("model", ""),
                "metadata": {
                    "main_topic": analysis.get("main_topic"),
                    "difficulty_comment": analysis.get("difficulty_comment"),
                    "warnings": analysis.get("warnings", []) + generation_meta.get("warnings", []),
                },
            }
        )
    return rows


def analysis_to_edge_rows(analysis: dict[str, Any], generation_meta: dict[str, Any]) -> list[dict[str, Any]]:
    problem_id = analysis["problem_id"]
    node_key_to_id = {
        node["node_key"]: stable_id("llm", problem_id, node["node_key"])
        for node in analysis.get("nodes", [])
    }
    rows = []
    for order, edge in enumerate(analysis.get("edges", []), start=1):
        rows.append(
            {
                "edge_id": stable_id("edge", problem_id, order, edge.get("edge_type", "")),
                "global_problem_id": problem_id,
                "source_tree_node_id": node_key_to_id.get(edge.get("source_node_key", ""), ""),
                "target_tree_node_id": node_key_to_id.get(edge.get("target_node_key", ""), ""),
                "source_node_key": edge.get("source_node_key", ""),
                "target_node_key": edge.get("target_node_key", ""),
                "edge_type": edge.get("edge_type", ""),
                "reason": edge.get("reason", ""),
                "confidence": edge.get("confidence", 0.0),
                "generation_status": generation_meta.get("status", ""),
            }
        )
    return rows


def estimate_depth(node_key: str, parent_key: str, nodes: list[dict[str, Any]]) -> int:
    parent_map = {node.get("node_key"): node.get("parent_node_key", "") for node in nodes}
    depth = 0
    current = parent_key
    seen = {node_key}
    while current:
        if current in seen:
            return depth
        seen.add(current)
        depth += 1
        current = parent_map.get(current, "")
    return depth


def build_llm_tree_dataset(
    problems: pd.DataFrame,
    *,
    limit: int = 3,
    problem_ids: list[str] | None = None,
    force_fallback: bool = False,
    model: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    selected = problems.copy()
    if problem_ids:
        selected = selected[selected["global_problem_id"].astype(str).isin(problem_ids)].copy()
    selected = selected.head(limit).copy()
    client = GPTClient(model=model)

    analyses = []
    tree_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    statuses = []
    for _, problem in selected.iterrows():
        analysis, generation_meta = build_problem_analysis(problem, client, force_fallback=force_fallback)
        analyses.append({**analysis, "_generation": generation_meta})
        tree_rows.extend(analysis_to_tree_rows(analysis, problem, generation_meta))
        edge_rows.extend(analysis_to_edge_rows(analysis, generation_meta))
        statuses.append(
            {
                "global_problem_id": analysis["problem_id"],
                "title": analysis["title"],
                "generation_status": generation_meta.get("status", ""),
                "generation_model": generation_meta.get("model", ""),
                "node_count": len(analysis.get("nodes", [])),
                "edge_count": len(analysis.get("edges", [])),
                "warning_count": len(analysis.get("warnings", [])) + len(generation_meta.get("warnings", [])),
            }
        )

    tree_df = pd.DataFrame(tree_rows)
    edge_df = pd.DataFrame(edge_rows)
    report = {
        "selected_problem_count": int(len(selected)),
        "tree_node_count": int(len(tree_df)),
        "edge_count": int(len(edge_df)),
        "client_available": bool(client.available),
        "statuses": statuses,
    }
    return tree_df, edge_df, analyses, report


def serializable_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in out.columns:
        if out[column].dtype == "object":
            out[column] = out[column].apply(
                lambda value: json.dumps(value, ensure_ascii=False, default=str)
                if isinstance(value, (list, dict))
                else value
            )
    return out


def save_llm_tree_outputs(
    tree_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    analyses: list[dict[str, Any]],
    report: dict[str, Any],
    processed_dir: str | Path,
) -> dict[str, str]:
    processed = Path(processed_dir)
    processed.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    for stem, df in {
        "cp_llm_tree_nodes_dataset": tree_df,
        "cp_llm_tree_edges_dataset": edge_df,
    }.items():
        serial = serializable_df(df)
        csv_path = processed / f"{stem}.csv"
        json_path = processed / f"{stem}.json"
        parquet_path = processed / f"{stem}.parquet"
        serial.to_csv(csv_path, index=False)
        df.to_json(json_path, orient="records", force_ascii=False, indent=2, default_handler=str)
        paths[f"{stem}_csv"] = str(csv_path)
        paths[f"{stem}_json"] = str(json_path)
        try:
            serial.to_parquet(parquet_path, index=False)
            paths[f"{stem}_parquet"] = str(parquet_path)
        except Exception:
            pass

    analysis_path = processed / "cp_llm_problem_analysis.json"
    report_path = processed / "llm_tree_build_report.json"
    analysis_path.write_text(json.dumps(analyses, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    paths["cp_llm_problem_analysis_json"] = str(analysis_path)
    paths["llm_tree_build_report_json"] = str(report_path)
    return paths
