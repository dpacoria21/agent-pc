"""Pedagogical knowledge graph layer for the CP RAG prototype.

The project already stores problems and page nodes. This module adds a second
layer that is closer to how a tutor reasons:

- global topics and techniques;
- problem-specific approaches;
- prerequisites and common risks;
- cross-problem links for similar practice paths.

The output is intentionally tabular so it can be exported to PageIndex, FAISS,
ChromaDB, or a local prototype without requiring a graph database.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from text_formatting import normalize_math_text


TOPIC_DESCRIPTIONS = {
    "math": "Mathematical modeling, formulas, number theory, counting, and proof-oriented reasoning.",
    "graphs": "Graph, tree, traversal, connectivity, and structure-based reasoning.",
    "dynamic_programming": "State definition, recurrence, transitions, and optimization over subproblems.",
    "greedy": "Local choices, exchange arguments, sorting decisions, and proof of optimality.",
    "data_structures": "Efficient state maintenance with sets, heaps, DSU, Fenwick, or segment trees.",
    "techniques": "Reusable techniques such as binary search, two pointers, sorting, and prefix sums.",
    "implementation_constructive": "Implementation-heavy and constructive pattern building.",
    "uncategorized": "Problems without a reliable normalized topic.",
}


TECHNIQUE_KEYWORDS = {
    "binary_search": [
        "binary search",
        "lower_bound",
        "upper_bound",
        "bisect",
        "monotonic",
        "predicate",
        "search on answer",
        "maximum x",
    ],
    "direct_formula": [
        "formula",
        "closed form",
        "quadratic",
        "discriminant",
        "sqrt",
        "square root",
        "integer root",
        "x^2",
    ],
    "dynamic_programming": ["dp", "dynamic programming", "state", "transition", "recurrence", "memo"],
    "greedy": ["greedy", "always", "choose", "exchange", "sort and", "take the"],
    "graphs": ["graph", "tree", "dfs", "bfs", "subtree", "path", "connected"],
    "data_structures": ["segment tree", "fenwick", "dsu", "priority queue", "ordered set", "heap"],
    "constructive": ["construct", "construction", "build", "arrange", "pattern"],
    "number_theory": ["gcd", "lcm", "prime", "divisor", "mod", "modulo", "number theory"],
    "prefix_sums": ["prefix", "suffix", "cumulative", "partial sum"],
    "brute_force": ["brute force", "try all", "enumerate", "all pairs"],
    "two_pointers": ["two pointers", "sliding window", "left pointer", "right pointer"],
    "sorting": ["sort", "sorted", "order", "permutation"],
}


TECHNIQUE_TO_TOPIC = {
    "binary_search": "techniques",
    "direct_formula": "math",
    "dynamic_programming": "dynamic_programming",
    "greedy": "greedy",
    "graphs": "graphs",
    "data_structures": "data_structures",
    "constructive": "implementation_constructive",
    "number_theory": "math",
    "prefix_sums": "techniques",
    "brute_force": "implementation_constructive",
    "two_pointers": "techniques",
    "sorting": "techniques",
}


STRATEGY_SYNONYMS = {
    "BINARY_SEARCH": "binary_search",
    "binary_search": "binary_search",
    "search_on_answer": "binary_search",
    "MATH_FORMULA": "direct_formula",
    "FORMULA": "direct_formula",
    "direct_formula": "direct_formula",
    "closed_form_math": "direct_formula",
    "DP": "dynamic_programming",
    "dynamic_programming": "dynamic_programming",
    "GRAPH": "graphs",
    "tree_algorithm": "graphs",
    "GREEDY": "greedy",
    "CONSTRUCTIVE": "constructive",
    "constructive": "constructive",
    "DATA_STRUCTURES": "data_structures",
    "data_structure": "data_structures",
    "BRUTE_FORCE": "brute_force",
    "IMPLEMENTATION": "constructive",
}


PREREQUISITES_BY_TECHNIQUE = {
    "binary_search": ["monotonic_predicate", "bounds_reasoning", "integer_overflow_awareness"],
    "direct_formula": ["algebraic_modeling", "integer_square_root", "overflow_awareness"],
    "dynamic_programming": ["state_design", "transition_reasoning", "base_cases"],
    "greedy": ["exchange_argument", "invariant_reasoning", "counterexample_testing"],
    "graphs": ["graph_modeling", "dfs_bfs", "tree_invariants"],
    "data_structures": ["operation_invariants", "complexity_tradeoffs"],
    "constructive": ["pattern_recognition", "case_splitting", "validity_proof"],
    "number_theory": ["divisibility", "modular_arithmetic"],
    "prefix_sums": ["range_aggregation", "index_boundaries"],
    "brute_force": ["constraint_reading", "complexity_estimation"],
    "two_pointers": ["monotonic_window", "invariant_reasoning"],
    "sorting": ["ordering_argument", "tie_handling"],
}


RISKS_BY_TECHNIQUE = {
    "binary_search": ["wrong_bounds", "off_by_one", "missing_monotonicity_proof"],
    "direct_formula": ["precision_error", "overflow", "invalid_integer_root"],
    "dynamic_programming": ["wrong_state", "missing_base_case", "bad_complexity"],
    "greedy": ["wrong_proof", "hidden_counterexample", "edge_cases"],
    "graphs": ["visited_state_bug", "recursion_depth", "wrong_tree_root"],
    "data_structures": ["stale_update", "wrong_query_range", "complexity_bug"],
    "constructive": ["invalid_construction", "edge_cases", "proof_gap"],
    "number_theory": ["modulo_bug", "overflow", "divisibility_edge_case"],
    "prefix_sums": ["indexing_bug", "off_by_one"],
    "brute_force": ["tle", "bad_complexity"],
    "two_pointers": ["pointer_invariant_bug", "off_by_one"],
    "sorting": ["tie_handling", "wrong_ordering"],
}


RISK_KEYWORDS = {
    "tle": ["tle", "too slow", "time limit", "bad complexity"],
    "bad_complexity": ["o(n^2)", "complexity too high", "too slow"],
    "wrong_proof": ["cannot prove", "proof", "why always", "counterexample"],
    "edge_cases": ["edge", "corner", "n=1", "empty", "zero"],
    "precision_error": ["precision", "double", "floating", "sqrt"],
    "overflow": ["overflow", "long long", "x*x", "r*r", "1+8"],
    "off_by_one": ["off by one", "lower_bound", "upper_bound", "inclusive", "exclusive"],
}


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
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
    except Exception:
        pass
    if text.startswith("[") and text.endswith("]"):
        return [part.strip(" '\"") for part in text.strip("[]").split(",") if part.strip(" '\"")]
    return [text]


def normalize_strategy(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    key = raw.replace(" ", "_")
    return STRATEGY_SYNONYMS.get(key, STRATEGY_SYNONYMS.get(key.upper(), key.lower()))


def keyword_matches(text: str, keyword: str) -> bool:
    """Match keywords without letting short tokens leak into other words.

    For example, `dp` should match `dp transition`, but not `description`.
    Multi-word phrases still use substring matching because they are already
    specific enough for this prototype.
    """

    key = keyword.lower().strip()
    if not key:
        return False
    if re.fullmatch(r"[a-z0-9_+\-*/^()]+", key):
        return re.search(rf"(?<![a-z0-9_]){re.escape(key)}(?![a-z0-9_])", text) is not None
    return key in text


def any_keyword_matches(text: str, keywords: list[str]) -> bool:
    return any(keyword_matches(text, keyword) for keyword in keywords)


def first_evidence(text: str, keywords: list[str], max_chars: int = 360) -> str:
    cleaned = clean_text(text)
    if not cleaned:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+|\n+", cleaned)
    for keyword in keywords:
        for sentence in sentences:
            if keyword_matches(sentence.lower(), keyword) and len(sentence.strip()) > 20:
                return sentence.strip()[:max_chars]
    return cleaned[:max_chars].rsplit(" ", 1)[0] if len(cleaned) > max_chars else cleaned


def problem_text_bundle(
    problem: pd.Series,
    page_nodes_for_problem: pd.DataFrame | None = None,
    llm_nodes_for_problem: pd.DataFrame | None = None,
) -> str:
    parts = [
        problem.get("title", ""),
        problem.get("statement", ""),
        problem.get("constraints", ""),
        problem.get("official_editorial", ""),
        " ".join(parse_listish(problem.get("original_tags"))),
        " ".join(parse_listish(problem.get("normalized_tags"))),
        " ".join(parse_listish(problem.get("topic_group"))),
    ]
    if page_nodes_for_problem is not None and not page_nodes_for_problem.empty:
        parts.extend(page_nodes_for_problem["node_text"].fillna("").astype(str).tolist())
    if llm_nodes_for_problem is not None and not llm_nodes_for_problem.empty:
        for column in ["title", "summary", "evidence_text", "node_text", "skills", "strategies"]:
            if column in llm_nodes_for_problem.columns:
                parts.extend(llm_nodes_for_problem[column].fillna("").astype(str).tolist())
    return clean_text("\n\n".join(str(part) for part in parts if str(part).strip()))


def infer_problem_signals(
    problem: pd.Series,
    page_nodes_for_problem: pd.DataFrame | None = None,
    llm_nodes_for_problem: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Infer topics, techniques, risks, and prerequisites for one problem.

    This is deliberately deterministic. If GPT-generated LLM tree nodes are
    present, their `skills` and `strategies` are treated as extra evidence; if
    not, keyword and metadata rules still produce a usable pedagogy layer.
    """

    text = problem_text_bundle(problem, page_nodes_for_problem, llm_nodes_for_problem)
    lower = text.lower()
    original_tags = parse_listish(problem.get("original_tags"))
    normalized_tags = parse_listish(problem.get("normalized_tags"))
    topic_group = parse_listish(problem.get("topic_group"))

    topics = [topic.replace(" ", "_").lower() for topic in topic_group or normalized_tags]
    techniques: list[str] = []
    for tag in original_tags + normalized_tags + topic_group:
        norm = normalize_strategy(tag)
        if norm in TECHNIQUE_TO_TOPIC:
            techniques.append(norm)
    for technique, keywords in TECHNIQUE_KEYWORDS.items():
        if any_keyword_matches(lower, keywords):
            techniques.append(technique)

    if llm_nodes_for_problem is not None and not llm_nodes_for_problem.empty:
        for _, row in llm_nodes_for_problem.iterrows():
            for item in parse_listish(row.get("strategies")):
                norm = normalize_strategy(item)
                if norm in TECHNIQUE_TO_TOPIC:
                    techniques.append(norm)
            for item in parse_listish(row.get("skills")):
                norm = normalize_strategy(item)
                if norm in TECHNIQUE_TO_TOPIC:
                    techniques.append(norm)

    techniques = list(dict.fromkeys(item for item in techniques if item and item != "unknown"))
    if not techniques:
        techniques = ["constructive"] if "implementation_constructive" in topics else ["brute_force"]

    for technique in techniques:
        topic = TECHNIQUE_TO_TOPIC.get(technique, "uncategorized")
        if topic not in topics:
            topics.append(topic)
    topics = list(dict.fromkeys(topics or ["uncategorized"]))

    prerequisites = list(
        dict.fromkeys(
            prereq
            for technique in techniques
            for prereq in PREREQUISITES_BY_TECHNIQUE.get(technique, [])
        )
    )
    risks = list(
        dict.fromkeys(
            risk
            for technique in techniques
            for risk in RISKS_BY_TECHNIQUE.get(technique, [])
        )
    )
    for risk, keywords in RISK_KEYWORDS.items():
        if any_keyword_matches(lower, keywords):
            risks.append(risk)
    risks = list(dict.fromkeys(risks))

    evidence_by_technique = {
        technique: first_evidence(text, TECHNIQUE_KEYWORDS.get(technique, [technique]))
        for technique in techniques
    }

    primary_topic = topics[0] if topics else "uncategorized"
    return {
        "global_problem_id": str(problem.get("global_problem_id", "")),
        "title": str(problem.get("title", "")),
        "topics": topics,
        "primary_topic": primary_topic,
        "techniques": techniques,
        "prerequisites": prerequisites,
        "risks": risks,
        "evidence_by_technique": evidence_by_technique,
        "difficulty": problem.get("normalized_difficulty"),
        "tags": list(dict.fromkeys(original_tags + normalized_tags + topic_group)),
    }


def _metadata(extra: dict[str, Any]) -> str:
    return json.dumps(extra, ensure_ascii=False, default=str)


def _pageindex_node(
    *,
    node_id: str,
    parent_node_id: str,
    node_type: str,
    title: str,
    text: str,
    depth: int,
    order: int,
    source: str = "",
    global_problem_id: str = "",
    normalized_difficulty: Any = None,
    normalized_tags: list[str] | None = None,
    topic_group: list[str] | None = None,
    url: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "parent_node_id": parent_node_id,
        "global_problem_id": global_problem_id,
        "node_type": node_type,
        "title": title,
        "text": clean_text(text),
        "depth": depth,
        "order": order,
        "source": source,
        "normalized_difficulty": normalized_difficulty,
        "normalized_tags": json.dumps(normalized_tags or [], ensure_ascii=False),
        "topic_group": json.dumps(topic_group or [], ensure_ascii=False),
        "url": url,
        "metadata": _metadata(metadata or {}),
    }


def _edge(
    source_node_id: str,
    target_node_id: str,
    edge_type: str,
    reason: str,
    confidence: float = 0.7,
) -> dict[str, Any]:
    return {
        "edge_id": stable_id("pi_ped_edge", edge_type, source_node_id, target_node_id),
        "source_node_id": source_node_id,
        "target_node_id": target_node_id,
        "edge_type": edge_type,
        "reason": reason,
        "confidence": round(float(confidence), 3),
    }


def build_pedagogical_pageindex_layer(
    problems: pd.DataFrame,
    page_nodes: pd.DataFrame,
    llm_nodes: pd.DataFrame | None = None,
    *,
    root_node_id: str = "pi::root::competitive_programming",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build PageIndex-shaped pedagogical nodes, cross edges, and signals."""

    rows: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(row: dict[str, Any]) -> None:
        if row["node_id"] in seen:
            return
        seen.add(row["node_id"])
        rows.append(row)

    pedagogy_root = "pi::pedagogy::root"
    prereq_root = "pi::pedagogy::prerequisites"
    risk_root = "pi::pedagogy::risks"
    add(
        _pageindex_node(
            node_id=pedagogy_root,
            parent_node_id=root_node_id,
            node_type="PEDAGOGY_ROOT",
            title="Pedagogical Knowledge Layer",
            text="Connected layer of topics, techniques, prerequisites, risks, and student-facing approaches.",
            depth=1,
            order=0,
            metadata={"layer": "pedagogical_graph"},
        )
    )
    add(
        _pageindex_node(
            node_id=prereq_root,
            parent_node_id=pedagogy_root,
            node_type="PREREQUISITE_GROUP",
            title="Prerequisites",
            text="Reusable prerequisite skills used to explain why an approach is valid.",
            depth=2,
            order=1,
            metadata={"layer": "pedagogical_graph"},
        )
    )
    add(
        _pageindex_node(
            node_id=risk_root,
            parent_node_id=pedagogy_root,
            node_type="RISK_GROUP",
            title="Common Risks",
            text="Reusable error modes such as overflow, off-by-one, weak proof, and missing edge cases.",
            depth=2,
            order=2,
            metadata={"layer": "pedagogical_graph"},
        )
    )

    page_by_problem = {key: frame.copy() for key, frame in page_nodes.groupby("global_problem_id")}
    llm_by_problem = (
        {key: frame.copy() for key, frame in llm_nodes.groupby("global_problem_id")}
        if llm_nodes is not None and not llm_nodes.empty and "global_problem_id" in llm_nodes.columns
        else {}
    )

    signals_by_problem: dict[str, dict[str, Any]] = {}
    for order, (_, problem) in enumerate(problems.iterrows(), start=1):
        problem_id = str(problem.get("global_problem_id", ""))
        signals = infer_problem_signals(
            problem,
            page_by_problem.get(problem_id),
            llm_by_problem.get(problem_id),
        )
        signals_by_problem[problem_id] = signals
        signal_rows.append(
            {
                "global_problem_id": problem_id,
                "title": signals["title"],
                "primary_topic": signals["primary_topic"],
                "topics": json.dumps(signals["topics"], ensure_ascii=False),
                "techniques": json.dumps(signals["techniques"], ensure_ascii=False),
                "prerequisites": json.dumps(signals["prerequisites"], ensure_ascii=False),
                "risks": json.dumps(signals["risks"], ensure_ascii=False),
                "difficulty": signals["difficulty"],
            }
        )

        problem_node_id = stable_id("pi", "problem", problem_id)
        for topic in signals["topics"]:
            topic_id = stable_id("pi", "pedagogy", "topic", topic)
            add(
                _pageindex_node(
                    node_id=topic_id,
                    parent_node_id=pedagogy_root,
                    node_type="TOPIC",
                    title=topic.replace("_", " ").title(),
                    text=TOPIC_DESCRIPTIONS.get(topic, f"Competitive programming topic: {topic}."),
                    depth=2,
                    order=order,
                    normalized_tags=[topic],
                    topic_group=[topic],
                    metadata={"layer": "pedagogical_graph", "topic": topic},
                )
            )
            edges.append(_edge(topic_id, problem_node_id, "INDEXES_PROBLEM", f"Problem belongs to topic {topic}.", 0.68))

        for technique in signals["techniques"]:
            topic = TECHNIQUE_TO_TOPIC.get(technique, signals["primary_topic"])
            topic_id = stable_id("pi", "pedagogy", "topic", topic)
            technique_id = stable_id("pi", "pedagogy", "technique", technique)
            add(
                _pageindex_node(
                    node_id=technique_id,
                    parent_node_id=topic_id,
                    node_type="TECHNIQUE",
                    title=technique.replace("_", " ").title(),
                    text=(
                        f"Technique {technique.replace('_', ' ')}. "
                        f"Prerequisites: {', '.join(PREREQUISITES_BY_TECHNIQUE.get(technique, []))}. "
                        f"Common risks: {', '.join(RISKS_BY_TECHNIQUE.get(technique, []))}."
                    ),
                    depth=3,
                    order=order,
                    normalized_tags=[technique, topic],
                    topic_group=[topic],
                    metadata={"layer": "pedagogical_graph", "technique": technique},
                )
            )
            approach_id = stable_id("pi", "pedagogy", "approach", problem_id, technique)
            evidence = signals["evidence_by_technique"].get(technique, "")
            prereqs = PREREQUISITES_BY_TECHNIQUE.get(technique, [])
            risks = RISKS_BY_TECHNIQUE.get(technique, [])
            add(
                _pageindex_node(
                    node_id=approach_id,
                    parent_node_id=problem_node_id,
                    node_type="APPROACH",
                    title=f"{problem.get('title', problem_id)} - {technique.replace('_', ' ').title()} Approach",
                    text=(
                        f"This problem can be interpreted through {technique.replace('_', ' ')}. "
                        f"Evidence: {evidence}. "
                        f"Prerequisites: {', '.join(prereqs)}. "
                        f"Common risks: {', '.join(risks)}."
                    ),
                    depth=4,
                    order=order,
                    source=str(problem.get("source", "")),
                    global_problem_id=problem_id,
                    normalized_difficulty=problem.get("normalized_difficulty"),
                    normalized_tags=list(dict.fromkeys(parse_listish(problem.get("normalized_tags")) + [technique])),
                    topic_group=signals["topics"],
                    url=str(problem.get("url", "")),
                    metadata={
                        "layer": "pedagogical_graph",
                        "technique": technique,
                        "prerequisites": prereqs,
                        "risks": risks,
                        "evidence": evidence,
                    },
                )
            )
            edges.append(_edge(technique_id, approach_id, "EXPLAINS_APPROACH", "Global technique explains this problem-specific approach.", 0.76))
            edges.append(_edge(technique_id, problem_node_id, "INDEXES_PROBLEM", f"Problem can be reached through technique {technique}.", 0.74))

            for prereq in prereqs:
                prereq_id = stable_id("pi", "pedagogy", "prerequisite", prereq)
                add(
                    _pageindex_node(
                        node_id=prereq_id,
                        parent_node_id=prereq_root,
                        node_type="PREREQUISITE",
                        title=prereq.replace("_", " ").title(),
                        text=f"Prerequisite skill: {prereq.replace('_', ' ')}.",
                        depth=3,
                        order=order,
                        normalized_tags=[prereq],
                        topic_group=[topic],
                        metadata={"layer": "pedagogical_graph", "prerequisite": prereq},
                    )
                )
                edges.append(_edge(prereq_id, approach_id, "PREREQUISITE_OF", f"{prereq} is needed for {technique}.", 0.72))

            for risk in risks:
                risk_id = stable_id("pi", "pedagogy", "risk", risk)
                add(
                    _pageindex_node(
                        node_id=risk_id,
                        parent_node_id=risk_root,
                        node_type="RISK",
                        title=risk.replace("_", " ").title(),
                        text=f"Common risk: {risk.replace('_', ' ')}. Useful for hinting and answer validation.",
                        depth=3,
                        order=order,
                        normalized_tags=[risk],
                        topic_group=[topic],
                        metadata={"layer": "pedagogical_graph", "risk": risk},
                    )
                )
                edges.append(_edge(approach_id, risk_id, "HAS_RISK", f"The {technique} approach may fail through {risk}.", 0.64))

        if {"binary_search", "direct_formula"}.issubset(set(signals["techniques"])):
            left = stable_id("pi", "pedagogy", "approach", problem_id, "binary_search")
            right = stable_id("pi", "pedagogy", "approach", problem_id, "direct_formula")
            edges.append(_edge(left, right, "ALTERNATIVE_APPROACH", "Same problem can be reasoned by binary search or direct formula.", 0.82))
            edges.append(_edge(right, left, "ALTERNATIVE_APPROACH", "Same problem can be reasoned by direct formula or binary search.", 0.82))

    # Cross-problem practice links by shared technique. These make the tree less
    # isolated: a retrieved approach can point to similar or next-step problems.
    for technique in sorted({tech for signals in signals_by_problem.values() for tech in signals["techniques"]}):
        related = [
            (problem_id, signals)
            for problem_id, signals in signals_by_problem.items()
            if technique in signals["techniques"]
        ]
        related.sort(key=lambda item: (float(item[1]["difficulty"]) if pd.notna(item[1]["difficulty"]) else 10**9, item[0]))
        for (left_id, left_signals), (right_id, right_signals) in zip(related, related[1:]):
            left_approach = stable_id("pi", "pedagogy", "approach", left_id, technique)
            right_approach = stable_id("pi", "pedagogy", "approach", right_id, technique)
            diff_left = left_signals["difficulty"]
            diff_right = right_signals["difficulty"]
            relation = "NEXT_PRACTICE_STEP"
            reason = f"Both approaches use {technique}; sorted by normalized difficulty for curriculum-style traversal."
            confidence = 0.58
            if pd.notna(diff_left) and pd.notna(diff_right) and abs(float(diff_left) - float(diff_right)) <= 300:
                relation = "SAME_PATTERN_AS"
                reason = f"Both approaches use {technique} and have nearby difficulty."
                confidence = 0.66
            edges.append(_edge(left_approach, right_approach, relation, reason, confidence))

    nodes_df = pd.DataFrame(rows)
    edges_df = pd.DataFrame(edges).drop_duplicates("edge_id") if edges else pd.DataFrame(columns=["edge_id", "source_node_id", "target_node_id", "edge_type", "reason", "confidence"])
    signals_df = pd.DataFrame(signal_rows)
    return nodes_df, edges_df, signals_df


def save_pedagogical_graph_outputs(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    signals: pd.DataFrame,
    processed_dir: str | Path,
) -> dict[str, str]:
    processed = Path(processed_dir)
    processed.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for stem, df in {
        "cp_pedagogical_graph_nodes": nodes,
        "cp_pedagogical_graph_edges": edges,
        "cp_problem_pedagogical_signals": signals,
    }.items():
        csv_path = processed / f"{stem}.csv"
        json_path = processed / f"{stem}.json"
        df.to_csv(csv_path, index=False)
        df.to_json(json_path, orient="records", force_ascii=False, indent=2, default_handler=str)
        paths[f"{stem}_csv"] = str(csv_path)
        paths[f"{stem}_json"] = str(json_path)
        try:
            parquet_path = processed / f"{stem}.parquet"
            df.to_parquet(parquet_path, index=False)
            paths[f"{stem}_parquet"] = str(parquet_path)
        except Exception:
            pass
    return paths
