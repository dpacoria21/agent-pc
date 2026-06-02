"""Local PageIndex-like hybrid tree search prototype for CP RAG.

This module does not call the PageIndex service or any external LLM. It builds
the same kind of research prototype shape locally:

- a tree of problem/document nodes;
- content chunks attached to tree nodes;
- value-based search through chunk similarity;
- guided tree search through query intent and metadata;
- a deduplicated hybrid queue;
- interpretable problem recommendations.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


STAGE_NODE_TYPES = {
    "PROOF": {"EDITORIAL_PROOF", "EDITORIAL_OBSERVATION"},
    "ALGORITHM": {"EDITORIAL_ALGORITHM", "EDITORIAL_OBSERVATION", "EDITORIAL_COMPLEXITY"},
    "IMPLEMENTATION": {"IMPLEMENTATION_HINTS", "COMMON_MISTAKES", "INPUT", "OUTPUT"},
    "EDGE_CASES": {"COMMON_MISTAKES", "CONSTRAINTS", "EXAMPLES"},
    "UNDERSTANDING": {"STATEMENT", "CONSTRAINTS", "EXAMPLES"},
    "COMPLEXITY": {"EDITORIAL_COMPLEXITY", "EDITORIAL_ALGORITHM"},
}


APPROACH_KEYWORDS = {
    "BINARY_SEARCH": {"binary search", "bisect", "search on answer", "lower_bound", "upper_bound"},
    "FORMULA": {"formula", "equation", "quadratic", "discriminant", "sqrt", "closed form"},
    "MATH": {"math", "gcd", "prime", "mod", "number theory", "claim", "proof"},
    "DP": {"dp", "dynamic programming", "state", "transition"},
    "GRAPH": {"graph", "tree", "dfs", "bfs"},
    "GREEDY": {"greedy", "always", "choose", "exchange"},
    "CONSTRUCTIVE": {"construct", "construction", "build", "pattern"},
}


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


def difficulty_bucket(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "missing"
    difficulty = float(value)
    buckets = [(0, 799), (800, 1199), (1200, 1599), (1600, 1999), (2000, 2399)]
    for low, high in buckets:
        if low <= difficulty <= high:
            return f"{low}-{high}"
    return "2400+"


def primary_topic(row: pd.Series) -> str:
    topics = parse_listish(row.get("topic_group"))
    tags = parse_listish(row.get("normalized_tags"))
    original_tags = parse_listish(row.get("original_tags"))
    for collection in (topics, tags, original_tags):
        if collection:
            return collection[0].replace(" ", "_").lower()
    return "uncategorized"


def section_group(node_type: str) -> str:
    if node_type in {"STATEMENT", "INPUT", "OUTPUT", "CONSTRAINTS", "EXAMPLES", "NOTES"}:
        return "UNDERSTANDING"
    if node_type.startswith("EDITORIAL"):
        return "EDITORIAL_REASONING"
    if node_type in {"IMPLEMENTATION_HINTS", "COMMON_MISTAKES"}:
        return "IMPLEMENTATION_RISKS"
    return "OTHER"


def node_tags(node: pd.Series) -> list[str]:
    return (
        parse_listish(node.get("normalized_tags"))
        + parse_listish(node.get("topic_group"))
        + parse_listish(node.get("original_tags"))
    )


def summarize_text(text: str, max_words: int = 34) -> str:
    words = clean_text(text).split()
    if not words:
        return ""
    return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")


def infer_query_intent(query: str) -> dict[str, Any]:
    lower = query.lower()
    stage = "UNDERSTANDING"
    if any(word in lower for word in ["prove", "proof", "why", "correct", "invariant", "claim"]):
        stage = "PROOF"
    elif any(word in lower for word in ["edge", "corner", "n=1", "case"]):
        stage = "EDGE_CASES"
    elif any(word in lower for word in ["tle", "complexity", "too slow", "o("]):
        stage = "COMPLEXITY"
    elif any(word in lower for word in ["code", "implement", "bug", "wa", "runtime", "overflow"]):
        stage = "IMPLEMENTATION"
    elif any(word in lower for word in ["algorithm", "approach", "solve", "strategy"]):
        stage = "ALGORITHM"

    approaches = []
    for label, keywords in APPROACH_KEYWORDS.items():
        if any(keyword in lower for keyword in keywords):
            approaches.append(label)
    if not approaches:
        approaches = ["UNKNOWN"]

    risks = []
    if any(word in lower for word in ["tle", "too slow", "complexity"]):
        risks.append("TLE")
    if any(word in lower for word in ["wrong", "wa", "fails", "incorrect"]):
        risks.append("WA")
    if any(word in lower for word in ["edge", "corner", "n=1"]):
        risks.append("EDGE_CASES")
    if any(word in lower for word in ["prove", "proof", "why"]):
        risks.append("WRONG_PROOF")

    return {
        "query": query,
        "stage": stage,
        "approaches": approaches,
        "risks": risks or ["UNKNOWN"],
        "preferred_node_types": sorted(STAGE_NODE_TYPES.get(stage, set())),
    }


def build_tree_nodes(problems: pd.DataFrame, page_nodes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(row: dict[str, Any]) -> None:
        if row["tree_node_id"] in seen:
            return
        seen.add(row["tree_node_id"])
        rows.append(row)

    add(
        {
            "tree_node_id": "root::competitive_programming",
            "parent_tree_node_id": "",
            "global_problem_id": "",
            "node_role": "ROOT",
            "node_type": "ROOT",
            "title": "Competitive Programming Knowledge Base",
            "summary": "Corpus root for Codeforces and AtCoder problem knowledge.",
            "node_text": "",
            "depth": 0,
            "order": 0,
            "source": "",
            "normalized_difficulty": np.nan,
            "difficulty_bucket": "all",
            "original_tags": [],
            "normalized_tags": [],
            "topic_group": [],
            "url": "",
            "metadata": {},
        }
    )

    page_nodes_by_problem = {key: frame.copy() for key, frame in page_nodes.groupby("global_problem_id")}
    for problem_order, (_, problem) in enumerate(problems.iterrows(), start=1):
        global_problem_id = str(problem["global_problem_id"])
        source = str(problem.get("source", "unknown"))
        topic = primary_topic(problem)
        bucket = difficulty_bucket(problem.get("normalized_difficulty"))
        platform_id = stable_id("platform", source)
        topic_id = stable_id("topic", source, topic)
        bucket_id = stable_id("difficulty", source, topic, bucket)
        problem_id = stable_id("problem", global_problem_id)
        tags = parse_listish(problem.get("normalized_tags"))
        original_tags = parse_listish(problem.get("original_tags"))
        topic_group = parse_listish(problem.get("topic_group"))
        problem_text = clean_text(
            "\n\n".join(
                [
                    str(problem.get("title", "")),
                    str(problem.get("statement", "")),
                    str(problem.get("official_editorial", "")),
                ]
            )
        )

        add(
            {
                "tree_node_id": platform_id,
                "parent_tree_node_id": "root::competitive_programming",
                "global_problem_id": "",
                "node_role": "PLATFORM",
                "node_type": "PLATFORM",
                "title": source.title(),
                "summary": f"Problems from {source}.",
                "node_text": "",
                "depth": 1,
                "order": problem_order,
                "source": source,
                "normalized_difficulty": np.nan,
                "difficulty_bucket": "all",
                "original_tags": [],
                "normalized_tags": [],
                "topic_group": [],
                "url": "",
                "metadata": {"source": source},
            }
        )
        add(
            {
                "tree_node_id": topic_id,
                "parent_tree_node_id": platform_id,
                "global_problem_id": "",
                "node_role": "TOPIC",
                "node_type": "TOPIC",
                "title": topic.replace("_", " ").title(),
                "summary": f"Problems grouped under topic {topic}.",
                "node_text": topic.replace("_", " "),
                "depth": 2,
                "order": problem_order,
                "source": source,
                "normalized_difficulty": np.nan,
                "difficulty_bucket": "all",
                "original_tags": [],
                "normalized_tags": [topic],
                "topic_group": [topic],
                "url": "",
                "metadata": {"topic": topic},
            }
        )
        add(
            {
                "tree_node_id": bucket_id,
                "parent_tree_node_id": topic_id,
                "global_problem_id": "",
                "node_role": "DIFFICULTY_BUCKET",
                "node_type": "DIFFICULTY_BUCKET",
                "title": f"Difficulty {bucket}",
                "summary": f"Problems with normalized difficulty in {bucket}.",
                "node_text": bucket,
                "depth": 3,
                "order": problem_order,
                "source": source,
                "normalized_difficulty": problem.get("normalized_difficulty"),
                "difficulty_bucket": bucket,
                "original_tags": [],
                "normalized_tags": [topic],
                "topic_group": [topic],
                "url": "",
                "metadata": {"difficulty_bucket": bucket},
            }
        )
        add(
            {
                "tree_node_id": problem_id,
                "parent_tree_node_id": bucket_id,
                "global_problem_id": global_problem_id,
                "node_role": "PROBLEM",
                "node_type": "PROBLEM",
                "title": str(problem.get("title", global_problem_id)),
                "summary": summarize_text(problem_text, max_words=45),
                "node_text": problem_text,
                "depth": 4,
                "order": problem_order,
                "source": source,
                "normalized_difficulty": problem.get("normalized_difficulty"),
                "difficulty_bucket": bucket,
                "original_tags": original_tags,
                "normalized_tags": tags,
                "topic_group": topic_group,
                "url": problem.get("url", ""),
                "metadata": {
                    "rating": problem.get("rating"),
                    "editorial_status": problem.get("editorial_status"),
                    "editorial_parse_method": problem.get("editorial_parse_method"),
                },
            }
        )

        problem_page_nodes = page_nodes_by_problem.get(global_problem_id, pd.DataFrame()).copy()
        for group_order, group in enumerate(["UNDERSTANDING", "EDITORIAL_REASONING", "IMPLEMENTATION_RISKS"], start=1):
            group_id = stable_id("group", global_problem_id, group)
            add(
                {
                    "tree_node_id": group_id,
                    "parent_tree_node_id": problem_id,
                    "global_problem_id": global_problem_id,
                    "node_role": "SECTION_GROUP",
                    "node_type": group,
                    "title": group.replace("_", " ").title(),
                    "summary": f"{group.replace('_', ' ').title()} nodes for {problem.get('title')}.",
                    "node_text": "",
                    "depth": 5,
                    "order": group_order,
                    "source": source,
                    "normalized_difficulty": problem.get("normalized_difficulty"),
                    "difficulty_bucket": bucket,
                    "original_tags": original_tags,
                    "normalized_tags": tags,
                    "topic_group": topic_group,
                    "url": problem.get("url", ""),
                    "metadata": {},
                }
            )

        for _, page_node in problem_page_nodes.iterrows():
            node_type = str(page_node.get("node_type", "CONTENT"))
            group_id = stable_id("group", global_problem_id, section_group(node_type))
            node_text = clean_text(page_node.get("node_text", ""))
            add(
                {
                    "tree_node_id": stable_id("content", page_node.get("node_id")),
                    "parent_tree_node_id": group_id,
                    "global_problem_id": global_problem_id,
                    "node_role": "CONTENT",
                    "node_type": node_type,
                    "title": str(page_node.get("node_title", node_type)),
                    "summary": summarize_text(node_text),
                    "node_text": node_text,
                    "depth": 6,
                    "order": int(page_node.get("order", 0)) if pd.notna(page_node.get("order", np.nan)) else 0,
                    "source": source,
                    "normalized_difficulty": page_node.get("normalized_difficulty", problem.get("normalized_difficulty")),
                    "difficulty_bucket": bucket,
                    "original_tags": original_tags,
                    "normalized_tags": tags,
                    "topic_group": topic_group,
                    "url": page_node.get("url", problem.get("url", "")),
                    "metadata": {
                        "page_node_id": page_node.get("node_id"),
                        "parent_problem_node": problem_id,
                    },
                }
            )

    tree = pd.DataFrame(rows)
    return tree.sort_values(["depth", "order", "tree_node_id"]).reset_index(drop=True)


def build_tree_chunks(tree_nodes: pd.DataFrame, max_words: int = 90, overlap: int = 18) -> pd.DataFrame:
    chunks: list[dict[str, Any]] = []
    chunkable = tree_nodes[tree_nodes["node_text"].fillna("").astype(str).str.len() > 0].copy()
    for _, node in chunkable.iterrows():
        words = clean_text(node["node_text"]).split()
        if not words:
            continue
        step = max(1, max_words - overlap)
        for start in range(0, len(words), step):
            piece = " ".join(words[start : start + max_words])
            if len(piece.split()) < 8 and chunks:
                continue
            chunks.append(
                {
                    "chunk_id": stable_id("chunk", node["tree_node_id"], start),
                    "tree_node_id": node["tree_node_id"],
                    "global_problem_id": node.get("global_problem_id", ""),
                    "node_role": node.get("node_role", ""),
                    "node_type": node.get("node_type", ""),
                    "chunk_text": piece,
                    "chunk_word_count": len(piece.split()),
                }
            )
            if start + max_words >= len(words):
                break
    return pd.DataFrame(chunks)


@dataclass
class HybridTreeSearcher:
    tree_nodes: pd.DataFrame
    chunks: pd.DataFrame
    alpha: float = 0.62
    vectorizer: TfidfVectorizer | None = None
    svd: TruncatedSVD | None = None
    chunk_embeddings: np.ndarray | None = None
    embedding_backend: str = ""

    def fit(self) -> dict[str, Any]:
        docs = self.chunks.apply(
            lambda row: " ".join(
                [
                    str(row["node_type"]),
                    str(row["node_role"]),
                    str(row["chunk_text"]),
                ]
            ),
            axis=1,
        ).tolist()
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=1)
        tfidf = self.vectorizer.fit_transform(docs)
        if min(tfidf.shape) > 3:
            n_components = min(96, max(2, min(tfidf.shape) - 1))
            self.svd = TruncatedSVD(n_components=n_components, random_state=42)
            self.chunk_embeddings = normalize(self.svd.fit_transform(tfidf)).astype("float32")
            self.embedding_backend = "tfidf_svd_l2"
            dim = int(self.chunk_embeddings.shape[1])
        else:
            self.svd = None
            self.chunk_embeddings = normalize(tfidf).astype("float32")
            self.embedding_backend = "tfidf_l2"
            dim = int(self.chunk_embeddings.shape[1])
        return {
            "embedding_backend": self.embedding_backend,
            "chunks_indexed": int(len(self.chunks)),
            "embedding_dim": dim,
        }

    def embed_query(self, query: str) -> np.ndarray:
        if self.vectorizer is None or self.chunk_embeddings is None:
            self.fit()
        assert self.vectorizer is not None
        query_tfidf = self.vectorizer.transform([query])
        if self.svd is not None:
            return normalize(self.svd.transform(query_tfidf)).astype("float32")[0]
        return normalize(query_tfidf).astype("float32")[0]

    def value_search(self, query: str, top_k_chunks: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.chunk_embeddings is None:
            self.fit()
        assert self.chunk_embeddings is not None
        query_embedding = self.embed_query(query)
        scores = self.chunk_embeddings @ query_embedding
        order = np.argsort(-scores)[: min(top_k_chunks, len(scores))]
        chunk_hits = self.chunks.iloc[order].copy()
        chunk_hits["chunk_score"] = scores[order]

        direct: dict[str, list[float]] = {}
        for _, hit in chunk_hits.iterrows():
            direct.setdefault(hit["tree_node_id"], []).append(float(hit["chunk_score"]))

        parent_map = self.tree_nodes.set_index("tree_node_id")["parent_tree_node_id"].to_dict()
        node_scores: dict[str, dict[str, float]] = {}
        for node_id, values in direct.items():
            score = sum(values) / math.sqrt(len(values) + 1)
            node_scores.setdefault(node_id, {"direct_value_score": 0.0, "propagated_value_score": 0.0})
            node_scores[node_id]["direct_value_score"] += score
            current = parent_map.get(node_id, "")
            depth = 1
            while current:
                propagated = score * (0.62**depth)
                node_scores.setdefault(current, {"direct_value_score": 0.0, "propagated_value_score": 0.0})
                node_scores[current]["propagated_value_score"] += propagated
                current = parent_map.get(current, "")
                depth += 1

        rows = []
        node_lookup = self.tree_nodes.set_index("tree_node_id")
        for node_id, value in node_scores.items():
            if node_id not in node_lookup.index:
                continue
            node = node_lookup.loc[node_id]
            rows.append(
                {
                    "tree_node_id": node_id,
                    "value_score": value["direct_value_score"] + value["propagated_value_score"],
                    "direct_value_score": value["direct_value_score"],
                    "propagated_value_score": value["propagated_value_score"],
                    "node_role": node["node_role"],
                    "node_type": node["node_type"],
                    "global_problem_id": node["global_problem_id"],
                    "title": node["title"],
                }
            )
        value_nodes = pd.DataFrame(rows)
        if value_nodes.empty:
            return value_nodes, chunk_hits
        value_nodes = value_nodes.sort_values("value_score", ascending=False).reset_index(drop=True)
        return value_nodes, chunk_hits

    def guided_tree_search(self, query: str) -> pd.DataFrame:
        intent = infer_query_intent(query)
        query_terms = set(re.findall(r"[a-zA-Z0-9_]+", query.lower()))
        preferred_types = set(intent["preferred_node_types"])
        approaches = set(intent["approaches"])
        rows = []
        for _, node in self.tree_nodes.iterrows():
            haystack = " ".join(
                [
                    str(node.get("title", "")),
                    str(node.get("summary", "")),
                    str(node.get("node_type", "")),
                    " ".join(node_tags(node)),
                ]
            ).lower()
            node_terms = set(re.findall(r"[a-zA-Z0-9_]+", haystack))
            overlap = len(query_terms.intersection(node_terms)) / max(1, len(query_terms))
            score = overlap
            if node.get("node_type") in preferred_types:
                score += 0.38
            if node.get("node_role") == "PROBLEM":
                score += 0.08
            if node.get("node_role") == "CONTENT":
                score += 0.12
            if "BINARY_SEARCH" in approaches and "binary" in haystack:
                score += 0.25
            if "FORMULA" in approaches and any(term in haystack for term in ["formula", "quadratic", "sqrt", "equation"]):
                score += 0.25
            if "PROOF" == intent["stage"] and any(term in haystack for term in ["proof", "claim", "observation"]):
                score += 0.2
            if "EDGE_CASES" == intent["stage"] and any(term in haystack for term in ["mistake", "edge", "corner", "example"]):
                score += 0.2
            if score > 0:
                rows.append(
                    {
                        "tree_node_id": node["tree_node_id"],
                        "guided_score": score,
                        "node_role": node["node_role"],
                        "node_type": node["node_type"],
                        "global_problem_id": node["global_problem_id"],
                        "title": node["title"],
                    }
                )
        guided = pd.DataFrame(rows)
        if guided.empty:
            return guided
        return guided.sort_values("guided_score", ascending=False).reset_index(drop=True)

    def metadata_bonus(self, node: pd.Series, intent: dict[str, Any], filters: dict[str, Any] | None) -> float:
        bonus = 0.0
        if node.get("node_type") in set(intent["preferred_node_types"]):
            bonus += 0.13
        tags = set(node_tags(node))
        query_terms = set(re.findall(r"[a-zA-Z0-9_]+", intent["query"].lower()))
        normalized_query = intent["query"].lower().replace("_", " ")
        if tags.intersection(query_terms) or any(tag.lower() in normalized_query for tag in tags):
            bonus += 0.1
        if filters:
            difficulty = node.get("normalized_difficulty")
            if pd.notna(difficulty):
                if filters.get("min_difficulty") is not None and difficulty >= filters["min_difficulty"]:
                    bonus += 0.03
                if filters.get("max_difficulty") is not None and difficulty <= filters["max_difficulty"]:
                    bonus += 0.03
            filter_tags = {str(tag).lower() for tag in filters.get("tags", [])}
            lowered_tags = {tag.lower() for tag in tags}
            if filter_tags and (lowered_tags.intersection(filter_tags) or any(tag in lowered_tags for tag in filter_tags)):
                bonus += 0.12
        return bonus

    def hybrid_search(
        self,
        query: str,
        top_k_nodes: int = 12,
        top_k_chunks: int = 24,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        intent = infer_query_intent(query)
        value_nodes, chunk_hits = self.value_search(query, top_k_chunks=top_k_chunks)
        guided_nodes = self.guided_tree_search(query)

        merged: dict[str, dict[str, Any]] = {}
        for _, row in value_nodes.iterrows():
            merged.setdefault(row["tree_node_id"], {}).update(row.to_dict())
        for _, row in guided_nodes.iterrows():
            merged.setdefault(row["tree_node_id"], {}).update(row.to_dict())

        node_lookup = self.tree_nodes.set_index("tree_node_id")
        result_rows = []
        for node_id, partial in merged.items():
            if node_id not in node_lookup.index:
                continue
            node = node_lookup.loc[node_id]
            if filters:
                difficulty = node.get("normalized_difficulty")
                if filters.get("min_difficulty") is not None and pd.notna(difficulty) and difficulty < filters["min_difficulty"]:
                    continue
                if filters.get("max_difficulty") is not None and pd.notna(difficulty) and difficulty > filters["max_difficulty"]:
                    continue
                filter_tags = set(filters.get("tags", []))
                if filter_tags:
                    tags = {tag.lower() for tag in node_tags(node)}
                    requested = {str(tag).lower() for tag in filter_tags}
                    if not tags.intersection(requested):
                        continue
            value_score = float(partial.get("value_score", 0.0))
            guided_score = float(partial.get("guided_score", 0.0))
            meta = self.metadata_bonus(node, intent, filters)
            hybrid_score = self.alpha * value_score + (1 - self.alpha) * guided_score + meta
            result_rows.append(
                {
                    "tree_node_id": node_id,
                    "parent_tree_node_id": node.get("parent_tree_node_id", ""),
                    "global_problem_id": node.get("global_problem_id", ""),
                    "node_role": node.get("node_role", ""),
                    "node_type": node.get("node_type", ""),
                    "title": node.get("title", ""),
                    "summary": node.get("summary", ""),
                    "value_score": round(value_score, 6),
                    "guided_score": round(guided_score, 6),
                    "metadata_bonus": round(meta, 6),
                    "hybrid_score": round(float(hybrid_score), 6),
                    "normalized_difficulty": node.get("normalized_difficulty"),
                    "normalized_tags": node.get("normalized_tags"),
                    "url": node.get("url", ""),
                    "reason": explain_node_reason(node, intent, value_score, guided_score, meta),
                }
            )

        results = pd.DataFrame(result_rows)
        if not results.empty:
            results = results.sort_values("hybrid_score", ascending=False).head(top_k_nodes).reset_index(drop=True)

        recommendations = recommend_problems_from_results(results, self.tree_nodes, intent)
        elapsed = time.perf_counter() - start
        return {
            "query": query,
            "intent": intent,
            "results": results,
            "chunk_hits": chunk_hits,
            "value_nodes": value_nodes,
            "guided_nodes": guided_nodes,
            "recommendations": recommendations,
            "elapsed_seconds": elapsed,
            "enough_information": enough_information(results, intent),
        }


def explain_node_reason(node: pd.Series, intent: dict[str, Any], value_score: float, guided_score: float, metadata_bonus: float) -> str:
    pieces = []
    if value_score > 0:
        pieces.append("recibio score vectorial desde chunks similares")
    if guided_score > 0:
        pieces.append("coincide con la intencion detectada de la query")
    if node.get("node_type") in intent.get("preferred_node_types", []):
        pieces.append(f"su tipo {node.get('node_type')} encaja con la etapa {intent.get('stage')}")
    if metadata_bonus > 0:
        pieces.append("recibio bonus por metadata/tags/filtros")
    return "; ".join(pieces) or "nodo candidato por estructura del arbol"


def enough_information(results: pd.DataFrame, intent: dict[str, Any]) -> bool:
    if results.empty:
        return False
    preferred = set(intent.get("preferred_node_types", []))
    top = results.head(6)
    has_preferred = bool(set(top["node_type"]).intersection(preferred))
    has_content = bool((top["node_role"] == "CONTENT").any())
    return has_preferred and has_content and float(top["hybrid_score"].max()) > 0.2


def recommend_problems_from_results(results: pd.DataFrame, tree_nodes: pd.DataFrame, intent: dict[str, Any], top_k: int = 5) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    problem_scores = results[results["global_problem_id"].astype(str) != ""].groupby("global_problem_id")[
        ["hybrid_score", "value_score", "guided_score", "metadata_bonus"]
    ].max()
    problem_nodes = tree_nodes[tree_nodes["node_role"] == "PROBLEM"].set_index("global_problem_id")
    rows = []
    for problem_id, scores in problem_scores.iterrows():
        if problem_id not in problem_nodes.index:
            continue
        problem = problem_nodes.loc[problem_id]
        tags = list(dict.fromkeys(parse_listish(problem.get("normalized_tags")) + parse_listish(problem.get("topic_group"))))
        stage = intent.get("stage", "UNDERSTANDING")
        reason = f"Recomendado porque recupero nodos relevantes para {stage.lower()}"
        if tags:
            reason += f" y coincide con temas {', '.join(tags[:3])}"
        rows.append(
            {
                "global_problem_id": problem_id,
                "title": problem.get("title"),
                "normalized_difficulty": problem.get("normalized_difficulty"),
                "tags": tags,
                "recommendation_score": round(float(scores["hybrid_score"]), 6),
                "reason": reason + ".",
                "url": problem.get("url", ""),
            }
        )
    recs = pd.DataFrame(rows)
    if recs.empty:
        return recs
    return recs.sort_values("recommendation_score", ascending=False).head(top_k).reset_index(drop=True)
