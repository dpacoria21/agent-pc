"""LangChain RAG runner for the CP dataset.

Retrieval stays local and deterministic (TF-IDF + SVD + cosine-equivalent
dot product). Answer generation and judging use ChatOpenAI through LangChain.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from llm.env_config import DEFAULT_OPENAI_MODEL, OpenAISettings, resolve_openai_settings


DEFAULT_METRIC_THRESHOLDS = {"Faithfulness": 0.75, "Answer Relevancy": 0.75}


def parse_jsonish(value: Any) -> list[str]:
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
    text = "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_document(row: pd.Series) -> str:
    tags = " ".join(parse_jsonish(row.get("normalized_tags")))
    topic = " ".join(parse_jsonish(row.get("topic_group")))
    return clean_text(
        " ".join(
            [
                str(row.get("node_title", "")),
                str(row.get("title", "")),
                str(row.get("node_type", "")),
                str(row.get("node_text", "")),
                str(row.get("chunk_text", "")),
                str(row.get("text", "")),
                tags,
                topic,
            ]
        )
    )


@dataclass
class LocalNodeRetriever:
    nodes: pd.DataFrame
    vectorizer: TfidfVectorizer
    svd: TruncatedSVD | None
    embeddings: np.ndarray

    @classmethod
    def from_nodes(cls, nodes: pd.DataFrame) -> "LocalNodeRetriever":
        filtered = nodes.copy()
        if "node_text" not in filtered.columns:
            if "chunk_text" in filtered.columns:
                filtered["node_text"] = filtered["chunk_text"]
            elif "text" in filtered.columns:
                filtered["node_text"] = filtered["text"]
            else:
                filtered["node_text"] = ""
        if "node_id" not in filtered.columns and "chunk_id" in filtered.columns:
            filtered["node_id"] = filtered["chunk_id"]
        filtered["node_text"] = filtered["node_text"].fillna("").astype(str)
        filtered = filtered[filtered["node_text"].str.len() > 0].reset_index(drop=True)
        docs = [build_document(row) for _, row in filtered.iterrows()]
        vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=1)
        tfidf = vectorizer.fit_transform(docs)
        if min(tfidf.shape) > 3:
            n_components = min(128, max(2, min(tfidf.shape) - 1))
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            embeddings = normalize(svd.fit_transform(tfidf)).astype("float32")
        else:
            svd = None
            embeddings = normalize(tfidf).astype("float32")
        return cls(nodes=filtered, vectorizer=vectorizer, svd=svd, embeddings=embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        vec = self.vectorizer.transform([query])
        if self.svd is not None:
            return normalize(self.svd.transform(vec)).astype("float32")[0]
        return normalize(vec).astype("float32")[0]

    def retrieve(self, query: str, top_k: int = 5) -> pd.DataFrame:
        query_embedding = self.embed_query(query)
        scores = self.embeddings @ query_embedding
        order = np.argsort(-scores)[: min(top_k, len(scores))]
        hits = self.nodes.iloc[order].copy()
        hits["retrieval_score"] = scores[order]
        hits["rank"] = range(1, len(hits) + 1)
        return hits.reset_index(drop=True)


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un tutor RAG de programacion competitiva. Responde solo con la evidencia del contexto. "
            "Si el contexto no alcanza, dilo claramente. Da ayuda pedagogica, no una solucion completa innecesaria. "
            "Antes de dar una formula, verifica que la explicacion verbal y el codigo del contexto sean consistentes. "
            "Si una frase del editorial es ambigua, reconciliala con el codigo o con una derivacion algebraica breve.",
        ),
        (
            "human",
            "Consulta del estudiante:\n{query}\n\nContexto recuperado:\n{context}\n\n"
            "Respuesta breve y fundamentada:",
        ),
    ]
)


JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un evaluador academico de RAG. Devuelve solo JSON valido con claves score y reason. "
            "score debe estar entre 0 y 1.",
        ),
        (
            "human",
            "Metrica: {metric}\n\nConsulta:\n{query}\n\nContexto:\n{context}\n\nRespuesta:\n{answer}\n\n"
            "Evalua estrictamente la metrica indicada.",
        ),
    ]
)


def make_llm(settings: OpenAISettings, *, temperature: float = 0.0) -> ChatOpenAI:
    kwargs: dict[str, Any] = {
        "model": settings.model or DEFAULT_OPENAI_MODEL,
        "api_key": settings.api_key,
        "temperature": temperature,
    }
    if settings.base_url:
        kwargs["base_url"] = settings.base_url
    return ChatOpenAI(**kwargs)


def format_context(hits: pd.DataFrame, max_chars: int = 6000) -> str:
    chunks = []
    used = 0
    for _, row in hits.iterrows():
        piece = (
            f"[rank={row.get('rank')} node_id={row.get('node_id')} "
            f"problem={row.get('global_problem_id')} type={row.get('node_type')} "
            f"score={float(row.get('retrieval_score', 0.0)):.4f}]\n"
            f"{clean_text(row.get('node_text'))}"
        )
        if used + len(piece) > max_chars:
            break
        chunks.append(piece)
        used += len(piece)
    return "\n\n".join(chunks)


def parse_judge_json(text: str) -> tuple[float, str]:
    raw = text.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.I | re.M).strip()
    try:
        data = json.loads(raw)
        score = max(0.0, min(1.0, float(data.get("score", 0.0))))
        return score, str(data.get("reason", ""))
    except Exception:
        match = re.search(r"([01](?:\.\d+)?)", raw)
        score = float(match.group(1)) if match else 0.0
        return max(0.0, min(1.0, score)), raw[:300]


def build_eval_queries(problems: pd.DataFrame, limit: int = 8) -> list[dict[str, str]]:
    selected = problems.head(limit).copy()
    queries = []
    for _, problem in selected.iterrows():
        tags = ", ".join(parse_jsonish(problem.get("normalized_tags"))[:4])
        title = problem.get("title", problem.get("global_problem_id", ""))
        query = (
            f"I am solving {title}. I need the key observation, proof idea, algorithm, "
            f"and common implementation risks. Tags: {tags}."
        )
        queries.append(
            {
                "query_id": f"q_{len(queries) + 1:02d}",
                "global_problem_id": str(problem.get("global_problem_id", "")),
                "query": query,
            }
        )
    return queries


def run_langchain_rag_eval(
    problems: pd.DataFrame,
    page_nodes: pd.DataFrame,
    *,
    model: str | None = None,
    top_k: int = 5,
    query_limit: int = 8,
    metric_thresholds: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    settings = resolve_openai_settings(requested_model=model or DEFAULT_OPENAI_MODEL)
    if not settings.available:
        raise RuntimeError("OPENAI_API_KEY is not configured. Add it to .env or environment variables.")

    retriever = LocalNodeRetriever.from_nodes(page_nodes)
    llm = make_llm(settings)
    answer_chain = ANSWER_PROMPT | llm | StrOutputParser()
    judge_chain = JUDGE_PROMPT | llm | StrOutputParser()

    rows = []
    queries = build_eval_queries(problems, limit=query_limit)
    start_all = time.perf_counter()
    for spec in queries:
        start = time.perf_counter()
        hits = retriever.retrieve(spec["query"], top_k=top_k)
        context = format_context(hits)
        answer = answer_chain.invoke({"query": spec["query"], "context": context})
        metric_scores = {}
        metric_reasons = {}
        for metric in ["Faithfulness", "Answer Relevancy"]:
            judge_text = judge_chain.invoke(
                {
                    "metric": metric,
                    "query": spec["query"],
                    "context": context,
                    "answer": answer,
                }
            )
            score, reason = parse_judge_json(judge_text)
            metric_scores[metric] = score
            metric_reasons[f"{metric} reason"] = reason
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "query_id": spec["query_id"],
                "global_problem_id": spec["global_problem_id"],
                "query": spec["query"],
                "answer": answer,
                "top_node_ids": json.dumps(hits["node_id"].head(top_k).tolist(), ensure_ascii=False),
                "top_problem_ids": json.dumps(hits["global_problem_id"].head(top_k).tolist(), ensure_ascii=False),
                "top_node_types": json.dumps(hits["node_type"].head(top_k).tolist(), ensure_ascii=False),
                "avg_retrieval_score": round(float(hits["retrieval_score"].mean()) if not hits.empty else 0.0, 4),
                "latency_seconds": round(elapsed, 3),
                **{key: round(value, 4) for key, value in metric_scores.items()},
                **metric_reasons,
            }
        )

    results = pd.DataFrame(rows)
    summary_rows = []
    thresholds = metric_thresholds or DEFAULT_METRIC_THRESHOLDS
    for metric, threshold in thresholds.items():
        values = pd.to_numeric(results[metric], errors="coerce")
        summary_rows.append(
            {
                "Metrica": metric,
                "Umbral": f"{threshold:.2f}",
                "Promedio": f"{values.mean():.2f}",
                "Desviacion estandar": f"{values.std(ddof=0):.3f}",
                "% >= umbral": f"{(values >= threshold).mean() * 100:.0f} %",
            }
        )
    summary = pd.DataFrame(summary_rows)
    metadata = {
        "model": settings.model,
        "key_source": settings.key_source,
        "queries": len(results),
        "top_k": top_k,
        "nodes_indexed": int(len(retriever.nodes)),
        "total_elapsed_seconds": round(time.perf_counter() - start_all, 3),
    }
    return results, summary, metadata
