"""Cross-model audit of persisted RAG answers with Gemini.

This script does not regenerate answers. It reuses the existing GPT-4o-mini
outputs and asks Gemini to score Faithfulness and Answer Relevancy against the
retrieved Codeforces context. The result is an independent-provider audit of a
persisted subset, not a replacement for human validation.
"""

from __future__ import annotations

import json
import math
import os
import re
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
DEFAULT_MODEL = "gemini-2.5-flash"
THRESHOLDS = {"Faithfulness": 0.80, "Answer Relevancy": 0.85}

MODE_CONFIG = {
    "Page Nodes": {
        "results": PROCESSED / "langchain_openai_rag_eval_results_page_nodes.csv",
        "units": PROCESSED / "cp_page_nodes_dataset.csv",
        "text_column": "node_text",
    },
    "PageIndex Chunks": {
        "results": PROCESSED / "langchain_openai_rag_eval_results_pageindex_chunks.csv",
        "units": PROCESSED / "cp_pageindex_ready_chunks.csv",
        "text_column": "chunk_text",
    },
}


def load_env_value(names: tuple[str, ...]) -> tuple[str, str]:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value, name
    for env_name in (".env.local", ".env"):
        path = ROOT / env_name
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() in names and value.strip().strip("\"'"):
                return value.strip().strip("\"'"), key.strip()
    raise RuntimeError(f"No se encontró ninguna variable utilizable: {', '.join(names)}")


def parse_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass
    return [text]


def tokens(text: Any) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z0-9_]+", str(text).lower()) if len(token) > 2}


def reconstruct_context(result: pd.Series, units: pd.DataFrame, text_column: str, max_chars: int = 10000) -> str:
    node_ids = parse_list(result.get("top_node_ids"))
    query_tokens = tokens(result.get("query"))
    pieces: list[str] = []
    used_rows: set[int] = set()
    total = 0
    for rank, node_id in enumerate(node_ids, 1):
        candidates = units[units["node_id"].astype(str).eq(str(node_id))]
        if candidates.empty:
            continue
        ranked = []
        for idx, row in candidates.iterrows():
            if idx in used_rows:
                continue
            text = str(row.get(text_column, "") or "")
            overlap = len(query_tokens.intersection(tokens(text)))
            ranked.append((overlap, len(text), idx, row))
        if not ranked:
            continue
        _, _, idx, row = max(ranked, key=lambda item: (item[0], item[1]))
        used_rows.add(idx)
        text = re.sub(r"\s+", " ", str(row.get(text_column, "") or "")).strip()
        piece = (
            f"[rank={rank} node_id={node_id} problem={row.get('global_problem_id', '')} "
            f"type={row.get('node_type', '')}]\n{text}"
        )
        if total + len(piece) > max_chars:
            remaining = max_chars - total
            if remaining > 200:
                pieces.append(piece[:remaining])
            break
        pieces.append(piece)
        total += len(piece)
    return "\n\n".join(pieces)


def judge_prompt(query: str, context: str, answer: str) -> str:
    return f"""Actúa como evaluador independiente de un sistema RAG educativo.
No uses conocimiento externo y no intentes mejorar la respuesta. Evalúa solo la salida entregada.

Definiciones:
- Faithfulness: proporción de afirmaciones verificables de la respuesta sustentadas por el contexto recuperado. Penaliza afirmaciones no sustentadas o contradictorias.
- Answer Relevancy: grado en que la respuesta atiende directamente la consulta, sin información irrelevante ni omisiones centrales.

Asigna a cada métrica un score continuo entre 0 y 1. Sé estricto y explica brevemente la evidencia.

CONSULTA:
{query}

CONTEXTO OFICIAL RECUPERADO:
{context}

RESPUESTA DEL SISTEMA:
{answer}
"""


def call_gemini(api_key: str, model: str, prompt: str, max_retries: int = 4) -> dict[str, Any]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    schema = {
        "type": "OBJECT",
        "properties": {
            "faithfulness": {
                "type": "OBJECT",
                "properties": {"score": {"type": "NUMBER"}, "reason": {"type": "STRING"}},
                "required": ["score", "reason"],
            },
            "answer_relevancy": {
                "type": "OBJECT",
                "properties": {"score": {"type": "NUMBER"}, "reason": {"type": "STRING"}},
                "required": ["score", "reason"],
            },
        },
        "required": ["faithfulness", "answer_relevancy"],
    }
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": schema,
        },
    }
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    for attempt in range(max_retries):
        request = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                data = json.loads(response.read().decode("utf-8"))
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            parsed = json.loads(text)
            return parsed
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            if attempt + 1 >= max_retries:
                raise RuntimeError(f"Gemini no pudo evaluar el caso después de {max_retries} intentos") from exc
            time.sleep(8 * (attempt + 1))
    raise AssertionError("unreachable")


def clamp(value: Any) -> float:
    try:
        return round(max(0.0, min(1.0, float(value))), 4)
    except (TypeError, ValueError):
        return 0.0


def correlation(left: pd.Series, right: pd.Series) -> float | None:
    if len(left) < 2 or left.nunique() < 2 or right.nunique() < 2:
        return None
    value = left.corr(right)
    return None if pd.isna(value) else round(float(value), 4)


def build_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode in ["Overall", *MODE_CONFIG.keys()]:
        subset = results if mode == "Overall" else results[results["mode"].eq(mode)]
        for metric, threshold in THRESHOLDS.items():
            original = pd.to_numeric(subset[f"original_{metric}"], errors="coerce")
            gemini = pd.to_numeric(subset[f"gemini_{metric}"], errors="coerce")
            rows.append(
                {
                    "Modo": mode,
                    "Métrica": metric,
                    "n": int(len(subset)),
                    "Umbral": threshold,
                    "Promedio juez original": round(float(original.mean()), 4),
                    "Promedio Gemini": round(float(gemini.mean()), 4),
                    "Desviación Gemini": round(float(gemini.std(ddof=0)), 4),
                    "% Gemini >= umbral": round(float((gemini >= threshold).mean() * 100), 2),
                    "MAE entre jueces": round(float((original - gemini).abs().mean()), 4),
                    "Correlación de Pearson": correlation(original, gemini),
                    "Acuerdo de clasificación": round(
                        float(((original >= threshold) == (gemini >= threshold)).mean() * 100), 2
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_human_protocol(results: pd.DataFrame) -> pd.DataFrame:
    selected = results.sort_values(["mode", "query_id"]).groupby("mode", as_index=False).head(4).copy()
    return pd.DataFrame(
        {
            "review_id": [f"human_{idx + 1:02d}" for idx in range(len(selected))],
            "mode": selected["mode"],
            "query_id": selected["query_id"],
            "global_problem_id": selected["global_problem_id"],
            "query": selected["query"],
            "answer": selected["answer"],
            "context_excerpt": selected["context"].str.slice(0, 1200),
            "reviewer_1_faithfulness_1_to_5": "",
            "reviewer_1_relevancy_1_to_5": "",
            "reviewer_1_pedagogical_utility_1_to_5": "",
            "reviewer_2_faithfulness_1_to_5": "",
            "reviewer_2_relevancy_1_to_5": "",
            "reviewer_2_pedagogical_utility_1_to_5": "",
            "adjudication_notes": "",
        }
    )


def main() -> None:
    api_key, key_source = load_env_value(("GOOGLE_API_KEY", "GEMINI_API_KEY"))
    model = os.getenv("GEMINI_JUDGE_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    rows = []
    for mode, config in MODE_CONFIG.items():
        persisted = pd.read_csv(config["results"])
        units = pd.read_csv(config["units"])
        for _, result in persisted.iterrows():
            context = reconstruct_context(result, units, config["text_column"])
            started = time.perf_counter()
            judgment = call_gemini(
                api_key,
                model,
                judge_prompt(str(result["query"]), context, str(result["answer"])),
            )
            faithfulness = judgment.get("faithfulness", {})
            relevancy = judgment.get("answer_relevancy", {})
            expected_problem = str(result.get("global_problem_id", ""))
            top_problem_ids = parse_list(result.get("top_problem_ids"))
            context_problem_precision = (
                sum(problem_id == expected_problem for problem_id in top_problem_ids) / len(top_problem_ids)
                if top_problem_ids
                else 0.0
            )
            rows.append(
                {
                    "mode": mode,
                    "query_id": result.get("query_id", ""),
                    "global_problem_id": expected_problem,
                    "query": result.get("query", ""),
                    "answer": result.get("answer", ""),
                    "context": context,
                    "context_problem_precision": round(context_problem_precision, 4),
                    "original_Faithfulness": clamp(result.get("Faithfulness")),
                    "original_Answer Relevancy": clamp(result.get("Answer Relevancy")),
                    "gemini_Faithfulness": clamp(faithfulness.get("score")),
                    "gemini_Answer Relevancy": clamp(relevancy.get("score")),
                    "gemini_Faithfulness_reason": str(faithfulness.get("reason", "")),
                    "gemini_Answer Relevancy_reason": str(relevancy.get("reason", "")),
                    "gemini_latency_seconds": round(time.perf_counter() - started, 3),
                }
            )
            print(f"Gemini audit: {mode} / {result.get('query_id', '')}", flush=True)
            time.sleep(6.5)

    results = pd.DataFrame(rows)
    summary = build_summary(results)
    protocol = build_human_protocol(results)
    results_path = PROCESSED / "gemini_cross_judge_results.csv"
    summary_path = PROCESSED / "gemini_cross_judge_summary.csv"
    protocol_path = PROCESSED / "human_validation_protocol.csv"
    results.to_csv(results_path, index=False)
    results.to_json(PROCESSED / "gemini_cross_judge_results.json", orient="records", force_ascii=False, indent=2)
    summary.to_csv(summary_path, index=False)
    summary.to_json(PROCESSED / "gemini_cross_judge_summary.json", orient="records", force_ascii=False, indent=2)
    protocol.to_csv(protocol_path, index=False)
    report = {
        "status": "ok",
        "generator_model": "gpt-4o-mini (persisted answers)",
        "independent_judge_model": model,
        "judge_provider": "Google",
        "key_source": key_source,
        "evaluated_answers": int(len(results)),
        "problems_per_mode": int(results.groupby("mode")["global_problem_id"].nunique().min()),
        "temperature": 0,
        "validation_type": "cross-model external automated audit grounded in retrieved official context",
        "human_validation_status": "protocol prepared; ratings pending",
        "mean_context_problem_precision": round(float(results["context_problem_precision"].mean()), 4),
        "outputs": {
            "results_csv": str(results_path),
            "summary_csv": str(summary_path),
            "human_protocol_csv": str(protocol_path),
        },
    }
    (PROCESSED / "gemini_cross_judge_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
