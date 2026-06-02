"""Prompts and JSON Schemas for GPT-assisted problem structuring."""

from __future__ import annotations

from typing import Any


LLM_TREE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "problem_id": {"type": "string"},
        "title": {"type": "string"},
        "main_topic": {"type": "string"},
        "difficulty_comment": {"type": "string"},
        "strategies": {"type": "array", "items": {"type": "string"}},
        "student_skills": {"type": "array", "items": {"type": "string"}},
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "node_key": {"type": "string"},
                    "parent_node_key": {"type": "string"},
                    "node_type": {
                        "type": "string",
                        "enum": [
                            "PROBLEM",
                            "UNDERSTANDING",
                            "STATEMENT_SUMMARY",
                            "MATHEMATICAL_MODEL",
                            "OBSERVATION",
                            "PROOF",
                            "PROOF_STEP",
                            "ALGORITHM",
                            "COMPLEXITY",
                            "IMPLEMENTATION",
                            "COMMON_MISTAKE",
                            "HINT",
                            "EDGE_CASE",
                            "PREREQUISITE",
                            "UNKNOWN",
                        ],
                    },
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "evidence_text": {"type": "string"},
                    "source_section": {"type": "string"},
                    "skills": {"type": "array", "items": {"type": "string"}},
                    "strategies": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": [
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
                ],
            },
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "source_node_key": {"type": "string"},
                    "target_node_key": {"type": "string"},
                    "edge_type": {
                        "type": "string",
                        "enum": [
                            "PREREQUISITE_OF",
                            "SUPPORTS",
                            "ALTERNATIVE_APPROACH",
                            "SAME_STRATEGY",
                            "PROOF_DEPENDS_ON",
                            "IMPLEMENTATION_DEPENDS_ON",
                        ],
                    },
                    "reason": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["source_node_key", "target_node_key", "edge_type", "reason", "confidence"],
            },
        },
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "problem_id",
        "title",
        "main_topic",
        "difficulty_comment",
        "strategies",
        "student_skills",
        "nodes",
        "edges",
        "warnings",
    ],
}


SYSTEM_PROMPT = """Actua como investigador en IA educativa y experto en programacion competitiva.
Tu tarea es estructurar un problema y su editorial oficial en un arbol pedagogico para un sistema RAG.

Reglas obligatorias:
- No inventes contenido que no aparezca en el statement o editorial.
- No escribas solucion nueva si no esta soportada por evidence_text.
- Separa observacion, modelo matematico, prueba, algoritmo, complejidad, implementacion y errores comunes si existen.
- Usa evidence_text corto y literal/parafraseado desde los textos dados.
- Si falta informacion, deja warnings.
- Devuelve solo JSON que cumpla el esquema.
"""


def truncate_text(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "\n[TRUNCATED]"


def build_problem_tree_prompt(problem: dict[str, Any], max_statement_chars: int = 4500, max_editorial_chars: int = 8500) -> str:
    return f"""Estructura este problema en nodos pedagogicos.

METADATA
problem_id: {problem.get("global_problem_id", "")}
source: {problem.get("source", "")}
title: {problem.get("title", "")}
rating: {problem.get("rating", "")}
normalized_difficulty: {problem.get("normalized_difficulty", "")}
tags: {problem.get("normalized_tags", "")}
original_tags: {problem.get("original_tags", "")}

STATEMENT
{truncate_text(problem.get("statement", ""), max_statement_chars)}

CONSTRAINTS
{truncate_text(problem.get("constraints", ""), 1800)}

SAMPLES
{truncate_text(problem.get("samples", ""), 1800)}

OFFICIAL_EDITORIAL
{truncate_text(problem.get("official_editorial", ""), max_editorial_chars)}

Devuelve un arbol en formato plano:
- Debe existir un nodo raiz node_type=PROBLEM con parent_node_key="".
- Los demas nodos deben apuntar a parent_node_key de otro nodo.
- Usa node_key corto y estable, por ejemplo "problem", "model", "proof", "algorithm".
- Incluye edge_type solo cuando la relacion sea clara.
"""

