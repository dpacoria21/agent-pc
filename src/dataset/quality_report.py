"""Dataset quality reporting for Phase 2."""

from __future__ import annotations

from typing import Any

import pandas as pd

from dataset.schema import IMPORTANT_EDITORIAL_NODE_TYPES, parse_listish, text_len


def value_counts_dict(series: pd.Series, limit: int | None = None) -> dict[str, int]:
    counts = series.fillna("missing").astype(str).value_counts()
    if limit:
        counts = counts.head(limit)
    return {str(key): int(value) for key, value in counts.items()}


def explode_list_counts(df: pd.DataFrame, column: str, limit: int = 25) -> dict[str, int]:
    values: list[str] = []
    if column not in df.columns:
        return {}
    for value in df[column].tolist():
        values.extend(parse_listish(value))
    if not values:
        return {}
    return {str(key): int(value) for key, value in pd.Series(values).value_counts().head(limit).items()}


def difficulty_buckets(series: pd.Series) -> dict[str, int]:
    values = pd.to_numeric(series, errors="coerce")
    labels = ["0-799", "800-1199", "1200-1599", "1600-1999", "2000-2399", "2400+"]
    buckets = pd.cut(
        values,
        bins=[-float("inf"), 799, 1199, 1599, 1999, 2399, float("inf")],
        labels=labels,
    )
    return {str(key): int(value) for key, value in buckets.value_counts(sort=False).items()}


def build_quality_report(problems: pd.DataFrame, page_nodes: pd.DataFrame, contract_report: dict[str, Any]) -> dict[str, Any]:
    editorial_downloaded = problems["editorial_status"].astype(str).eq("downloaded") if "editorial_status" in problems else pd.Series(dtype=bool)
    statement_downloaded = problems["statement_status"].astype(str).eq("downloaded") if "statement_status" in problems else pd.Series(dtype=bool)

    editorial_lengths = problems["official_editorial"].apply(text_len) if "official_editorial" in problems else pd.Series(dtype=int)
    statement_lengths = problems["statement"].apply(text_len) if "statement" in problems else pd.Series(dtype=int)

    nonempty_nodes = page_nodes[page_nodes["node_text"].fillna("").astype(str).str.strip() != ""].copy()
    editorial_nodes = page_nodes[page_nodes["node_type"].isin(IMPORTANT_EDITORIAL_NODE_TYPES)].copy()
    nonempty_editorial_nodes = editorial_nodes[editorial_nodes["node_text"].fillna("").astype(str).str.strip() != ""]

    report = {
        "contract_status": contract_report["status"],
        "problem_count": int(len(problems)),
        "page_node_count": int(len(page_nodes)),
        "nonempty_page_node_count": int(len(nonempty_nodes)),
        "source_counts": value_counts_dict(problems["source"]) if "source" in problems else {},
        "difficulty_buckets": difficulty_buckets(problems["normalized_difficulty"]) if "normalized_difficulty" in problems else {},
        "top_normalized_tags": explode_list_counts(problems, "normalized_tags"),
        "top_topic_groups": explode_list_counts(problems, "topic_group"),
        "statement_status_counts": value_counts_dict(problems["statement_status"]) if "statement_status" in problems else {},
        "editorial_status_counts": value_counts_dict(problems["editorial_status"]) if "editorial_status" in problems else {},
        "editorial_parse_method_counts": value_counts_dict(problems["editorial_parse_method"]) if "editorial_parse_method" in problems else {},
        "statement_download_rate": round(float(statement_downloaded.mean()) if len(problems) else 0.0, 4),
        "editorial_download_rate": round(float(editorial_downloaded.mean()) if len(problems) else 0.0, 4),
        "avg_statement_chars": round(float(statement_lengths.mean()) if len(statement_lengths) else 0.0, 2),
        "avg_editorial_chars": round(float(editorial_lengths.mean()) if len(editorial_lengths) else 0.0, 2),
        "node_type_counts": value_counts_dict(page_nodes["node_type"]) if "node_type" in page_nodes else {},
        "nonempty_node_type_counts": value_counts_dict(nonempty_nodes["node_type"]) if "node_type" in nonempty_nodes else {},
        "editorial_related_node_count": int(len(editorial_nodes)),
        "nonempty_editorial_related_node_count": int(len(nonempty_editorial_nodes)),
        "issues_by_severity": contract_report.get("severity_counts", {}),
    }
    return report


def build_issue_dataframe(contract_report: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(contract_report.get("issues", []), columns=["severity", "dataset", "check", "message", "row_id"])

