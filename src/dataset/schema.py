"""Dataset schema contract for the CP RAG prototype.

Phase 2 freezes the minimum shape expected by later modules. The checks here
are intentionally strict enough to catch broken scraping/indexing, but they do
not assume a huge dataset or require AtCoder to be present.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_PROBLEM_COLUMNS = [
    "global_problem_id",
    "source",
    "platform_problem_id",
    "contest_id",
    "problem_index",
    "title",
    "url",
    "rating",
    "difficulty",
    "points",
    "normalized_difficulty",
    "difficulty_source",
    "original_tags",
    "normalized_tags",
    "topic_group",
    "solved_count",
    "statement",
    "official_editorial",
    "statement_status",
    "editorial_status",
]


OPTIONAL_PROBLEM_COLUMNS = [
    "task_id",
    "time_limit",
    "memory_limit",
    "input_description",
    "output_description",
    "constraints",
    "samples",
    "notes",
    "editorial_url",
    "editorial_parse_method",
    "editorial_problem_code",
    "editorial_toggle_count",
    "parse_status",
    "language",
    "raw_metadata",
]


REQUIRED_PAGE_NODE_COLUMNS = [
    "node_id",
    "global_problem_id",
    "source",
    "contest_id",
    "platform_problem_id",
    "node_type",
    "node_title",
    "node_text",
    "parent_node_id",
    "order",
    "normalized_difficulty",
    "original_tags",
    "normalized_tags",
    "topic_group",
    "url",
    "metadata",
]


IMPORTANT_EDITORIAL_NODE_TYPES = [
    "EDITORIAL_FULL",
    "EDITORIAL_OBSERVATION",
    "EDITORIAL_PROOF",
    "EDITORIAL_ALGORITHM",
    "EDITORIAL_COMPLEXITY",
    "IMPLEMENTATION_HINTS",
    "COMMON_MISTAKES",
]


@dataclass
class DatasetIssue:
    severity: str
    dataset: str
    check: str
    message: str
    row_id: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "dataset": self.dataset,
            "check": self.check,
            "message": self.message,
            "row_id": self.row_id,
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


def text_len(value: Any) -> int:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0
    return len(str(value).strip())


def load_processed_datasets(processed_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    processed = Path(processed_dir)
    problems_path = processed / "cp_problems_dataset.csv"
    nodes_path = processed / "cp_page_nodes_dataset.csv"
    if not problems_path.exists():
        raise FileNotFoundError(f"Missing {problems_path}")
    if not nodes_path.exists():
        raise FileNotFoundError(f"Missing {nodes_path}")
    return pd.read_csv(problems_path), pd.read_csv(nodes_path)


def missing_columns(df: pd.DataFrame, required: list[str]) -> list[str]:
    return [column for column in required if column not in df.columns]


def validate_problem_schema(problems: pd.DataFrame) -> list[DatasetIssue]:
    issues: list[DatasetIssue] = []
    for column in missing_columns(problems, REQUIRED_PROBLEM_COLUMNS):
        issues.append(DatasetIssue("error", "problems", "required_column", f"Missing required column {column}"))
    if issues:
        return issues

    if problems.empty:
        issues.append(DatasetIssue("error", "problems", "non_empty", "Problem dataset is empty"))
        return issues

    duplicated = problems[problems["global_problem_id"].duplicated(keep=False)]["global_problem_id"].tolist()
    for problem_id in duplicated:
        issues.append(DatasetIssue("error", "problems", "unique_global_problem_id", "Duplicated global_problem_id", str(problem_id)))

    required_nonempty = ["global_problem_id", "source", "title", "url"]
    for column in required_nonempty:
        missing = problems[problems[column].isna() | (problems[column].astype(str).str.strip() == "")]
        for _, row in missing.iterrows():
            issues.append(DatasetIssue("error", "problems", f"nonempty_{column}", f"{column} is empty", str(row.get("global_problem_id", ""))))

    placeholder_mask = problems["official_editorial"].fillna("").astype(str).str.contains("Tutorial is loading", case=False, regex=False)
    for _, row in problems[placeholder_mask].iterrows():
        issues.append(DatasetIssue("error", "problems", "editorial_placeholder", "Editorial still contains Codeforces placeholder text", str(row["global_problem_id"])))

    downloaded = problems[problems["editorial_status"].astype(str) == "downloaded"]
    for _, row in downloaded.iterrows():
        if text_len(row.get("official_editorial")) < 80:
            issues.append(DatasetIssue("warning", "problems", "short_editorial", "Downloaded editorial is unusually short", str(row["global_problem_id"])))
        if row.get("source") == "codeforces" and "editorial_problem_code" in problems.columns:
            expected_code = f"{row.get('contest_id')}{row.get('problem_index')}"
            actual_code = str(row.get("editorial_problem_code", "")).strip()
            if actual_code and actual_code != expected_code:
                issues.append(
                    DatasetIssue(
                        "error",
                        "problems",
                        "editorial_problem_code_match",
                        f"Expected editorial code {expected_code}, got {actual_code}",
                        str(row["global_problem_id"]),
                    )
                )

    for _, row in problems.iterrows():
        if not parse_listish(row.get("normalized_tags")):
            issues.append(DatasetIssue("warning", "problems", "normalized_tags_present", "Problem has no normalized tags", str(row["global_problem_id"])))
        if pd.isna(row.get("normalized_difficulty")):
            issues.append(DatasetIssue("warning", "problems", "difficulty_present", "Problem has no normalized difficulty", str(row["global_problem_id"])))

    return issues


def validate_page_node_schema(page_nodes: pd.DataFrame, problems: pd.DataFrame) -> list[DatasetIssue]:
    issues: list[DatasetIssue] = []
    for column in missing_columns(page_nodes, REQUIRED_PAGE_NODE_COLUMNS):
        issues.append(DatasetIssue("error", "page_nodes", "required_column", f"Missing required column {column}"))
    if issues:
        return issues

    if page_nodes.empty:
        issues.append(DatasetIssue("error", "page_nodes", "non_empty", "Page node dataset is empty"))
        return issues

    duplicated = page_nodes[page_nodes["node_id"].duplicated(keep=False)]["node_id"].tolist()
    for node_id in duplicated:
        issues.append(DatasetIssue("error", "page_nodes", "unique_node_id", "Duplicated node_id", str(node_id)))

    problem_ids = set(problems["global_problem_id"].astype(str))
    orphan_nodes = page_nodes[~page_nodes["global_problem_id"].astype(str).isin(problem_ids)]
    for _, row in orphan_nodes.iterrows():
        issues.append(DatasetIssue("error", "page_nodes", "problem_reference", "Node references a missing problem", str(row["node_id"])))

    empty_text = page_nodes[
        page_nodes["node_type"].isin(["STATEMENT", "EDITORIAL_FULL"])
        & (page_nodes["node_text"].fillna("").astype(str).str.strip() == "")
    ]
    for _, row in empty_text.iterrows():
        issues.append(DatasetIssue("warning", "page_nodes", "important_node_text", "Important node has empty node_text", str(row["node_id"])))

    editorial_full = page_nodes[page_nodes["node_type"] == "EDITORIAL_FULL"].copy()
    if not editorial_full.empty:
        placeholders = editorial_full["node_text"].fillna("").astype(str).str.contains("Tutorial is loading", case=False, regex=False)
        for _, row in editorial_full[placeholders].iterrows():
            issues.append(DatasetIssue("error", "page_nodes", "editorial_placeholder", "Editorial node still contains placeholder text", str(row["node_id"])))

    counts = page_nodes.groupby("global_problem_id")["node_type"].apply(set)
    for problem_id in problem_ids:
        available = counts.get(problem_id, set())
        if "STATEMENT" not in available:
            issues.append(DatasetIssue("warning", "page_nodes", "statement_node_present", "Problem has no STATEMENT node", problem_id))
        if "EDITORIAL_FULL" not in available:
            issues.append(DatasetIssue("warning", "page_nodes", "editorial_full_node_present", "Problem has no EDITORIAL_FULL node", problem_id))

    return issues


def build_contract_report(problems: pd.DataFrame, page_nodes: pd.DataFrame) -> dict[str, Any]:
    issues = validate_problem_schema(problems) + validate_page_node_schema(page_nodes, problems)
    issue_rows = [issue.to_dict() for issue in issues]
    severity_counts = pd.Series([issue["severity"] for issue in issue_rows]).value_counts().to_dict() if issue_rows else {}
    return {
        "status": "failed" if severity_counts.get("error", 0) else "passed",
        "problem_count": int(len(problems)),
        "page_node_count": int(len(page_nodes)),
        "required_problem_columns": REQUIRED_PROBLEM_COLUMNS,
        "required_page_node_columns": REQUIRED_PAGE_NODE_COLUMNS,
        "optional_problem_columns_present": [column for column in OPTIONAL_PROBLEM_COLUMNS if column in problems.columns],
        "severity_counts": severity_counts,
        "issues": issue_rows,
    }

