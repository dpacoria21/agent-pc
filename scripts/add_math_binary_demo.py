"""Add a math-vs-binary-search demo slice to the local CP RAG dataset.

The demo focuses on Codeforces problems where students may reasonably choose
either an algebraic/formula route or an algorithmic binary-search route.
It appends three curated problems to the processed datasets and writes a
classification report for two simulated students.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cp_dataset_scraper import build_page_nodes_dataset, stable_json_dumps

PROCESSED = ROOT / "data" / "processed"
ASSETS = ROOT / "comparison_assets"
PROCESSED.mkdir(parents=True, exist_ok=True)
ASSETS.mkdir(parents=True, exist_ok=True)

PROBLEMS_PATH = PROCESSED / "cp_problems_dataset.csv"
NODES_PATH = PROCESSED / "cp_page_nodes_dataset.csv"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def current_agent_heuristic(idea_text: str) -> dict[str, str]:
    """Mirror the current analyze_student_idea() heuristic.

    This intentionally keeps the current limitations: there is no explicit
    BINARY_SEARCH or MATH_FORMULA class yet.
    """
    text = normalize_text(idea_text)

    approach = "UNKNOWN"
    if contains_any(text, ["segment tree", "fenwick", "dsu", "ordered set", "priority queue"]):
        approach = "DATA_STRUCTURES"
    elif contains_any(text, ["dp", "state", "transition", "memo", "base case"]):
        approach = "DP"
    elif contains_any(text, ["graph", "bfs", "dfs", "tree", "component", "shortest path"]):
        approach = "GRAPH"
    elif contains_any(text, ["sort", "greedy", "choose", "always", "exchange"]):
        approach = "GREEDY"
    elif contains_any(text, ["math", "modulo", "parity", "formula", "gcd", "combinatorics"]):
        approach = "MATH"
    elif contains_any(text, ["brute force", "try all", "enumerate all"]):
        approach = "BRUTE_FORCE"
    elif contains_any(text, ["implement", "case", "branch", "parse", "debug"]):
        approach = "IMPLEMENTATION"

    reasoning_stage = "UNDERSTANDING"
    if contains_any(text, ["debug", "wa", "tle", "runtime", "wrong answer"]):
        reasoning_stage = "DEBUGGING"
    elif contains_any(text, ["implement", "code", "branch", "parse"]):
        reasoning_stage = "IMPLEMENTATION"
    elif contains_any(text, ["prove", "proof", "invariant", "exchange"]):
        reasoning_stage = "PROOF"
    elif contains_any(text, ["algorithm", "transition", "bfs", "dfs", "sort", "compute"]):
        reasoning_stage = "ALGORITHM"
    elif contains_any(text, ["maybe", "i think", "hypothesis"]):
        reasoning_stage = "HYPOTHESIS"
    elif contains_any(text, ["observe", "notice", "property"]):
        reasoning_stage = "OBSERVATION"

    risk_type = "UNKNOWN"
    if contains_any(text, ["o(n^2)", "too slow", "tle", "complexity too high"]):
        risk_type = "BAD_COMPLEXITY"
    elif contains_any(text, ["edge", "n=1", "corner", "zero values", "empty"]):
        risk_type = "EDGE_CASES"
    elif contains_any(text, ["prove", "proof", "always valid", "invariant"]):
        risk_type = "WRONG_PROOF"
    elif contains_any(text, ["brute force", "try all"]):
        risk_type = "TLE"
    elif contains_any(text, ["wrong answer", "wa"]):
        risk_type = "WA"
    elif contains_any(text, ["correct", "safe", "works generally"]):
        risk_type = "NONE"

    return {
        "current_approach": approach,
        "current_reasoning_stage": reasoning_stage,
        "current_risk_type": risk_type,
    }


def fine_grained_strategy(idea_text: str) -> dict[str, Any]:
    """Small diagnostic layer for the thesis demo.

    This is not an LLM; it shows the kind of labels the future LLM evaluator
    should produce in a structured way.
    """
    text = normalize_text(idea_text)
    concepts = []
    if contains_any(text, ["triangular", "x*(x+1)/2", "x(x+1)/2"]):
        concepts.append("triangular_numbers")
    if contains_any(text, ["quadratic", "discriminant", "x^2", "sqrt(1+8"]):
        concepts.append("quadratic_formula")
    if contains_any(text, ["sqrt", "square root", "root"]):
        concepts.append("integer_sqrt")
    if contains_any(text, ["perfect square", "r*r", "root*root"]):
        concepts.append("perfect_square")
    if contains_any(text, ["monotonic", "predicate", "binary search", "maximum x"]):
        concepts.append("monotonic_predicate")

    if contains_any(text, ["binary search", "monotonic", "predicate", "lower_bound", "upper_bound"]):
        strategy = "BINARY_SEARCH"
    elif contains_any(text, ["formula", "quadratic", "discriminant", "sqrt", "square root", "root"]):
        strategy = "MATH_FORMULA"
    elif contains_any(text, ["precompute", "set", "enumerate", "for each"]):
        strategy = "PRECOMPUTE_ENUMERATION"
    else:
        strategy = "UNKNOWN"

    return {
        "fine_strategy": strategy,
        "math_concepts": concepts,
    }


DEMO_PROBLEMS = [
    {
        "global_problem_id": "codeforces_750_A",
        "source": "codeforces",
        "platform_problem_id": "750A",
        "contest_id": 750,
        "problem_index": "A",
        "task_id": "",
        "title": "New Year and Hurry",
        "url": "https://codeforces.com/problemset/problem/750/A",
        "rating": 800,
        "difficulty": None,
        "points": 500.0,
        "normalized_difficulty": 800,
        "difficulty_source": "rating",
        "original_tags": ["binary search", "brute force", "implementation", "math"],
        "normalized_tags": ["techniques", "implementation_constructive", "math"],
        "topic_group": ["techniques", "math"],
        "solved_count": 92630,
        "time_limit": "",
        "memory_limit": "",
        "statement": (
            "Given a limited amount of contest time, each solved problem i costs 5*i minutes. "
            "Find the maximum number of first problems that can be solved without exceeding the remaining time."
        ),
        "input_description": "n is the number of available problems; k is the time already reserved.",
        "output_description": "Maximum x such that x <= n and the cumulative time fits.",
        "constraints": "Small n, k within the contest time window.",
        "samples": [{"note": "Paraphrased demo sample omitted; see source URL."}],
        "notes": "Core inequality: 5*x*(x+1)/2 <= 240-k.",
        "official_editorial": (
            "The cumulative time is triangular. A binary-search solution checks whether a candidate x fits. "
            "A formula solution solves x^2 + x - 2*T <= 0, where T=(240-k)/5, then caps the result by n. "
            "Common mistakes: forgetting the cap by n or using floating point without flooring carefully."
        ),
        "editorial_url": "https://codeforces.com/problemset/problem/750/A",
        "statement_status": "curated_paraphrase",
        "editorial_status": "curated_observation",
        "parse_status": "manual_demo",
        "language": "en",
        "raw_metadata": {"demo_reason": "quadratic inequality can be solved by formula or binary search"},
        "accepted_strategies": ["BINARY_SEARCH", "MATH_FORMULA"],
    },
    {
        "global_problem_id": "codeforces_1915_C",
        "source": "codeforces",
        "platform_problem_id": "1915C",
        "contest_id": 1915,
        "problem_index": "C",
        "task_id": "",
        "title": "Can I Square?",
        "url": "https://codeforces.com/problemset/problem/1915/C",
        "rating": 800,
        "difficulty": None,
        "points": None,
        "normalized_difficulty": 800,
        "difficulty_source": "rating",
        "original_tags": ["binary search", "implementation"],
        "normalized_tags": ["techniques", "implementation_constructive"],
        "topic_group": ["techniques"],
        "solved_count": 54264,
        "time_limit": "",
        "memory_limit": "",
        "statement": (
            "Given numbers in a test case, decide whether their sum can be represented as the area of an integer-sided square."
        ),
        "input_description": "Several test cases with an array of positive integers.",
        "output_description": "YES if the sum is a perfect square; otherwise NO.",
        "constraints": "The sum should be handled with integer arithmetic.",
        "samples": [{"note": "Paraphrased demo sample omitted; see source URL."}],
        "notes": "Core condition: exists integer r such that r*r equals the total sum.",
        "official_editorial": (
            "Compute the total sum. A formula route uses integer square root and verifies root*root == sum. "
            "A binary-search route searches r and compares r*r against the sum. "
            "Common mistakes: using imprecise floating point sqrt or overflowing r*r in other languages."
        ),
        "editorial_url": "https://codeforces.com/problemset/problem/1915/C",
        "statement_status": "curated_paraphrase",
        "editorial_status": "curated_observation",
        "parse_status": "manual_demo",
        "language": "en",
        "raw_metadata": {"demo_reason": "perfect square detection via integer sqrt or binary search"},
        "accepted_strategies": ["BINARY_SEARCH", "MATH_FORMULA"],
    },
    {
        "global_problem_id": "codeforces_192_A",
        "source": "codeforces",
        "platform_problem_id": "192A",
        "contest_id": 192,
        "problem_index": "A",
        "task_id": "",
        "title": "Funky Numbers",
        "url": "https://codeforces.com/problemset/problem/192/A",
        "rating": 1300,
        "difficulty": None,
        "points": 500.0,
        "normalized_difficulty": 1300,
        "difficulty_source": "rating",
        "original_tags": ["binary search", "brute force", "implementation"],
        "normalized_tags": ["techniques", "implementation_constructive"],
        "topic_group": ["techniques", "math"],
        "solved_count": 11565,
        "time_limit": "",
        "memory_limit": "",
        "statement": (
            "Decide whether a number can be written as the sum of two triangular numbers."
        ),
        "input_description": "A single integer n.",
        "output_description": "YES if n = T_a + T_b for some triangular numbers; otherwise NO.",
        "constraints": "n is large enough that naive unbounded enumeration is not ideal.",
        "samples": [{"note": "Paraphrased demo sample omitted; see source URL."}],
        "notes": "Triangular number: T_i = i*(i+1)/2.",
        "official_editorial": (
            "Generate triangular numbers up to n and test complements with a set or binary search. "
            "A formula-oriented route tests if m is triangular using discriminant 1+8*m being a perfect square. "
            "Common mistakes: not limiting generated triangular numbers or confusing square numbers with triangular numbers."
        ),
        "editorial_url": "https://codeforces.com/problemset/problem/192/A",
        "statement_status": "curated_paraphrase",
        "editorial_status": "curated_observation",
        "parse_status": "manual_demo",
        "language": "en",
        "raw_metadata": {"demo_reason": "triangular representation via binary search or quadratic discriminant"},
        "accepted_strategies": ["BINARY_SEARCH", "MATH_FORMULA", "PRECOMPUTE_ENUMERATION"],
    },
]


SIMULATED_IDEAS = [
    {
        "student_id": "student_binary",
        "student_profile": "prefiere razonamiento algoritmico",
        "global_problem_id": "codeforces_750_A",
        "query_step": 1,
        "created_second": 80,
        "idea_text": "I will binary search the maximum x. The check is 5*x*(x+1)/2 <= 240-k and x <= n.",
    },
    {
        "student_id": "student_binary",
        "student_profile": "prefiere razonamiento algoritmico",
        "global_problem_id": "codeforces_1915_C",
        "query_step": 2,
        "created_second": 140,
        "idea_text": "I can binary search the integer root r and compare r*r with the sum to avoid floating point sqrt.",
    },
    {
        "student_id": "student_binary",
        "student_profile": "prefiere razonamiento algoritmico",
        "global_problem_id": "codeforces_192_A",
        "query_step": 3,
        "created_second": 260,
        "idea_text": "I will precompute triangular numbers and for each value binary search if n minus it is also triangular.",
    },
    {
        "student_id": "student_formula",
        "student_profile": "prefiere razonamiento algebraico",
        "global_problem_id": "codeforces_750_A",
        "query_step": 1,
        "created_second": 70,
        "idea_text": "I want to solve the quadratic formula for x^2+x-2*T <= 0 and use sqrt(1+8*T).",
    },
    {
        "student_id": "student_formula",
        "student_profile": "prefiere razonamiento algebraico",
        "global_problem_id": "codeforces_1915_C",
        "query_step": 2,
        "created_second": 130,
        "idea_text": "I compute the integer sqrt of the total sum and check root*root == sum as the formula condition.",
    },
    {
        "student_id": "student_formula",
        "student_profile": "prefiere razonamiento algebraico",
        "global_problem_id": "codeforces_192_A",
        "query_step": 3,
        "created_second": 250,
        "idea_text": "For each triangular t, I test n-t with the discriminant 1+8*m being a perfect square.",
    },
]


def stringify_for_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].apply(
            lambda v: stable_json_dumps(v) if isinstance(v, (list, dict, tuple)) else v
        )
    return out


def prepare_for_storage(df: pd.DataFrame) -> pd.DataFrame:
    out = stringify_for_dataset(df)
    for col in out.columns:
        dtype_name = str(out[col].dtype)
        if dtype_name not in {"object", "string"} and not dtype_name.startswith("str"):
            continue
        out[col] = out[col].apply(lambda v: None if pd.isna(v) else str(v))
    return out


def save_dataset(df: pd.DataFrame, stem: str) -> None:
    csv_path = PROCESSED / f"{stem}.csv"
    json_path = PROCESSED / f"{stem}.json"
    parquet_path = PROCESSED / f"{stem}.parquet"
    storage_df = prepare_for_storage(df)
    storage_df.to_csv(csv_path, index=False)
    storage_df.to_json(json_path, orient="records", force_ascii=False, indent=2)
    try:
        storage_df.to_parquet(parquet_path, index=False)
    except Exception as exc:
        print(f"Skipping parquet for {stem}: {exc}")


def load_main_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    problems = pd.read_csv(PROBLEMS_PATH)
    nodes = pd.read_csv(NODES_PATH)
    return problems, nodes


def parse_maybe_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return value
    if pd.isna(value) or value == "":
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    return [str(value)]


def add_demo_problems() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    problems, _ = load_main_dataset()
    demo_df = pd.DataFrame(DEMO_PROBLEMS)

    # Keep the public dataset schema clean; accepted_strategies is saved only in
    # the demo report and raw_metadata, not in the main problems table.
    demo_for_main = demo_df.drop(columns=["accepted_strategies"])
    demo_for_main = demo_for_main[problems.columns]

    ids = set(demo_for_main["global_problem_id"])
    problems_without_demo = problems[~problems["global_problem_id"].isin(ids)].copy()
    updated_problems = pd.concat([problems_without_demo, stringify_for_dataset(demo_for_main)], ignore_index=True)

    demo_nodes = build_page_nodes_dataset(demo_for_main)
    nodes = pd.read_csv(NODES_PATH)
    nodes_without_demo = nodes[~nodes["global_problem_id"].isin(ids)].copy()
    updated_nodes = pd.concat([nodes_without_demo, stringify_for_dataset(demo_nodes)], ignore_index=True)

    save_dataset(updated_problems, "cp_problems_dataset")
    save_dataset(updated_nodes, "cp_page_nodes_dataset")
    save_dataset(demo_for_main, "math_binary_demo_problems")
    save_dataset(demo_nodes, "math_binary_demo_page_nodes")
    return updated_problems, updated_nodes, demo_df


def build_classification_report(demo_df: pd.DataFrame) -> pd.DataFrame:
    problem_lookup = demo_df.set_index("global_problem_id").to_dict("index")
    rows = []
    for item in SIMULATED_IDEAS:
        current = current_agent_heuristic(item["idea_text"])
        fine = fine_grained_strategy(item["idea_text"])
        problem = problem_lookup[item["global_problem_id"]]
        accepted = problem["accepted_strategies"]
        strategy_supported = fine["fine_strategy"] in accepted
        agent_gap = (
            "current_heuristic_has_no_binary_search_class"
            if fine["fine_strategy"] == "BINARY_SEARCH" and current["current_approach"] == "UNKNOWN"
            else "current_heuristic_collapses_formula_to_math"
            if fine["fine_strategy"] == "MATH_FORMULA" and current["current_approach"] == "MATH"
            else "current_heuristic_partial"
        )
        rows.append(
            {
                **item,
                "problem_title": problem["title"],
                "rating": problem["rating"],
                "accepted_strategies": accepted,
                **current,
                **fine,
                "strategy_supported_by_problem": strategy_supported,
                "agent_gap": agent_gap,
            }
        )

    report = pd.DataFrame(rows)
    save_dataset(report, "math_binary_classification_report")
    (PROCESSED / "math_binary_student_queries.csv").write_text(
        report[
            [
                "student_id",
                "global_problem_id",
                "problem_title",
                "query_step",
                "created_second",
                "idea_text",
            ]
        ].to_csv(index=False),
        encoding="utf-8",
    )
    return report


def plot_report(report: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    pd.crosstab(report["student_id"], report["current_approach"]).plot(
        kind="bar", ax=axes[0], color=["#64748b", "#2563eb", "#f59e0b"]
    )
    axes[0].set_title("Clasificacion actual del agente")
    axes[0].set_xlabel("student")
    axes[0].set_ylabel("ideas")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].grid(axis="y", alpha=0.25)

    pd.crosstab(report["student_id"], report["fine_strategy"]).plot(
        kind="bar", ax=axes[1], color=["#14b8a6", "#8b5cf6", "#f59e0b"]
    )
    axes[1].set_title("Clasificacion fina propuesta")
    axes[1].set_xlabel("student")
    axes[1].set_ylabel("ideas")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(ASSETS / "math_binary_strategy_classification.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    report.groupby(["problem_title", "fine_strategy"]).size().unstack(fill_value=0).plot(
        kind="barh", ax=ax, color=["#14b8a6", "#8b5cf6", "#f59e0b"]
    )
    ax.set_title("Estrategias detectadas por problema demo")
    ax.set_xlabel("ideas simuladas")
    ax.set_ylabel("problem")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ASSETS / "math_binary_problem_strategy_map.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_summary(updated_problems: pd.DataFrame, updated_nodes: pd.DataFrame, report: pd.DataFrame) -> None:
    summary = {
        "demo": "math_vs_binary_search",
        "added_problem_ids": [p["global_problem_id"] for p in DEMO_PROBLEMS],
        "total_problems_after_update": int(len(updated_problems)),
        "total_page_nodes_after_update": int(len(updated_nodes)),
        "simulated_students": sorted(report["student_id"].unique().tolist()),
        "simulated_ideas": int(len(report)),
        "current_agent_observation": (
            "The current heuristic separates broad MATH from UNKNOWN/IMPLEMENTATION, "
            "but it does not explicitly model BINARY_SEARCH vs MATH_FORMULA."
        ),
        "recommended_next_step": (
            "Add an LLM or structured evaluator that outputs strategy_family, math_concepts, "
            "proof_gap, complexity_risk, and confidence."
        ),
        "outputs": {
            "problems": str(PROCESSED / "math_binary_demo_problems.csv"),
            "nodes": str(PROCESSED / "math_binary_demo_page_nodes.csv"),
            "queries": str(PROCESSED / "math_binary_student_queries.csv"),
            "classification": str(PROCESSED / "math_binary_classification_report.csv"),
            "chart_classification": str(ASSETS / "math_binary_strategy_classification.png"),
            "chart_problem_map": str(ASSETS / "math_binary_problem_strategy_map.png"),
        },
    }
    (PROCESSED / "math_binary_demo_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    updated_problems, updated_nodes, demo_df = add_demo_problems()
    report = build_classification_report(demo_df)
    plot_report(report)
    write_summary(updated_problems, updated_nodes, report)

    print(json.dumps({
        "added": [p["global_problem_id"] for p in DEMO_PROBLEMS],
        "total_problems": int(len(updated_problems)),
        "total_page_nodes": int(len(updated_nodes)),
        "classification_rows": int(len(report)),
        "current_agent_counts": report["current_approach"].value_counts().to_dict(),
        "fine_strategy_counts": report["fine_strategy"].value_counts().to_dict(),
    }, ensure_ascii=False, indent=2))
    print(report[[
        "student_id",
        "problem_title",
        "current_approach",
        "current_reasoning_stage",
        "fine_strategy",
        "math_concepts",
        "agent_gap",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
