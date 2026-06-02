"""Verify fresh Codeforces editorial scraping problem by problem.

This script is intentionally narrow: it checks whether each selected
Codeforces problem resolves to its own official tutorial/editorial content,
including content hidden behind Codeforces spoiler/toggle blocks.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cp_dataset_scraper as cpds


DEFAULT_PROBLEMS = [
    "2219:A",
    "2219:B1",
    "2219:C",
    "2219:D",
    "2219:E",
    "2220:A",
    "2220:B",
]


def parse_problem_specs(raw: str) -> list[tuple[int, str]]:
    specs: list[tuple[int, str]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid problem spec {item!r}. Use contest:index, e.g. 2219:A.")
        contest_id, problem_index = item.split(":", 1)
        specs.append((int(contest_id.strip()), problem_index.strip().upper()))
    return specs


def build_config(request_delay: float, timeout: float, use_cache: bool) -> dict[str, Any]:
    cfg = copy.deepcopy(cpds.CONFIG)
    cfg["platforms"] = ["codeforces"]
    cfg["codeforces"]["enabled"] = True
    cfg["codeforces"]["download_statements"] = True
    cfg["codeforces"]["download_editorials"] = True
    cfg["codeforces"]["require_rating"] = False
    cfg["codeforces"]["tags"] = []
    cfg["codeforces"]["min_rating"] = None
    cfg["codeforces"]["max_rating"] = None
    cfg["scraping"]["enabled"] = True
    cfg["scraping"]["respect_robots_txt"] = True
    cfg["scraping"]["request_delay_seconds"] = request_delay
    cfg["scraping"]["timeout_seconds"] = timeout
    cfg["scraping"]["max_retries"] = 1
    cfg["scraping"]["use_cache"] = use_cache
    cfg["scraping"]["cache_dir"] = str(ROOT / "data" / "cache")
    cfg["output"]["raw_dir"] = str(ROOT / "data" / "raw")
    cfg["output"]["processed_dir"] = str(ROOT / "data" / "processed")
    cfg["output"]["save_csv"] = True
    cfg["output"]["save_json"] = True
    cfg["output"]["save_parquet"] = True
    return cfg


def select_metadata(metadata: pd.DataFrame, specs: list[tuple[int, str]]) -> pd.DataFrame:
    ordered_keys = [(contest_id, problem_index) for contest_id, problem_index in specs]
    keys = set(ordered_keys)
    selected = metadata[
        metadata.apply(lambda row: (int(row["contestId"]), str(row["index"]).upper()) in keys, axis=1)
    ].copy()
    order = {key: idx for idx, key in enumerate(ordered_keys)}
    selected["_order"] = selected.apply(lambda row: order.get((int(row["contestId"]), str(row["index"]).upper()), 9999), axis=1)
    selected = selected.sort_values("_order").drop(columns=["_order"])
    return selected


def verify_row(row: pd.Series, editorial: dict[str, Any], min_chars: int) -> dict[str, Any]:
    expected_problem_code = f"{row.get('contestId')}{row.get('index')}"
    editorial_text = cpds.clean_text(editorial.get("official_editorial", ""))
    status = editorial.get("editorial_status", "")
    problem_code = editorial.get("editorial_problem_code", "")
    placeholder_removed = "tutorial is loading" not in editorial_text.lower()
    has_solution = len(editorial_text) >= min_chars
    problem_code_ok = problem_code == expected_problem_code
    verified = status == "downloaded" and has_solution and problem_code_ok and placeholder_removed
    return {
        "global_problem_id": f"codeforces_{row.get('contestId')}_{row.get('index')}",
        "problem_code": expected_problem_code,
        "title": row.get("name"),
        "rating": row.get("rating"),
        "tags": row.get("tags"),
        "scrape_url": row.get("scrape_url") or row.get("url"),
        "editorial_url": editorial.get("editorial_url", ""),
        "editorial_status": status,
        "editorial_parse_method": editorial.get("editorial_parse_method", ""),
        "editorial_problem_code": problem_code,
        "editorial_text_chars": len(editorial_text),
        "editorial_toggle_count": editorial.get("editorial_toggle_count", 0),
        "problem_code_ok": problem_code_ok,
        "placeholder_removed": placeholder_removed,
        "has_min_text": has_solution,
        "verified": verified,
        "preview": editorial_text[:240].replace("\n", " "),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--problems",
        default=",".join(DEFAULT_PROBLEMS),
        help="Comma-separated list like 2219:A,2219:B1,2220:A.",
    )
    parser.add_argument("--request-delay", type=float, default=1.5)
    parser.add_argument("--timeout", type=float, default=25)
    parser.add_argument("--min-editorial-chars", type=int, default=120)
    parser.add_argument("--no-cache", action="store_true", help="Ignore cache while verifying.")
    args = parser.parse_args()

    specs = parse_problem_specs(args.problems)
    cfg = build_config(args.request_delay, args.timeout, use_cache=not args.no_cache)
    cpds.ensure_dirs(cfg)
    client = cpds.CachedHttpClient(cfg)

    print(f"Checking Codeforces editorials for: {', '.join(f'{c}{i}' for c, i in specs)}")
    metadata = cpds.fetch_codeforces_problemset(cfg, client)
    selected = select_metadata(metadata, specs)
    if selected.empty:
        raise SystemExit("No selected Codeforces problems were found in the public API metadata.")

    requested_keys = {f"{contest_id}{problem_index}" for contest_id, problem_index in specs}
    found_keys = {f"{row.get('contestId')}{row.get('index')}" for _, row in selected.iterrows()}
    missing_keys = sorted(requested_keys - found_keys)
    if missing_keys:
        print(f"Warning: these requested problems were not found in metadata: {missing_keys}")

    problem_rows: list[dict[str, Any]] = []
    verification_rows: list[dict[str, Any]] = []

    for _, row in tqdm(selected.iterrows(), total=len(selected), desc="Editorial verification"):
        scrape_url = row.get("scrape_url") or row.get("url")
        scraped = cpds.scrape_codeforces_problem_statement(scrape_url, client=client, config=cfg)
        editorial = cpds.scrape_codeforces_editorial(
            row.get("contestId"),
            row.get("index"),
            row.get("name"),
            client=client,
            config=cfg,
            problem_url=scrape_url,
        )
        problem_rows.append(cpds.codeforces_row_to_unified(row, scraped, editorial))
        verification_rows.append(verify_row(row, editorial, args.min_editorial_chars))

    problems_dataset = pd.DataFrame(problem_rows)
    page_nodes_dataset = cpds.build_page_nodes_dataset(problems_dataset)
    paths = cpds.save_outputs(problems_dataset, page_nodes_dataset, cfg)

    processed_dir = Path(cfg["output"]["processed_dir"])
    verification_df = pd.DataFrame(verification_rows)
    verification_csv = processed_dir / "codeforces_editorial_verification.csv"
    verification_json = processed_dir / "codeforces_editorial_verification.json"
    verification_df.to_csv(verification_csv, index=False)
    verification_df.to_json(verification_json, orient="records", force_ascii=False, indent=2, default_handler=str)

    summary = {
        "requested_count": len(specs),
        "found_count": int(len(selected)),
        "verified_count": int(verification_df["verified"].sum()),
        "downloaded_count": int((verification_df["editorial_status"] == "downloaded").sum()),
        "failed_count": int((~verification_df["verified"]).sum()),
        "problems_dataset": paths,
        "verification_csv": str(verification_csv),
        "verification_json": str(verification_json),
    }
    summary_path = processed_dir / "codeforces_editorial_verification_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("\nVerification summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print("\nProblem-by-problem status")
    cols = [
        "problem_code",
        "title",
        "editorial_status",
        "editorial_parse_method",
        "editorial_text_chars",
        "editorial_toggle_count",
        "problem_code_ok",
        "verified",
    ]
    print(verification_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
