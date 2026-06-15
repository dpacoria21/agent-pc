"""Build a Codeforces dataset from the latest Div. 4 contests.

Default run:

    python scripts/build_latest_div4_dataset.py --target-problems 80

The script uses Codeforces public APIs for metadata and the existing respectful
scraper for statements/editorials.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cp_dataset_scraper as cpds


PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
CONTEST_LIST_URL = "https://codeforces.com/api/contest.list?gym=false"


def is_div4_contest(name: str) -> bool:
    text = str(name or "").lower()
    return bool(re.search(r"\bdiv\.?\s*4\b|\bdivision\s*4\b", text))


def fetch_latest_div4_contests(config: dict[str, Any], limit: int = 20) -> pd.DataFrame:
    client = cpds.CachedHttpClient(config)
    payload, status = client.get_json(CONTEST_LIST_URL)
    if payload is None or payload.get("status") != "OK":
        raise RuntimeError(f"Could not fetch Codeforces contests: {status}")
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "codeforces_contest_list_raw.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    contests = pd.DataFrame(payload.get("result", []))
    if contests.empty:
        raise RuntimeError("Codeforces contest list was empty.")
    contests = contests[
        contests["phase"].eq("FINISHED") & contests["name"].apply(is_div4_contest)
    ].copy()
    contests = contests.sort_values("startTimeSeconds", ascending=False).head(limit).reset_index(drop=True)
    return contests


def choose_contests_for_problem_count(
    contests: pd.DataFrame,
    problemset: pd.DataFrame,
    target_problems: int,
) -> pd.DataFrame:
    rows = []
    total = 0
    for _, contest in contests.iterrows():
        contest_id = contest["id"]
        count = int((problemset["contestId"].astype(str) == str(contest_id)).sum())
        if count <= 0:
            continue
        rows.append(
            {
                "contest_id": contest_id,
                "name": contest["name"],
                "startTimeSeconds": contest.get("startTimeSeconds"),
                "problem_count_in_problemset": count,
            }
        )
        total += count
        if total >= target_problems:
            break
    selected = pd.DataFrame(rows)
    if selected.empty:
        raise RuntimeError("No Div. 4 contests had problems in problemset.problems.")
    return selected


def build_div4_config(
    selected_contest_ids: list[int | str],
    *,
    target_problems: int,
    request_delay: float,
    download_content: bool,
) -> dict[str, Any]:
    config = copy.deepcopy(cpds.CONFIG)
    config["platforms"] = ["codeforces"]
    config["codeforces"].update(
        {
            "enabled": True,
            "max_problems": target_problems,
            "tags": [],
            "exclude_tags": [],
            "min_rating": None,
            "max_rating": None,
            "contest_ids": [int(item) for item in selected_contest_ids],
            "problem_indices": [],
            "download_statements": download_content,
            "download_editorials": download_content,
            "allow_community_editorials": False,
            "tag_match_mode": "any",
            "require_rating": False,
            "sort_by": "contestId",
            "sort_order": "desc",
        }
    )
    config["atcoder"]["enabled"] = False
    config["scraping"].update(
        {
            "enabled": download_content,
            "respect_robots_txt": True,
            "request_delay_seconds": request_delay,
            "timeout_seconds": 20,
            "max_retries": 1,
            "use_cache": True,
            "cache_dir": str(ROOT / "data" / "cache"),
        }
    )
    config["output"].update(
        {
            "raw_dir": str(ROOT / "data" / "raw"),
            "processed_dir": str(PROCESSED),
            "save_raw": True,
            "save_csv": True,
            "save_json": True,
            "save_parquet": True,
            "build_page_nodes": True,
        }
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build latest Codeforces Div. 4 dataset.")
    parser.add_argument("--target-problems", type=int, default=80)
    parser.add_argument("--contest-lookback", type=int, default=24)
    parser.add_argument("--request-delay", type=float, default=1.0)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()

    base_config = copy.deepcopy(cpds.CONFIG)
    base_config["platforms"] = ["codeforces"]
    base_config["atcoder"]["enabled"] = False
    base_config["scraping"]["cache_dir"] = str(ROOT / "data" / "cache")
    base_config["output"]["raw_dir"] = str(ROOT / "data" / "raw")
    base_config["output"]["processed_dir"] = str(PROCESSED)

    contests = fetch_latest_div4_contests(base_config, limit=args.contest_lookback)
    client = cpds.CachedHttpClient(base_config)
    problemset = cpds.fetch_codeforces_problemset(base_config, client)
    selected = choose_contests_for_problem_count(contests, problemset, args.target_problems)
    selected_ids = selected["contest_id"].tolist()
    config = build_div4_config(
        selected_ids,
        target_problems=args.target_problems,
        request_delay=args.request_delay,
        download_content=not args.metadata_only,
    )
    result = cpds.build_cp_dataset(config)
    problems = result["problems_dataset"]
    nodes = result["page_nodes_dataset"]

    selected.to_csv(PROCESSED / "latest_div4_contests.csv", index=False)
    selected.to_json(PROCESSED / "latest_div4_contests.json", orient="records", force_ascii=False, indent=2)
    report = {
        "status": "ok",
        "target_problems": args.target_problems,
        "obtained_problems": int(len(problems)),
        "page_nodes": int(len(nodes)),
        "selected_contests": selected.to_dict("records"),
        "download_content": not args.metadata_only,
        "statement_status": problems["statement_status"].value_counts().to_dict() if not problems.empty else {},
        "editorial_status": problems["editorial_status"].value_counts().to_dict() if not problems.empty else {},
        "paths": result["paths"],
    }
    (PROCESSED / "latest_div4_dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
