"""Dataset builder for competitive-programming RAG prototypes.

The module focuses on metadata collection, optional respectful scraping,
normalization, Page Indexing, persistence, and quality reporting.

It intentionally does not create embeddings and does not call an LLM.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin, urlparse
from urllib import robotparser

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm.auto import tqdm


CODEFORCES_API_URL = "https://codeforces.com/api/problemset.problems"
ATCODER_PROBLEMS_URL = "https://kenkoooo.com/atcoder/resources/problems.json"
ATCODER_MODELS_URL = "https://kenkoooo.com/atcoder/resources/problem-models.json"
ATCODER_MERGED_PROBLEMS_URL = "https://kenkoooo.com/atcoder/resources/merged-problems.json"


CONFIG: Dict[str, Any] = {
    "platforms": ["codeforces", "atcoder"],
    "codeforces": {
        "enabled": True,
        "max_problems": 20,
        "tags": ["dp", "greedy", "graphs"],
        "exclude_tags": [],
        "min_rating": 800,
        "max_rating": 1800,
        "contest_ids": [],
        "problem_indices": [],
        "problemset_name": None,
        "download_statements": True,
        "download_editorials": True,
        "allow_community_editorials": False,
        "tag_match_mode": "any",
        "require_rating": True,
        "sort_by": "rating",
        "sort_order": "asc",
        "random_seed": 42,
    },
    "atcoder": {
        "enabled": True,
        "max_problems": 20,
        "contest_prefixes": ["abc", "arc", "dp"],
        "contest_ids": [],
        "task_ids": [],
        "min_difficulty": 0,
        "max_difficulty": 1800,
        "min_points": None,
        "max_points": None,
        "tags": [],
        "exclude_tags": [],
        "download_statements": True,
        "download_editorials": True,
        "preferred_language": "en",
        "require_difficulty": False,
        "sort_by": "difficulty",
        "sort_order": "asc",
        "random_seed": 42,
        "infer_atcoder_tags_from_text": False,
    },
    "scraping": {
        "enabled": True,
        "respect_robots_txt": True,
        "request_delay_seconds": 1.5,
        "timeout_seconds": 20,
        "max_retries": 2,
        "use_cache": True,
        "cache_dir": "data/cache",
        "user_agent": "CP-RAG-Research-Prototype/0.1 educational dataset builder",
    },
    "output": {
        "raw_dir": "data/raw",
        "processed_dir": "data/processed",
        "save_raw": True,
        "save_csv": True,
        "save_parquet": True,
        "save_json": True,
        "build_page_nodes": True,
    },
}


PRESETS: Dict[str, Dict[str, Any]] = {
    "cf_beginner_dp": {
        "platforms": ["codeforces"],
        "codeforces": {
            "tags": ["dp"],
            "min_rating": 800,
            "max_rating": 1300,
            "max_problems": 100,
        },
    },
    "cf_intermediate_greedy_graphs": {
        "platforms": ["codeforces"],
        "codeforces": {
            "tags": ["greedy", "graphs"],
            "min_rating": 1200,
            "max_rating": 1800,
            "max_problems": 150,
        },
    },
    "atcoder_abc_beginner": {
        "platforms": ["atcoder"],
        "atcoder": {
            "contest_prefixes": ["abc"],
            "min_difficulty": 0,
            "max_difficulty": 1200,
            "max_problems": 150,
        },
    },
    "mixed_dp_training": {
        "platforms": ["codeforces", "atcoder"],
        "codeforces": {
            "tags": ["dp"],
            "min_rating": 800,
            "max_rating": 1800,
            "max_problems": 150,
        },
        "atcoder": {
            "contest_prefixes": ["abc", "dp"],
            "min_difficulty": 0,
            "max_difficulty": 1800,
            "max_problems": 150,
        },
    },
}


TAG_GROUP_MAP = {
    "dp": "dynamic_programming",
    "graphs": "graphs",
    "dfs and similar": "graphs",
    "shortest paths": "graphs",
    "trees": "graphs",
    "flows": "graphs",
    "greedy": "greedy",
    "math": "math",
    "number theory": "math",
    "combinatorics": "math",
    "probabilities": "math",
    "geometry": "math",
    "data structures": "data_structures",
    "dsu": "data_structures",
    "bitmasks": "data_structures",
    "hashing": "data_structures",
    "binary search": "techniques",
    "two pointers": "techniques",
    "sortings": "techniques",
    "divide and conquer": "techniques",
    "implementation": "implementation_constructive",
    "brute force": "implementation_constructive",
    "constructive algorithms": "implementation_constructive",
    "strings": "strings",
    "fft": "advanced",
    "matrices": "math",
    "games": "math",
    "interactive": "interactive",
}


PAGE_NODE_TYPES = [
    "STATEMENT",
    "INPUT",
    "OUTPUT",
    "CONSTRAINTS",
    "EXAMPLES",
    "NOTES",
    "EDITORIAL_FULL",
    "EDITORIAL_OBSERVATION",
    "EDITORIAL_PROOF",
    "EDITORIAL_ALGORITHM",
    "EDITORIAL_COMPLEXITY",
    "IMPLEMENTATION_HINTS",
    "COMMON_MISTAKES",
]


def deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update a dict without mutating the original."""
    result = copy.deepcopy(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def apply_preset(preset_name: str, base_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge a named preset into CONFIG-like settings.

    Missing keys are preserved from base_config, so partial presets are safe.
    """
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {sorted(PRESETS)}")
    return deep_update(base_config, PRESETS[preset_name])


def normalize_list(value: Any) -> List[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        if not value.strip():
            return []
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                value = parsed
            else:
                value = [value]
        except Exception:
            value = [part.strip() for part in value.split(",")]
    if not isinstance(value, Iterable) or isinstance(value, (dict, bytes)):
        return [str(value).strip().lower()]
    return list(dict.fromkeys(str(item).strip().lower() for item in value if str(item).strip()))


def normalize_tags(tags: Any) -> Tuple[List[str], List[str], List[str]]:
    original_tags = normalize_list(tags)
    normalized_tags = []
    for tag in original_tags:
        normalized_tags.append(TAG_GROUP_MAP.get(tag, re.sub(r"[^a-z0-9]+", "_", tag).strip("_")))
    normalized_tags = list(dict.fromkeys(tag for tag in normalized_tags if tag))
    topic_group = list(dict.fromkeys(normalized_tags))
    return original_tags, normalized_tags, topic_group


def infer_tags_from_text(text: str) -> Tuple[List[str], List[str]]:
    text = (text or "").lower()
    inferred = []
    rules = {
        "dynamic_programming": ["dp", "dynamic programming", "transition", "state"],
        "graphs": ["graph", "tree", "dfs", "bfs", "shortest path", "component"],
        "greedy": ["sort", "greedy", "always choose", "exchange argument"],
        "math": ["mod", "prime", "gcd", "lcm", "combinatorics", "parity"],
        "data_structures": ["segment tree", "fenwick", "binary indexed tree", "dsu"],
        "techniques": ["binary search", "two pointers", "prefix sum"],
    }
    for tag, keywords in rules.items():
        if any(keyword in text for keyword in keywords):
            inferred.append(tag)
    inferred = list(dict.fromkeys(inferred))
    return inferred, inferred


def ensure_dirs(config: Dict[str, Any]) -> None:
    for key in ["cache_dir"]:
        Path(config["scraping"][key]).mkdir(parents=True, exist_ok=True)
    Path(config["output"]["raw_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["output"]["processed_dir"]).mkdir(parents=True, exist_ok=True)


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def clean_text(text: Any) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    text = str(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def first_nonempty(*values: Any) -> Any:
    for value in values:
        if value is not None and not (isinstance(value, float) and np.isnan(value)) and value != "":
            return value
    return None


@dataclass
class CachedHttpClient:
    config: Dict[str, Any]
    session: requests.Session = field(default_factory=requests.Session)
    _last_request_ts: float = 0.0
    _robots_cache: Dict[str, robotparser.RobotFileParser] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.session.headers.update({"User-Agent": self.config["scraping"]["user_agent"]})
        Path(self.config["scraping"]["cache_dir"]).mkdir(parents=True, exist_ok=True)

    def cache_path(self, url: str, suffix: str = ".txt") -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return Path(self.config["scraping"]["cache_dir"]) / f"{digest}{suffix}"

    def allowed_by_robots(self, url: str) -> bool:
        if not self.config["scraping"].get("respect_robots_txt", True):
            return True
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in self._robots_cache:
            rp = robotparser.RobotFileParser()
            rp.set_url(urljoin(base, "/robots.txt"))
            try:
                rp.read()
            except Exception:
                return True
            self._robots_cache[base] = rp
        return self._robots_cache[base].can_fetch(self.config["scraping"]["user_agent"], url)

    def _respect_delay(self) -> None:
        delay = float(self.config["scraping"].get("request_delay_seconds", 1.5))
        elapsed = time.time() - self._last_request_ts
        if elapsed < delay:
            time.sleep(delay - elapsed)

    def get_text(self, url: str, *, use_cache: Optional[bool] = None, is_json: bool = False) -> Tuple[Optional[str], str]:
        if use_cache is None:
            use_cache = bool(self.config["scraping"].get("use_cache", True))
        suffix = ".json" if is_json else ".html"
        path = self.cache_path(url, suffix=suffix)
        if use_cache and path.exists():
            return path.read_text(encoding="utf-8", errors="ignore"), "cache"

        # The configured JSON endpoints are public metadata resources. Robots
        # checks are applied to HTML scraping targets, where accidental load or
        # policy mismatch is more likely to matter.
        if not is_json and not self.allowed_by_robots(url):
            return None, "robots_disallowed"

        max_retries = int(self.config["scraping"].get("max_retries", 2))
        timeout = float(self.config["scraping"].get("timeout_seconds", 20))
        last_status = "unavailable"
        for attempt in range(max_retries + 1):
            try:
                self._respect_delay()
                response = self.session.get(url, timeout=timeout)
                self._last_request_ts = time.time()
                if response.status_code == 404:
                    return None, "missing"
                response.raise_for_status()
                response.encoding = response.encoding or "utf-8"
                text = response.text
                if use_cache:
                    path.write_text(text, encoding="utf-8")
                return text, "downloaded"
            except Exception as exc:
                last_status = f"error:{type(exc).__name__}"
                if attempt < max_retries:
                    time.sleep(max(1.0, float(self.config["scraping"].get("request_delay_seconds", 1.5))))
        return None, last_status

    def get_json(self, url: str) -> Tuple[Optional[Any], str]:
        text, status = self.get_text(url, is_json=True)
        if text is None:
            return None, status
        try:
            return json.loads(text), status
        except Exception:
            return None, "parse_failed"


def save_raw_payload(payload: Any, name: str, config: Dict[str, Any]) -> None:
    if not config["output"].get("save_raw", True):
        return
    raw_path = Path(config["output"]["raw_dir"]) / name
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def fetch_codeforces_problemset(config: Dict[str, Any], client: Optional[CachedHttpClient] = None) -> pd.DataFrame:
    client = client or CachedHttpClient(config)
    cf_cfg = config["codeforces"]
    params = []
    # Codeforces accepts a semicolon-separated tags parameter, but local
    # filtering is needed to preserve our explicit any/all semantics.
    if cf_cfg.get("tags") and cf_cfg.get("tag_match_mode", "any") == "all":
        params.append(("tags", ";".join(cf_cfg["tags"])))
    if cf_cfg.get("problemset_name"):
        params.append(("problemsetName", cf_cfg["problemset_name"]))
    url = CODEFORCES_API_URL
    if params:
        query = "&".join(f"{key}={requests.utils.quote(str(value))}" for key, value in params)
        url = f"{url}?{query}"

    payload, status = client.get_json(url)
    if payload is None or payload.get("status") != "OK":
        print(f"Codeforces metadata unavailable: {status}")
        return pd.DataFrame()
    save_raw_payload(payload, "codeforces_problemset_raw.json", config)

    problems = payload.get("result", {}).get("problems", [])
    stats = payload.get("result", {}).get("problemStatistics", [])
    stats_map = {
        (item.get("contestId"), item.get("index")): item.get("solvedCount")
        for item in stats
    }
    rows = []
    for problem in problems:
        contest_id = problem.get("contestId")
        index = problem.get("index")
        rows.append(
            {
                "contestId": contest_id,
                "index": index,
                "name": problem.get("name"),
                "type": problem.get("type"),
                "points": problem.get("points"),
                "rating": problem.get("rating"),
                "tags": normalize_list(problem.get("tags", [])),
                "solvedCount": stats_map.get((contest_id, index), 0),
                "url": f"https://codeforces.com/problemset/problem/{contest_id}/{index}" if contest_id and index else None,
                "raw_metadata": problem,
            }
        )
    return pd.DataFrame(rows)


def filter_codeforces_problems(cf_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    if cf_df.empty:
        return cf_df.copy()
    cfg = config["codeforces"]
    df = cf_df.copy()
    mask = pd.Series(True, index=df.index)

    if "type" in df.columns:
        mask &= df["type"].fillna("PROGRAMMING").eq("PROGRAMMING")
    if cfg.get("require_rating", True):
        mask &= df["rating"].notna()
    if cfg.get("min_rating") is not None:
        mask &= pd.to_numeric(df["rating"], errors="coerce") >= cfg["min_rating"]
    if cfg.get("max_rating") is not None:
        mask &= pd.to_numeric(df["rating"], errors="coerce") <= cfg["max_rating"]
    if cfg.get("contest_ids"):
        contest_ids = set(str(item) for item in cfg["contest_ids"])
        mask &= df["contestId"].astype(str).isin(contest_ids)
    if cfg.get("problem_indices"):
        indices = set(str(item).upper() for item in cfg["problem_indices"])
        mask &= df["index"].astype(str).str.upper().isin(indices)

    include_tags = set(normalize_list(cfg.get("tags", [])))
    if include_tags:
        if cfg.get("tag_match_mode", "any") == "all":
            mask &= df["tags"].apply(lambda tags: include_tags.issubset(set(normalize_list(tags))))
        else:
            mask &= df["tags"].apply(lambda tags: bool(include_tags.intersection(set(normalize_list(tags)))))

    exclude_tags = set(normalize_list(cfg.get("exclude_tags", [])))
    if exclude_tags:
        mask &= ~df["tags"].apply(lambda tags: bool(exclude_tags.intersection(set(normalize_list(tags)))))

    df = df.loc[mask].copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["solvedCount"] = pd.to_numeric(df["solvedCount"], errors="coerce").fillna(0).astype(int)

    sort_by = cfg.get("sort_by", "rating")
    ascending = cfg.get("sort_order", "asc") == "asc"
    if sort_by == "random":
        df = df.sample(frac=1, random_state=int(cfg.get("random_seed", 42)))
    elif sort_by == "solved_count":
        df = df.sort_values("solvedCount", ascending=ascending, na_position="last")
    elif sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending, na_position="last")

    max_problems = cfg.get("max_problems")
    if max_problems:
        df = df.head(int(max_problems))
    return df.reset_index(drop=True)


def fetch_atcoder_metadata(config: Dict[str, Any], client: Optional[CachedHttpClient] = None) -> pd.DataFrame:
    client = client or CachedHttpClient(config)
    problems, problems_status = client.get_json(ATCODER_PROBLEMS_URL)
    models, models_status = client.get_json(ATCODER_MODELS_URL)
    merged, merged_status = client.get_json(ATCODER_MERGED_PROBLEMS_URL)

    if problems is None:
        print(f"AtCoder problems metadata unavailable: {problems_status}")
        return pd.DataFrame()
    if models is None:
        models = {}
    if merged is None:
        merged = []

    save_raw_payload(problems, "atcoder_problems_raw.json", config)
    save_raw_payload(models, "atcoder_problem_models_raw.json", config)
    save_raw_payload(merged, "atcoder_merged_problems_raw.json", config)

    merged_map = {item.get("id"): item for item in merged if isinstance(item, dict)}
    rows = []
    for problem in problems:
        task_id = problem.get("id")
        model = models.get(task_id, {}) if isinstance(models, dict) else {}
        merged_item = merged_map.get(task_id, {})
        rows.append(
            {
                "id": task_id,
                "contest_id": problem.get("contest_id"),
                "problem_index": problem.get("problem_index"),
                "name": problem.get("name"),
                "title": problem.get("title") or problem.get("name"),
                "difficulty": model.get("difficulty"),
                "points": merged_item.get("point"),
                "solver_count": merged_item.get("solver_count"),
                "source_code_length": merged_item.get("source_code_length"),
                "execution_time": merged_item.get("execution_time"),
                "tags": [],
                "url": f"https://atcoder.jp/contests/{problem.get('contest_id')}/tasks/{task_id}",
                "raw_metadata": {"problem": problem, "model": model, "merged": merged_item},
            }
        )
    return pd.DataFrame(rows)


def contest_prefix(contest_id: Any) -> str:
    contest_id = str(contest_id or "").lower()
    match = re.match(r"([a-z]+)", contest_id)
    return match.group(1) if match else contest_id


def filter_atcoder_problems(atcoder_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    if atcoder_df.empty:
        return atcoder_df.copy()
    cfg = config["atcoder"]
    df = atcoder_df.copy()
    mask = pd.Series(True, index=df.index)

    df["difficulty"] = pd.to_numeric(df["difficulty"], errors="coerce")
    df["points"] = pd.to_numeric(df["points"], errors="coerce")
    df["contest_prefix"] = df["contest_id"].apply(contest_prefix)

    if cfg.get("require_difficulty", False):
        mask &= df["difficulty"].notna()
    if cfg.get("contest_prefixes"):
        prefixes = set(str(item).lower() for item in cfg["contest_prefixes"])
        mask &= df["contest_prefix"].isin(prefixes)
    if cfg.get("contest_ids"):
        contest_ids = set(str(item) for item in cfg["contest_ids"])
        mask &= df["contest_id"].astype(str).isin(contest_ids)
    if cfg.get("task_ids"):
        task_ids = set(str(item) for item in cfg["task_ids"])
        mask &= df["id"].astype(str).isin(task_ids)
    if cfg.get("min_difficulty") is not None:
        mask &= df["difficulty"].fillna(df["points"]).fillna(-10**9) >= cfg["min_difficulty"]
    if cfg.get("max_difficulty") is not None:
        mask &= df["difficulty"].fillna(df["points"]).fillna(10**9) <= cfg["max_difficulty"]
    if cfg.get("min_points") is not None:
        mask &= df["points"].fillna(-10**9) >= cfg["min_points"]
    if cfg.get("max_points") is not None:
        mask &= df["points"].fillna(10**9) <= cfg["max_points"]

    include_tags = set(normalize_list(cfg.get("tags", [])))
    if include_tags and "tags" in df.columns:
        mask &= df["tags"].apply(lambda tags: bool(include_tags.intersection(set(normalize_list(tags)))))
    exclude_tags = set(normalize_list(cfg.get("exclude_tags", [])))
    if exclude_tags and "tags" in df.columns:
        mask &= ~df["tags"].apply(lambda tags: bool(exclude_tags.intersection(set(normalize_list(tags)))))

    df = df.loc[mask].copy()
    sort_by = cfg.get("sort_by", "difficulty")
    ascending = cfg.get("sort_order", "asc") == "asc"
    if sort_by == "random":
        df = df.sample(frac=1, random_state=int(cfg.get("random_seed", 42)))
    elif sort_by in ["difficulty", "points", "contest_id"] and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending, na_position="last")
    max_problems = cfg.get("max_problems")
    if max_problems:
        df = df.head(int(max_problems))
    return df.reset_index(drop=True)


def extract_constraints_from_text(text: str) -> str:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    candidates = []
    for line in lines:
        if re.search(r"(\d+\s*(<=|≤|<)\s*[a-zA-Z]|[a-zA-Z]\s*(<=|≤|<)\s*\d+|constraints?)", line, re.I):
            candidates.append(line)
    return clean_text("\n".join(candidates[:12]))


def parse_sample_pairs(input_blocks: List[str], output_blocks: List[str]) -> List[Dict[str, str]]:
    samples = []
    for idx in range(max(len(input_blocks), len(output_blocks))):
        samples.append(
            {
                "sample_index": idx + 1,
                "input": clean_text(input_blocks[idx]) if idx < len(input_blocks) else "",
                "output": clean_text(output_blocks[idx]) if idx < len(output_blocks) else "",
            }
        )
    return samples


def scrape_codeforces_problem_statement(
    problem_url: str,
    client: Optional[CachedHttpClient] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    config = config or CONFIG
    if not config["scraping"].get("enabled", True):
        return {"statement_status": "skipped", "parse_status": "skipped"}
    client = client or CachedHttpClient(config)
    html, fetch_status = client.get_text(problem_url)
    if html is None:
        status = "missing" if fetch_status == "missing" else "unavailable"
        return {"statement_status": status, "parse_status": fetch_status}

    try:
        soup = BeautifulSoup(html, "html.parser")
        container = soup.select_one("div.problem-statement")
        if container is None:
            return {"statement_status": "parse_failed", "parse_status": "parse_failed"}

        time_limit = clean_text(container.select_one(".time-limit").get_text(" ", strip=True)) if container.select_one(".time-limit") else ""
        memory_limit = clean_text(container.select_one(".memory-limit").get_text(" ", strip=True)) if container.select_one(".memory-limit") else ""

        input_node = container.select_one(".input-specification")
        output_node = container.select_one(".output-specification")
        note_node = container.select_one(".note")
        sample_node = container.select_one(".sample-tests")

        input_description = clean_text(input_node.get_text("\n", strip=True)) if input_node else ""
        output_description = clean_text(output_node.get_text("\n", strip=True)) if output_node else ""
        notes = clean_text(note_node.get_text("\n", strip=True)) if note_node else ""

        input_blocks = [pre.get_text("\n", strip=True) for pre in container.select(".sample-tests .input pre")]
        output_blocks = [pre.get_text("\n", strip=True) for pre in container.select(".sample-tests .output pre")]
        samples = parse_sample_pairs(input_blocks, output_blocks)

        # Main statement: text before input/output/sample/note sections.
        stop_classes = {"input-specification", "output-specification", "sample-tests", "note"}
        statement_parts = []
        for child in container.children:
            classes = set(getattr(child, "get", lambda *_: [])("class", []))
            if classes.intersection(stop_classes):
                break
            text = clean_text(child.get_text("\n", strip=True)) if hasattr(child, "get_text") else clean_text(child)
            if text and not text.lower().startswith(("time limit", "memory limit")):
                statement_parts.append(text)
        statement = clean_text("\n\n".join(statement_parts))
        constraints = extract_constraints_from_text(statement + "\n" + input_description)

        return {
            "statement": statement,
            "input_description": input_description,
            "output_description": output_description,
            "constraints": constraints,
            "samples": samples,
            "notes": notes,
            "time_limit": time_limit,
            "memory_limit": memory_limit,
            "statement_status": "downloaded" if statement else "missing",
            "parse_status": "downloaded",
            "language": "en",
        }
    except Exception as exc:
        return {
            "statement_status": "parse_failed",
            "parse_status": f"parse_failed:{type(exc).__name__}",
        }


def find_codeforces_editorial_url(
    contest_id: Any,
    client: CachedHttpClient,
    config: Dict[str, Any],
) -> Optional[str]:
    contest_url = f"https://codeforces.com/contest/{contest_id}"
    html, status = client.get_text(contest_url)
    if html is None:
        return None
    soup = BeautifulSoup(html, "html.parser")
    candidates = []
    for link in soup.select("a[href]"):
        href = link.get("href")
        text = link.get_text(" ", strip=True).lower()
        if "tutorial" in text or "editorial" in text:
            candidates.append(urljoin(contest_url, href))
        elif href and "/blog/entry/" in href and not config["codeforces"].get("allow_community_editorials", False):
            candidates.append(urljoin(contest_url, href))
    return candidates[0] if candidates else None


def scrape_codeforces_editorial(
    contest_id: Any,
    problem_index: Any,
    title: str,
    client: Optional[CachedHttpClient] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    config = config or CONFIG
    if not config["scraping"].get("enabled", True) or not config["codeforces"].get("download_editorials", True):
        return {"official_editorial": "", "editorial_url": "", "editorial_status": "skipped"}
    client = client or CachedHttpClient(config)
    editorial_url = find_codeforces_editorial_url(contest_id, client, config)
    if not editorial_url:
        return {"official_editorial": "", "editorial_url": "", "editorial_status": "unavailable"}
    html, fetch_status = client.get_text(editorial_url)
    if html is None:
        return {"official_editorial": "", "editorial_url": editorial_url, "editorial_status": "missing" if fetch_status == "missing" else "unavailable"}
    try:
        soup = BeautifulSoup(html, "html.parser")
        body = soup.select_one(".ttypography") or soup.select_one(".blog-entry-content") or soup.body
        text = clean_text(body.get_text("\n", strip=True)) if body else ""
        problem_marker = str(problem_index).lower()
        title_marker = (title or "").lower()
        if text and (problem_marker in text.lower() or (title_marker and title_marker[:20] in text.lower())):
            status = "downloaded"
        elif text:
            status = "unmatched"
        else:
            status = "missing"
        return {"official_editorial": text, "editorial_url": editorial_url, "editorial_status": status}
    except Exception as exc:
        return {"official_editorial": "", "editorial_url": editorial_url, "editorial_status": f"parse_failed:{type(exc).__name__}"}


def choose_atcoder_language_container(soup: BeautifulSoup, preferred_language: str = "en") -> Optional[Any]:
    statement = soup.select_one("#task-statement")
    if statement is None:
        return None
    preferred_class = ".lang-en" if preferred_language == "en" else ".lang-ja"
    preferred = statement.select_one(preferred_class)
    if preferred:
        return preferred
    fallback = statement.select_one(".lang-en") or statement.select_one(".lang-ja")
    return fallback or statement


def extract_atcoder_sections(container: Any) -> Dict[str, Any]:
    h3_nodes = container.find_all("h3")
    sections: Dict[str, List[str]] = {}
    samples_in: Dict[int, str] = {}
    samples_out: Dict[int, str] = {}

    statement_intro = []
    for child in container.children:
        if getattr(child, "name", None) == "h3":
            break
        text = clean_text(child.get_text("\n", strip=True)) if hasattr(child, "get_text") else clean_text(child)
        if text:
            statement_intro.append(text)

    for h3 in h3_nodes:
        heading = clean_text(h3.get_text(" ", strip=True))
        body_parts = []
        for sibling in h3.next_siblings:
            if getattr(sibling, "name", None) == "h3":
                break
            text = clean_text(sibling.get_text("\n", strip=True)) if hasattr(sibling, "get_text") else clean_text(sibling)
            if text:
                body_parts.append(text)
        body = clean_text("\n\n".join(body_parts))
        key = heading.lower()
        sample_match = re.search(r"sample input\s*(\d+)", key)
        if sample_match:
            samples_in[int(sample_match.group(1))] = body
            continue
        sample_match = re.search(r"sample output\s*(\d+)", key)
        if sample_match:
            samples_out[int(sample_match.group(1))] = body
            continue
        sections.setdefault(key, []).append(body)

    sample_indices = sorted(set(samples_in).union(samples_out))
    samples = [
        {"sample_index": idx, "input": samples_in.get(idx, ""), "output": samples_out.get(idx, "")}
        for idx in sample_indices
    ]

    def find_section(*names: str) -> str:
        for name in names:
            for key, values in sections.items():
                if name in key:
                    return clean_text("\n\n".join(values))
        return ""

    intro = clean_text("\n\n".join(statement_intro))
    statement = clean_text(intro or find_section("problem statement", "statement"))
    constraints = find_section("constraints")
    input_description = find_section("input")
    output_description = find_section("output")
    notes = find_section("note", "explanation")
    return {
        "statement": statement,
        "constraints": constraints,
        "input_description": input_description,
        "output_description": output_description,
        "samples": samples,
        "notes": notes,
    }


def scrape_atcoder_problem_statement(
    problem_url: str,
    preferred_language: str = "en",
    client: Optional[CachedHttpClient] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    config = config or CONFIG
    if not config["scraping"].get("enabled", True):
        return {"statement_status": "skipped", "parse_status": "skipped"}
    client = client or CachedHttpClient(config)
    html, fetch_status = client.get_text(problem_url)
    if html is None:
        status = "missing" if fetch_status == "missing" else "unavailable"
        return {"statement_status": status, "parse_status": fetch_status}

    try:
        soup = BeautifulSoup(html, "html.parser")
        container = choose_atcoder_language_container(soup, preferred_language=preferred_language)
        if container is None:
            return {"statement_status": "parse_failed", "parse_status": "parse_failed"}
        sections = extract_atcoder_sections(container)
        all_text = clean_text(soup.get_text("\n", strip=True))
        time_match = re.search(r"Time Limit:\s*([^/\n]+)", all_text)
        mem_match = re.search(r"Memory Limit:\s*([^\n]+)", all_text)
        sections.update(
            {
                "time_limit": clean_text(time_match.group(1)) if time_match else "",
                "memory_limit": clean_text(mem_match.group(1)) if mem_match else "",
                "statement_status": "downloaded" if sections.get("statement") else "missing",
                "parse_status": "downloaded",
                "language": preferred_language,
            }
        )
        return sections
    except Exception as exc:
        return {
            "statement_status": "parse_failed",
            "parse_status": f"parse_failed:{type(exc).__name__}",
        }


def scrape_atcoder_editorials(
    contest_id: Any,
    preferred_language: str = "en",
    client: Optional[CachedHttpClient] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    config = config or CONFIG
    client = client or CachedHttpClient(config)
    url = f"https://atcoder.jp/contests/{contest_id}/editorial"
    if not config["scraping"].get("enabled", True):
        return {"contest_id": contest_id, "editorial_index_url": url, "editorial_status": "skipped", "links": []}
    html, fetch_status = client.get_text(url)
    if html is None:
        return {"contest_id": contest_id, "editorial_index_url": url, "editorial_status": fetch_status, "links": []}
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for link in soup.select("a[href]"):
        href = link.get("href", "")
        if f"/contests/{contest_id}/editorial/" in href or re.search(r"/editorial/\d+", href):
            links.append({"text": clean_text(link.get_text(" ", strip=True)), "url": urljoin(url, href)})
    unique_links = []
    seen = set()
    for item in links:
        if item["url"] not in seen:
            unique_links.append(item)
            seen.add(item["url"])
    return {"contest_id": contest_id, "editorial_index_url": url, "editorial_status": "downloaded", "links": unique_links}


def match_atcoder_editorial_for_problem(
    contest_id: Any,
    task_id: str,
    problem_index: str,
    title: str,
    preferred_language: str,
    client: CachedHttpClient,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    if not config["scraping"].get("enabled", True) or not config["atcoder"].get("download_editorials", True):
        return {"official_editorial": "", "editorial_url": "", "editorial_status": "skipped"}

    index = scrape_atcoder_editorials(contest_id, preferred_language, client, config)
    if index.get("editorial_status") != "downloaded":
        return {"official_editorial": "", "editorial_url": index.get("editorial_index_url", ""), "editorial_status": "unavailable"}

    title_tokens = [token for token in re.findall(r"[a-zA-Z0-9]+", title or "") if len(token) >= 3]
    problem_index = str(problem_index or "").lower()
    selected = None
    for link in index.get("links", []):
        text = link.get("text", "").lower()
        url = link.get("url", "")
        if task_id and task_id.lower() in url.lower():
            selected = link
            break
        if problem_index and re.search(rf"\b{re.escape(problem_index)}\b", text):
            selected = link
            break
        if title_tokens and any(token.lower() in text for token in title_tokens[:4]):
            selected = link
            break

    if selected is None:
        return {"official_editorial": "", "editorial_url": index.get("editorial_index_url", ""), "editorial_status": "unmatched"}

    html, fetch_status = client.get_text(selected["url"])
    if html is None:
        return {"official_editorial": "", "editorial_url": selected["url"], "editorial_status": "missing" if fetch_status == "missing" else "unavailable"}
    try:
        soup = BeautifulSoup(html, "html.parser")
        container = soup.select_one(".lang-en" if preferred_language == "en" else ".lang-ja") or soup.select_one("#main-container") or soup.body
        text = clean_text(container.get_text("\n", strip=True)) if container else ""
        return {"official_editorial": text, "editorial_url": selected["url"], "editorial_status": "downloaded" if text else "missing"}
    except Exception as exc:
        return {"official_editorial": "", "editorial_url": selected["url"], "editorial_status": f"parse_failed:{type(exc).__name__}"}


def codeforces_row_to_unified(row: pd.Series, scraped: Dict[str, Any], editorial: Dict[str, Any]) -> Dict[str, Any]:
    original_tags, normalized_tags, topic_group = normalize_tags(row.get("tags", []))
    contest_id = row.get("contestId")
    problem_index = row.get("index")
    platform_problem_id = f"{contest_id}{problem_index}"
    rating = row.get("rating")
    return {
        "global_problem_id": f"codeforces_{contest_id}_{problem_index}",
        "source": "codeforces",
        "platform_problem_id": platform_problem_id,
        "contest_id": contest_id,
        "problem_index": problem_index,
        "task_id": None,
        "title": row.get("name"),
        "url": row.get("url"),
        "rating": rating,
        "difficulty": None,
        "points": row.get("points"),
        "normalized_difficulty": rating,
        "difficulty_source": "codeforces_rating" if pd.notna(rating) else "missing",
        "original_tags": original_tags,
        "normalized_tags": normalized_tags,
        "topic_group": topic_group,
        "solved_count": row.get("solvedCount"),
        "time_limit": scraped.get("time_limit", ""),
        "memory_limit": scraped.get("memory_limit", ""),
        "statement": scraped.get("statement", ""),
        "input_description": scraped.get("input_description", ""),
        "output_description": scraped.get("output_description", ""),
        "constraints": scraped.get("constraints", ""),
        "samples": scraped.get("samples", []),
        "notes": scraped.get("notes", ""),
        "official_editorial": editorial.get("official_editorial", ""),
        "editorial_url": editorial.get("editorial_url", ""),
        "statement_status": scraped.get("statement_status", "skipped"),
        "editorial_status": editorial.get("editorial_status", "skipped"),
        "parse_status": scraped.get("parse_status", "skipped"),
        "language": scraped.get("language", "en"),
        "raw_metadata": row.get("raw_metadata", {}),
    }


def atcoder_row_to_unified(row: pd.Series, scraped: Dict[str, Any], editorial: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    original_tags, normalized_tags, topic_group = normalize_tags(row.get("tags", []))
    if config["atcoder"].get("infer_atcoder_tags_from_text", False) and not normalized_tags:
        text_for_tags = " ".join(
            [
                scraped.get("statement", ""),
                scraped.get("constraints", ""),
                editorial.get("official_editorial", ""),
            ]
        )
        normalized_tags, topic_group = infer_tags_from_text(text_for_tags)
        original_tags = []

    difficulty = row.get("difficulty")
    points = row.get("points")
    if pd.notna(difficulty):
        normalized_difficulty = difficulty
        difficulty_source = "atcoder_difficulty"
    elif pd.notna(points):
        normalized_difficulty = points
        difficulty_source = "points_proxy"
    else:
        normalized_difficulty = None
        difficulty_source = "missing"

    return {
        "global_problem_id": f"atcoder_{row.get('id')}",
        "source": "atcoder",
        "platform_problem_id": row.get("id"),
        "contest_id": row.get("contest_id"),
        "problem_index": row.get("problem_index"),
        "task_id": row.get("id"),
        "title": first_nonempty(row.get("title"), row.get("name")),
        "url": row.get("url"),
        "rating": None,
        "difficulty": difficulty if pd.notna(difficulty) else None,
        "points": points if pd.notna(points) else None,
        "normalized_difficulty": normalized_difficulty,
        "difficulty_source": difficulty_source,
        "original_tags": original_tags,
        "normalized_tags": normalized_tags,
        "topic_group": topic_group,
        "solved_count": row.get("solver_count"),
        "time_limit": scraped.get("time_limit", ""),
        "memory_limit": scraped.get("memory_limit", ""),
        "statement": scraped.get("statement", ""),
        "input_description": scraped.get("input_description", ""),
        "output_description": scraped.get("output_description", ""),
        "constraints": scraped.get("constraints", ""),
        "samples": scraped.get("samples", []),
        "notes": scraped.get("notes", ""),
        "official_editorial": editorial.get("official_editorial", ""),
        "editorial_url": editorial.get("editorial_url", ""),
        "statement_status": scraped.get("statement_status", "skipped"),
        "editorial_status": editorial.get("editorial_status", "skipped"),
        "parse_status": scraped.get("parse_status", "skipped"),
        "language": scraped.get("language", config["atcoder"].get("preferred_language", "en")),
        "raw_metadata": row.get("raw_metadata", {}),
    }


def build_codeforces_dataset(config: Dict[str, Any], client: CachedHttpClient) -> pd.DataFrame:
    metadata = fetch_codeforces_problemset(config, client)
    filtered = filter_codeforces_problems(metadata, config)
    if filtered.empty:
        return pd.DataFrame()

    rows = []
    for _, row in tqdm(filtered.iterrows(), total=len(filtered), desc="Codeforces problems"):
        if config["codeforces"].get("download_statements", True):
            scraped = scrape_codeforces_problem_statement(row["url"], client=client, config=config)
        else:
            scraped = {"statement_status": "skipped", "parse_status": "skipped"}
        if config["codeforces"].get("download_editorials", True):
            editorial = scrape_codeforces_editorial(row.get("contestId"), row.get("index"), row.get("name"), client=client, config=config)
        else:
            editorial = {"official_editorial": "", "editorial_url": "", "editorial_status": "skipped"}
        rows.append(codeforces_row_to_unified(row, scraped, editorial))
    return pd.DataFrame(rows)


def build_atcoder_dataset(config: Dict[str, Any], client: CachedHttpClient) -> pd.DataFrame:
    metadata = fetch_atcoder_metadata(config, client)
    filtered = filter_atcoder_problems(metadata, config)
    if filtered.empty:
        return pd.DataFrame()

    rows = []
    preferred_language = config["atcoder"].get("preferred_language", "en")
    for _, row in tqdm(filtered.iterrows(), total=len(filtered), desc="AtCoder problems"):
        if config["atcoder"].get("download_statements", True):
            scraped = scrape_atcoder_problem_statement(row["url"], preferred_language=preferred_language, client=client, config=config)
        else:
            scraped = {"statement_status": "skipped", "parse_status": "skipped", "language": preferred_language}
        if config["atcoder"].get("download_editorials", True):
            editorial = match_atcoder_editorial_for_problem(
                row.get("contest_id"),
                row.get("id"),
                row.get("problem_index"),
                first_nonempty(row.get("title"), row.get("name")) or "",
                preferred_language,
                client,
                config,
            )
        else:
            editorial = {"official_editorial": "", "editorial_url": "", "editorial_status": "skipped"}
        rows.append(atcoder_row_to_unified(row, scraped, editorial, config))
    return pd.DataFrame(rows)


def subsection_by_keywords(text: str, keywords: List[str]) -> str:
    text = clean_text(text)
    if not text:
        return ""
    blocks = re.split(r"\n{2,}|(?<=\.)\s+(?=[A-Z])", text)
    selected = [block for block in blocks if any(keyword in block.lower() for keyword in keywords)]
    return clean_text("\n\n".join(selected[:8]))


def build_page_nodes_dataset(problems_dataset: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in problems_dataset.iterrows():
        parent_id = f"{row['global_problem_id']}::ROOT"
        samples_text = stable_json_dumps(row.get("samples", [])) if row.get("samples") not in [None, ""] else ""
        editorial = clean_text(row.get("official_editorial", ""))
        node_texts = {
            "STATEMENT": row.get("statement", ""),
            "INPUT": row.get("input_description", ""),
            "OUTPUT": row.get("output_description", ""),
            "CONSTRAINTS": row.get("constraints", ""),
            "EXAMPLES": samples_text,
            "NOTES": row.get("notes", ""),
            "EDITORIAL_FULL": editorial,
            "EDITORIAL_OBSERVATION": subsection_by_keywords(editorial, ["observation", "notice", "intuition", "key idea", "insight"]),
            "EDITORIAL_PROOF": subsection_by_keywords(editorial, ["proof", "prove", "invariant", "exchange", "correctness"]),
            "EDITORIAL_ALGORITHM": subsection_by_keywords(editorial, ["algorithm", "solution", "compute", "transition", "dfs", "bfs", "sort"]),
            "EDITORIAL_COMPLEXITY": subsection_by_keywords(editorial, ["complexity", "time", "memory", "o("]),
            "IMPLEMENTATION_HINTS": subsection_by_keywords(editorial + "\n" + clean_text(row.get("notes", "")), ["implement", "careful", "array", "index", "mod"]),
            "COMMON_MISTAKES": subsection_by_keywords(editorial + "\n" + clean_text(row.get("notes", "")), ["mistake", "wrong", "corner", "edge", "overflow", "off-by-one"]),
        }
        for order, node_type in enumerate(PAGE_NODE_TYPES, start=1):
            node_text = clean_text(node_texts.get(node_type, ""))
            metadata = {
                "is_empty": not bool(node_text),
                "statement_status": row.get("statement_status"),
                "editorial_status": row.get("editorial_status"),
                "parse_status": row.get("parse_status"),
                "time_limit": row.get("time_limit"),
                "memory_limit": row.get("memory_limit"),
            }
            rows.append(
                {
                    "node_id": f"{row['global_problem_id']}::{order:02d}_{node_type}",
                    "global_problem_id": row.get("global_problem_id"),
                    "source": row.get("source"),
                    "contest_id": row.get("contest_id"),
                    "platform_problem_id": row.get("platform_problem_id"),
                    "node_type": node_type,
                    "node_title": f"{row.get('title')} - {node_type.replace('_', ' ').title()}",
                    "node_text": node_text,
                    "parent_node_id": parent_id,
                    "order": order,
                    "rating": row.get("rating"),
                    "difficulty": row.get("difficulty"),
                    "points": row.get("points"),
                    "normalized_difficulty": row.get("normalized_difficulty"),
                    "original_tags": row.get("original_tags"),
                    "normalized_tags": row.get("normalized_tags"),
                    "topic_group": row.get("topic_group"),
                    "url": row.get("url"),
                    "editorial_url": row.get("editorial_url"),
                    "language": row.get("language"),
                    "metadata": metadata,
                }
            )
    return pd.DataFrame(rows)


def serializable_df(df: pd.DataFrame, stringify_object_columns: bool = False) -> pd.DataFrame:
    out = df.copy()
    for column in out.columns:
        if out[column].apply(lambda value: isinstance(value, (list, dict, tuple))).any():
            out[column] = out[column].apply(lambda value: stable_json_dumps(value) if isinstance(value, (list, dict, tuple)) else value)
        if stringify_object_columns and out[column].dtype == "object":
            out[column] = out[column].apply(lambda value: "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value))
    return out


def save_outputs(problems_dataset: pd.DataFrame, page_nodes_dataset: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, str]:
    processed_dir = Path(config["output"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    datasets = {
        "cp_problems_dataset": problems_dataset,
        "cp_page_nodes_dataset": page_nodes_dataset,
    }
    for name, df in datasets.items():
        serial = serializable_df(df)
        if config["output"].get("save_csv", True):
            path = processed_dir / f"{name}.csv"
            serial.to_csv(path, index=False)
            paths[f"{name}_csv"] = str(path)
        if config["output"].get("save_json", True):
            path = processed_dir / f"{name}.json"
            df.to_json(path, orient="records", force_ascii=False, indent=2, default_handler=str)
            paths[f"{name}_json"] = str(path)
        if config["output"].get("save_parquet", True):
            path = processed_dir / f"{name}.parquet"
            try:
                serializable_df(df, stringify_object_columns=True).to_parquet(path, index=False)
                paths[f"{name}_parquet"] = str(path)
            except Exception as exc:
                print(f"Could not save {path.name}: {type(exc).__name__}. Install pyarrow or fastparquet to enable parquet.")
    return paths


def build_cp_dataset(config: Optional[Dict[str, Any]] = None, preset_name: Optional[str] = None) -> Dict[str, Any]:
    active_config = copy.deepcopy(config or CONFIG)
    if preset_name:
        active_config = apply_preset(preset_name, active_config)
    ensure_dirs(active_config)
    client = CachedHttpClient(active_config)

    frames = []
    platforms = set(active_config.get("platforms", []))
    if "codeforces" in platforms and active_config["codeforces"].get("enabled", True):
        frames.append(build_codeforces_dataset(active_config, client))
    if "atcoder" in platforms and active_config["atcoder"].get("enabled", True):
        frames.append(build_atcoder_dataset(active_config, client))

    frames = [frame for frame in frames if frame is not None and not frame.empty]
    problems_dataset = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not problems_dataset.empty:
        problems_dataset["normalized_difficulty"] = pd.to_numeric(problems_dataset["normalized_difficulty"], errors="coerce")
        problems_dataset["solved_count"] = pd.to_numeric(problems_dataset["solved_count"], errors="coerce")

    if active_config["output"].get("build_page_nodes", True):
        page_nodes_dataset = build_page_nodes_dataset(problems_dataset)
    else:
        page_nodes_dataset = pd.DataFrame()

    paths = save_outputs(problems_dataset, page_nodes_dataset, active_config)
    return {
        "config": active_config,
        "problems_dataset": problems_dataset,
        "page_nodes_dataset": page_nodes_dataset,
        "paths": paths,
    }


def difficulty_bucket(value: Any) -> str:
    if pd.isna(value):
        return "missing"
    value = float(value)
    if value < 800:
        return "0-799"
    if value < 1200:
        return "800-1199"
    if value < 1600:
        return "1200-1599"
    if value < 2000:
        return "1600-1999"
    if value < 2400:
        return "2000-2399"
    return "2400+"


def flatten_counts(series: pd.Series) -> pd.Series:
    counter: Dict[str, int] = {}
    for value in series.dropna():
        for item in normalize_list(value):
            counter[item] = counter.get(item, 0) + 1
    return pd.Series(counter).sort_values(ascending=False)


def _display(obj: Any) -> None:
    try:
        from IPython.display import display

        display(obj)
    except Exception:
        print(obj)


def dataset_quality_report(problems_dataset: pd.DataFrame, page_nodes_dataset: pd.DataFrame) -> Dict[str, Any]:
    if problems_dataset is None or problems_dataset.empty:
        print("No problems in dataset.")
        return {}

    problems = problems_dataset.copy()
    nodes = page_nodes_dataset.copy() if page_nodes_dataset is not None else pd.DataFrame()
    problems["difficulty_bucket"] = problems["normalized_difficulty"].apply(difficulty_bucket)
    problems["contest_prefix"] = problems["contest_id"].apply(contest_prefix)

    report = {
        "total_problems": int(len(problems)),
        "total_by_platform": problems["source"].value_counts().to_dict(),
        "total_by_difficulty_bucket": problems["difficulty_bucket"].value_counts().sort_index().to_dict(),
        "total_by_normalized_tag": flatten_counts(problems["normalized_tags"]).to_dict(),
        "total_by_topic_group": flatten_counts(problems["topic_group"]).to_dict(),
        "statement_download_rate": float((problems["statement_status"] == "downloaded").mean()),
        "editorial_download_rate": float((problems["editorial_status"] == "downloaded").mean()),
        "total_page_nodes": int(len(nodes)),
        "node_type_distribution": nodes["node_type"].value_counts().to_dict() if not nodes.empty else {},
        "problems_without_rating_or_difficulty": int(problems["normalized_difficulty"].isna().sum()),
        "problems_without_tags": int(problems["normalized_tags"].apply(lambda tags: len(normalize_list(tags)) == 0).sum()),
        "problems_with_parse_failed": int(problems["parse_status"].astype(str).str.contains("parse_failed", na=False).sum()),
        "atcoder_by_contest_prefix": problems.loc[problems["source"] == "atcoder", "contest_prefix"].value_counts().to_dict(),
        "codeforces_by_rating_bucket": problems.loc[problems["source"] == "codeforces", "difficulty_bucket"].value_counts().sort_index().to_dict(),
    }

    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))

    _display(pd.DataFrame(problems["source"].value_counts()).rename(columns={"count": "problems"}))
    _display(pd.DataFrame(problems["difficulty_bucket"].value_counts().sort_index()).rename(columns={"count": "problems"}))
    _display(flatten_counts(problems["normalized_tags"]).head(20).rename_axis("normalized_tag").reset_index(name="count"))
    if not nodes.empty:
        _display(nodes["node_type"].value_counts().rename_axis("node_type").reset_index(name="count"))

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        problems["source"].value_counts().plot(kind="bar", ax=axes[0, 0], title="Distribucion de plataformas")
        problems["normalized_difficulty"].dropna().plot(kind="hist", bins=16, ax=axes[0, 1], title="Normalized difficulty")
        flatten_counts(problems["normalized_tags"]).head(12).plot(kind="bar", ax=axes[0, 2], title="Top tags")
        pd.crosstab(problems["source"], problems["editorial_status"]).plot(kind="bar", stacked=True, ax=axes[1, 0], title="Editorial status")
        pd.crosstab(problems["source"], problems["statement_status"]).plot(kind="bar", stacked=True, ax=axes[1, 1], title="Statement status")
        if not nodes.empty:
            nodes["node_type"].value_counts().plot(kind="bar", ax=axes[1, 2], title="Node type distribution")
        else:
            axes[1, 2].set_title("Node type distribution")
        plt.tight_layout()
        plt.show()
    except Exception as exc:
        print(f"Plotting skipped: {type(exc).__name__}")

    return report


def final_build_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    config = result["config"]
    problems = result["problems_dataset"]
    nodes = result["page_nodes_dataset"]
    if problems.empty:
        summary = {
            "active_config": config,
            "platforms_used": config.get("platforms", []),
            "requested_problem_count": 0,
            "obtained_problem_count": 0,
            "output_paths": result.get("paths", {}),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return summary

    requested = 0
    if "codeforces" in config.get("platforms", []):
        requested += int(config["codeforces"].get("max_problems") or 0)
    if "atcoder" in config.get("platforms", []):
        requested += int(config["atcoder"].get("max_problems") or 0)
    included_tags = {
        "codeforces": config["codeforces"].get("tags", []),
        "atcoder": config["atcoder"].get("tags", []),
    }
    excluded_tags = {
        "codeforces": config["codeforces"].get("exclude_tags", []),
        "atcoder": config["atcoder"].get("exclude_tags", []),
    }
    summary = {
        "active_config": config,
        "platforms_used": config.get("platforms", []),
        "requested_problem_count": requested,
        "obtained_problem_count": int(len(problems)),
        "difficulty_range_final": [
            float(problems["normalized_difficulty"].min()) if problems["normalized_difficulty"].notna().any() else None,
            float(problems["normalized_difficulty"].max()) if problems["normalized_difficulty"].notna().any() else None,
        ],
        "tags_included": included_tags,
        "tags_excluded": excluded_tags,
        "total_statements_downloaded": int((problems["statement_status"] == "downloaded").sum()),
        "total_editorials_downloaded": int((problems["editorial_status"] == "downloaded").sum()),
        "total_page_nodes": int(len(nodes)),
        "output_paths": result.get("paths", {}),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return summary


def load_processed_dataset(processed_dir: str = "data/processed") -> Tuple[pd.DataFrame, pd.DataFrame]:
    processed = Path(processed_dir)
    problems_path = processed / "cp_problems_dataset.csv"
    nodes_path = processed / "cp_page_nodes_dataset.csv"
    problems = pd.read_csv(problems_path) if problems_path.exists() else pd.DataFrame()
    nodes = pd.read_csv(nodes_path) if nodes_path.exists() else pd.DataFrame()
    return problems, nodes


if __name__ == "__main__":
    result = build_cp_dataset(CONFIG)
    dataset_quality_report(result["problems_dataset"], result["page_nodes_dataset"])
    final_build_summary(result)
