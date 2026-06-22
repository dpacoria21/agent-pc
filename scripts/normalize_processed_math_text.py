"""Normalize math text in already processed dataset artifacts."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from text_formatting import normalize_math_text  # noqa: E402


PROCESSED = ROOT / "data" / "processed"

DATASET_TEXT_COLUMNS: dict[str, list[str]] = {
    "cp_problems_dataset": [
        "title",
        "statement",
        "input_description",
        "output_description",
        "constraints",
        "notes",
        "official_editorial",
    ],
    "cp_page_nodes_dataset": ["node_title", "node_text"],
    "cp_llm_tree_nodes_dataset": ["title", "summary", "evidence_text", "node_text"],
    "cp_pageindex_ready_nodes": ["title", "text"],
    "cp_pageindex_ready_chunks": ["chunk_text"],
}


def read_existing_dataset(stem: str) -> pd.DataFrame | None:
    parquet_path = PROCESSED / f"{stem}.parquet"
    csv_path = PROCESSED / f"{stem}.csv"
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except Exception:
            pass
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def normalize_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            continue
        out[column] = out[column].apply(
            lambda value: value if value is None or (isinstance(value, float) and pd.isna(value)) else normalize_math_text(value)
        )
    return out


def write_existing_formats(stem: str, df: pd.DataFrame) -> list[str]:
    written: list[str] = []
    csv_path = PROCESSED / f"{stem}.csv"
    json_path = PROCESSED / f"{stem}.json"
    parquet_path = PROCESSED / f"{stem}.parquet"
    if csv_path.exists():
        df.to_csv(csv_path, index=False)
        written.append(str(csv_path))
    if json_path.exists():
        df.to_json(json_path, orient="records", force_ascii=False, indent=2, default_handler=str)
        written.append(str(json_path))
    if parquet_path.exists():
        try:
            df.to_parquet(parquet_path, index=False)
            written.append(str(parquet_path))
        except ImportError:
            print(f"{stem}: skipped parquet rewrite because pyarrow/fastparquet is not installed")
    return written


def main() -> None:
    for stem, columns in DATASET_TEXT_COLUMNS.items():
        df = read_existing_dataset(stem)
        if df is None:
            continue
        normalized = normalize_columns(df, columns)
        written = write_existing_formats(stem, normalized)
        print(f"{stem}: normalized {len(normalized)} rows; wrote {len(written)} files")


if __name__ == "__main__":
    main()
