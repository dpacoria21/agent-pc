"""Phase 2 runner: validate dataset contract and quality."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset.quality_report import build_issue_dataframe, build_quality_report
from dataset.schema import build_contract_report, load_processed_datasets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--fail-on-error", action="store_true", help="Exit with status 2 when contract errors exist.")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    problems, page_nodes = load_processed_datasets(processed_dir)
    contract_report = build_contract_report(problems, page_nodes)
    quality_report = build_quality_report(problems, page_nodes, contract_report)
    issues = build_issue_dataframe(contract_report)

    contract_path = processed_dir / "dataset_contract_report.json"
    quality_path = processed_dir / "dataset_quality_report.json"
    issues_path = processed_dir / "dataset_contract_issues.csv"

    contract_path.write_text(json.dumps(contract_report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    quality_path.write_text(json.dumps(quality_report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    issues.to_csv(issues_path, index=False)

    print("Phase 2 dataset contract")
    print(json.dumps(
        {
            "contract_status": contract_report["status"],
            "problem_count": contract_report["problem_count"],
            "page_node_count": contract_report["page_node_count"],
            "severity_counts": contract_report["severity_counts"],
            "contract_report": str(contract_path),
            "quality_report": str(quality_path),
            "issues_csv": str(issues_path),
        },
        ensure_ascii=False,
        indent=2,
        default=str,
    ))

    if args.fail_on_error and contract_report["status"] == "failed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

