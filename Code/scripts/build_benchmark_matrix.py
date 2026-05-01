#!/usr/bin/env python
"""Build the formal benchmark matrix from existing speedup report JSON files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dinov3_trt.reports.benchmark_matrix import (  # noqa: E402
    build_benchmark_matrix,
    render_benchmark_matrix_markdown,
    write_benchmark_matrix,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("Artifacts") / "reports",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("Artifacts") / "reports" / "formal_benchmark_matrix.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("Artifacts") / "reports" / "formal_benchmark_matrix.csv",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("Artifacts") / "reports" / "formal_benchmark_matrix.md",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write a partial matrix when some expected speedup reports are missing.",
    )
    args = parser.parse_args()

    matrix = build_benchmark_matrix(
        args.reports_dir,
        allow_missing=args.allow_missing,
    )
    write_benchmark_matrix(
        matrix,
        json_output=args.output_json,
        csv_output=args.output_csv,
        markdown_output=args.output_md,
    )
    print(render_benchmark_matrix_markdown(matrix))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
