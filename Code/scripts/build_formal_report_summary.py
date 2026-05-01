#!/usr/bin/env python
"""Build a consolidated formal benchmark summary from existing report JSON files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dinov3_trt.reports.formal_summary import (  # noqa: E402
    build_formal_summary,
    render_formal_summary_markdown,
    write_formal_summary,
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
        default=Path("Artifacts") / "reports" / "formal_summary.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("Artifacts") / "reports" / "formal_summary.md",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write a partial summary when some expected report files are missing.",
    )
    args = parser.parse_args()

    summary = build_formal_summary(
        args.reports_dir,
        allow_missing=args.allow_missing,
    )
    write_formal_summary(
        summary,
        json_output=args.output_json,
        markdown_output=args.output_md,
    )
    print(render_formal_summary_markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

