#!/usr/bin/env python
"""Build SVG benchmark figures from the formal benchmark matrix CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dinov3_trt.reports.benchmark_figures import build_benchmark_figures  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix-csv",
        type=Path,
        default=Path("Artifacts") / "reports" / "formal_benchmark_matrix.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Artifacts") / "reports" / "figures",
    )
    args = parser.parse_args()

    manifest = build_benchmark_figures(args.matrix_csv, args.output_dir)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
