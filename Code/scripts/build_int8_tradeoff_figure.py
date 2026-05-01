#!/usr/bin/env python
"""Generate cosine-vs-speedup tradeoff scatter SVGs from real-image eval + speedup reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dinov3_trt.reports.benchmark_figures import (  # noqa: E402
    DEFAULT_TRADEOFF_FIGURE_SPECS,
    build_tradeoff_figures,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=Path("Artifacts") / "reports")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Artifacts") / "reports" / "figures",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip figures whose source reports are not yet available rather than failing.",
    )
    args = parser.parse_args()

    manifest = build_tradeoff_figures(
        args.reports_dir,
        args.output_dir,
        specs=DEFAULT_TRADEOFF_FIGURE_SPECS,
        allow_missing=args.allow_missing,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
