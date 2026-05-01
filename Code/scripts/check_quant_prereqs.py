#!/usr/bin/env python
"""Check readiness for the ModelOpt INT8 quantization path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dinov3_trt.quantization.preflight import build_preflight_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calib-manifest",
        type=Path,
        default=Path("Artifacts") / "manifests" / "calib_imagenet500.json",
    )
    parser.add_argument(
        "--eval-manifest",
        type=Path,
        default=Path("Artifacts") / "manifests" / "eval_imagenet1000.json",
    )
    parser.add_argument(
        "--allow-missing-data",
        action="store_true",
        help="Return success when dependencies/CUDA are ready but manifests are missing.",
    )
    args = parser.parse_args()

    report = build_preflight_report(
        calib_manifest=args.calib_manifest,
        eval_manifest=args.eval_manifest,
    )
    print(json.dumps(report.to_json(), indent=2))
    if report.ready:
        return 0
    if args.allow_missing_data and report.dependencies_ready and report.cuda.available:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

