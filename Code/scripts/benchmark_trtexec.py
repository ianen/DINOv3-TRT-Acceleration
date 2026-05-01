#!/usr/bin/env python
"""Run TensorRT `trtexec --loadEngine` benchmarks and write a JSON report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dinov3_trt.benchmarks.trtexec import run_trtexec_benchmarks  # noqa: E402


def parse_batches(value: str) -> tuple[int, ...]:
    try:
        batches = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("batches must be a comma-separated integer list") from exc
    if not batches:
        raise argparse.ArgumentTypeError("at least one batch size is required")
    if any(batch < 1 for batch in batches):
        raise argparse.ArgumentTypeError("batch sizes must be >= 1")
    return batches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batches", type=parse_batches, default=(1, 8, 32))
    parser.add_argument("--duration", type=int, default=10)
    parser.add_argument("--warmup-ms", type=int, default=200)
    parser.add_argument("--input-name", default="pixel_values")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--trtexec", default="trtexec")
    parser.add_argument("--use-spin-wait", action="store_true")
    args = parser.parse_args()

    report = run_trtexec_benchmarks(
        engine_path=args.engine,
        batch_sizes=args.batches,
        duration_seconds=args.duration,
        warmup_ms=args.warmup_ms,
        trtexec=args.trtexec,
        input_name=args.input_name,
        image_size=args.image_size,
        use_spin_wait=args.use_spin_wait,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if all(result["returncode"] == 0 for result in report["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
