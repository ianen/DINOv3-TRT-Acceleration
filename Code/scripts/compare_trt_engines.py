#!/usr/bin/env python
"""Compare two TensorRT engines on the same deterministic input."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dinov3_trt.infer.compare import compare_output_tensors, make_input  # noqa: E402
from dinov3_trt.infer.trt_runtime import TensorRTEngineRunConfig, run_engine  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-engine", required=True, type=Path)
    parser.add_argument("--candidate-engine", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--input-mode",
        choices=("random-normal", "uniform-0-1", "zeros", "ones"),
        default="random-normal",
    )
    parser.add_argument("--input-name", default="pixel_values")
    args = parser.parse_args()

    input_tensor = make_input(
        batch_size=args.batch_size,
        image_size=args.image_size,
        seed=args.seed,
        mode=args.input_mode,
    )
    reference = run_engine(
        TensorRTEngineRunConfig(
            engine_path=args.reference_engine,
            input_name=args.input_name,
        ),
        input_tensor,
    )
    candidate = run_engine(
        TensorRTEngineRunConfig(
            engine_path=args.candidate_engine,
            input_name=args.input_name,
        ),
        input_tensor,
    )
    comparisons = compare_output_tensors(reference, candidate)
    report = {
        "reference_engine": str(args.reference_engine),
        "candidate_engine": str(args.candidate_engine),
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "seed": args.seed,
        "input_mode": args.input_mode,
        "outputs": [comparison.to_json() for comparison in comparisons],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
