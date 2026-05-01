#!/usr/bin/env python
"""Compare two ONNX models on the same deterministic input."""

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
from dinov3_trt.infer.onnx_runtime import (  # noqa: E402
    OnnxModelRunConfig,
    parse_ort_providers,
    run_onnx_model,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-onnx", required=True, type=Path)
    parser.add_argument("--candidate-onnx", required=True, type=Path)
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
    parser.add_argument(
        "--providers",
        default="CUDAExecutionProvider,CPUExecutionProvider",
        help="Comma-separated ONNX Runtime execution providers.",
    )
    args = parser.parse_args()

    providers = parse_ort_providers(args.providers)
    input_tensor = make_input(
        batch_size=args.batch_size,
        image_size=args.image_size,
        seed=args.seed,
        mode=args.input_mode,
    )
    reference = run_onnx_model(
        OnnxModelRunConfig(
            onnx_path=args.reference_onnx,
            input_name=args.input_name,
            providers=providers,
        ),
        input_tensor,
    )
    candidate = run_onnx_model(
        OnnxModelRunConfig(
            onnx_path=args.candidate_onnx,
            input_name=args.input_name,
            providers=providers,
        ),
        input_tensor,
    )
    comparisons = compare_output_tensors(reference, candidate)
    report = {
        "reference_onnx": str(args.reference_onnx),
        "candidate_onnx": str(args.candidate_onnx),
        "providers": list(providers),
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
