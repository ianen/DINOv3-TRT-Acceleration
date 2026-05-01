#!/usr/bin/env python
"""Evaluate two TensorRT engines on an image directory and aggregate metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dinov3_trt.infer.compare import compare_output_tensors  # noqa: E402
from dinov3_trt.infer.image_eval import (  # noqa: E402
    OutputMetricAccumulator,
    chunk_paths,
    list_image_paths,
    load_image_batch,
    read_image_manifest,
)
from dinov3_trt.infer.trt_runtime import TensorRTEngineRunConfig, run_engine  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-engine", required=True, type=Path)
    parser.add_argument("--candidate-engine", required=True, type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--input-name", default="pixel_values")
    args = parser.parse_args()

    if (args.image_root is None) == (args.manifest is None):
        parser.error("provide exactly one of --image-root or --manifest")
    if args.manifest is not None:
        image_paths = read_image_manifest(args.manifest)
        if args.max_images is not None:
            image_paths = image_paths[: args.max_images]
        image_root = args.manifest
    else:
        image_paths = list_image_paths(
            args.image_root,
            recursive=not args.no_recursive,
            limit=args.max_images,
        )
        image_root = args.image_root
    accumulators: dict[str, OutputMetricAccumulator] = {}
    batch_reports: list[dict[str, object]] = []
    for batch_index, path_batch in enumerate(chunk_paths(image_paths, args.batch_size)):
        image_batch = load_image_batch(path_batch, image_size=args.image_size)
        reference = run_engine(
            TensorRTEngineRunConfig(
                engine_path=args.reference_engine,
                input_name=args.input_name,
            ),
            image_batch.tensor,
        )
        candidate = run_engine(
            TensorRTEngineRunConfig(
                engine_path=args.candidate_engine,
                input_name=args.input_name,
            ),
            image_batch.tensor,
        )
        comparisons = compare_output_tensors(reference, candidate)
        for comparison in comparisons:
            accumulator = accumulators.setdefault(
                comparison.name,
                OutputMetricAccumulator(name=comparison.name),
            )
            accumulator.update(comparison)
        batch_reports.append(
            {
                "batch_index": batch_index,
                "images": [str(path) for path in path_batch],
                "outputs": [comparison.to_json() for comparison in comparisons],
            }
        )

    report = {
        "reference_engine": str(args.reference_engine),
        "candidate_engine": str(args.candidate_engine),
        "image_root": str(image_root),
        "image_count": len(image_paths),
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "images": [str(path) for path in image_paths],
        "outputs": [accumulators[name].to_json() for name in accumulators],
        "batches": batch_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
