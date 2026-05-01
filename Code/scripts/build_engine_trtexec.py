#!/usr/bin/env python
"""Build a TensorRT engine using the existing `trtexec` installation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dinov3_trt.engine.trtexec import (  # noqa: E402
    ShapeProfile,
    TrtExecConfig,
    build_trtexec_command,
    quote_for_display,
)
from dinov3_trt.contracts import make_dinov3_vitl16_contract  # noqa: E402


def parse_block_indices(value: str) -> tuple[int, ...]:
    """Parse comma-separated transformer block indices and inclusive ranges."""

    indices: set[int] = set()
    try:
        parts = [part.strip() for part in value.split(",") if part.strip()]
        for part in parts:
            if "-" in part:
                start_text, end_text = part.split("-", maxsplit=1)
                start = int(start_text)
                end = int(end_text)
                if start > end:
                    raise ValueError
                indices.update(range(start, end + 1))
            else:
                indices.add(int(part))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "block list must contain integers or inclusive ranges, e.g. 0-3,8,12"
        ) from exc
    if not indices:
        raise argparse.ArgumentTypeError("at least one block index is required")
    if min(indices) < 0 or max(indices) > 19:
        raise argparse.ArgumentTypeError("DINOv3 ViT-L/16 block indices must be in 0..19")
    return tuple(sorted(indices))


def make_fp32_block_specs(indices: tuple[int, ...]) -> tuple[str, ...]:
    return tuple(f"/model/layer.{index}/*:fp32" for index in indices)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16", "int8", "fp8"),
        required=True,
    )
    parser.add_argument("--input-name", default="pixel_values")
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=8)
    parser.add_argument("--max-batch", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workspace-gb", type=int, default=4)
    parser.add_argument("--timing-cache", type=Path, default=None)
    parser.add_argument("--precision-constraints", choices=("prefer", "obey"), default=None)
    parser.add_argument(
        "--layer-precision",
        action="append",
        default=[],
        help="TensorRT layer precision spec, e.g. /model/layer.0/*:fp32",
    )
    parser.add_argument(
        "--layer-output-type",
        action="append",
        default=[],
        help="TensorRT layer output type spec, e.g. /model/layer.0/*:fp32",
    )
    parser.add_argument(
        "--fp32-transformer-blocks",
        type=parse_block_indices,
        default=(),
        help="Convenience range for full transformer blocks to keep in FP32, e.g. 0-19",
    )
    parser.add_argument("--run-inference", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    fp32_block_specs = make_fp32_block_specs(args.fp32_transformer_blocks)
    layer_precisions = tuple(args.layer_precision) + fp32_block_specs
    layer_output_types = tuple(args.layer_output_type) + fp32_block_specs
    profile = ShapeProfile(
        input_name=args.input_name,
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
        height=args.image_size,
        width=args.image_size,
    )
    contract = make_dinov3_vitl16_contract(args.image_size)
    config = TrtExecConfig(
        onnx_path=args.onnx,
        engine_path=args.engine,
        precision=args.precision,
        profile=profile,
        contract=contract,
        workspace_gb=args.workspace_gb,
        timing_cache_path=args.timing_cache,
        skip_inference=not args.run_inference,
        precision_constraints=args.precision_constraints,
        layer_precisions=layer_precisions,
        layer_output_types=layer_output_types,
    )
    command = build_trtexec_command(config)
    payload = {
        "command": command,
        "display": quote_for_display(command),
        "precision": args.precision,
        "engine": str(args.engine),
        "precision_constraints": args.precision_constraints,
        "layer_precisions": list(layer_precisions),
        "layer_output_types": list(layer_output_types),
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return 0

    args.engine.parent.mkdir(parents=True, exist_ok=True)
    if args.timing_cache is not None:
        args.timing_cache.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command, check=False)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
