#!/usr/bin/env python
"""Quantize the project ONNX model with NVIDIA ModelOpt ONNX PTQ."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dinov3_trt.artifacts import ArtifactLayout  # noqa: E402
from dinov3_trt.quantization.modelopt_onnx import (  # noqa: E402
    ModelOptOnnxPtqConfig,
    load_calibration_array,
    parse_execution_providers,
    parse_optional_csv,
    run_modelopt_onnx_ptq,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("Artifacts"))
    parser.add_argument("--onnx", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--calib-manifest",
        type=Path,
        default=Path("Artifacts") / "manifests" / "imagenet_calib_500.json",
    )
    parser.add_argument("--input-name", default="pixel_values")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-calibration-images", type=int, default=None)
    parser.add_argument("--load-batch-size", type=int, default=16)
    parser.add_argument(
        "--quantize-mode",
        choices=("int8", "fp8"),
        default="int8",
        help=(
            "Quantization data type. ModelOpt 0.43 maps 'fp8' to E4M3 weights "
            "+ E4M3 activations on Blackwell sm_120 5th-gen Tensor Core."
        ),
    )
    parser.add_argument("--calibration-method", choices=("max", "minmax", "entropy"), default="max")
    parser.add_argument("--calibration-eps", default="cuda:0,cpu")
    parser.add_argument("--high-precision-dtype", choices=("fp16", "fp32", "bf16"), default="fp32")
    parser.add_argument(
        "--mha-accumulation-dtype",
        choices=("fp16", "fp32", "bf16"),
        default="fp32",
    )
    parser.add_argument("--op-types-to-quantize", default=None)
    parser.add_argument("--op-types-to-exclude", default=None)
    parser.add_argument("--nodes-to-quantize", default=None)
    parser.add_argument("--nodes-to-exclude", default=None)
    parser.add_argument("--disable-mha-qdq", action="store_true")
    parser.add_argument("--dq-only", action="store_true")
    parser.add_argument("--simplify", action="store_true")
    parser.add_argument("--use-external-data-format", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load calibration images and print the planned ModelOpt call without quantizing.",
    )
    args = parser.parse_args()

    layout = ArtifactLayout(args.artifact_root)
    onnx_path = args.onnx or layout.onnx_path
    output_path = args.output or (
        layout.root / "onnx" / f"dinov3_vitl16_4out.{args.quantize_mode}.modelopt.onnx"
    )
    config = ModelOptOnnxPtqConfig(
        onnx_path=onnx_path,
        output_path=output_path,
        calibration_manifest=args.calib_manifest,
        input_name=args.input_name,
        image_size=args.image_size,
        max_calibration_images=args.max_calibration_images,
        load_batch_size=args.load_batch_size,
        quantize_mode=args.quantize_mode,
        calibration_method=args.calibration_method,
        calibration_eps=parse_execution_providers(args.calibration_eps),
        high_precision_dtype=args.high_precision_dtype,
        mha_accumulation_dtype=args.mha_accumulation_dtype,
        op_types_to_quantize=parse_optional_csv(args.op_types_to_quantize),
        op_types_to_exclude=parse_optional_csv(args.op_types_to_exclude),
        nodes_to_quantize=parse_optional_csv(args.nodes_to_quantize),
        nodes_to_exclude=parse_optional_csv(args.nodes_to_exclude),
        disable_mha_qdq=args.disable_mha_qdq,
        dq_only=args.dq_only,
        simplify=args.simplify,
        use_external_data_format=args.use_external_data_format,
        log_level=args.log_level,
    )
    if args.dry_run:
        calibration = load_calibration_array(config)
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "config": config.to_json(),
                    "calibration": {
                        "shape": list(calibration.shape),
                        "dtype": str(calibration.dtype),
                    },
                },
                indent=2,
            )
        )
        return 0

    report = run_modelopt_onnx_ptq(config)
    print(json.dumps(report, indent=2))
    return 0 if report["output_exists"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
