#!/usr/bin/env python
"""Run MatMul-only ModelOpt ONNX block sweep variants and ONNX Runtime checks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dinov3_trt.quantization.matmul_sweep import (  # noqa: E402
    MatMulSweepPaths,
    MatMulSweepVariant,
    make_sweep_paths,
    parse_variant_specs,
)


def _command_to_json(command: Sequence[str]) -> list[str]:
    return [str(part) for part in command]


def _run_command(command: Sequence[str], stdout_path: Path) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [str(part) for part in command],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    stdout_path.write_text(
        "\n".join(part for part in (result.stdout, result.stderr) if part),
        encoding="utf-8",
    )
    return result.returncode


def _quantize_command(args: argparse.Namespace, paths: MatMulSweepPaths, nodes: str) -> list[str]:
    command = [
        sys.executable,
        "scripts/quantize_onnx_modelopt.py",
        "--onnx",
        str(args.onnx),
        "--output",
        str(paths.quantized_onnx),
        "--calib-manifest",
        str(args.calib_manifest),
        "--max-calibration-images",
        str(args.max_calibration_images),
        "--load-batch-size",
        str(args.load_batch_size),
        "--quantize-mode",
        args.quantize_mode,
        "--calibration-method",
        args.calibration_method,
        "--calibration-eps",
        args.calibration_eps,
        "--op-types-to-quantize",
        "MatMul",
        "--nodes-to-quantize",
        nodes,
        "--high-precision-dtype",
        args.high_precision_dtype,
        "--mha-accumulation-dtype",
        args.mha_accumulation_dtype,
        "--log-level",
        args.log_level,
    ]
    if args.simplify:
        command.append("--simplify")
    return command


def _random_compare_command(args: argparse.Namespace, paths: MatMulSweepPaths) -> list[str]:
    return [
        sys.executable,
        "scripts/compare_onnx_outputs.py",
        "--reference-onnx",
        str(args.onnx),
        "--candidate-onnx",
        str(paths.quantized_onnx),
        "--output",
        str(paths.random_compare),
        "--batch-size",
        str(args.random_batch_size),
        "--seed",
        str(args.seed),
        "--input-mode",
        args.input_mode,
        "--providers",
        args.providers,
    ]


def _image_eval_command(args: argparse.Namespace, paths: MatMulSweepPaths) -> list[str]:
    return [
        sys.executable,
        "scripts/evaluate_onnx_pair_on_images.py",
        "--reference-onnx",
        str(args.onnx),
        "--candidate-onnx",
        str(paths.quantized_onnx),
        "--manifest",
        str(args.eval_manifest),
        "--output",
        str(paths.image_eval),
        "--batch-size",
        str(args.eval_batch_size),
        "--max-images",
        str(args.max_eval_images),
        "--providers",
        args.providers,
    ]


def _load_image_eval_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"image eval report must be a JSON object: {path}")
    outputs = payload.get("outputs")
    if not isinstance(outputs, list):
        raise ValueError(f"image eval report missing outputs: {path}")
    rows: list[dict[str, Any]] = []
    for item in outputs:
        if not isinstance(item, dict):
            raise ValueError(f"image eval output must be a JSON object: {path}")
        rows.append(
            {
                "name": item.get("name"),
                "cosine_similarity_mean": item.get("cosine_similarity_mean"),
                "cosine_similarity_min": item.get("cosine_similarity_min"),
                "candidate_l2_norm_min": item.get("candidate_l2_norm_min"),
            }
        )
    return {
        "image_count": payload.get("image_count"),
        "batch_size": payload.get("batch_size"),
        "outputs": rows,
    }


def _run_variant(
    args: argparse.Namespace,
    *,
    variant: MatMulSweepVariant,
    paths: MatMulSweepPaths,
) -> dict[str, Any]:
    nodes = ",".join(variant.nodes_to_quantize)
    quantize = _quantize_command(args, paths, nodes)
    random_compare = _random_compare_command(args, paths)
    image_eval = _image_eval_command(args, paths)

    result: dict[str, Any] = {
        "label": variant.label,
        "blocks": list(variant.blocks),
        "node_group": variant.node_group,
        "node_suffixes": list(variant.node_suffixes),
        "nodes_to_quantize": list(variant.nodes_to_quantize),
        "paths": {
            "quantized_onnx": str(paths.quantized_onnx),
            "random_compare": str(paths.random_compare),
            "image_eval": str(paths.image_eval),
        },
        "commands": {
            "quantize": _command_to_json(quantize),
            "random_compare": _command_to_json(random_compare),
            "image_eval": _command_to_json(image_eval),
        },
    }
    if args.dry_run:
        result["dry_run"] = True
        return result

    result["quantize_returncode"] = (
        0
        if paths.quantized_onnx.exists() and args.skip_existing
        else _run_command(quantize, paths.quantize_stdout)
    )
    if result["quantize_returncode"] != 0:
        return result

    result["random_compare_returncode"] = _run_command(
        random_compare,
        paths.random_compare_stdout,
    )
    result["image_eval_returncode"] = _run_command(image_eval, paths.image_eval_stdout)
    result["image_eval_summary"] = _load_image_eval_summary(paths.image_eval)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=Path("Artifacts/onnx/dinov3_vitl16_4out.onnx"))
    parser.add_argument(
        "--calib-manifest",
        type=Path,
        default=Path("Artifacts/manifests/imagenette_selected_calib_500.json"),
    )
    parser.add_argument(
        "--eval-manifest",
        type=Path,
        default=Path("Artifacts/manifests/imagenette_selected_eval_1000.json"),
    )
    parser.add_argument("--onnx-dir", type=Path, default=Path("Artifacts/onnx"))
    parser.add_argument("--reports-dir", type=Path, default=Path("Artifacts/reports"))
    parser.add_argument("--model-stem", default="dinov3_vitl16_4out")
    parser.add_argument("--prefix", default="imagenette64_matmul")
    parser.add_argument("--variant", action="append")
    parser.add_argument(
        "--node-group",
        action="append",
        choices=(
            "all",
            "attention",
            "qkv",
            "attention-core",
            "attention-out",
            "mlp",
            "mlp-up",
            "mlp-down",
        ),
        help=(
            "MatMul subset to quantize for plain --variant values. Inline "
            "variant specs like 19:mlp override this option."
        ),
    )
    parser.add_argument("--max-calibration-images", type=int, default=64)
    parser.add_argument("--load-batch-size", type=int, default=16)
    parser.add_argument("--calibration-method", choices=("max", "minmax", "entropy"), default="max")
    parser.add_argument("--calibration-eps", default="cpu")
    parser.add_argument("--high-precision-dtype", choices=("fp16", "fp32", "bf16"), default="fp32")
    parser.add_argument("--mha-accumulation-dtype", choices=("fp16", "fp32", "bf16"), default="fp32")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--providers", default="CPUExecutionProvider")
    parser.add_argument("--random-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-eval-images", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument(
        "--input-mode",
        choices=("random-normal", "uniform-0-1", "zeros", "ones"),
        default="random-normal",
    )
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--simplify", action="store_true")
    parser.add_argument(
        "--quantize-mode",
        choices=("int8", "fp8"),
        default="int8",
        help=(
            "Quantization data type forwarded to scripts/quantize_onnx_modelopt.py. "
            "FP8 reuses the same node-level partial sweep methodology that recovered "
            "INT8 correctness for layer19 / layer19_attention; output paths include "
            "the mode so int8 and fp8 sweeps coexist."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    variant_specs = tuple(args.variant) if args.variant else ("19", "18-19", "17-19", "16-19")
    node_groups = tuple(args.node_group) if args.node_group else ("all",)
    variants = parse_variant_specs(variant_specs, node_groups=node_groups)
    results = [
        _run_variant(
            args,
            variant=variant,
            paths=make_sweep_paths(
                onnx_dir=args.onnx_dir,
                reports_dir=args.reports_dir,
                model_stem=args.model_stem,
                prefix=args.prefix,
                variant=variant,
                quantize_mode=args.quantize_mode,
            ),
        )
        for variant in variants
    ]
    summary = {
        "onnx": str(args.onnx),
        "calib_manifest": str(args.calib_manifest),
        "eval_manifest": str(args.eval_manifest),
        "prefix": args.prefix,
        "variants": results,
    }
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if args.dry_run:
        return 0
    return 0 if all(_variant_succeeded(result) for result in results) else 1


def _variant_succeeded(result: dict[str, Any]) -> bool:
    return (
        result.get("quantize_returncode") == 0
        and result.get("random_compare_returncode") == 0
        and result.get("image_eval_returncode") == 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
