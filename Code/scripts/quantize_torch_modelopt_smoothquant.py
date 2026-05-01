#!/usr/bin/env python
"""Apply ModelOpt SmoothQuant (PyTorch path) to the HF DINOv3 ViT-L/16 wrapper.

ModelOpt 0.43 only ships SmoothQuant for the PyTorch quantization path
(``modelopt.torch.quantization``). The ONNX PTQ entry point used by
``scripts/quantize_onnx_modelopt.py`` does not expose a SmoothQuant flag.

This script bridges that gap:

1. Loads the HF DINOv3 model + ``make_hf_export_module`` ``nn.Module`` wrapper
   so the 4-output contract is preserved through quantization and export.
2. Builds a tiny calibration ``forward_loop`` from a project image manifest.
3. Calls ``mtq.quantize(wrapper, INT8_SMOOTHQUANT_CFG, forward_loop=...)``
   in-place, which inserts smoothing scales + INT8 Q/DQ quantizers.
4. Exports the now-quantized wrapper with ``torch.onnx.export``. The exported
   graph carries Q/DQ nodes that TensorRT consumes via ``trtexec --int8``.

The output stays compatible with the existing speedup / eval / matrix tooling.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dinov3_trt.artifacts import ArtifactLayout  # noqa: E402
from dinov3_trt.contracts import (  # noqa: E402
    DINO_VITL16_224_CONTRACT,
    expected_token_count,
    make_dinov3_vitl16_contract,
)
from dinov3_trt.export.hf_model import (  # noqa: E402
    create_hf_dinov3_model,
    freeze_module_parameters,
    make_hf_export_module,
    patch_hf_dinov3_rope_for_onnx_export,
)
from dinov3_trt.infer.image_eval import chunk_paths, load_image_batch, read_image_manifest  # noqa: E402


def _resolve_model_path(args: argparse.Namespace, layout: ArtifactLayout) -> str:
    if args.model_path:
        return str(args.model_path)
    if (layout.weights_dir / "config.json").exists():
        return str(layout.weights_dir)
    return DINO_VITL16_224_CONTRACT.model_id


def _load_calibration_batches(
    manifest: Path,
    *,
    image_size: int,
    max_calibration_images: int,
    load_batch_size: int,
    device: str,
) -> list[Any]:
    torch = importlib.import_module("torch")
    image_paths = read_image_manifest(manifest)[:max_calibration_images]
    if not image_paths:
        raise ValueError(f"calibration manifest produced no images: {manifest}")
    batches: list[Any] = []
    for path_batch in chunk_paths(image_paths, load_batch_size):
        image_batch = load_image_batch(path_batch, image_size=image_size)
        tensor = torch.from_numpy(image_batch.tensor).to(device=device, dtype=torch.float32)
        batches.append(tensor)
    return batches


def _parse_skip_blocks(value: str | None) -> tuple[int, ...]:
    """Parse a CSV/range spec like ``"16-19"`` or ``"16,17,19"`` into block indices.

    Empty / ``None`` returns an empty tuple. Each block index must be in 0..23
    (DINOv3 ViT-L has 24 transformer blocks).
    """

    if not value:
        return ()
    indices: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", maxsplit=1)
            start = int(start_text)
            end = int(end_text)
            if start > end:
                raise ValueError(f"invalid block range: {part}")
            indices.update(range(start, end + 1))
        else:
            indices.add(int(part))
    if any(b < 0 or b > 23 for b in indices):
        raise ValueError("DINOv3 ViT-L block indices must be in 0..23")
    return tuple(sorted(indices))


def _smoothquant_config(alpha: float, *, skip_blocks: tuple[int, ...] = ()) -> dict[str, Any]:
    """Return a copy of ``INT8_SMOOTHQUANT_CFG`` with a custom alpha.

    When ``skip_blocks`` is non-empty, the matching transformer blocks have
    their input + weight quantizers disabled at calibration time, so those
    blocks fall back to the surrounding (BF16/FP32) execution. This is the
    "sensitive-layer mixed precision" recipe that targets feat_layer_16/20
    deep-layer cosine drift while keeping shallow blocks INT8.
    """

    mtq = importlib.import_module("modelopt.torch.quantization")
    base = mtq.INT8_SMOOTHQUANT_CFG
    config: dict[str, Any]
    if isinstance(base, dict):
        config = dict(base)
    elif hasattr(base, "model_dump"):
        config = base.model_dump()
    else:
        # Pydantic v1 fallback
        config = dict(base) if hasattr(base, "__iter__") else {"quant_cfg": {}, "algorithm": {}}
    algorithm = config.get("algorithm")
    if isinstance(algorithm, dict):
        algorithm = dict(algorithm)
        algorithm["alpha"] = alpha
        config["algorithm"] = algorithm
    elif isinstance(algorithm, str):
        # ModelOpt accepts either string or dict; promote to dict so we can set alpha.
        config["algorithm"] = {"method": "smoothquant", "alpha": alpha}
    else:
        config["algorithm"] = {"method": "smoothquant", "alpha": alpha}

    if skip_blocks:
        # Insert per-block disable wildcards into ``quant_cfg``. Wildcard format
        # mirrors the existing ``*lm_head*`` / ``*router*`` entries shipped in
        # INT8_SMOOTHQUANT_CFG; ModelOpt resolves them against quantizer module
        # names (``*input_quantizer`` / ``*weight_quantizer``).
        quant_cfg = config.get("quant_cfg")
        quant_cfg = dict(quant_cfg) if isinstance(quant_cfg, dict) else {}
        for block in skip_blocks:
            quant_cfg[f"*model.layer.{block}.*"] = {"enable": False}
        config["quant_cfg"] = quant_cfg
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("Artifacts"))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--token", default=None)
    parser.add_argument(
        "--attn-implementation",
        default="eager",
        help="Transformers attention implementation; matches ONNX export default.",
    )
    parser.add_argument(
        "--calib-manifest",
        type=Path,
        default=Path("Artifacts") / "manifests" / "imagenette_selected_calib_500.json",
    )
    parser.add_argument("--max-calibration-images", type=int, default=16)
    parser.add_argument("--load-batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help=(
            "SmoothQuant migration strength. 0.5 is the SmoothQuant paper default. "
            "Higher values move more of the activation outliers into weights."
        ),
    )
    parser.add_argument(
        "--skip-blocks",
        default=None,
        help=(
            "Comma/range spec (e.g. '16-19' or '16,17,19') of DINOv3 ViT-L "
            "transformer blocks whose input + weight quantizers should be "
            "disabled. Those blocks then run in the surrounding (FP32/BF16) "
            "precision while the rest of the model goes INT8 SmoothQuant. "
            "Block indices are 0-based and must be in 0..23."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--opset", type=int, default=19)
    parser.add_argument("--use-external-data-format", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip both quantization and export; emit the planned config.",
    )
    args = parser.parse_args()

    layout = ArtifactLayout(args.artifact_root)
    contract = make_dinov3_vitl16_contract(args.image_size)
    model_path = _resolve_model_path(args, layout)
    skip_blocks = _parse_skip_blocks(args.skip_blocks)
    output_path = args.output or (
        layout.root / "onnx" / "dinov3_vitl16_4out.int8.modelopt.smoothquant.onnx"
    )

    plan: dict[str, Any] = {
        "model_path": str(model_path),
        "calib_manifest": str(args.calib_manifest),
        "max_calibration_images": args.max_calibration_images,
        "load_batch_size": args.load_batch_size,
        "image_size": args.image_size,
        "alpha": args.alpha,
        "skip_blocks": list(skip_blocks),
        "output_path": str(output_path),
        "device": args.device,
        "opset": args.opset,
        "use_external_data_format": args.use_external_data_format,
        "expected_tokens": expected_token_count(contract),
    }

    if args.dry_run:
        plan["dry_run"] = True
        try:
            plan["config"] = _smoothquant_config(args.alpha, skip_blocks=skip_blocks)
        except ModuleNotFoundError as exc:
            # ModelOpt is only installed on the RTX 5080 host. Local CI / unit
            # tests should still be able to inspect argparse plumbing in dry-run
            # without pulling in the heavy GPU stack.
            plan["config"] = None
            plan["config_error"] = (
                f"modelopt unavailable in this environment ({exc.name}); "
                "config will be materialised on the RTX 5080 host."
            )
        print(json.dumps(plan, indent=2, default=str))
        return 0

    torch = importlib.import_module("torch")
    mtq = importlib.import_module("modelopt.torch.quantization")

    # 1. Load HF model + RoPE patch
    model = create_hf_dinov3_model(
        model_path,
        token=args.token,
        local_files_only=args.local_files_only,
        attn_implementation=args.attn_implementation,
    )
    model = model.to(device=args.device, dtype=torch.float32).eval()
    rope_patches = patch_hf_dinov3_rope_for_onnx_export(model)
    freeze_module_parameters(model)

    # 2. Wrap as nn.Module exposing the 4-output contract
    wrapper = make_hf_export_module(model, contract=contract).to(args.device).eval()
    freeze_module_parameters(wrapper)

    # 3. Calibration data + forward loop
    batches = _load_calibration_batches(
        args.calib_manifest,
        image_size=args.image_size,
        max_calibration_images=args.max_calibration_images,
        load_batch_size=args.load_batch_size,
        device=args.device,
    )

    def forward_loop(m: Any) -> None:
        with torch.no_grad():
            for batch in batches:
                m(batch)

    # 4. Quantize + calibrate (in-place SmoothQuant)
    config = _smoothquant_config(args.alpha, skip_blocks=skip_blocks)
    quantized = mtq.quantize(wrapper, config, forward_loop=forward_loop)

    # 5. Export to ONNX (the quantizers are picked up as Q/DQ nodes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(
        1, 3, args.image_size, args.image_size, device=args.device, dtype=torch.float32
    )
    dynamic_axes = {
        "pixel_values": {0: "batch"},
        **{name: {0: "batch"} for name in contract.output_names},
    }
    with torch.no_grad():
        torch.onnx.export(
            quantized,
            dummy_input,
            str(output_path),
            input_names=["pixel_values"],
            output_names=list(contract.output_names),
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )

    plan["rope_patches"] = rope_patches
    plan["calibration_image_count"] = sum(int(b.shape[0]) for b in batches)
    plan["output_size_bytes"] = output_path.stat().st_size if output_path.exists() else None
    plan["output_exists"] = output_path.exists()
    print(json.dumps(plan, indent=2, default=str))
    return 0 if output_path.exists() else 1


if __name__ == "__main__":
    raise SystemExit(main())
