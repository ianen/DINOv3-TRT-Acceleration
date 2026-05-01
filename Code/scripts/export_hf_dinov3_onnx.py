#!/usr/bin/env python
"""Export Hugging Face DINOv3 ViT-L/16 to the project 4-output ONNX contract."""

from __future__ import annotations

import argparse
import importlib
import json
import os
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
    ModelContract,
    expected_token_count,
    make_dinov3_vitl16_contract,
)
from dinov3_trt.export.hf_model import (  # noqa: E402
    create_hf_dinov3_model,
    freeze_module_parameters,
    make_hf_export_module,
    patch_hf_dinov3_rope_for_onnx_export,
)
from dinov3_trt.export.onnx_export import (  # noqa: E402
    OnnxExportConfig,
    assert_no_onnx_if_nodes,
    build_dynamic_axes,
    make_dummy_input,
)


def torch_dtype(name: str) -> object:
    torch = importlib.import_module("torch")
    mapping = {"float32": torch.float32, "float16": torch.float16}
    return mapping[name]


def default_model_path(layout: ArtifactLayout) -> str:
    if (layout.weights_dir / "config.json").exists():
        return str(layout.weights_dir)
    return DINO_VITL16_224_CONTRACT.model_id


def export_hf_model_to_onnx(
    model: Any,
    config: OnnxExportConfig,
    *,
    num_register_tokens: int | None = None,
    contract: ModelContract = DINO_VITL16_224_CONTRACT,
) -> dict[str, object]:
    config.validate()
    torch = importlib.import_module("torch")
    wrapper = make_hf_export_module(
        model,
        contract=contract,
        num_register_tokens=num_register_tokens,
    ).eval()
    freeze_module_parameters(wrapper)
    dummy_input = make_dummy_input(config, contract)
    dynamic_axes = build_dynamic_axes(config, contract)

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_input,
            config.output_path,
            input_names=[config.input_name],
            output_names=list(contract.output_names),
            dynamic_axes=dynamic_axes,
            opset_version=config.opset,
            do_constant_folding=config.do_constant_folding,
            dynamo=config.dynamo,
        )

    return {
        "output_path": str(config.output_path),
        "output_names": list(contract.output_names),
        "dynamic_axes": dynamic_axes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("Artifacts"))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN") or None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--attn-implementation",
        default="eager",
        help="Transformers attention implementation to use during export.",
    )
    parser.add_argument("--num-register-tokens", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=DINO_VITL16_224_CONTRACT.image_size)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--static-batch", action="store_true")
    parser.add_argument("--dynamo", action="store_true")
    parser.add_argument("--validate-no-if", action="store_true")
    args = parser.parse_args()

    layout = ArtifactLayout(args.artifact_root)
    contract = make_dinov3_vitl16_contract(args.image_size)
    model_path = args.model_path or default_model_path(layout)
    output_path = args.output or layout.onnx_path

    try:
        torch = importlib.import_module("torch")
        model = create_hf_dinov3_model(
            model_path,
            token=args.token,
            revision=args.revision,
            local_files_only=args.local_files_only,
            attn_implementation=args.attn_implementation,
        )
        dtype = torch_dtype(args.dtype)
        model = model.to(device=args.device, dtype=dtype).eval()
        rope_patch_count = patch_hf_dinov3_rope_for_onnx_export(model)
        freeze_module_parameters(model)
        with torch.no_grad():
            payload = export_hf_model_to_onnx(
                model,
                OnnxExportConfig(
                    output_path=output_path,
                    opset=args.opset,
                    batch_size=args.batch_size,
                    device=args.device,
                    dtype=args.dtype,
                    dynamic_batch=not args.static_batch,
                    apply_rope_patch=False,
                    dynamo=args.dynamo,
                ),
                num_register_tokens=args.num_register_tokens,
                contract=contract,
            )
        if args.validate_no_if:
            assert_no_onnx_if_nodes(output_path)
    except Exception as exc:
        print(f"HF DINOv3 ONNX export failed: {exc}", file=sys.stderr)
        return 1

    payload["model_path"] = str(model_path)
    payload["device"] = args.device
    payload["dtype"] = args.dtype
    payload["image_size"] = args.image_size
    payload["expected_tokens"] = expected_token_count(contract)
    payload["hf_rope_export_patch_count"] = rope_patch_count
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
