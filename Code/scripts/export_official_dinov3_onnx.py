#!/usr/bin/env python
"""Export official local DINOv3 ViT-L/16 to the project 4-output ONNX contract."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dinov3_trt.artifacts import ArtifactLayout  # noqa: E402
from dinov3_trt.contracts import make_dinov3_vitl16_contract  # noqa: E402
from dinov3_trt.export.official_model import (  # noqa: E402
    add_dinov3_source_to_path,
    create_official_vitl16_model,
    resolve_official_pth_weight,
)
from dinov3_trt.export.onnx_export import (  # noqa: E402
    OnnxExportConfig,
    assert_no_onnx_if_nodes,
    export_model_to_onnx,
)
from dinov3_trt.export.rope_patch import apply_dinov3_export_patches  # noqa: E402


def torch_dtype(name: str) -> object:
    torch = importlib.import_module("torch")
    mapping = {"float32": torch.float32, "float16": torch.float16}
    return mapping[name]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("Artifacts"))
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help=(
            "Square input resolution (default 224). Must be ≥ patch_size (16) and ideally a "
            "multiple of 16 for clean patch grid. V1.0.4 uses 512."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--static-batch", action="store_true")
    parser.add_argument("--skip-rope-patch", action="store_true")
    parser.add_argument("--dynamo", action="store_true")
    parser.add_argument("--validate-no-if", action="store_true")
    args = parser.parse_args()

    torch = importlib.import_module("torch")
    layout = ArtifactLayout(args.artifact_root)
    source_dir = args.source_dir or layout.source_dir
    output_path = args.output or layout.onnx_path
    try:
        add_dinov3_source_to_path(source_dir)
        patch_report = None if args.skip_rope_patch else apply_dinov3_export_patches()
        weights_path = resolve_official_pth_weight(
            layout.weights_dir,
            weights_path=args.weights,
            pretrained=args.pretrained,
        )
        model = create_official_vitl16_model(
            source_dir=source_dir,
            weights_path=weights_path,
            pretrained=args.pretrained,
        )
        dtype = torch_dtype(args.dtype)
        model = model.to(device=args.device, dtype=dtype).eval()
        contract = make_dinov3_vitl16_contract(image_size=args.image_size)
        with torch.no_grad():
            result = export_model_to_onnx(
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
                contract=contract,
            )
        if args.validate_no_if:
            assert_no_onnx_if_nodes(output_path)
    except Exception as exc:
        print(f"official DINOv3 ONNX export failed: {exc}", file=sys.stderr)
        return 1

    payload = result.to_json()
    payload["source_dir"] = str(source_dir)
    payload["weights"] = None if weights_path is None else str(weights_path)
    payload["pretrained"] = args.pretrained
    payload["device"] = args.device
    payload["dtype"] = args.dtype
    payload["patch_report"] = None if patch_report is None else patch_report.to_json()
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
