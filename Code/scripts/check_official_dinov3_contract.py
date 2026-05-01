#!/usr/bin/env python
"""Validate the 4-output contract using the official local DINOv3 source tree."""

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
from dinov3_trt.contracts import DINO_VITL16_224_CONTRACT, validate_output_shapes  # noqa: E402
from dinov3_trt.export.official_model import (  # noqa: E402
    add_dinov3_source_to_path,
    create_official_vitl16_model,
    resolve_official_pth_weight,
)
from dinov3_trt.export.rope_patch import apply_dinov3_export_patches  # noqa: E402
from dinov3_trt.export.wrapper import DinoV3IntermediateLayerWrapper, ordered_output_mapping  # noqa: E402


def torch_dtype(name: str) -> object:
    torch = importlib.import_module("torch")
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("Artifacts"))
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--skip-rope-patch", action="store_true")
    args = parser.parse_args()

    torch = importlib.import_module("torch")
    layout = ArtifactLayout(args.artifact_root)
    source_dir = args.source_dir or layout.source_dir
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
        wrapper = DinoV3IntermediateLayerWrapper(model).eval()
        pixel_values = torch.zeros(
            (
                args.batch_size,
                3,
                DINO_VITL16_224_CONTRACT.image_size,
                DINO_VITL16_224_CONTRACT.image_size,
            ),
            dtype=dtype,
            device=args.device,
        )
        with torch.no_grad():
            outputs = wrapper(pixel_values)
        mapping = ordered_output_mapping(wrapper.output_names, outputs)
        validate_output_shapes(mapping, batch_size=args.batch_size)
    except Exception as exc:
        print(f"official DINOv3 contract check failed: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "source_dir": str(source_dir),
                "weights": None if weights_path is None else str(weights_path),
                "pretrained": args.pretrained,
                "device": args.device,
                "dtype": args.dtype,
                "batch_size": args.batch_size,
                "patch_report": None if patch_report is None else patch_report.to_json(),
                "outputs": {name: list(value.shape) for name, value in mapping.items()},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
