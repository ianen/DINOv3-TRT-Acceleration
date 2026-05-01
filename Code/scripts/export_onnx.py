#!/usr/bin/env python
"""Export an importable DINOv3 model factory to the project 4-output ONNX."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, cast

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dinov3_trt.artifacts import ArtifactLayout  # noqa: E402
from dinov3_trt.export.onnx_export import (  # noqa: E402
    OnnxExportConfig,
    assert_no_onnx_if_nodes,
    export_model_to_onnx,
)


def import_factory(dotted_path: str) -> Callable[[], Any]:
    module_name, _, attr = dotted_path.partition(":")
    if not module_name or not attr:
        raise ValueError("factory must use module:function syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, attr)
    if not callable(factory):
        raise TypeError(f"{dotted_path} is not callable")
    return cast(Callable[[], Any], factory)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory", required=True, help="Import path like my_module:create_model")
    parser.add_argument("--artifact-root", type=Path, default=Path("Artifacts"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--static-batch", action="store_true")
    parser.add_argument("--skip-rope-patch", action="store_true")
    parser.add_argument("--dynamo", action="store_true")
    parser.add_argument("--validate-no-if", action="store_true")
    args = parser.parse_args()

    layout = ArtifactLayout(args.artifact_root)
    output_path = args.output or layout.onnx_path
    try:
        factory = import_factory(args.factory)
        model = factory()
        result = export_model_to_onnx(
            model,
            OnnxExportConfig(
                output_path=output_path,
                opset=args.opset,
                batch_size=args.batch_size,
                device=args.device,
                dtype=args.dtype,
                dynamic_batch=not args.static_batch,
                apply_rope_patch=not args.skip_rope_patch,
                dynamo=args.dynamo,
            ),
        )
        if args.validate_no_if:
            assert_no_onnx_if_nodes(output_path)
    except Exception as exc:
        print(f"ONNX export failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result.to_json(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
