#!/usr/bin/env python
"""Validate DINOv3 4-output PyTorch contract for an importable model factory."""

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

from dinov3_trt.contracts import DINO_VITL16_224_CONTRACT, validate_output_shapes  # noqa: E402
from dinov3_trt.export.wrapper import DinoV3IntermediateLayerWrapper, ordered_output_mapping  # noqa: E402


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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    try:
        torch = importlib.import_module("torch")
    except ImportError:
        print("torch is required for this check", file=sys.stderr)
        return 2

    factory = import_factory(args.factory)
    model = factory()
    wrapper = DinoV3IntermediateLayerWrapper(model).eval()
    pixel_values = torch.zeros(
        (
            args.batch_size,
            3,
            DINO_VITL16_224_CONTRACT.image_size,
            DINO_VITL16_224_CONTRACT.image_size,
        ),
        dtype=torch.float32,
        device=args.device,
    )
    with torch.no_grad():
        outputs = wrapper(pixel_values)
    mapping = ordered_output_mapping(wrapper.output_names, outputs)
    validate_output_shapes(mapping, batch_size=args.batch_size)
    print(
        json.dumps(
            {
                "factory": args.factory,
                "device": args.device,
                "batch_size": args.batch_size,
                "outputs": {name: list(value.shape) for name, value in mapping.items()},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
