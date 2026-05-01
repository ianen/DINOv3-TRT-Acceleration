#!/usr/bin/env python
"""Inspect ONNX graph outputs and top-level operator counts."""

from __future__ import annotations

import argparse
import collections
import importlib
import json
from pathlib import Path
from typing import Any


def value_info_shape(value_info: Any) -> list[int | str]:
    dims = value_info.type.tensor_type.shape.dim
    return [dim.dim_param or dim.dim_value for dim in dims]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx", type=Path)
    parser.add_argument("--top-k", type=int, default=30)
    args = parser.parse_args()

    onnx = importlib.import_module("onnx")
    model = onnx.load(args.onnx)
    op_counts = collections.Counter(node.op_type for node in model.graph.node)
    payload = {
        "path": str(args.onnx),
        "size_bytes": args.onnx.stat().st_size,
        "outputs": [
            {
                "name": output.name,
                "shape": value_info_shape(output),
            }
            for output in model.graph.output
        ],
        "top_level_node_count": len(model.graph.node),
        "top_level_op_counts": op_counts.most_common(args.top_k),
        "has_top_level_if": "If" in op_counts,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
