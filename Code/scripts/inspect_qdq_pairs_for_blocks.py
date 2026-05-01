#!/usr/bin/env python
"""Inspect Q/DQ pair distribution for given DINOv3 transformer blocks (ADR-010 § 5.1).

Reads an ONNX model, extracts every QuantizeLinear→DequantizeLinear adjacent
pair, classifies each pair against the requested 0-based block range, and
writes a JSON report with internal / boundary_input / boundary_output counts
plus the full pair list. The report becomes input for the V1.2 strip
operation (not implemented yet — see ADR-010 § 5.1 step 1).

Usage::

    python scripts/inspect_qdq_pairs_for_blocks.py \\
        --onnx Artifacts/onnx/dinov3_vitl16_4out.int8.modelopt.smoothquant.alpha080.imagenette500.onnx \\
        --blocks 16-19 \\
        --output Artifacts/reports/qdq_pairs_inspect_blocks_16-19.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dinov3_trt.quantization.onnx_qdq_stripper import (  # noqa: E402
    OnnxNodeWithEdges,
    find_block_qdq_pairs,
    split_strippable_and_preserved,
    summarise_pairs,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument(
        "--blocks",
        required=True,
        help="0-based block indices, e.g. '16-19' or '16,17,18,19' or '16-19,22'.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--load-external-data",
        action="store_true",
        help="Set when ONNX has external weights that you need fully loaded.",
    )
    return parser.parse_args(argv)


def parse_blocks_spec(value: str) -> tuple[int, ...]:
    """Parse '16-19' / '16,17,18,19' / '16-19,22' into sorted unique ints.

    Identical semantics to ``build_layer_precisions_arg.py.parse_blocks_spec``.
    """

    if not value or not value.strip():
        raise ValueError("blocks spec must be non-empty")
    indices: set[int] = set()
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo_str, hi_str = chunk.split("-", 1)
            lo, hi = int(lo_str), int(hi_str)
            if lo > hi:
                raise ValueError(f"range '{chunk}' is reversed (lo > hi)")
            indices.update(range(lo, hi + 1))
        else:
            indices.add(int(chunk))
    if not indices:
        raise ValueError(f"blocks spec '{value}' resolved to empty index set")
    return tuple(sorted(indices))


def load_onnx_node_records(
    path: Path,
    *,
    load_external_data: bool,
) -> tuple[OnnxNodeWithEdges, ...]:
    import onnx  # type: ignore[import-not-found]  # noqa: PLC0415

    model = onnx.load(str(path), load_external_data=load_external_data)
    return tuple(
        OnnxNodeWithEdges(
            name=node.name,
            op_type=node.op_type,
            inputs=tuple(node.input),
            outputs=tuple(node.output),
        )
        for node in model.graph.node
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    block_indices = parse_blocks_spec(args.blocks)
    nodes = load_onnx_node_records(args.onnx, load_external_data=args.load_external_data)
    pairs = find_block_qdq_pairs(nodes, block_indices=list(block_indices))
    strippable, preserved = split_strippable_and_preserved(pairs)
    counts = summarise_pairs(pairs)

    payload: dict[str, object] = {
        "onnx_path": str(args.onnx),
        "blocks_spec": args.blocks,
        "block_indices_resolved": list(block_indices),
        "node_count_total": len(nodes),
        "qdq_counts": dict(counts),
        "strippable_internal_pairs": [
            {
                "quantize_node": pair.quantize_node,
                "dequantize_node": pair.dequantize_node,
                "block": pair.quantize_block,
            }
            for pair in strippable
        ],
        "preserved_boundary_pairs": [
            {
                "quantize_node": pair.quantize_node,
                "dequantize_node": pair.dequantize_node,
                "quantize_block": pair.quantize_block,
                "dequantize_block": pair.dequantize_block,
                "location": pair.location,
            }
            for pair in preserved
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"qdq pairs in blocks {list(block_indices)}: "
        f"internal={counts['internal']} boundary_input={counts['boundary_input']} "
        f"boundary_output={counts['boundary_output']} total={counts['total']}"
    )
    print(f"report -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
