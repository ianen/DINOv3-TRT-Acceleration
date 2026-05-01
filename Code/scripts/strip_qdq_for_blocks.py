#!/usr/bin/env python
"""Strip internal Q/DQ pairs from selected DINOv3 blocks (ADR-010 § 5.1 step 2).

Reads a SmoothQuant Q/DQ ONNX, computes a strip plan via the pure-Python
planner, applies it to the loaded ONNX (delete + rewire only — no
onnx-graphsurgeon dependency), and writes a new ONNX file with the requested
blocks rendered Q/DQ-free. The new ONNX can then be fed to trtexec with
``--int8`` and the surrounding INT8 path will keep the boundary blocks in
INT8 while the requested range falls back to fp precision (FP32 / BF16
depending on builder flags).

Companion sidecar JSON records the strip plan (deleted nodes + rewires +
counts) for traceability.
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

from dinov3_trt.quantization.onnx_qdq_strip_planner import (  # noqa: E402
    StripPlan,
    plan_strip_operations,
)
from dinov3_trt.quantization.onnx_qdq_stripper import OnnxNodeWithEdges  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--output-onnx", required=True, type=Path)
    parser.add_argument(
        "--blocks",
        required=True,
        help="0-based block indices, e.g. '16-19' or '16,17,18,19' or '16-19,22'.",
    )
    parser.add_argument(
        "--plan-json",
        type=Path,
        help="Optional sidecar JSON path. Default: <output-onnx>.strip_plan.json.",
    )
    parser.add_argument(
        "--load-external-data",
        action="store_true",
        help="Load external tensors when reading the source ONNX.",
    )
    return parser.parse_args(argv)


def parse_blocks_spec(value: str) -> tuple[int, ...]:
    """Parse '16-19' / '16,17,18,19' / '16-19,22' into sorted unique ints."""

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


def project_graph_nodes(graph: object) -> tuple[OnnxNodeWithEdges, ...]:
    """Project an ONNX GraphProto into the planner's `OnnxNodeWithEdges` records."""

    return tuple(
        OnnxNodeWithEdges(
            name=node.name,
            op_type=node.op_type,
            inputs=tuple(node.input),
            outputs=tuple(node.output),
        )
        for node in graph.node  # type: ignore[attr-defined]
    )


def apply_plan_in_place(graph: object, plan: StripPlan) -> tuple[int, int]:
    """Mutate a loaded `onnx.GraphProto` according to the plan.

    Returns ``(deleted_node_count, rewired_input_count)`` for caller logging.

    Implementation: repeated-field protobuf APIs do not support index-based
    delete cleanly when iterating, so we rebuild ``graph.node`` from a kept
    list and reset the repeated field. Input rewires are done in-place by
    overwriting individual ``node.input[i]`` slots.
    """

    rewires = plan.tensor_rewires
    rewired_count = 0
    kept_nodes = []
    for node in graph.node:  # type: ignore[attr-defined]
        if node.name in plan.nodes_to_delete:
            continue
        for index in range(len(node.input)):
            new_tensor = rewires.get(node.input[index])
            if new_tensor is None or new_tensor == node.input[index]:
                continue
            node.input[index] = new_tensor
            rewired_count += 1
        kept_nodes.append(node)

    deleted = len(graph.node) - len(kept_nodes)  # type: ignore[attr-defined]
    del graph.node[:]  # type: ignore[attr-defined]
    graph.node.extend(kept_nodes)  # type: ignore[attr-defined]
    return deleted, rewired_count


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    block_indices = parse_blocks_spec(args.blocks)

    import onnx  # type: ignore[import-not-found]  # noqa: PLC0415

    model = onnx.load(str(args.onnx), load_external_data=args.load_external_data)
    nodes = project_graph_nodes(model.graph)

    plan = plan_strip_operations(nodes, block_indices=list(block_indices))
    if plan.stripped_pair_count == 0:
        print(
            f"[strip] no internal Q/DQ pairs found in blocks {list(block_indices)}; "
            f"nothing to do."
        )
    deleted, rewired = apply_plan_in_place(model.graph, plan)
    print(
        f"[strip] deleted {deleted} nodes, rewired {rewired} input slots "
        f"(planned strip {plan.stripped_pair_count} pairs)"
    )

    args.output_onnx.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output_onnx))
    print(f"[strip] saved -> {args.output_onnx}")

    plan_path = args.plan_json or args.output_onnx.with_suffix(args.output_onnx.suffix + ".strip_plan.json")
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_payload = {
        "source_onnx": str(args.onnx),
        "output_onnx": str(args.output_onnx),
        "blocks_spec": args.blocks,
        "block_indices_resolved": list(block_indices),
        "deleted_node_count": deleted,
        "rewired_input_slots": rewired,
        "plan": plan.to_dict(),
    }
    plan_path.write_text(json.dumps(plan_payload, indent=2), encoding="utf-8")
    print(f"[strip] plan -> {plan_path}")

    if deleted != 2 * plan.stripped_pair_count:
        print(
            f"[strip] WARNING: deleted={deleted} but planned 2*{plan.stripped_pair_count}="
            f"{2 * plan.stripped_pair_count} — review plan JSON."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
