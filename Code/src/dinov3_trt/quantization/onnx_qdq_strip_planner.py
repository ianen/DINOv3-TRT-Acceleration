"""V1.2 ONNX Q/DQ strip planner (ADR-010 § 5.1 step 2 — pure-Python core).

Step 2 of the V1.2 path: given the node list of a SmoothQuant Q/DQ ONNX and
the requested 0-based block range, decide *exactly* which nodes to delete and
which downstream tensor references to rewire so the surrounding INT8 path
keeps working after the internal Q/DQ pairs in those blocks are removed.

The planner is pure-Python — it consumes the same `OnnxNodeWithEdges`
projection produced by `onnx_qdq_stripper.py` and outputs a `StripPlan` data
object. The thin remote-only driver script (`strip_qdq_for_blocks.py`) loads
the real ONNX, calls this planner, and applies the plan using only the
`onnx` library (no onnx-graphsurgeon dependency).

Why split planning vs application?
==================================
- The planning logic is the bug-prone part (which node deletes which edge?).
  Splitting it out makes that logic *unit-testable on macOS* without onnx
  installed.
- The application layer is mechanical (delete from node list + replace input
  references). It can be reviewed by reading <30 lines of imperative code in
  the driver script.
- This pattern mirrors `layer_precision.py` ↔ `build_layer_precisions_arg.py`
  used in V1.1, so reviewers don't need to learn a new convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from dinov3_trt.quantization.onnx_qdq_stripper import (
    OnnxNodeWithEdges,
    QDQPair,
    find_block_qdq_pairs,
    split_strippable_and_preserved,
)


@dataclass(frozen=True)
class StripPlan:
    """Deterministic description of how to mutate an ONNX graph.

    * ``nodes_to_delete`` — exact ONNX node names that should be removed.
    * ``tensor_rewires`` — for every internal Q→DQ pair, the ``DequantizeLinear``
      output tensor must be replaced everywhere downstream by the corresponding
      ``QuantizeLinear`` input tensor (so the path skips the Q/DQ pair entirely
      and runs at fp precision).
    * ``preserved_pairs`` — pairs that the planner refused to strip (boundary
      input/output). For SmoothQuant α=0.8 in the layer 16-19 range this is
      empty (see ADR-010 § 4.3 修订) but the field is kept for forward
      compatibility with non-SmoothQuant ONNX layouts.
    * ``stripped_pair_count`` — convenience integer for sanity checking.
    """

    nodes_to_delete: frozenset[str]
    tensor_rewires: Mapping[str, str]
    preserved_pairs: tuple[QDQPair, ...]
    stripped_pair_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "nodes_to_delete": sorted(self.nodes_to_delete),
            "tensor_rewires": dict(self.tensor_rewires),
            "preserved_pairs": [
                {
                    "quantize_node": pair.quantize_node,
                    "dequantize_node": pair.dequantize_node,
                    "location": pair.location,
                }
                for pair in self.preserved_pairs
            ],
            "stripped_pair_count": self.stripped_pair_count,
        }


def plan_strip_operations(
    nodes: Sequence[OnnxNodeWithEdges],
    *,
    block_indices: Sequence[int],
) -> StripPlan:
    """Build a :class:`StripPlan` for stripping internal Q/DQ pairs in a block range.

    The plan only touches *internal* pairs (per
    :func:`onnx_qdq_stripper.classify_pair`). Boundary pairs are recorded in
    ``preserved_pairs`` and left intact, so callers can audit what the
    planner declined to strip.
    """

    if not block_indices:
        raise ValueError("block_indices must be non-empty")

    pairs = find_block_qdq_pairs(nodes, block_indices=list(block_indices))
    strippable, preserved = split_strippable_and_preserved(pairs)

    nodes_by_name: dict[str, OnnxNodeWithEdges] = {node.name: node for node in nodes}

    delete: set[str] = set()
    rewires: dict[str, str] = {}
    for pair in strippable:
        q_node = nodes_by_name.get(pair.quantize_node)
        dq_node = nodes_by_name.get(pair.dequantize_node)
        if q_node is None or dq_node is None:
            raise ValueError(
                f"strippable pair references unknown node: q={pair.quantize_node} "
                f"dq={pair.dequantize_node}"
            )
        if not q_node.inputs:
            raise ValueError(f"QuantizeLinear node has no inputs: {q_node.name}")
        if not dq_node.outputs:
            raise ValueError(f"DequantizeLinear node has no outputs: {dq_node.name}")

        upstream_tensor = q_node.inputs[0]
        downstream_tensor = dq_node.outputs[0]
        delete.add(q_node.name)
        delete.add(dq_node.name)
        # Rewire: anywhere an input was the DQ output tensor, point it to the
        # Q input tensor instead. The pair's middle tensor (Q's output / DQ's
        # input) becomes orphaned and is dropped when callers serialise.
        if downstream_tensor in rewires and rewires[downstream_tensor] != upstream_tensor:
            raise ValueError(
                f"conflicting rewire for tensor '{downstream_tensor}': "
                f"existing -> '{rewires[downstream_tensor]}', new -> '{upstream_tensor}'"
            )
        rewires[downstream_tensor] = upstream_tensor

    return StripPlan(
        nodes_to_delete=frozenset(delete),
        tensor_rewires=dict(rewires),
        preserved_pairs=tuple(preserved),
        stripped_pair_count=len(strippable),
    )


def apply_plan_to_node_list(
    nodes: Sequence[OnnxNodeWithEdges],
    plan: StripPlan,
) -> list[OnnxNodeWithEdges]:
    """Return a new node list with the plan applied (pure data, no ONNX I/O).

    Useful for in-memory tests that round-trip a StripPlan over the
    OnnxNodeWithEdges projection and compare to a hand-computed expected list.
    The real driver script does the equivalent on `onnx.GraphProto.node`.
    """

    rewires = plan.tensor_rewires
    rewritten: list[OnnxNodeWithEdges] = []
    for node in nodes:
        if node.name in plan.nodes_to_delete:
            continue
        new_inputs = tuple(rewires.get(inp, inp) for inp in node.inputs)
        if new_inputs == node.inputs:
            rewritten.append(node)
        else:
            rewritten.append(
                OnnxNodeWithEdges(
                    name=node.name,
                    op_type=node.op_type,
                    inputs=new_inputs,
                    outputs=node.outputs,
                )
            )
    return rewritten
