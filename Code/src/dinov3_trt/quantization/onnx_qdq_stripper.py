"""V1.2 ONNX Q/DQ stripping helpers (ADR-010, Proposed status).

This module is the pure-Python core of the V1.2 mixed-precision path: identify
which Quantize/Dequantize node pairs in a SmoothQuant Q/DQ ONNX belong to which
DINOv3 transformer block, and classify each pair as **internal** (safe to
strip) or **boundary** (must be preserved to keep the surrounding INT8 path).

Why pure-Python (no ``onnx`` import)
====================================
Same pattern as ``layer_precision.py``: the actual ONNX I/O / graph rewrite
happens in a thin driver script that calls ``onnx.load`` / ``onnx_graphsurgeon``
once and feeds the projected node-info records into these pure functions. This
keeps unit tests free of native ONNX dependencies and runnable on macOS.

Scope (V1.2 implementation, not yet shipped)
============================================
This module implements the *identification + classification* layer described
in ADR-010 § 5.1 step 1. The actual ``strip_internal_qdq_pairs(model, ...)``
that mutates the ONNX graph and rewires edges is left for the V1.2
implementation phase — when the project is ready to spend the 0.5-1
engineering-day budget noted in ADR-010 § 7.
"""

from __future__ import annotations

import re
from typing import Iterable, Literal, Mapping, NamedTuple, Optional, Sequence

QDQ_OP_TYPES: tuple[str, ...] = ("QuantizeLinear", "DequantizeLinear")
"""ONNX op types that participate in explicit Q/DQ quantization."""

PairLocation = Literal["internal", "boundary_input", "boundary_output"]
"""Where a Q/DQ pair sits relative to the requested block range.

* ``internal`` — both Q and DQ belong to the requested block range; safe to
  strip when forcing those blocks to a higher precision.
* ``boundary_input`` — Q is outside the range, DQ is inside the first block of
  the range. Must be preserved so the INT8 → fp transition survives.
* ``boundary_output`` — Q is inside the last block of the range, DQ is outside
  it. Must be preserved so the fp → INT8 transition survives.
"""

_BLOCK_RE = re.compile(r"/layer\.(\d+)/")


class OnnxNodeWithEdges(NamedTuple):
    """A single ONNX node projected to (name, op_type, inputs, outputs).

    ``inputs`` and ``outputs`` are tensor names. They reference each other
    across nodes (one node's output is another node's input), which is how we
    detect Q→DQ adjacency.
    """

    name: str
    op_type: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


class QDQPair(NamedTuple):
    """A QuantizeLinear immediately feeding into a DequantizeLinear."""

    quantize_node: str
    dequantize_node: str
    quantize_block: Optional[int]
    dequantize_block: Optional[int]
    location: PairLocation


def parse_block_index(node_name: str) -> Optional[int]:
    """Return the 0-based DINOv3 block index for an ONNX node name, or None."""

    if not isinstance(node_name, str) or not node_name:
        return None
    match = _BLOCK_RE.search(node_name)
    if match is None:
        return None
    return int(match.group(1))


def find_qdq_pairs(nodes: Sequence[OnnxNodeWithEdges]) -> list[tuple[str, str]]:
    """Return all (Quantize → Dequantize) adjacent pairs in the graph.

    A pair is detected when a ``QuantizeLinear`` node's *first* output tensor
    is consumed as the *first* input of a ``DequantizeLinear`` node. Multi-fan
    out edges (rare in DINOv3 export) are skipped — explicit Q/DQ ONNX from
    Model Optimizer / SmoothQuant emits 1:1 Q→DQ chains.
    """

    output_to_q_node: dict[str, str] = {}
    for node in nodes:
        if node.op_type != "QuantizeLinear":
            continue
        if not node.outputs:
            continue
        output_to_q_node[node.outputs[0]] = node.name

    pairs: list[tuple[str, str]] = []
    for node in nodes:
        if node.op_type != "DequantizeLinear":
            continue
        if not node.inputs:
            continue
        producer = output_to_q_node.get(node.inputs[0])
        if producer is not None:
            pairs.append((producer, node.name))
    return pairs


def classify_pair(
    quantize_name: str,
    dequantize_name: str,
    *,
    block_indices: Sequence[int],
) -> Optional[QDQPair]:
    """Classify one Q/DQ pair against a 0-based block range.

    Returns ``None`` if neither node is in the requested block range — those
    pairs are unrelated to the strip operation. Otherwise returns a
    :class:`QDQPair` whose ``location`` is one of ``internal`` /
    ``boundary_input`` / ``boundary_output``.
    """

    if not block_indices:
        raise ValueError("block_indices must be non-empty")
    indices_set = set(int(idx) for idx in block_indices)
    if any(idx < 0 for idx in indices_set):
        raise ValueError("block_indices must be non-negative")

    q_block = parse_block_index(quantize_name)
    dq_block = parse_block_index(dequantize_name)
    q_in_range = q_block is not None and q_block in indices_set
    dq_in_range = dq_block is not None and dq_block in indices_set

    if not q_in_range and not dq_in_range:
        return None

    location: PairLocation
    if q_in_range and dq_in_range:
        location = "internal"
    elif dq_in_range and not q_in_range:
        location = "boundary_input"
    else:
        location = "boundary_output"

    return QDQPair(
        quantize_node=quantize_name,
        dequantize_node=dequantize_name,
        quantize_block=q_block,
        dequantize_block=dq_block,
        location=location,
    )


def find_block_qdq_pairs(
    nodes: Sequence[OnnxNodeWithEdges],
    *,
    block_indices: Sequence[int],
) -> list[QDQPair]:
    """Return every Q/DQ pair that touches the requested block range."""

    pairs = find_qdq_pairs(nodes)
    classified: list[QDQPair] = []
    for q_name, dq_name in pairs:
        pair = classify_pair(q_name, dq_name, block_indices=block_indices)
        if pair is not None:
            classified.append(pair)
    return classified


def split_strippable_and_preserved(
    pairs: Iterable[QDQPair],
) -> tuple[list[QDQPair], list[QDQPair]]:
    """Partition pairs into ``(strippable_internal, preserved_boundary)``.

    Per ADR-010 § 4.3, only ``internal`` pairs can be removed when forcing
    a block range to higher precision. ``boundary_input`` and
    ``boundary_output`` pairs must stay so the surrounding INT8 path keeps its
    precision-conversion nodes.
    """

    strippable: list[QDQPair] = []
    preserved: list[QDQPair] = []
    for pair in pairs:
        if pair.location == "internal":
            strippable.append(pair)
        else:
            preserved.append(pair)
    return strippable, preserved


def summarise_pairs(pairs: Iterable[QDQPair]) -> Mapping[str, int]:
    """Return ``{location: count}`` for a sequence of classified pairs."""

    counts: dict[str, int] = {
        "internal": 0,
        "boundary_input": 0,
        "boundary_output": 0,
    }
    for pair in pairs:
        counts[pair.location] = counts.get(pair.location, 0) + 1
    counts["total"] = sum(
        counts[key] for key in ("internal", "boundary_input", "boundary_output")
    )
    return counts
