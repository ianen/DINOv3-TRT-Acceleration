"""Helpers for trtexec ``--layerPrecisions`` per-layer override generation.

V1.1 stretch follow-up: SmoothQuant + skip 16-19 mixed-precision via the
``modelopt`` disable_quantizer interface delivered only +0.003 cos at the cost
of ~30% speedup (see `Wiki/0-项目计划/milestones/M1-progress.md`). The expected
fix is to let TensorRT itself force layers 16-19 onto BF16 via ``trtexec
--layerPrecisions=name:bf16,...``. This module builds the long argument value
deterministically from the project's ONNX node naming convention.

DINOv3 HF export emits ONNX nodes named ``/model/layer.{N}/...`` for each of
the 24 transformer blocks; this module parses that block index, filters by an
optional op-type allow-list, and emits the comma-separated trtexec value.

The module is pure-Python (no onnx import) so it stays cheap to test on the
local macOS workspace; the single driver script ``build_layer_precisions_arg
.py`` is the place where ``onnx.load`` is called.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple, Optional, Sequence

SUPPORTED_PRECISIONS: tuple[str, ...] = ("fp32", "fp16", "bf16", "int8", "fp8")
"""Precisions that trtexec ``--layerPrecisions`` accepts in this project."""

DINOV3_NUM_BLOCKS = 24
"""ViT-L/16 has 24 transformer blocks; block indices are 0-based."""

_BLOCK_RE = re.compile(r"/layer\.(\d+)/")

DEFAULT_COMPUTE_OP_TYPES: tuple[str, ...] = (
    "MatMul",
    "Gemm",
    "Add",
    "LayerNormalization",
    "Softmax",
    "Mul",
)
"""Op types that carry the per-block compute load (excludes Constant/Cast/etc).

When forcing a block onto BF16 it is sufficient to override these compute-heavy
op types; the surrounding shape/copy/Q-DQ ops follow naturally.
"""


class OnnxNodeInfo(NamedTuple):
    """Minimal projection of an ONNX node we need (name + op_type)."""

    name: str
    op_type: str


def parse_block_index(node_name: str) -> Optional[int]:
    """Return the 0-based DINOv3 block index for an ONNX node name, or None."""

    if not isinstance(node_name, str) or not node_name:
        return None
    match = _BLOCK_RE.search(node_name)
    if match is None:
        return None
    return int(match.group(1))


def select_block_node_names(
    nodes: Sequence[OnnxNodeInfo],
    *,
    block_indices: Sequence[int],
    op_types: Optional[Sequence[str]] = None,
) -> tuple[str, ...]:
    """Return ONNX node names belonging to the requested block indices.

    ``block_indices`` are 0-based and must be within ``[0, DINOV3_NUM_BLOCKS)``.
    ``op_types=None`` keeps every op type; otherwise only nodes whose
    ``op_type`` appears in the iterable are kept.
    """

    if not block_indices:
        raise ValueError("block_indices must be non-empty")
    indices_set: set[int] = set()
    for raw in block_indices:
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise TypeError("block_indices must be ints")
        if raw < 0 or raw >= DINOV3_NUM_BLOCKS:
            raise ValueError(
                f"block_indices must be within [0, {DINOV3_NUM_BLOCKS}); got {raw}"
            )
        indices_set.add(raw)
    op_filter: Optional[set[str]] = None
    if op_types is not None:
        op_filter = set(op_types)
        if not op_filter:
            raise ValueError("op_types, if provided, must be non-empty")

    selected: list[str] = []
    for node in nodes:
        block = parse_block_index(node.name)
        if block is None or block not in indices_set:
            continue
        if op_filter is not None and node.op_type not in op_filter:
            continue
        selected.append(node.name)
    return tuple(selected)


def build_layer_precisions_arg(
    node_names: Sequence[str],
    precision: str,
) -> str:
    """Construct the trtexec ``--layerPrecisions`` value.

    Format: ``layerName1:precision,layerName2:precision,...`` (comma-separated).
    The caller is responsible for shell-quoting; for very long values use
    `subprocess` with a list argv to bypass cmd.exe's 32 KB command-line limit.
    """

    if precision not in SUPPORTED_PRECISIONS:
        raise ValueError(
            f"unsupported precision '{precision}'; expected one of {SUPPORTED_PRECISIONS}"
        )
    if not node_names:
        raise ValueError("node_names must be non-empty")
    seen: set[str] = set()
    for name in node_names:
        if not isinstance(name, str) or not name:
            raise ValueError("node_names must be non-empty strings")
        if "," in name or ":" in name:
            raise ValueError(
                f"node name '{name}' contains a ',' or ':' separator; trtexec value would be ambiguous"
            )
        if name in seen:
            raise ValueError(f"duplicate node name: {name}")
        seen.add(name)
    return ",".join(f"{name}:{precision}" for name in node_names)


def write_layer_precisions_file(
    path: Path,
    *,
    arg_value: str,
    block_indices: Sequence[int],
    precision: str,
    op_types: Optional[Sequence[str]] = None,
) -> dict[str, object]:
    """Persist the trtexec value alongside a small JSON sidecar for traceability.

    Returns the metadata payload that was written next to the value file.
    """

    if not isinstance(path, Path):
        path = Path(path)
    if precision not in SUPPORTED_PRECISIONS:
        raise ValueError(f"unsupported precision: {precision}")
    if not arg_value:
        raise ValueError("arg_value must be non-empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(arg_value, encoding="utf-8")

    metadata: dict[str, object] = {
        "arg_value_path": str(path),
        "arg_value_chars": len(arg_value),
        "node_count": arg_value.count(",") + 1,
        "precision": precision,
        "block_indices": sorted(set(int(idx) for idx in block_indices)),
        "op_types": sorted(op_types) if op_types else None,
    }
    metadata_path = path.with_suffix(path.suffix + ".meta.json")
    import json

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
