#!/usr/bin/env python
"""Generate a trtexec ``--layerPrecisions`` value for given DINOv3 block range.

Reads an ONNX model, selects nodes belonging to the requested 0-based block
indices (and optional op-type filter), then writes the comma-separated
``layerName:precision`` value to a UTF-8 text file alongside a small JSON
sidecar with traceable metadata.

The text file content is suitable for ``trtexec --layerPrecisions=$(cat
file.txt)`` when invoked through Python ``subprocess`` (bypasses cmd.exe's
~32 KB command-line cap).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dinov3_trt.quantization.layer_precision import (  # noqa: E402
    DEFAULT_COMPUTE_OP_TYPES,
    OnnxNodeInfo,
    SUPPORTED_PRECISIONS,
    build_layer_precisions_arg,
    select_block_node_names,
    write_layer_precisions_file,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path, help="Source ONNX path.")
    parser.add_argument(
        "--blocks",
        required=True,
        help="0-based block indices, e.g. '16-19' or '16,17,18,19' or '16-19,22'.",
    )
    parser.add_argument(
        "--precision",
        required=True,
        choices=SUPPORTED_PRECISIONS,
        help="Target precision for the selected nodes.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination .txt file for the trtexec value (sidecar metadata is written next to it).",
    )
    parser.add_argument(
        "--op-types",
        default=",".join(DEFAULT_COMPUTE_OP_TYPES),
        help=(
            "Comma-separated ONNX op types to include. Default = compute-heavy ops "
            f"({','.join(DEFAULT_COMPUTE_OP_TYPES)}). Use 'all' to disable the filter."
        ),
    )
    parser.add_argument(
        "--load-external-data",
        action="store_true",
        help="If set, also load external tensor data when reading the ONNX file.",
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


def parse_op_types(value: str) -> tuple[str, ...] | None:
    """Parse the --op-types CLI value: 'all' -> None, else a CSV tuple."""

    cleaned = value.strip()
    if cleaned.lower() == "all":
        return None
    items = tuple(part.strip() for part in cleaned.split(",") if part.strip())
    if not items:
        raise ValueError("--op-types must be 'all' or a non-empty CSV")
    return items


def load_onnx_node_info(path: Path, *, load_external_data: bool) -> tuple[OnnxNodeInfo, ...]:
    import onnx  # type: ignore[import-not-found]  # noqa: PLC0415

    model = onnx.load(str(path), load_external_data=load_external_data)
    return tuple(OnnxNodeInfo(node.name, node.op_type) for node in model.graph.node)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    block_indices = parse_blocks_spec(args.blocks)
    op_types = parse_op_types(args.op_types)
    nodes = load_onnx_node_info(args.onnx, load_external_data=args.load_external_data)
    selected = select_block_node_names(
        nodes,
        block_indices=block_indices,
        op_types=op_types,
    )
    if not selected:
        print(
            f"[error] no nodes matched blocks={list(block_indices)} op_types={op_types}",
            file=sys.stderr,
        )
        return 2
    arg_value = build_layer_precisions_arg(selected, args.precision)
    metadata = write_layer_precisions_file(
        args.output,
        arg_value=arg_value,
        block_indices=block_indices,
        precision=args.precision,
        op_types=op_types,
    )
    print(
        f"wrote {metadata['node_count']} entries ({metadata['arg_value_chars']} chars) -> {args.output}"
    )
    print(f"sidecar: {args.output.with_suffix(args.output.suffix + '.meta.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
