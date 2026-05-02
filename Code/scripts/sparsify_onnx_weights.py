#!/usr/bin/env python
"""Apply 2:4 structured sparsity masks to ONNX weight initializers.

Loads a DINOv3 ViT-L/16 ONNX, identifies attention QKV / proj and MLP
fc1/fc2 weight initializers via configurable name patterns, computes 2:4
masks via :mod:`dinov3_trt.sparsity.sparsify`, applies the masks
(zero-filling pruned positions), and saves a new ONNX where the same
graph topology is preserved but the marked weight tensors are 50% sparse
along the input-channel dimension.

The output ONNX is then a drop-in replacement for the dense one when
building a TensorRT engine with ``--sparsity=enable``: TRT's sparse
Tensor Core kernel selector recognises the 2:4 pattern and dispatches
the halved-bandwidth GEMM path.

Usage
=====
::

    .venv/bin/python scripts/sparsify_onnx_weights.py \\
        --input  Artifacts/onnx/dinov3_vitl16_4out.onnx \\
        --output Artifacts/onnx/dinov3_vitl16_4out.sparse2to4.onnx \\
        --layer-pattern "blocks\\.\\d+\\.attn\\.qkv\\.weight" \\
        --layer-pattern "blocks\\.\\d+\\.attn\\.proj\\.weight" \\
        --layer-pattern "blocks\\.\\d+\\.mlp\\.fc1\\.weight" \\
        --layer-pattern "blocks\\.\\d+\\.mlp\\.fc2\\.weight"

Per-layer ablation
==================
For ADR-016 selective sparsity validation, pass ``--layer-pattern`` with a
single block index (e.g. ``blocks\\.0\\..*\\.weight``) and run cosine eval
to identify which layers tolerate the mask without dropping cos_min below
0.99. Then assemble the final ONNX with the union of tolerant layers.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dinov3_trt.sparsity.sparsify import (  # noqa: E402
    SparsityPattern,
    apply_2to4_mask,
    is_2to4_compatible,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, type=Path, help="Source dense ONNX path"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Destination sparse ONNX path"
    )
    parser.add_argument(
        "--layer-pattern",
        action="append",
        required=True,
        help=(
            "Regex matched against initializer names. Each matching weight "
            "tensor receives a 2:4 mask along axis -1. Repeatable."
        ),
    )
    parser.add_argument(
        "--axis",
        type=int,
        default=-1,
        help=(
            "Axis along which to apply the 2:4 group constraint. Default -1 "
            "= last axis (input-channel for [out, in] matmul weights)."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help=(
            "Optional JSON report path listing every modified tensor with its "
            "shape, total params, sparsified params, and energy retention "
            "ratio (Frobenius norm post / pre)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Identify matching tensors, compute masks in-memory, and print "
            "the would-modify list without writing the output ONNX."
        ),
    )
    return parser.parse_args(argv)


def find_matching_initializers(
    initializer_names: list[str], patterns: list[str]
) -> list[str]:
    """Return the subset of ``initializer_names`` matched by any pattern.

    Each pattern is anchored implicitly (``re.search`` semantics) so partial
    matches are supported, mirroring the project's existing layer-name
    convention (``blocks.0.attn.qkv.weight``).
    """
    compiled = [re.compile(p) for p in patterns]
    return [name for name in initializer_names if any(c.search(name) for c in compiled)]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # ``onnx`` is a heavyweight native dependency present on the remote build
    # host but optional on developer workstations; deferred import keeps the
    # CLI parser usable for ``--help`` even without onnx installed.
    try:
        import onnx  # type: ignore[import-not-found]
        from onnx import numpy_helper
    except ImportError as exc:
        raise SystemExit(
            "onnx package is required; install via 'pip install onnx' on the "
            "build host (already present in the project venv on Windows remote)."
        ) from exc

    print(f"[load] {args.input}", flush=True)
    model = onnx.load(str(args.input))
    initializers = list(model.graph.initializer)
    init_names = [t.name for t in initializers]
    matched = find_matching_initializers(init_names, args.layer_pattern)
    print(
        f"[match] {len(matched)} / {len(init_names)} initializers match patterns",
        flush=True,
    )

    pattern = SparsityPattern(axis=args.axis)
    report_entries: list[dict[str, object]] = []
    name_to_tensor = {t.name: t for t in initializers}

    for name in matched:
        tensor = name_to_tensor[name]
        weight = numpy_helper.to_array(tensor)
        if not is_2to4_compatible(weight.shape, axis=args.axis):
            print(
                f"[skip] {name} shape={weight.shape} not 2:4-compatible (last dim not div 4)",
                flush=True,
            )
            continue
        original_norm = float((weight.astype("float64") ** 2).sum() ** 0.5)
        masked = apply_2to4_mask(weight, pattern=pattern)
        masked_norm = float((masked.astype("float64") ** 2).sum() ** 0.5)
        retention = masked_norm / original_norm if original_norm > 0 else 1.0
        nonzero_post = int((masked != 0).sum())

        if not args.dry_run:
            new_tensor = numpy_helper.from_array(masked, name=tensor.name)
            tensor.CopyFrom(new_tensor)

        report_entries.append(
            {
                "name": name,
                "shape": list(weight.shape),
                "total_params": int(weight.size),
                "nonzero_after_mask": nonzero_post,
                "density": nonzero_post / weight.size,
                "frobenius_retention": retention,
            }
        )
        print(
            f"[mask]  {name:60s} shape={weight.shape} retention={retention:.4f}",
            flush=True,
        )

    if not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        print(f"[save] {args.output}", flush=True)
        onnx.save(model, str(args.output))

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "input_onnx": str(args.input),
            "output_onnx": str(args.output) if not args.dry_run else None,
            "patterns": list(args.layer_pattern),
            "axis": args.axis,
            "matched_count": len(matched),
            "modified_count": len(report_entries),
            "tensors": report_entries,
        }
        args.report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[report] {args.report}", flush=True)

    print(f"[done] modified {len(report_entries)} tensors", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
