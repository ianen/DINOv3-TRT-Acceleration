#!/usr/bin/env python
"""Export Hugging Face ImageNet parquet shards into an ImageNet-style image tree."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dinov3_trt.datasets.hf_imagenet_parquet import (  # noqa: E402
    export_rows_to_image_tree,
    iter_parquet_rows,
)
from dinov3_trt.infer.image_eval import write_image_manifest  # noqa: E402


def _resolve_parquet_paths(args: argparse.Namespace) -> tuple[Path, ...]:
    paths: list[Path] = []
    if args.parquet:
        paths.extend(args.parquet)
    if args.parquet_dir is not None:
        paths.extend(sorted(args.parquet_dir.glob(args.glob)))
    if not paths:
        raise ValueError("provide --parquet or --parquet-dir")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing parquet shard(s): {', '.join(missing)}")
    return tuple(paths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", action="append", type=Path, default=[])
    parser.add_argument("--parquet-dir", type=Path)
    parser.add_argument("--glob", default="validation-*.parquet")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    parquet_paths = _resolve_parquet_paths(args)
    rows = iter_parquet_rows(
        parquet_paths,
        image_column=args.image_column,
        label_column=args.label_column,
        batch_size=args.batch_size,
    )
    result = export_rows_to_image_tree(
        rows,
        output_root=args.output_root,
        split=args.split,
        image_column=args.image_column,
        label_column=args.label_column,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    if args.manifest_output is not None:
        write_image_manifest(
            args.manifest_output,
            image_root=args.output_root,
            images=result.paths,
            seed=args.seed,
            split=args.split,
        )

    payload = {
        "parquet_shards": [str(path) for path in parquet_paths],
        "output_root": str(result.output_root),
        "image_count": result.image_count,
        "manifest_output": None if args.manifest_output is None else str(args.manifest_output),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

