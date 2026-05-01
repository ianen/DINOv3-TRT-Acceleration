#!/usr/bin/env python
"""Prepare disjoint evaluation and calibration image manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dinov3_trt.infer.image_eval import (  # noqa: E402
    list_image_paths,
    stratified_eval_calib_split,
    write_image_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--eval-output", required=True, type=Path)
    parser.add_argument("--calib-output", required=True, type=Path)
    parser.add_argument("--eval-count", type=int, default=1000)
    parser.add_argument("--calib-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--no-recursive", action="store_true")
    args = parser.parse_args()

    paths = list_image_paths(args.image_root, recursive=not args.no_recursive)
    eval_paths, calib_paths = stratified_eval_calib_split(
        paths,
        image_root=args.image_root,
        eval_count=args.eval_count,
        calib_count=args.calib_count,
        seed=args.seed,
    )
    write_image_manifest(
        args.eval_output,
        image_root=args.image_root,
        images=eval_paths,
        seed=args.seed,
        split="eval",
    )
    write_image_manifest(
        args.calib_output,
        image_root=args.image_root,
        images=calib_paths,
        seed=args.seed,
        split="calib",
    )
    payload = {
        "image_root": str(args.image_root),
        "total_images": len(paths),
        "seed": args.seed,
        "eval_output": str(args.eval_output),
        "eval_count": len(eval_paths),
        "calib_output": str(args.calib_output),
        "calib_count": len(calib_paths),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
