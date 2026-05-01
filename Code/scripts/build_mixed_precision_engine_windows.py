#!/usr/bin/env python
"""Build a mixed-precision TensorRT engine via trtexec --layerPrecisions on Windows.

Reads a ``--layerPrecisions`` value file (produced by
``build_layer_precisions_arg.py``) and invokes trtexec via Python subprocess so
the long argument bypasses cmd.exe's command-line length cap (~32 KB).

Default flow targets the SmoothQuant α=0.8 ONNX with layers 16-19 forced to
BF16, but every flag is parameterised.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--timing-cache", type=Path)
    parser.add_argument("--layer-precisions-file", required=True, type=Path)
    parser.add_argument(
        "--precision-constraints",
        choices=("prefer", "obey"),
        default="obey",
    )
    parser.add_argument(
        "--enable-int8",
        action="store_true",
        help="Pass --int8 to trtexec (SmoothQuant Q/DQ ONNX needs this).",
    )
    parser.add_argument(
        "--enable-bf16",
        action="store_true",
        help="Pass --bf16 to trtexec (mixed-precision target).",
    )
    parser.add_argument("--workspace-gb", type=int, default=4)
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=8)
    parser.add_argument("--max-batch", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--profiling-verbosity",
        default="layer_names_only",
        choices=("layer_names_only", "detailed", "none"),
    )
    parser.add_argument(
        "--trtexec",
        default=r"C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT-10.13.2.6\bin\trtexec.exe",
    )
    parser.add_argument(
        "--no-skip-inference",
        action="store_true",
        help="Run a smoke benchmark after build (default: --skipInference).",
    )
    parser.add_argument(
        "--log",
        type=Path,
        help="Optional path to capture stdout+stderr (truncated trtexec output).",
    )
    return parser.parse_args(argv)


def build_argv(args: argparse.Namespace, layer_precisions_value: str) -> list[str]:
    if not args.enable_int8 and not args.enable_bf16:
        raise SystemExit("at least one of --enable-int8 / --enable-bf16 is required")
    if not (1 <= args.min_batch <= args.opt_batch <= args.max_batch):
        raise SystemExit("require 1 <= min-batch <= opt-batch <= max-batch")
    profile_template = "pixel_values:{batch}x3x{img}x{img}"
    image_size = int(args.image_size)
    cmd: list[str] = [
        args.trtexec,
        f"--onnx={args.onnx}",
        f"--saveEngine={args.engine}",
        f"--minShapes={profile_template.format(batch=args.min_batch, img=image_size)}",
        f"--optShapes={profile_template.format(batch=args.opt_batch, img=image_size)}",
        f"--maxShapes={profile_template.format(batch=args.max_batch, img=image_size)}",
        f"--memPoolSize=workspace:{args.workspace_gb}G",
        f"--profilingVerbosity={args.profiling_verbosity}",
        "--noTF32",
        "--verbose",
    ]
    if args.enable_int8:
        cmd.append("--int8")
    if args.enable_bf16:
        cmd.append("--bf16")
    cmd.append(f"--precisionConstraints={args.precision_constraints}")
    cmd.append(f"--layerPrecisions={layer_precisions_value}")
    if args.timing_cache is not None:
        cmd.append(f"--timingCacheFile={args.timing_cache}")
    if not args.no_skip_inference:
        cmd.append("--skipInference")
    return cmd


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    layer_precisions_value = args.layer_precisions_file.read_text(encoding="utf-8").strip()
    if not layer_precisions_value:
        raise SystemExit(f"layer-precisions file is empty: {args.layer_precisions_file}")

    cmd = build_argv(args, layer_precisions_value)
    args.engine.parent.mkdir(parents=True, exist_ok=True)
    if args.log is not None:
        args.log.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[build] trtexec layer_precisions chars={len(layer_precisions_value)} "
        f"node_count={layer_precisions_value.count(',') + 1}"
    )
    print("[build] argv (excluding layerPrecisions):")
    for part in cmd:
        if part.startswith("--layerPrecisions="):
            print(f"  --layerPrecisions=<{len(layer_precisions_value)} chars omitted>")
        else:
            print(f"  {part}")

    start = time.perf_counter()
    process = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ},
        check=False,
    )
    elapsed = time.perf_counter() - start

    print(f"[build] returncode={process.returncode} elapsed_s={elapsed:.2f}")
    if args.log is not None:
        args.log.write_text(
            (process.stdout or "") + "\n--- STDERR ---\n" + (process.stderr or ""),
            encoding="utf-8",
        )
        print(f"[build] log -> {args.log}")
    if process.returncode != 0:
        last_lines = (process.stderr or process.stdout or "").splitlines()[-30:]
        print("[build] last stderr lines:")
        for line in last_lines:
            print(f"  {line}")
    return int(process.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
