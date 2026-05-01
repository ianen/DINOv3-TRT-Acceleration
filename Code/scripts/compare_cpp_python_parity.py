#!/usr/bin/env python
"""Compare Python TensorRT runtime outputs with the C++ runtime for one engine."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dinov3_trt.infer.cpp_parity import build_cpp_python_parity_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument(
        "--cpp-runner",
        type=Path,
        default=Path("build") / "cpp-trt-inspect-msvc" / "dinov3_trt_dump_outputs.exe",
    )
    parser.add_argument(
        "--dump-dir",
        type=Path,
        default=Path("Artifacts") / "reports" / "cpp_python_parity_dump",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-name", default="pixel_values")
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help=(
            "Optional input image size to forward to the C++ dumper via "
            "--image-size N. When omitted, the C++ tool default (224) is used."
        ),
    )
    parser.add_argument(
        "--reuse-dump",
        action="store_true",
        help="Do not invoke the C++ runner; read an existing dump manifest instead.",
    )
    args = parser.parse_args()

    manifest_path = args.dump_dir / "manifest.json"
    if not args.reuse_dump:
        args.dump_dir.mkdir(parents=True, exist_ok=True)
        cmd: list[str] = [
            str(args.cpp_runner),
            str(args.engine),
            str(args.dump_dir),
            str(args.batch_size),
        ]
        if args.image_size is not None:
            cmd.extend(["--image-size", str(args.image_size)])
        subprocess.run(cmd, check=True)

    report = build_cpp_python_parity_report(
        engine_path=args.engine,
        cpp_manifest_path=manifest_path,
        input_name=args.input_name,
    )
    report["cpp_runner"] = str(args.cpp_runner)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

