#!/usr/bin/env python
"""Split a large artifact and emit a sequential reverse-HTTP PowerShell downloader."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dinov3_trt.remote_transfer import (  # noqa: E402
    DEFAULT_PART_PREFIX,
    DEFAULT_PART_WIDTH,
    render_sequential_http_downloader,
    render_windows_part_merger,
    split_file,
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--parts-dir", required=True, type=Path)
    parser.add_argument("--script-output", required=True, type=Path)
    parser.add_argument("--merge-script-output", type=Path, default=None)
    parser.add_argument("--remote-dir", required=True)
    parser.add_argument("--remote-output", default=None)
    parser.add_argument("--expected-sha256", default=None)
    parser.add_argument("--base-url", default="http://127.0.0.1:18765")
    parser.add_argument("--chunk-size-mib", type=positive_int, default=8)
    parser.add_argument("--prefix", default=DEFAULT_PART_PREFIX)
    parser.add_argument("--width", type=positive_int, default=DEFAULT_PART_WIDTH)
    parser.add_argument("--curl-max-time-seconds", type=positive_int, default=600)
    args = parser.parse_args()

    chunk_size = args.chunk_size_mib * 1024 * 1024
    parts = split_file(
        args.input,
        args.parts_dir,
        chunk_size=chunk_size,
        prefix=args.prefix,
        width=args.width,
    )
    script = render_sequential_http_downloader(
        base_url=args.base_url,
        remote_dir=args.remote_dir,
        total_size=args.input.stat().st_size,
        chunk_size=chunk_size,
        prefix=args.prefix,
        width=args.width,
        curl_max_time_seconds=args.curl_max_time_seconds,
    )
    args.script_output.parent.mkdir(parents=True, exist_ok=True)
    args.script_output.write_text(script, encoding="utf-8")

    if args.merge_script_output is not None:
        if args.remote_output is None:
            parser.error("--remote-output is required when --merge-script-output is set")
        if args.expected_sha256 is None:
            parser.error("--expected-sha256 is required when --merge-script-output is set")
        merge_script = render_windows_part_merger(
            parts_dir=args.remote_dir,
            output_path=args.remote_output,
            total_size=args.input.stat().st_size,
            chunk_size=chunk_size,
            expected_sha256=args.expected_sha256,
            prefix=args.prefix,
            width=args.width,
        )
        args.merge_script_output.parent.mkdir(parents=True, exist_ok=True)
        args.merge_script_output.write_text(merge_script, encoding="utf-8")

    payload = {
        "input": str(args.input),
        "parts_dir": str(args.parts_dir),
        "script_output": str(args.script_output),
        "total_size_bytes": args.input.stat().st_size,
        "chunk_size_bytes": chunk_size,
        "part_count": len(parts),
        "merge_script_output": None
        if args.merge_script_output is None
        else str(args.merge_script_output),
        "parts": [part.to_json() for part in parts],
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
