#!/usr/bin/env python
"""Check local DINOv3 source, weights, ONNX, engine, and report artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dinov3_trt.artifacts import (  # noqa: E402
    ArtifactLayout,
    RequiredAsset,
    missing_required_assets,
    scan_assets,
)

ALL_REQUIRED: tuple[RequiredAsset, ...] = (
    "source",
    "weights",
    "onnx",
    "fp32-engine",
    "fp16-engine",
    "reports",
)

SUPPORTED_REQUIRED: tuple[RequiredAsset, ...] = (
    "source",
    "weights",
    "onnx",
    "fp32-engine",
    "fp16-engine",
    "bf16-engine",
    "reports",
)


def parse_required(values: list[str] | None) -> tuple[RequiredAsset, ...]:
    if not values:
        return ()
    required: list[RequiredAsset] = []
    for value in values:
        if value == "all":
            required.extend(ALL_REQUIRED)
            continue
        if value not in SUPPORTED_REQUIRED:
            allowed = ", ".join((*SUPPORTED_REQUIRED, "all"))
            raise ValueError(f"unsupported required asset {value!r}; allowed: {allowed}")
        required.append(value)
    return tuple(dict.fromkeys(required))


def _atomic_write_text(target: Path, payload: str) -> None:
    """Write `payload` to `target` atomically by replacing it from a temp file."""

    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("Artifacts"))
    parser.add_argument(
        "--require",
        action="append",
        default=None,
        help=(
            "Required asset: source, weights, onnx, fp32-engine, fp16-engine, "
            "bf16-engine, reports, or all. The all alias keeps the core formal assets "
            "and does not require BF16."
        ),
    )
    parser.add_argument("--create-dirs", action="store_true")
    parser.add_argument(
        "--with-sha256",
        action="store_true",
        help="Include SHA256 digests in file_info. This can be slow for large ONNX/engine files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output file path for the manifest JSON. When set, the file is "
            "written atomically (temp file + os.replace) and is excluded from the "
            "report scan so the manifest does not record a self-referential 0-byte "
            "entry. When omitted, the JSON is printed to stdout (legacy behavior)."
        ),
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        help=(
            "Additional file paths to exclude from the report scan. Useful when the "
            "calling shell creates several sibling 0-byte sentinel files before this "
            "script runs."
        ),
    )
    args = parser.parse_args()

    layout = ArtifactLayout(args.artifact_root)
    if args.create_dirs:
        layout.create_directories()

    try:
        required = parse_required(args.require)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    exclude_files: list[Path] = []
    if args.output is not None:
        exclude_files.append(args.output)
    if args.exclude:
        exclude_files.extend(Path(item) for item in args.exclude)

    statuses = scan_assets(layout, exclude_files=exclude_files)
    missing = missing_required_assets(layout, required, exclude_files=exclude_files)
    payload = {
        "artifact_root": str(layout.root),
        "assets": {
            name: status.to_json(
                include_file_info=True,
                include_sha256=args.with_sha256,
            )
            for name, status in statuses.items()
        },
        "required": list(required),
        "missing_required": [
            status.to_json(include_file_info=True, include_sha256=args.with_sha256)
            for status in missing
        ],
    }
    serialized = json.dumps(payload, indent=2)
    if args.output is None:
        print(serialized)
    else:
        _atomic_write_text(args.output, serialized + "\n")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
