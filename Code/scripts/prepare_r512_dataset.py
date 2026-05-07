#!/usr/bin/env python
"""V1.0.4 ADR-024 — Offline 1024 → 512 dataset resize + SHA256 manifest.

把 ``Artifacts/datasets/good/`` 144 张 1024×1024 印花布 JPG 离线 resize 为
512×512 落盘到 ``Artifacts/datasets/good_r512/``，附 SHA256 manifest 用于
完整性校验。

Resize 算法 = PIL Image.LANCZOS（高质量降采样，保留印花纹理细节避免混叠）。
JPEG quality = 95（保留细节，文件体积可接受）。

resize 一次性完成 — V1.0.4 production_benchmark 后续直接读 r=512 数据集，
不把 resize 算入 inference 时序。

Usage
=====
::

    python Code/scripts/prepare_r512_dataset.py \\
        --input Artifacts/datasets/good \\
        --output Artifacts/datasets/good_r512 \\
        --target-size 512
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

try:
    from PIL import Image
except ImportError as e:
    sys.stderr.write("PIL not installed; pip install Pillow\n")
    raise SystemExit(1) from e


@dataclass(frozen=True)
class ManifestEntry:
    """Per-image manifest entry."""

    file: str
    src_path: str
    src_size: tuple[int, int]
    dst_size: tuple[int, int]
    sha256: str
    bytes_written: int


@dataclass
class ResizeReport:
    """Aggregate report after batch resize."""

    target_size: int
    jpeg_quality: int
    resample: str
    entries: list[ManifestEntry] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input dataset directory (containing *.jpg)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output dataset directory (will be created if absent)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=512,
        help="Target square size (default 512). Must be patch-aligned (multiple of 16) for DINOv3 ViT-L/16.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality (default 95). 90-95 preserves print-fabric detail without bloat.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest JSON path (default: <output>/manifest.json)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files (default: skip if same SHA)",
    )
    parser.add_argument(
        "--ext",
        default=".jpg",
        help="Input file extension to scan (default .jpg, also tries .jpeg .JPG .JPEG)",
    )
    return parser.parse_args(argv)


def iter_input_images(input_dir: Path, ext: str) -> Iterable[Path]:
    """Yield input image paths from dir, sorted by name."""

    patterns = {ext.lower(), ext.upper(), ".jpeg", ".JPEG"} if ext.lower() in {".jpg", ".jpeg"} else {ext}
    seen: set[Path] = set()
    files: list[Path] = []
    for pat in patterns:
        for p in input_dir.glob(f"*{pat}"):
            if p in seen:
                continue
            seen.add(p)
            files.append(p)
    files.sort(key=lambda p: p.name)
    yield from files


def sha256_of_file(path: Path) -> str:
    """Compute SHA256 of a file (chunked, memory-friendly)."""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resize_one(
    src_path: Path,
    dst_path: Path,
    target_size: int,
    jpeg_quality: int,
    overwrite: bool,
) -> tuple[ManifestEntry | None, dict | None]:
    """Resize one image. Returns (manifest_entry, error_dict). error_dict non-None → failed."""

    try:
        with Image.open(src_path) as img:
            src_size = img.size  # (W, H)
            img_rgb = img.convert("RGB")
    except Exception as exc:  # noqa: BLE001
        return None, {"file": src_path.name, "stage": "open", "error": str(exc)}

    if dst_path.exists() and not overwrite:
        # Skip if existing file same target size
        try:
            with Image.open(dst_path) as exist:
                if exist.size == (target_size, target_size):
                    sha = sha256_of_file(dst_path)
                    return (
                        ManifestEntry(
                            file=dst_path.name,
                            src_path=str(src_path),
                            src_size=src_size,
                            dst_size=(target_size, target_size),
                            sha256=sha,
                            bytes_written=dst_path.stat().st_size,
                        ),
                        None,
                    )
        except Exception:
            pass  # corrupt existing → re-resize below

    try:
        resized = img_rgb.resize((target_size, target_size), Image.LANCZOS)
    except Exception as exc:  # noqa: BLE001
        return None, {"file": src_path.name, "stage": "resize", "error": str(exc)}

    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        resized.save(dst_path, "JPEG", quality=jpeg_quality, optimize=True)
    except Exception as exc:  # noqa: BLE001
        return None, {"file": src_path.name, "stage": "save", "error": str(exc)}

    sha = sha256_of_file(dst_path)
    return (
        ManifestEntry(
            file=dst_path.name,
            src_path=str(src_path),
            src_size=src_size,
            dst_size=(target_size, target_size),
            sha256=sha,
            bytes_written=dst_path.stat().st_size,
        ),
        None,
    )


def write_manifest(report: ResizeReport, manifest_path: Path) -> None:
    """Atomic write manifest JSON (tempfile + os.replace)."""

    payload = {
        "target_size": report.target_size,
        "jpeg_quality": report.jpeg_quality,
        "resample": report.resample,
        "image_count": len(report.entries),
        "skipped_count": len(report.skipped),
        "error_count": len(report.errors),
        "entries": [
            {
                "file": e.file,
                "src_path": e.src_path,
                "src_size": list(e.src_size),
                "dst_size": list(e.dst_size),
                "sha256": e.sha256,
                "bytes_written": e.bytes_written,
            }
            for e in report.entries
        ],
        "skipped": report.skipped,
        "errors": report.errors,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(manifest_path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.target_size % 16 != 0:
        sys.stderr.write(
            f"warning: target-size {args.target_size} is not patch-aligned (multiple of 16). "
            f"DINOv3 ViT-L/16 will floor to {(args.target_size // 16) * 16}.\n"
        )

    if not args.input.is_dir():
        sys.stderr.write(f"error: input directory does not exist: {args.input}\n")
        return 1

    input_files = list(iter_input_images(args.input, args.ext))
    if not input_files:
        sys.stderr.write(f"error: no images found in {args.input} (ext={args.ext})\n")
        return 1

    print(f"[prepare_r512_dataset] input_dir = {args.input}")
    print(f"[prepare_r512_dataset] output_dir = {args.output}")
    print(f"[prepare_r512_dataset] target_size = {args.target_size}x{args.target_size}")
    print(f"[prepare_r512_dataset] jpeg_quality = {args.jpeg_quality}")
    print(f"[prepare_r512_dataset] images = {len(input_files)}")

    args.output.mkdir(parents=True, exist_ok=True)

    report = ResizeReport(
        target_size=args.target_size,
        jpeg_quality=args.jpeg_quality,
        resample="LANCZOS",
    )

    for idx, src_path in enumerate(input_files, start=1):
        dst_path = args.output / src_path.name
        entry, err = resize_one(
            src_path,
            dst_path,
            args.target_size,
            args.jpeg_quality,
            args.overwrite,
        )
        if err is not None:
            report.errors.append(err)
            sys.stderr.write(f"  [{idx:3d}/{len(input_files)}] FAIL {src_path.name}: {err['error']}\n")
            continue
        assert entry is not None
        report.entries.append(entry)
        if idx == 1 or idx % 20 == 0 or idx == len(input_files):
            print(
                f"  [{idx:3d}/{len(input_files)}] {src_path.name} "
                f"{entry.src_size[0]}x{entry.src_size[1]} → "
                f"{entry.dst_size[0]}x{entry.dst_size[1]} "
                f"({entry.bytes_written // 1024} KB)"
            )

    manifest_path = args.manifest or (args.output / "manifest.json")
    write_manifest(report, manifest_path)
    print(f"[prepare_r512_dataset] manifest → {manifest_path}")
    print(
        f"[prepare_r512_dataset] done: "
        f"{len(report.entries)} ok, {len(report.skipped)} skipped, {len(report.errors)} errors"
    )

    return 0 if not report.errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
