#!/usr/bin/env python
"""Download ImageNet val 50K via Kaggle as workaround for HF 403 GatedRepoError.

Background
==========
HuggingFace Hub `ILSVRC/imagenet-1k` is gated and inaccessible from the project's
Windows + RTX 5080 work-station via the cpolar SSH tunnel (verified in round 48:
`Test-NetConnection huggingface.co:443` returns False; the hf-mirror.com mirror
inherits the same gating with HTTP 403 on the LFS download path). Kaggle.com is
reachable from the same network (verified TCP 443 connect = True), making
Kaggle a viable alternative for the standard ILSVRC2012 validation split.

Multiple Kaggle datasets contain ImageNet val 50K. This script supports the
most reliable mirror by default and exposes the dataset slug as a CLI flag.

User-side prerequisites (one-time setup)
========================================
Kaggle now ships two API token formats; both are accepted:

* **New (KGAT)** — Settings -> API -> Create New Token shows a single
  ``KGAT_*`` string and the helper command
  ``echo KGAT_xxx > ~/.kaggle/access_token``.
* **Legacy (kaggle.json)** — older accounts still receive a downloaded
  ``kaggle.json`` containing ``{"username": ..., "key": ...}``.

Place either file at one of:
    Linux/macOS: ``~/.kaggle/access_token`` *or* ``~/.kaggle/kaggle.json``
    Windows:     ``C:\\Users\\USER\\.kaggle\\access_token`` *or* ``kaggle.json``

Then ``chmod 600`` on Linux/macOS (Windows ACL default is fine).

Some Kaggle datasets require accepting their terms once via the web UI
before API download will succeed.

Usage
=====
::

    python scripts/download_imagenet_val_via_kaggle.py \\
        --kaggle-dataset titericz/imagenet1k-val \\
        --output-dir Artifacts/datasets/imagenet_val_kaggle

The script then:
    1. verifies kaggle.json is readable and authenticates
    2. downloads the dataset zip into the target directory
    3. unpacks images
    4. emits a manifest JSON listing all extracted JPEG paths
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kaggle-dataset",
        default="titericz/imagenet1k-val",
        help=(
            "Kaggle dataset slug. Default: 'titericz/imagenet1k-val' "
            "(50,000 ILSVRC2012 val images). Other known mirrors include "
            "'tusonggao/imagenet-validation-dataset' (alternative 50K val) "
            "and 'lijiyu/imagenet' (full train+val) — pick the smallest that "
            "covers val if download size is a concern."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Artifacts") / "datasets" / "imagenet_val_kaggle",
        help="Destination directory for the unpacked images.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=None,
        help="Optional manifest JSON path. Default: <output-dir>/manifest.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Verify kaggle.json + Kaggle API authentication and print the planned "
            "command without performing the download. Useful for verifying setup "
            "on a fresh machine."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the output directory already contains files.",
    )
    return parser.parse_args(argv)


CREDENTIAL_FILENAMES: tuple[str, ...] = ("access_token", "kaggle.json")


def find_kaggle_credentials() -> Path | None:
    """Return the path to a Kaggle credential file in standard locations.

    Detects either the new ``access_token`` (single-string KGAT_* PAT) or the
    legacy ``kaggle.json`` (username+key JSON). Search order: new format first
    (current Kaggle UI default), then legacy. Honors ``KAGGLE_CONFIG_DIR``
    when set, falling back to ``~/.kaggle``.
    """

    search_dirs: list[Path] = []
    env_dir = os.environ.get("KAGGLE_CONFIG_DIR")
    if env_dir:
        search_dirs.append(Path(env_dir))
    search_dirs.append(Path.home() / ".kaggle")

    for directory in search_dirs:
        for filename in CREDENTIAL_FILENAMES:
            candidate = directory / filename
            if candidate.is_file():
                return candidate
    return None


def kaggle_setup_instructions() -> str:
    return (
        "Kaggle credentials not found.\n"
        "\n"
        "  1. Sign up at https://www.kaggle.com (free).\n"
        "  2. Go to Settings -> 'API' -> 'Create New Token'.\n"
        "     Current UI shows a single KGAT_* string and the helper command:\n"
        "         echo KGAT_xxx > ~/.kaggle/access_token\n"
        "         chmod 600 ~/.kaggle/access_token\n"
        "     Older accounts may instead receive a downloaded kaggle.json.\n"
        "  3. Place either file at:\n"
        "         Windows: C:\\Users\\USER\\.kaggle\\{access_token,kaggle.json}\n"
        "         Linux/macOS: ~/.kaggle/{access_token,kaggle.json}\n"
        "  4. Re-run this script.\n"
        "\n"
        "The script does not transmit your token outside your machine."
    )


def authenticate_kaggle_api() -> object:
    """Authenticate the Kaggle Python API. Returns the api object or raises."""

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore[import-not-found]  # noqa: PLC0415
    except ImportError as exc:
        raise SystemExit(
            "kaggle Python package is not installed. Run "
            "`python -m pip install kaggle` and try again."
        ) from exc

    api = KaggleApi()
    try:
        api.authenticate()
    except OSError as exc:
        raise SystemExit(f"{kaggle_setup_instructions()}\n\nUnderlying error: {exc}") from exc
    return api


def perform_download(
    api: object,
    *,
    dataset: str,
    output_dir: Path,
    force: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not force and any(output_dir.iterdir()):
        print(f"[download] {output_dir} already non-empty; pass --force to re-download.")
        return

    print(f"[download] kaggle datasets download -d {dataset} -p {output_dir}")
    api.dataset_download_files(  # type: ignore[attr-defined]
        dataset=dataset,
        path=str(output_dir),
        force=force,
        quiet=False,
        unzip=False,
    )


def unpack_zip_archives(output_dir: Path) -> list[Path]:
    """Unzip every .zip in output_dir; return list of extracted directories."""

    extracted_roots: list[Path] = []
    for archive in sorted(output_dir.glob("*.zip")):
        target = output_dir / archive.stem
        target.mkdir(exist_ok=True)
        print(f"[unpack] {archive.name} -> {target}")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
        extracted_roots.append(target)
    return extracted_roots


def write_manifest(
    output_dir: Path, manifest_path: Path | None, dataset: str
) -> Path:
    manifest_path = manifest_path or (output_dir / "manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in output_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    payload = {
        "kaggle_dataset": dataset,
        "image_root": str(output_dir),
        "image_count": len(images),
        "images": [str(p.relative_to(output_dir)) for p in images],
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[manifest] wrote {manifest_path} (image_count={len(images)})")
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    creds = find_kaggle_credentials()
    if creds is None:
        print(kaggle_setup_instructions(), file=sys.stderr)
        return 2
    print(f"[auth] credential file located at: {creds}")

    if args.dry_run:
        # Authenticate to verify the token is valid, then exit.
        api = authenticate_kaggle_api()
        del api  # noqa: F841 — only used for credential validation
        print(
            f"[dry-run] would download dataset='{args.kaggle_dataset}' to "
            f"'{args.output_dir}', then unpack zip(s) and emit manifest at "
            f"'{args.manifest_output or (args.output_dir / 'manifest.json')}'."
        )
        return 0

    api = authenticate_kaggle_api()
    perform_download(
        api,
        dataset=args.kaggle_dataset,
        output_dir=args.output_dir,
        force=args.force,
    )
    extracted = unpack_zip_archives(args.output_dir)
    if not extracted:
        print(
            "[unpack] no .zip archives found; the dataset may be already unpacked.",
            file=sys.stderr,
        )
    write_manifest(args.output_dir, args.manifest_output, args.kaggle_dataset)

    free_bytes = shutil.disk_usage(args.output_dir).free
    print(f"[done] disk free remaining: {free_bytes / (1024**3):.1f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
