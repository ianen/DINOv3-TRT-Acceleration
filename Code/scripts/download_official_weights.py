#!/usr/bin/env python
"""Download official source-compatible DINOv3 ViT-L/16 LVD-1689M weights."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dinov3_trt.artifacts import ArtifactLayout  # noqa: E402

DEFAULT_URL = (
    "https://dl.fbaipublicfiles.com/dinov3/dinov3_vitl16/"
    "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
)
DEFAULT_FILENAME = "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"


def download_file(url: str, output_path: Path, *, overwrite: bool, timeout: float) -> dict[str, object]:
    if output_path.exists() and not overwrite:
        return {
            "status": "exists",
            "url": url,
            "output_path": str(output_path),
            "bytes": output_path.stat().st_size,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "dinov3-trt-acceleration/0.1"})
    bytes_written = 0
    with urllib.request.urlopen(request, timeout=timeout) as response:
        with temp_path.open("wb") as handle:
            while True:
                chunk = response.read(16 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                bytes_written += len(chunk)
    temp_path.replace(output_path)
    return {
        "status": "downloaded",
        "url": url,
        "output_path": str(output_path),
        "bytes": bytes_written,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("Artifacts"))
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    layout = ArtifactLayout(args.artifact_root)
    output_path = args.output or (layout.weights_dir / DEFAULT_FILENAME)
    try:
        payload = download_file(args.url, output_path, overwrite=args.overwrite, timeout=args.timeout)
    except Exception as exc:
        print(f"official weight download failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
