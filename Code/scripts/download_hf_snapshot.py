#!/usr/bin/env python
"""Download a Hugging Face repository snapshot into an artifact directory."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dinov3_trt.artifacts import ArtifactLayout  # noqa: E402
from dinov3_trt.contracts import DINO_VITL16_224_CONTRACT  # noqa: E402

DEFAULT_REPO_ID = DINO_VITL16_224_CONTRACT.model_id
DEFAULT_LOCAL_DIR = ArtifactLayout(Path("Artifacts")).weights_dir
DEFAULT_PATTERNS = ("*.safetensors", "config.json", "README.md", "LICENSE*", "*.md")


def parse_patterns(values: Sequence[str] | None) -> list[str]:
    if not values:
        return list(DEFAULT_PATTERNS)
    patterns: list[str] = []
    for value in values:
        patterns.extend(part.strip() for part in value.split(",") if part.strip())
    return patterns


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--include", action="append", default=None)
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT"))
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN") or None)
    args = parser.parse_args()

    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint

    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import GatedRepoError, LocalEntryNotFoundError, RepositoryNotFoundError
    except ImportError as exc:
        print(
            "huggingface_hub is not installed. Install the Code package or run "
            "`python -m pip install huggingface-hub`.",
            file=sys.stderr,
        )
        print(str(exc), file=sys.stderr)
        return 2

    args.local_dir.mkdir(parents=True, exist_ok=True)
    try:
        path = snapshot_download(
            repo_id=str(args.repo_id),
            revision=args.revision,
            local_dir=args.local_dir,
            allow_patterns=parse_patterns(args.include),
            token=args.token,
        )
    except GatedRepoError as exc:
        print(
            "The repository is gated. Accept the model license and run `hf auth login` "
            "on the target machine, or provide HF_TOKEN.",
            file=sys.stderr,
        )
        print(str(exc), file=sys.stderr)
        return 3
    except RepositoryNotFoundError as exc:
        print("Repository not found or not accessible with the current token.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 4
    except LocalEntryNotFoundError as exc:
        print("Could not reach Hugging Face and no matching local snapshot is cached.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 5
    except ImportError as exc:
        print(
            "Hugging Face download could not initialize the HTTP client. "
            "If the environment uses a SOCKS proxy, install the project dependency "
            "`socksio` or rerun `python -m pip install -e .`.",
            file=sys.stderr,
        )
        print(str(exc), file=sys.stderr)
        return 2

    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
