#!/usr/bin/env python
"""Clone or validate the official DINOv3 source tree under Artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dinov3_trt.artifacts import ArtifactLayout  # noqa: E402

DEFAULT_REMOTE_URL = "https://github.com/facebookresearch/dinov3.git"


def has_contents(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def run_command(command: Sequence[str], *, dry_run: bool) -> int:
    if dry_run:
        return 0
    result = subprocess.run(list(command), check=False)
    return int(result.returncode)


def git_text(command: Sequence[str], cwd: Path) -> str:
    result = subprocess.run(
        list(command),
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def clone_command(
    remote_url: str,
    source_dir: Path,
    *,
    revision: str | None,
    full_history: bool,
    git_config_args: Sequence[str] = (),
) -> list[str]:
    command = ["git", *git_config_args, "clone"]
    if not full_history:
        command.extend(["--depth", "1"])
    if revision is not None:
        command.extend(["--branch", revision])
    command.extend([remote_url, str(source_dir)])
    return command


def git_proxy_config_args(*, git_proxy: str | None, disable_git_proxy: bool) -> tuple[str, ...]:
    if git_proxy is not None and disable_git_proxy:
        raise ValueError("--git-proxy and --disable-git-proxy cannot be used together")
    if disable_git_proxy:
        return ("-c", "http.proxy=", "-c", "https.proxy=")
    if git_proxy is not None:
        return ("-c", f"http.proxy={git_proxy}", "-c", f"https.proxy={git_proxy}")
    return ()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("Artifacts"))
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--remote-url", default=DEFAULT_REMOTE_URL)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--full-history", action="store_true")
    parser.add_argument("--git-proxy", default=None)
    parser.add_argument("--disable-git-proxy", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        git_config_args = git_proxy_config_args(
            git_proxy=args.git_proxy,
            disable_git_proxy=args.disable_git_proxy,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    layout = ArtifactLayout(args.artifact_root)
    source_dir = args.source_dir or layout.source_dir
    source_parent = source_dir.parent
    source_parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, object] = {
        "source_dir": str(source_dir),
        "remote_url": args.remote_url,
        "revision": args.revision,
        "git_config_args": list(git_config_args),
        "dry_run": args.dry_run,
    }

    if has_contents(source_dir):
        git_dir = source_dir / ".git"
        if not git_dir.exists():
            payload["status"] = "non_empty_non_git"
            payload["message"] = "source_dir is not empty and is not a git repository"
            print(json.dumps(payload, indent=2))
            return 3

        head = git_text(["git", "rev-parse", "--short", "HEAD"], source_dir)
        payload["status"] = "exists"
        payload["head"] = head
        if args.revision is not None:
            fetch = ["git", *git_config_args, "-C", str(source_dir), "fetch"]
            if not args.full_history:
                fetch.extend(["--depth", "1"])
            fetch.extend(["origin", args.revision])
            checkout = ["git", "-C", str(source_dir), "checkout", "FETCH_HEAD"]
            payload["commands"] = [fetch, checkout]
            print(json.dumps(payload, indent=2))
            if run_command(fetch, dry_run=args.dry_run) != 0:
                return 4
            if run_command(checkout, dry_run=args.dry_run) != 0:
                return 5
        else:
            print(json.dumps(payload, indent=2))
        return 0

    command = clone_command(
        args.remote_url,
        source_dir,
        revision=args.revision,
            full_history=args.full_history,
            git_config_args=git_config_args,
        )
    payload["status"] = "clone"
    payload["commands"] = [command]
    print(json.dumps(payload, indent=2))
    return run_command(command, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
