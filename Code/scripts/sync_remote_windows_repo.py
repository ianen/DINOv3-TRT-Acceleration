#!/usr/bin/env python
"""Sync the current source workspace to the Windows RTX 5080 host over SSH/SCP.

Default direction is local -> remote (push the source tree). When invoked with
``--pull-reports`` the script instead pulls ``Artifacts/reports/`` (text-only:
``.json/.md/.csv/.svg/.png/.jpg/.log/.txt``) from the remote host back to the
local workspace, which is useful when the remote is the only place that runs
the actual benchmarks but the local checkout still needs the freshly generated
SVG / matrix / manifest artefacts.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dinov3_trt.remote_sync import (  # noqa: E402
    DEFAULT_REPORT_INCLUDE_EXTENSIONS,
    DEFAULT_SYNC_ITEMS,
    build_remote_cleanup_powershell,
    build_remote_pack_powershell,
    build_remote_powershell,
    create_zip_archive,
    extract_pulled_archive,
    windows_join,
)


DEFAULT_PROJECT_DIR = r"D:\WorkPlace\ZMP\DINOv3-TRT-Acceleration"
DEFAULT_REMOTE_REPORTS_DIR = r"D:\WorkPlace\ZMP\DINOv3-TRT-Acceleration\Code\Artifacts\reports"
DEFAULT_REMOTE_PULL_ARCHIVE = r"D:\WorkPlace\ZMP\DINOv3-TRT-Acceleration\dinov3-reports-pull.zip"


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def powershell_encoded_command(command: str) -> str:
    return base64.b64encode(command.encode("utf-16le")).decode("ascii")


def _build_ssh_command(host: str, powershell_script: str) -> list[str]:
    return [
        "ssh",
        host,
        "powershell",
        "-NoProfile",
        "-EncodedCommand",
        powershell_encoded_command(powershell_script),
    ]


def push_repo(args: argparse.Namespace, repo_root: Path) -> int:
    with tempfile.TemporaryDirectory(prefix="dinov3-sync-") as temp_dir:
        archive_path = Path(temp_dir) / args.remote_zip_name
        manifest = create_zip_archive(repo_root, archive_path, items=DEFAULT_SYNC_ITEMS)
        remote_zip_path = windows_join(args.project_dir, args.remote_zip_name)
        powershell = build_remote_powershell(
            remote_zip_path=remote_zip_path,
            project_dir=args.project_dir,
            init_git=not args.no_git_init,
        )

        payload = {
            "direction": "push",
            "repo_root": str(repo_root),
            "archive_path": str(archive_path),
            "archive_size_bytes": archive_path.stat().st_size,
            "manifest_count": len(manifest),
            "manifest_preview": manifest[:20],
            "scp_command": ["scp", str(archive_path), f"{args.host}:{remote_zip_path}"],
            "ssh_command": _build_ssh_command(args.host, powershell),
        }
        if args.dry_run:
            print(json.dumps(payload, indent=2))
            return 0

        scp_result = run_command(payload["scp_command"])
        ssh_result = run_command(payload["ssh_command"])
        payload["scp_result"] = {
            "returncode": scp_result.returncode,
            "stdout": scp_result.stdout.strip(),
            "stderr": scp_result.stderr.strip(),
        }
        payload["ssh_result"] = {
            "returncode": ssh_result.returncode,
            "stdout": ssh_result.stdout.strip(),
            "stderr": ssh_result.stderr.strip(),
        }
        print(json.dumps(payload, indent=2))
        return scp_result.returncode or ssh_result.returncode


def pull_reports(args: argparse.Namespace, repo_root: Path) -> int:
    pack_command = build_remote_pack_powershell(
        remote_source_dir=args.remote_reports_dir,
        remote_archive_path=args.remote_pull_archive,
        include_extensions=DEFAULT_REPORT_INCLUDE_EXTENSIONS,
    )
    cleanup_command = build_remote_cleanup_powershell(args.remote_pull_archive)
    local_destination = (repo_root / args.local_reports_dir).resolve()

    pack_ssh_command = _build_ssh_command(args.host, pack_command)
    cleanup_ssh_command = _build_ssh_command(args.host, cleanup_command)

    payload: dict[str, object] = {
        "direction": "pull-reports",
        "remote_reports_dir": args.remote_reports_dir,
        "remote_pull_archive": args.remote_pull_archive,
        "local_destination": str(local_destination),
        "include_extensions": list(DEFAULT_REPORT_INCLUDE_EXTENSIONS),
        "pack_ssh_command": pack_ssh_command,
        "cleanup_ssh_command": cleanup_ssh_command,
    }

    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return 0

    pack_result = run_command(pack_ssh_command)
    payload["pack_result"] = {
        "returncode": pack_result.returncode,
        "stdout": pack_result.stdout.strip(),
        "stderr": pack_result.stderr.strip()[-400:],
    }
    if pack_result.returncode != 0 or "PACK_OK" not in pack_result.stdout:
        print(json.dumps(payload, indent=2))
        return pack_result.returncode or 1

    # OpenSSH `scp` on macOS chokes on Windows-style backslashes in the remote
    # path (returns "No such file or directory"). The Windows side accepts
    # forward slashes equally well, so normalise here for the wire path only.
    scp_remote_path = args.remote_pull_archive.replace("\\", "/")
    with tempfile.TemporaryDirectory(prefix="dinov3-pull-") as temp_dir:
        local_archive_path = Path(temp_dir) / "dinov3-reports-pull.zip"
        scp_command = [
            "scp",
            f"{args.host}:{scp_remote_path}",
            str(local_archive_path),
        ]
        payload["scp_command"] = scp_command
        scp_result = run_command(scp_command)
        payload["scp_result"] = {
            "returncode": scp_result.returncode,
            "stdout": scp_result.stdout.strip(),
            "stderr": scp_result.stderr.strip()[-400:],
        }
        if scp_result.returncode != 0 or not local_archive_path.exists():
            cleanup_result = run_command(cleanup_ssh_command)
            payload["cleanup_result"] = {
                "returncode": cleanup_result.returncode,
                "stdout": cleanup_result.stdout.strip(),
                "stderr": cleanup_result.stderr.strip()[-400:],
            }
            print(json.dumps(payload, indent=2))
            return scp_result.returncode or 1

        payload["local_archive_size_bytes"] = local_archive_path.stat().st_size
        extracted = extract_pulled_archive(
            local_archive_path,
            local_destination,
            include_extensions=DEFAULT_REPORT_INCLUDE_EXTENSIONS,
        )
        payload["extracted_count"] = len(extracted)
        payload["extracted_preview"] = [path.as_posix() for path in extracted[:20]]

    cleanup_result = run_command(cleanup_ssh_command)
    payload["cleanup_result"] = {
        "returncode": cleanup_result.returncode,
        "stdout": cleanup_result.stdout.strip(),
        "stderr": cleanup_result.stderr.strip()[-400:],
    }

    print(json.dumps(payload, indent=2))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="windows-pc")
    parser.add_argument("--project-dir", default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--remote-zip-name", default="dinov3-sync.zip")
    parser.add_argument("--no-git-init", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--pull-reports",
        action="store_true",
        help=(
            "Reverse direction: pull Artifacts/reports/ (text-only, zip-packed) "
            "from the remote host back to the local Code/Artifacts/reports/."
        ),
    )
    parser.add_argument(
        "--remote-reports-dir",
        default=DEFAULT_REMOTE_REPORTS_DIR,
        help="Remote source directory for --pull-reports.",
    )
    parser.add_argument(
        "--remote-pull-archive",
        default=DEFAULT_REMOTE_PULL_ARCHIVE,
        help="Remote scratch path for the temporary pull zip.",
    )
    parser.add_argument(
        "--local-reports-dir",
        default="Code/Artifacts/reports",
        help="Local destination (relative to repo root) for --pull-reports.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    if args.pull_reports:
        sys.exit(pull_reports(args, repo_root))
    sys.exit(push_repo(args, repo_root))


if __name__ == "__main__":
    main()
