"""Tests for the `check_assets.py` CLI manifest writer.

These cover the regression where the manifest was previously self-referencing
itself with a `size_bytes=0` / empty-string SHA256 entry, because the calling
shell pre-created the target file via `>` redirection before the Python
process scanned the reports directory.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_assets.py"


def _run_script(args: list[str], *, env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    src_dir = str(ROOT / "src")
    env["PYTHONPATH"] = (
        src_dir + (os.pathsep + env["PYTHONPATH"]) if "PYTHONPATH" in env else src_dir
    )
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def _make_layout_skeleton(artifact_root: Path) -> None:
    (artifact_root / "source" / "dinov3").mkdir(parents=True)
    (artifact_root / "source" / "dinov3" / "README.md").write_text("dinov3", encoding="utf-8")
    (artifact_root / "weights" / "dinov3-vitl16-pretrain-lvd1689m").mkdir(parents=True)
    (
        artifact_root
        / "weights"
        / "dinov3-vitl16-pretrain-lvd1689m"
        / "model.safetensors"
    ).write_bytes(b"weights")
    (artifact_root / "onnx").mkdir(parents=True)
    (artifact_root / "onnx" / "dinov3_vitl16_4out.onnx").write_bytes(b"onnx")
    (artifact_root / "engines").mkdir(parents=True)
    (artifact_root / "reports").mkdir(parents=True)
    (artifact_root / "reports" / "keep.json").write_text("{}", encoding="utf-8")


def test_output_atomic_write_excludes_self_from_report_scan(tmp_path: Path) -> None:
    artifact_root = tmp_path / "Artifacts"
    _make_layout_skeleton(artifact_root)

    output_path = artifact_root / "reports" / "artifact_manifest_formal_with_sha256.json"
    # Simulate the shell `>` redirect: the empty file already exists when the
    # Python process starts. Without the fix, this 0-byte sentinel would have
    # been recorded inside its own manifest.
    output_path.write_bytes(b"")

    result = _run_script(
        [
            "--artifact-root",
            str(artifact_root),
            "--with-sha256",
            "--output",
            str(output_path),
        ]
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    report_files = payload["assets"]["reports"]["files"]
    assert str(output_path) not in report_files
    file_info_paths = [item["path"] for item in payload["assets"]["reports"]["file_info"]]
    assert str(output_path) not in file_info_paths
    # The sibling file we wanted to keep must still appear.
    keep_path = artifact_root / "reports" / "keep.json"
    assert str(keep_path) in report_files


def test_output_writes_atomically_without_pre_existing_file(tmp_path: Path) -> None:
    artifact_root = tmp_path / "Artifacts"
    _make_layout_skeleton(artifact_root)
    output_path = artifact_root / "reports" / "artifact_manifest_formal_with_sha256.json"

    result = _run_script(
        [
            "--artifact-root",
            str(artifact_root),
            "--output",
            str(output_path),
        ]
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["artifact_root"] == str(artifact_root)
    # No leftover temp files after a successful atomic write.
    leftovers = [
        path
        for path in output_path.parent.iterdir()
        if path.name.startswith(f".{output_path.name}.")
    ]
    assert leftovers == []


def test_stdout_legacy_path_still_works(tmp_path: Path) -> None:
    artifact_root = tmp_path / "Artifacts"
    _make_layout_skeleton(artifact_root)

    result = _run_script(["--artifact-root", str(artifact_root)])

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["assets"]["reports"]["present"] is True


def test_exclude_flag_filters_additional_paths(tmp_path: Path) -> None:
    artifact_root = tmp_path / "Artifacts"
    _make_layout_skeleton(artifact_root)
    extra = artifact_root / "reports" / "side_effect.json"
    extra.write_text("{}", encoding="utf-8")

    result = _run_script(
        [
            "--artifact-root",
            str(artifact_root),
            "--exclude",
            str(extra),
        ]
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert str(extra) not in payload["assets"]["reports"]["files"]
