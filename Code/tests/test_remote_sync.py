from __future__ import annotations

from pathlib import Path

import zipfile

import pytest

from dinov3_trt.remote_sync import (
    DEFAULT_REPORT_INCLUDE_EXTENSIONS,
    DEFAULT_SYNC_ITEMS,
    build_manifest,
    build_remote_cleanup_powershell,
    build_remote_pack_powershell,
    build_remote_powershell,
    create_zip_archive,
    extract_pulled_archive,
    windows_join,
)


def test_build_manifest_keeps_project_sources_and_skips_generated_files(tmp_path: Path) -> None:
    repo_root = tmp_path
    (repo_root / "Code" / "src").mkdir(parents=True)
    (repo_root / "Code" / "src" / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (repo_root / "Code" / "src" / "package" / "reports").mkdir(parents=True)
    (repo_root / "Code" / "src" / "package" / "reports" / "summary.py").write_text(
        "VALUE = 1\n",
        encoding="utf-8",
    )
    (repo_root / "Code" / ".venv" / "bin").mkdir(parents=True)
    (repo_root / "Code" / ".venv" / "bin" / "python").write_text("", encoding="utf-8")
    (repo_root / "Code" / "__pycache__").mkdir()
    (repo_root / "Code" / "__pycache__" / "x.pyc").write_bytes(b"")
    (repo_root / "reports").mkdir()
    (repo_root / "reports" / "bench.csv").write_text("bench\n", encoding="utf-8")
    (repo_root / "CLAUDE.md").write_text("# repo\n", encoding="utf-8")

    manifest = build_manifest(repo_root, items=("CLAUDE.md", "Code", "reports"))

    assert [path.as_posix() for path in manifest] == [
        "CLAUDE.md",
        "Code/src/main.py",
        "Code/src/package/reports/summary.py",
    ]


def test_create_zip_archive_uses_relative_paths(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "Code").mkdir()
    (repo_root / "Code" / "README.md").write_text("hello\n", encoding="utf-8")
    archive_path = tmp_path / "bundle.zip"

    written = create_zip_archive(repo_root, archive_path, items=("Code",))

    assert written == ["Code/README.md"]


def test_default_sync_items_include_project_readme_and_license_copy() -> None:
    assert "README.md" in DEFAULT_SYNC_ITEMS
    assert "LICENSES" in DEFAULT_SYNC_ITEMS


def test_windows_join_normalizes_separators() -> None:
    assert windows_join(r"D:\WorkPlace", "ZMP", "Repo") == r"D:\WorkPlace\ZMP\Repo"


def test_build_remote_powershell_includes_sync_and_optional_git_init() -> None:
    command = build_remote_powershell(
        remote_zip_path=r"D:\Temp\dinov3-sync.zip",
        project_dir=r"D:\WorkPlace\ZMP\DINOv3-TRT-Acceleration",
        init_git=True,
    )

    assert "Expand-Archive" in command
    assert "git init" in command
    assert "Remove-Item" in command
    assert str(DEFAULT_SYNC_ITEMS[0]) not in command


def test_default_report_include_extensions_cover_text_artifacts() -> None:
    assert ".json" in DEFAULT_REPORT_INCLUDE_EXTENSIONS
    assert ".svg" in DEFAULT_REPORT_INCLUDE_EXTENSIONS
    assert ".csv" in DEFAULT_REPORT_INCLUDE_EXTENSIONS
    assert ".md" in DEFAULT_REPORT_INCLUDE_EXTENSIONS
    # Binary engine/weight extensions must never accidentally land here.
    assert ".engine" not in DEFAULT_REPORT_INCLUDE_EXTENSIONS
    assert ".onnx" not in DEFAULT_REPORT_INCLUDE_EXTENSIONS


def test_build_remote_pack_powershell_filters_by_extension_and_emits_marker() -> None:
    command = build_remote_pack_powershell(
        remote_source_dir=r"D:\WorkPlace\ZMP\DINOv3-TRT-Acceleration\Code\Artifacts\reports",
        remote_archive_path=r"D:\WorkPlace\ZMP\DINOv3-TRT-Acceleration\dinov3-reports-pull.zip",
        include_extensions=(".json", ".svg"),
    )

    assert "$includeExtensions = @('.json', '.svg')" in command
    assert "Get-ChildItem" in command
    assert "ZipArchive" in command
    assert "PACK_OK" in command
    # Defensive: cleanup of any stale archive must precede creation.
    assert "Remove-Item -LiteralPath $archive -Force" in command


def test_build_remote_pack_powershell_rejects_empty_extensions() -> None:
    with pytest.raises(ValueError, match="at least one include extension"):
        build_remote_pack_powershell(
            remote_source_dir="D:\\repo",
            remote_archive_path="D:\\out.zip",
            include_extensions=(),
        )


def test_build_remote_cleanup_powershell_emits_status_marker() -> None:
    command = build_remote_cleanup_powershell(
        r"D:\WorkPlace\ZMP\DINOv3-TRT-Acceleration\dinov3-reports-pull.zip"
    )

    assert "Remove-Item" in command
    assert "CLEANUP_OK" in command
    assert "CLEANUP_MISSING" in command


def _build_test_archive(archive_path: Path, files: dict[str, bytes]) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)


def test_extract_pulled_archive_only_writes_allowed_extensions(tmp_path: Path) -> None:
    archive_path = tmp_path / "pull.zip"
    _build_test_archive(
        archive_path,
        {
            "formal_summary.md": b"# summary\n",
            "figures/benchmark.svg": b"<svg/>",
            "figures/benchmark.engine": b"\x00\x01",  # blocked: not in default extensions
            "stray.cache": b"junk",  # blocked
        },
    )
    destination = tmp_path / "reports"

    written = extract_pulled_archive(archive_path, destination)

    assert sorted(p.as_posix() for p in written) == [
        "figures/benchmark.svg",
        "formal_summary.md",
    ]
    assert (destination / "formal_summary.md").read_bytes() == b"# summary\n"
    assert (destination / "figures" / "benchmark.svg").read_bytes() == b"<svg/>"
    assert not (destination / "figures" / "benchmark.engine").exists()
    assert not (destination / "stray.cache").exists()


def test_extract_pulled_archive_rejects_zip_slip_paths(tmp_path: Path) -> None:
    archive_path = tmp_path / "evil.zip"
    _build_test_archive(
        archive_path,
        {
            "../escape.md": b"nope",
            "ok.md": b"ok",
        },
    )
    destination = tmp_path / "reports"

    written = extract_pulled_archive(archive_path, destination)

    assert [p.as_posix() for p in written] == ["ok.md"]
    # Ensure nothing landed outside `destination`.
    assert not (tmp_path / "escape.md").exists()


def test_extract_pulled_archive_translates_windows_separators(tmp_path: Path) -> None:
    archive_path = tmp_path / "winpaths.zip"
    _build_test_archive(
        archive_path,
        {
            r"figures\benchmark.svg": b"<svg/>",
        },
    )
    destination = tmp_path / "reports"

    written = extract_pulled_archive(archive_path, destination)

    assert [p.as_posix() for p in written] == ["figures/benchmark.svg"]
    assert (destination / "figures" / "benchmark.svg").read_bytes() == b"<svg/>"
