"""Utilities for syncing the source workspace to the Windows RTX 5080 host."""

from __future__ import annotations

from pathlib import Path, PureWindowsPath
from typing import Iterable
import zipfile


DEFAULT_SYNC_ITEMS = ("CLAUDE.md", "README.md", ".gitignore", ".claude", "Wiki", "LICENSES", "Code")

EXCLUDED_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".hypothesis",
    ".tox",
    ".nox",
    "Artifacts",
    "artifacts",
    "weights",
    "data",
    "datasets",
    "engines",
    "checkpoints",
    "calibration",
    "runs",
    "wandb",
    "mlruns",
    "tensorboard",
}

EXCLUDED_SUFFIXES = (
    ".pyc",
    ".pyo",
    ".pyd",
    ".onnx",
    ".engine",
    ".plan",
    ".trt",
    ".pth",
    ".pt",
    ".bin",
    ".safetensors",
    ".cache",
    ".log",
    ".tar",
    ".tar.gz",
    ".zip",
)


def _is_excluded(relative_path: Path) -> bool:
    if relative_path.parts and relative_path.parts[0] == "reports":
        return True
    for part in relative_path.parts:
        if part in EXCLUDED_DIR_NAMES or part.endswith(".egg-info"):
            return True
    name = relative_path.name
    return any(name.endswith(suffix) for suffix in EXCLUDED_SUFFIXES)


def build_manifest(repo_root: Path, items: tuple[str, ...] = DEFAULT_SYNC_ITEMS) -> list[Path]:
    """Return source files to ship, relative to `repo_root`."""

    manifest: list[Path] = []
    for item in items:
        source = repo_root / item
        if not source.exists():
            continue
        if source.is_file():
            relative_file = source.relative_to(repo_root)
            if not _is_excluded(relative_file):
                manifest.append(relative_file)
            continue

        for file_path in sorted(path for path in source.rglob("*") if path.is_file()):
            relative_file = file_path.relative_to(repo_root)
            if _is_excluded(relative_file):
                continue
            manifest.append(relative_file)
    return manifest


def create_zip_archive(
    repo_root: Path,
    archive_path: Path,
    items: tuple[str, ...] = DEFAULT_SYNC_ITEMS,
) -> list[str]:
    """Write a zip archive containing the filtered source tree."""

    manifest = build_manifest(repo_root, items=items)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative_path in manifest:
            archive.write(repo_root / relative_path, arcname=relative_path.as_posix())
    return [path.as_posix() for path in manifest]


def windows_join(base: str, *parts: str) -> str:
    """Join Windows path segments and normalize separators."""

    current = PureWindowsPath(base)
    for part in parts:
        current /= part
    return str(current)


def build_remote_powershell(
    *,
    remote_zip_path: str,
    project_dir: str,
    init_git: bool,
) -> str:
    """Return a single PowerShell command string for extraction and optional git init."""

    project = project_dir.replace("'", "''")
    archive = remote_zip_path.replace("'", "''")
    git_block = ""
    if init_git:
        git_block = (
            "if (-not (Test-Path -LiteralPath (Join-Path $project '.git'))) { "
            "if (Get-Command git -ErrorAction SilentlyContinue) { "
            "Push-Location $project; git init | Out-Null; Pop-Location } } "
        )

    return (
        "$ErrorActionPreference = 'Stop'; "
        f"$project = '{project}'; "
        f"$archive = '{archive}'; "
        "New-Item -ItemType Directory -Force -Path $project | Out-Null; "
        "Expand-Archive -LiteralPath $archive -DestinationPath $project -Force; "
        + git_block
        + "Remove-Item -LiteralPath $archive -Force; "
        "Write-Output 'SYNC_OK'"
    )


# ---------------------------------------------------------------------------
# Reverse-pull (remote -> local) for report artifacts
# ---------------------------------------------------------------------------

# Default file extensions to include in a reverse pull. Stays small/text-only —
# we don't want to drag back engine binaries or weights.
DEFAULT_REPORT_INCLUDE_EXTENSIONS: tuple[str, ...] = (
    ".json",
    ".md",
    ".csv",
    ".svg",
    ".png",
    ".jpg",
    ".log",
    ".txt",
)


def build_remote_pack_powershell(
    *,
    remote_source_dir: str,
    remote_archive_path: str,
    include_extensions: Iterable[str] = DEFAULT_REPORT_INCLUDE_EXTENSIONS,
) -> str:
    """Return a PowerShell command that zips a remote directory's text-only files.

    The remote command:
    1. Validates that `remote_source_dir` exists.
    2. Filters files under it by extension (recursive).
    3. Writes them into `remote_archive_path` with paths relative to the source dir.
    4. Echoes `PACK_OK <count> <archive>` on success.
    """

    source = remote_source_dir.replace("'", "''")
    archive = remote_archive_path.replace("'", "''")
    extensions = tuple(ext.lower() for ext in include_extensions)
    if not extensions:
        raise ValueError("at least one include extension is required")
    extensions_literal = ", ".join("'" + ext.replace("'", "''") + "'" for ext in extensions)

    # NOTE: stays compatible with PowerShell 5.1 / .NET Framework 4.x. We avoid
    # `[System.IO.Path]::GetRelativePath` (only on .NET Core 2.0+) and instead
    # slice the source prefix by string length, which works on all PS versions.
    return (
        "$ErrorActionPreference = 'Stop'; "
        f"$source = '{source}'; "
        f"$archive = '{archive}'; "
        f"$includeExtensions = @({extensions_literal}); "
        "if (-not (Test-Path -LiteralPath $source)) { "
        "throw \"remote source directory not found: $source\" } "
        "$resolvedSource = (Resolve-Path -LiteralPath $source).Path; "
        "$prefixLen = $resolvedSource.Length; "
        "$archiveDir = Split-Path -Parent $archive; "
        "if ($archiveDir) { "
        "New-Item -ItemType Directory -Force -Path $archiveDir | Out-Null } "
        "if (Test-Path -LiteralPath $archive) { "
        "Remove-Item -LiteralPath $archive -Force } "
        "Add-Type -AssemblyName System.IO.Compression; "
        "Add-Type -AssemblyName System.IO.Compression.FileSystem; "
        "$zipStream = [System.IO.File]::Open("
        "$archive, [System.IO.FileMode]::CreateNew); "
        "$zip = New-Object System.IO.Compression.ZipArchive("
        "$zipStream, [System.IO.Compression.ZipArchiveMode]::Create); "
        "try { "
        "$count = 0; "
        "Get-ChildItem -LiteralPath $resolvedSource -Recurse -File | "
        "Where-Object { $includeExtensions -contains $_.Extension.ToLower() } | "
        "ForEach-Object { "
        "$rel = $_.FullName.Substring($prefixLen); "
        "$rel = $rel.TrimStart('\\').TrimStart('/'); "
        "$rel = $rel -replace '\\\\', '/'; "
        "$entry = [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile("
        "$zip, $_.FullName, $rel, "
        "[System.IO.Compression.CompressionLevel]::Optimal); "
        "$count++ } "
        "} finally { $zip.Dispose(); $zipStream.Dispose() } "
        "Write-Output \"PACK_OK $count $archive\""
    )


def build_remote_cleanup_powershell(remote_archive_path: str) -> str:
    """Return a PowerShell command that removes a remote archive after scp."""

    archive = remote_archive_path.replace("'", "''")
    return (
        "$ErrorActionPreference = 'Stop'; "
        f"$archive = '{archive}'; "
        "if (Test-Path -LiteralPath $archive) { "
        "Remove-Item -LiteralPath $archive -Force; "
        "Write-Output 'CLEANUP_OK' } "
        "else { Write-Output 'CLEANUP_MISSING' }"
    )


def extract_pulled_archive(
    archive_path: Path,
    destination_dir: Path,
    *,
    include_extensions: Iterable[str] = DEFAULT_REPORT_INCLUDE_EXTENSIONS,
) -> list[Path]:
    """Extract a pulled zip into `destination_dir`, filtering by extension.

    Returns the list of relative paths actually written. The extension filter
    is applied a second time as defence-in-depth — the remote PowerShell already
    filtered, but we don't trust user-supplied archives.
    """

    destination_dir.mkdir(parents=True, exist_ok=True)
    allowed_extensions = tuple(ext.lower() for ext in include_extensions)
    written: list[Path] = []
    with zipfile.ZipFile(archive_path, "r") as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            relative_name = member.filename
            if relative_name.startswith("/") or ".." in PureWindowsPath(relative_name).parts:
                # Defensive: refuse zip-slip / absolute paths.
                continue
            relative_path = Path(relative_name.replace("\\", "/"))
            if relative_path.suffix.lower() not in allowed_extensions:
                continue
            target_path = destination_dir / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source_stream, target_path.open("wb") as out_stream:
                out_stream.write(source_stream.read())
            written.append(relative_path)
    return written
