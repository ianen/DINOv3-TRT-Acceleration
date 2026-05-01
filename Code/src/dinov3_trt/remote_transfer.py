"""Helpers for resumable large artifact transfer over a reverse HTTP tunnel."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_PART_PREFIX = "model.safetensors.part."
DEFAULT_PART_WIDTH = 4
DEFAULT_BUFFER_SIZE = 1024 * 1024


@dataclass(frozen=True)
class PartSpec:
    """One fixed-size file part expected by the remote downloader."""

    index: int
    name: str
    size_bytes: int

    def to_json(self) -> dict[str, object]:
        return {
            "index": self.index,
            "name": self.name,
            "size_bytes": self.size_bytes,
        }


def build_part_specs(
    *,
    total_size: int,
    chunk_size: int,
    prefix: str = DEFAULT_PART_PREFIX,
    width: int = DEFAULT_PART_WIDTH,
) -> tuple[PartSpec, ...]:
    """Return deterministic part names and expected sizes for a fixed chunk size."""

    if total_size <= 0:
        raise ValueError("total_size must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if width <= 0:
        raise ValueError("width must be positive")

    count = (total_size + chunk_size - 1) // chunk_size
    parts: list[PartSpec] = []
    for index in range(count):
        remaining = total_size - (index * chunk_size)
        size_bytes = min(chunk_size, remaining)
        parts.append(
            PartSpec(
                index=index,
                name=f"{prefix}{index:0{width}d}",
                size_bytes=size_bytes,
            )
        )
    return tuple(parts)


def split_file(
    input_path: Path,
    output_dir: Path,
    *,
    chunk_size: int,
    prefix: str = DEFAULT_PART_PREFIX,
    width: int = DEFAULT_PART_WIDTH,
    buffer_size: int = DEFAULT_BUFFER_SIZE,
) -> tuple[PartSpec, ...]:
    """Split `input_path` into deterministic parts and return the part manifest."""

    if buffer_size <= 0:
        raise ValueError("buffer_size must be positive")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    total_size = input_path.stat().st_size
    parts = build_part_specs(
        total_size=total_size,
        chunk_size=chunk_size,
        prefix=prefix,
        width=width,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("rb") as source:
        for part in parts:
            remaining = part.size_bytes
            output_path = output_dir / part.name
            with output_path.open("wb") as target:
                while remaining > 0:
                    payload = source.read(min(buffer_size, remaining))
                    if not payload:
                        raise IOError(f"unexpected EOF while writing {output_path}")
                    target.write(payload)
                    remaining -= len(payload)
    return parts


def powershell_single_quote(value: str) -> str:
    """Return a PowerShell single-quoted string literal."""

    return "'" + value.replace("'", "''") + "'"


def render_sequential_http_downloader(
    *,
    base_url: str,
    remote_dir: str,
    total_size: int,
    chunk_size: int,
    prefix: str = DEFAULT_PART_PREFIX,
    width: int = DEFAULT_PART_WIDTH,
    retries: int = 5,
    curl_retry: int = 2,
    curl_retry_delay_seconds: int = 3,
    curl_connect_timeout_seconds: int = 30,
    curl_max_time_seconds: int = 600,
) -> str:
    """Render a PowerShell script that downloads and verifies all parts sequentially."""

    build_part_specs(total_size=total_size, chunk_size=chunk_size, prefix=prefix, width=width)
    if retries <= 0:
        raise ValueError("retries must be positive")
    if curl_retry < 0:
        raise ValueError("curl_retry must be non-negative")
    if curl_retry_delay_seconds < 0:
        raise ValueError("curl_retry_delay_seconds must be non-negative")
    if curl_connect_timeout_seconds <= 0:
        raise ValueError("curl_connect_timeout_seconds must be positive")
    if curl_max_time_seconds <= 0:
        raise ValueError("curl_max_time_seconds must be positive")

    return f"""$ErrorActionPreference = 'Stop'
$baseUrl = {powershell_single_quote(base_url.rstrip("/"))}
$dst = {powershell_single_quote(remote_dir)}
$prefix = {powershell_single_quote(prefix)}
$indexFormat = 'D{width}'
$chunkSize = [int64]{chunk_size}
$totalSize = [int64]{total_size}
$count = [int][Math]::Ceiling($totalSize / [double]$chunkSize)
New-Item -ItemType Directory -Force $dst | Out-Null
$completed = 0
for ($i = 0; $i -lt $count; $i++) {{
  $name = $prefix + $i.ToString($indexFormat)
  if ($i -eq ($count - 1)) {{ $expected = $totalSize - ($chunkSize * ($count - 1)) }} else {{ $expected = $chunkSize }}
  $file = Join-Path $dst $name
  if (Test-Path $file) {{
    $existing = (Get-Item $file).Length
    if ($existing -eq $expected) {{
      $completed += 1
      Write-Output "SKIP $name $existing completed=$completed/$count"
      continue
    }}
    Remove-Item -Force $file
  }}
  $url = "$baseUrl/$name"
  $ok = $false
  for ($attempt = 1; $attempt -le {retries}; $attempt++) {{
    Write-Output "GET $name attempt=$attempt completed=$completed/$count"
    & curl.exe -L --fail --retry {curl_retry} --retry-delay {curl_retry_delay_seconds} --connect-timeout {curl_connect_timeout_seconds} --max-time {curl_max_time_seconds} -sS -o $file $url
    $code = $LASTEXITCODE
    $size = 0
    if (Test-Path $file) {{ $size = (Get-Item $file).Length }}
    if ($code -eq 0 -and $size -eq $expected) {{
      $completed += 1
      Write-Output "OK $name $size completed=$completed/$count"
      $ok = $true
      break
    }}
    Write-Output "BAD $name code=$code size=$size expected=$expected"
    Remove-Item -Force $file -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 5
  }}
  if (-not $ok) {{
    Write-Output "FAILED $name"
    exit 1
  }}
}}
$total = (Get-ChildItem $dst -Filter "$prefix*" -File | Measure-Object Length -Sum).Sum
Write-Output "PART_TOTAL=$total"
if ($total -ne $totalSize) {{ exit 1 }}
Write-Output 'ALL_PARTS_OK'
exit 0
"""


def render_windows_part_merger(
    *,
    parts_dir: str,
    output_path: str,
    total_size: int,
    chunk_size: int,
    expected_sha256: str,
    prefix: str = DEFAULT_PART_PREFIX,
    width: int = DEFAULT_PART_WIDTH,
) -> str:
    """Render a PowerShell script that validates, merges, and hashes downloaded parts."""

    build_part_specs(total_size=total_size, chunk_size=chunk_size, prefix=prefix, width=width)
    normalized_sha256 = expected_sha256.strip().lower()
    if len(normalized_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in normalized_sha256
    ):
        raise ValueError("expected_sha256 must be a 64-character hex digest")

    return f"""$ErrorActionPreference = 'Stop'
$partsDir = {powershell_single_quote(parts_dir)}
$outputPath = {powershell_single_quote(output_path)}
$prefix = {powershell_single_quote(prefix)}
$indexFormat = 'D{width}'
$chunkSize = [int64]{chunk_size}
$totalSize = [int64]{total_size}
$expectedSha256 = {powershell_single_quote(normalized_sha256)}
$count = [int][Math]::Ceiling($totalSize / [double]$chunkSize)
if (-not (Test-Path $partsDir)) {{ throw "parts directory does not exist: $partsDir" }}
New-Item -ItemType Directory -Force (Split-Path -Parent $outputPath) | Out-Null
Remove-Item -Force $outputPath -ErrorAction SilentlyContinue
$out = [System.IO.File]::Open($outputPath, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write)
try {{
  for ($i = 0; $i -lt $count; $i++) {{
    $name = $prefix + $i.ToString($indexFormat)
    if ($i -eq ($count - 1)) {{ $expected = $totalSize - ($chunkSize * ($count - 1)) }} else {{ $expected = $chunkSize }}
    $partPath = Join-Path $partsDir $name
    if (-not (Test-Path $partPath)) {{ throw "missing part: $name" }}
    $actual = (Get-Item $partPath).Length
    if ($actual -ne $expected) {{ throw "bad part size: $name actual=$actual expected=$expected" }}
    Write-Output "MERGE $name $actual"
    $input = [System.IO.File]::OpenRead($partPath)
    try {{ $input.CopyTo($out) }} finally {{ $input.Dispose() }}
  }}
}} finally {{
  $out.Dispose()
}}
$mergedSize = (Get-Item $outputPath).Length
Write-Output "MERGED_SIZE=$mergedSize"
if ($mergedSize -ne $totalSize) {{ throw "bad merged size: $mergedSize expected=$totalSize" }}
$actualSha256 = (Get-FileHash -Algorithm SHA256 $outputPath).Hash.ToLowerInvariant()
Write-Output "SHA256=$actualSha256"
if ($actualSha256 -ne $expectedSha256) {{ throw "sha256 mismatch: $actualSha256 expected=$expectedSha256" }}
Write-Output 'MERGE_OK'
exit 0
"""
