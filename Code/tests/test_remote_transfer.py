from __future__ import annotations

from pathlib import Path

import pytest

from dinov3_trt.remote_transfer import (
    build_part_specs,
    powershell_single_quote,
    render_sequential_http_downloader,
    render_windows_part_merger,
    split_file,
)


def test_build_part_specs_uses_fixed_chunks_and_last_remainder() -> None:
    parts = build_part_specs(total_size=20, chunk_size=8, prefix="part.", width=3)

    assert [part.name for part in parts] == ["part.000", "part.001", "part.002"]
    assert [part.size_bytes for part in parts] == [8, 8, 4]


def test_build_part_specs_rejects_invalid_sizes() -> None:
    with pytest.raises(ValueError, match="total_size"):
        build_part_specs(total_size=0, chunk_size=8)
    with pytest.raises(ValueError, match="chunk_size"):
        build_part_specs(total_size=8, chunk_size=0)


def test_split_file_writes_deterministic_parts(tmp_path: Path) -> None:
    source = tmp_path / "model.safetensors"
    source.write_bytes(b"abcdefghijklmnopqrst")
    parts_dir = tmp_path / "parts"

    parts = split_file(source, parts_dir, chunk_size=8, prefix="part.", width=2, buffer_size=3)

    assert [part.name for part in parts] == ["part.00", "part.01", "part.02"]
    assert (parts_dir / "part.00").read_bytes() == b"abcdefgh"
    assert (parts_dir / "part.01").read_bytes() == b"ijklmnop"
    assert (parts_dir / "part.02").read_bytes() == b"qrst"


def test_powershell_single_quote_escapes_embedded_quotes() -> None:
    assert powershell_single_quote(r"C:\Users\O'Hara") == r"'C:\Users\O''Hara'"


def test_render_sequential_http_downloader_contains_transfer_contract() -> None:
    script = render_sequential_http_downloader(
        base_url="http://127.0.0.1:18765/",
        remote_dir=r"C:\Users\USER\parts",
        total_size=20,
        chunk_size=8,
        prefix="part.",
        width=3,
        curl_max_time_seconds=60,
    )

    assert "$baseUrl = 'http://127.0.0.1:18765'" in script
    assert "$dst = 'C:\\Users\\USER\\parts'" in script
    assert "$prefix = 'part.'" in script
    assert "$indexFormat = 'D3'" in script
    assert "$chunkSize = [int64]8" in script
    assert "$totalSize = [int64]20" in script
    assert "--max-time 60" in script
    assert "ALL_PARTS_OK" in script


def test_render_windows_part_merger_contains_size_and_hash_checks() -> None:
    script = render_windows_part_merger(
        parts_dir=r"C:\Users\USER\parts",
        output_path=r"D:\repo\Code\Artifacts\weights\model.safetensors",
        total_size=20,
        chunk_size=8,
        expected_sha256="A" * 64,
        prefix="part.",
        width=3,
    )

    assert "$partsDir = 'C:\\Users\\USER\\parts'" in script
    assert "$outputPath = 'D:\\repo\\Code\\Artifacts\\weights\\model.safetensors'" in script
    assert "$prefix = 'part.'" in script
    assert "$indexFormat = 'D3'" in script
    assert "$chunkSize = [int64]8" in script
    assert "$totalSize = [int64]20" in script
    assert "$expectedSha256 = '" + ("a" * 64) + "'" in script
    assert "bad part size" in script
    assert "Get-FileHash -Algorithm SHA256" in script
    assert "MERGE_OK" in script


def test_render_windows_part_merger_rejects_bad_sha256() -> None:
    with pytest.raises(ValueError, match="expected_sha256"):
        render_windows_part_merger(
            parts_dir=r"C:\parts",
            output_path=r"D:\model.safetensors",
            total_size=20,
            chunk_size=8,
            expected_sha256="not-a-digest",
        )
