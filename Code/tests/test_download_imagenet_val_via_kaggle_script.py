"""Tests for `scripts/download_imagenet_val_via_kaggle.py`."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "download_imagenet_val_via_kaggle.py"


def _load_script_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "download_imagenet_val_via_kaggle", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load script")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


SCRIPT = _load_script_module()


def test_parse_args_defaults() -> None:
    args = SCRIPT.parse_args(["--output-dir", "/tmp/x"])
    assert args.kaggle_dataset == "titericz/imagenet1k-val"
    assert args.output_dir == Path("/tmp/x")
    assert args.dry_run is False
    assert args.force is False


def test_parse_args_dry_run() -> None:
    args = SCRIPT.parse_args(
        [
            "--output-dir",
            "/tmp/x",
            "--dry-run",
            "--kaggle-dataset",
            "custom/slug",
        ]
    )
    assert args.dry_run is True
    assert args.kaggle_dataset == "custom/slug"


def test_find_kaggle_credentials_none_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("KAGGLE_CONFIG_DIR", raising=False)
    with patch("pathlib.Path.home", return_value=tmp_path):
        creds = SCRIPT.find_kaggle_credentials()
    assert creds is None


def test_find_kaggle_credentials_found_in_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home_kaggle = tmp_path / ".kaggle"
    home_kaggle.mkdir()
    token = home_kaggle / "kaggle.json"
    token.write_text('{"username":"x","key":"y"}', encoding="utf-8")

    monkeypatch.delenv("KAGGLE_CONFIG_DIR", raising=False)
    with patch("pathlib.Path.home", return_value=tmp_path):
        creds = SCRIPT.find_kaggle_credentials()

    assert creds == token


def test_find_kaggle_credentials_finds_new_access_token_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Detect the new Kaggle ``access_token`` (single-string KGAT_*) format."""
    home_kaggle = tmp_path / ".kaggle"
    home_kaggle.mkdir()
    token = home_kaggle / "access_token"
    token.write_text("KGAT_dummytokenforcoveragetestingonly", encoding="utf-8")

    monkeypatch.delenv("KAGGLE_CONFIG_DIR", raising=False)
    with patch("pathlib.Path.home", return_value=tmp_path):
        creds = SCRIPT.find_kaggle_credentials()

    assert creds == token


def test_find_kaggle_credentials_prefers_access_token_over_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When both formats exist, the new ``access_token`` is preferred."""
    home_kaggle = tmp_path / ".kaggle"
    home_kaggle.mkdir()
    new_token = home_kaggle / "access_token"
    new_token.write_text("KGAT_dummy", encoding="utf-8")
    legacy_token = home_kaggle / "kaggle.json"
    legacy_token.write_text('{"username":"x","key":"y"}', encoding="utf-8")

    monkeypatch.delenv("KAGGLE_CONFIG_DIR", raising=False)
    with patch("pathlib.Path.home", return_value=tmp_path):
        creds = SCRIPT.find_kaggle_credentials()

    assert creds == new_token


def test_find_kaggle_credentials_via_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom_dir = tmp_path / "custom_kaggle"
    custom_dir.mkdir()
    token = custom_dir / "kaggle.json"
    token.write_text('{"username":"x","key":"y"}', encoding="utf-8")

    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(custom_dir))
    with patch("pathlib.Path.home", return_value=tmp_path / "no-home"):
        creds = SCRIPT.find_kaggle_credentials()

    assert creds == token


def test_kaggle_setup_instructions_mentions_both_formats() -> None:
    text = SCRIPT.kaggle_setup_instructions()
    assert "kaggle.com" in text
    assert "access_token" in text
    assert "kaggle.json" in text
    assert "KGAT_" in text


def test_main_dry_run_when_creds_missing_returns_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "out"
    with patch.object(SCRIPT, "find_kaggle_credentials", return_value=None):
        rc = SCRIPT.main(["--output-dir", str(output), "--dry-run"])

    assert rc == 2


def test_main_dry_run_with_creds_authenticates_and_returns_0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "out"
    fake_creds = tmp_path / "kaggle.json"
    fake_creds.write_text("{}", encoding="utf-8")

    with patch.object(SCRIPT, "find_kaggle_credentials", return_value=fake_creds), patch.object(
        SCRIPT, "authenticate_kaggle_api", return_value=MagicMock()
    ):
        rc = SCRIPT.main(["--output-dir", str(output), "--dry-run"])

    assert rc == 0


def test_write_manifest_lists_extracted_images(tmp_path: Path) -> None:
    img1 = tmp_path / "ILSVRC2012_val_00000001.JPEG"
    img2 = tmp_path / "subdir" / "ILSVRC2012_val_00000002.JPEG"
    img2.parent.mkdir()
    img1.write_bytes(b"\xff\xd8")
    img2.write_bytes(b"\xff\xd8")

    manifest_path = SCRIPT.write_manifest(tmp_path, None, "titericz/imagenet1k-val")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["kaggle_dataset"] == "titericz/imagenet1k-val"
    assert payload["image_count"] == 2
    assert sorted(payload["images"]) == sorted(
        [
            "ILSVRC2012_val_00000001.JPEG",
            "subdir/ILSVRC2012_val_00000002.JPEG",
        ]
    )


def test_write_manifest_uses_custom_path(tmp_path: Path) -> None:
    img = tmp_path / "a.jpg"
    img.write_bytes(b"x")
    custom_path = tmp_path / "elsewhere" / "manifest.json"

    SCRIPT.write_manifest(tmp_path, custom_path, "x/y")

    assert custom_path.exists()


def test_unpack_zip_archives_extracts_each_zip(tmp_path: Path) -> None:
    import zipfile

    zip_path = tmp_path / "data.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("a.jpg", b"\xff\xd8")
        zf.writestr("b.jpg", b"\xff\xd8")

    roots = SCRIPT.unpack_zip_archives(tmp_path)

    assert len(roots) == 1
    assert roots[0].name == "data"
    assert (roots[0] / "a.jpg").exists()
    assert (roots[0] / "b.jpg").exists()


def test_perform_download_skips_when_output_non_empty_without_force(
    tmp_path: Path,
) -> None:
    output = tmp_path / "out"
    output.mkdir()
    (output / "existing.zip").write_bytes(b"x")
    api = MagicMock()

    SCRIPT.perform_download(api, dataset="x/y", output_dir=output, force=False)

    api.dataset_download_files.assert_not_called()


def test_perform_download_calls_api_with_force(tmp_path: Path) -> None:
    output = tmp_path / "out"
    output.mkdir()
    (output / "existing.zip").write_bytes(b"x")
    api = MagicMock()

    SCRIPT.perform_download(api, dataset="x/y", output_dir=output, force=True)

    api.dataset_download_files.assert_called_once_with(
        dataset="x/y",
        path=str(output),
        force=True,
        quiet=False,
        unzip=False,
    )
