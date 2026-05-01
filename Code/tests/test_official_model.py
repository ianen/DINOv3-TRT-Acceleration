from pathlib import Path

import pytest

from dinov3_trt.export.official_model import (
    add_dinov3_source_to_path,
    find_official_pth_weight,
    resolve_official_pth_weight,
)


def test_find_official_pth_weight_prefers_pth(tmp_path: Path) -> None:
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    (weights_dir / "model.safetensors").write_bytes(b"safe")
    expected = weights_dir / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    expected.write_bytes(b"pth")

    assert find_official_pth_weight(weights_dir) == expected


def test_resolve_official_pth_weight_uses_explicit_pth(tmp_path: Path) -> None:
    expected = tmp_path / "model-8aa4cbdd.pth"
    expected.write_bytes(b"pth")

    assert (
        resolve_official_pth_weight(tmp_path, weights_path=expected, pretrained=True)
        == expected
    )


def test_resolve_official_pth_weight_rejects_explicit_safetensors(tmp_path: Path) -> None:
    weights_path = tmp_path / "model.safetensors"
    weights_path.write_bytes(b"safe")

    with pytest.raises(ValueError, match="requires a source-compatible .pth"):
        resolve_official_pth_weight(tmp_path, weights_path=weights_path, pretrained=True)


def test_resolve_official_pth_weight_explains_safetensors_only_dir(tmp_path: Path) -> None:
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    (weights_dir / "model.safetensors").write_bytes(b"safe")

    with pytest.raises(ValueError, match="HF safetensors are not loadable"):
        resolve_official_pth_weight(weights_dir, weights_path=None, pretrained=True)


def test_resolve_official_pth_weight_allows_random_weight_mode(tmp_path: Path) -> None:
    assert resolve_official_pth_weight(tmp_path, weights_path=None, pretrained=False) is None


def test_add_dinov3_source_to_path_rejects_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="DINOv3 source directory"):
        add_dinov3_source_to_path(tmp_path / "missing")
