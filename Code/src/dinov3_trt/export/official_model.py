"""Load the official DINOv3 source tree for local export and contract checks."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

from dinov3_trt.artifacts import ArtifactLayout, find_weight_files


def add_dinov3_source_to_path(source_dir: Path) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f"DINOv3 source directory does not exist: {source_dir}")
    source_dir_str = str(source_dir.resolve())
    if source_dir_str not in sys.path:
        sys.path.insert(0, source_dir_str)


def find_official_pth_weight(weights_dir: Path) -> Path | None:
    for path in find_weight_files(weights_dir):
        if path.suffix == ".pth":
            return path
    return None


def resolve_official_pth_weight(
    weights_dir: Path,
    *,
    weights_path: Path | None,
    pretrained: bool,
) -> Path | None:
    """Resolve a source-compatible official `.pth` weight for the local DINOv3 loader."""

    if not pretrained:
        return weights_path
    if weights_path is not None:
        if weights_path.suffix != ".pth":
            raise ValueError(
                "The official DINOv3 source loader requires a source-compatible .pth "
                f"checkpoint, got: {weights_path}"
            )
        return weights_path

    found = find_official_pth_weight(weights_dir)
    if found is not None:
        return found

    detected = find_weight_files(weights_dir)
    if detected:
        names = ", ".join(path.name for path in detected)
        raise ValueError(
            "No source-compatible .pth checkpoint found. "
            "HF safetensors are not loadable through the official source loader path; "
            f"detected: {names}"
        )
    raise ValueError(f"No source-compatible .pth checkpoint found in {weights_dir}")


def create_official_vitl16_model(
    *,
    source_dir: Path,
    weights_path: Path | None = None,
    pretrained: bool | None = None,
) -> Any:
    add_dinov3_source_to_path(source_dir)
    backbones = importlib.import_module("dinov3.hub.backbones")
    use_pretrained = weights_path is not None if pretrained is None else pretrained
    if use_pretrained:
        if weights_path is None:
            raise ValueError("weights_path is required when pretrained=True")
        return backbones.dinov3_vitl16(pretrained=True, weights=str(weights_path))
    return backbones.dinov3_vitl16(pretrained=False)


def create_official_vitl16_from_layout(
    layout: ArtifactLayout,
    *,
    pretrained: bool,
) -> Any:
    weights_path = resolve_official_pth_weight(
        layout.weights_dir,
        weights_path=None,
        pretrained=pretrained,
    )
    return create_official_vitl16_model(
        source_dir=layout.source_dir,
        weights_path=weights_path,
        pretrained=pretrained,
    )
