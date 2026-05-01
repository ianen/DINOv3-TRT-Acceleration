from pathlib import Path

import pytest

from dinov3_trt.datasets.hf_imagenet_parquet import (
    detect_image_extension,
    export_rows_to_image_tree,
    normalize_label,
)


JPEG_BYTES = b"\xff\xd8\xff\xe0fake-jpeg"


def test_export_rows_to_image_tree_writes_label_directories(tmp_path: Path) -> None:
    result = export_rows_to_image_tree(
        [
            {
                "image": {"bytes": JPEG_BYTES, "path": "ILSVRC2012_val_00000001.JPEG"},
                "label": 42,
            }
        ],
        output_root=tmp_path,
        split="validation",
    )

    assert result.image_count == 1
    exported = result.images[0]
    assert exported.row_index == 0
    assert exported.label == "label_0042"
    assert exported.path == tmp_path / "label_0042" / "validation_00000000.jpg"
    assert exported.path.read_bytes() == JPEG_BYTES


def test_export_rows_to_image_tree_honors_limit_and_start_index(tmp_path: Path) -> None:
    rows = [
        {"image": JPEG_BYTES, "label": 1},
        {"image": JPEG_BYTES, "label": 2},
    ]

    result = export_rows_to_image_tree(
        rows,
        output_root=tmp_path,
        split="calib",
        start_index=10,
        limit=1,
    )

    assert result.image_count == 1
    assert result.paths == (tmp_path / "label_0001" / "calib_00000010.jpg",)


def test_export_rows_to_image_tree_rejects_existing_files(tmp_path: Path) -> None:
    target = tmp_path / "label_0001"
    target.mkdir()
    (target / "validation_00000000.jpg").write_bytes(JPEG_BYTES)

    with pytest.raises(FileExistsError, match="image already exists"):
        export_rows_to_image_tree(
            [{"image": JPEG_BYTES, "label": 1}],
            output_root=tmp_path,
        )


def test_detect_image_extension_uses_supported_source_suffix() -> None:
    assert detect_image_extension(b"not-magic", source_path="sample.PNG") == ".png"


def test_normalize_label_sanitizes_strings() -> None:
    assert normalize_label("n01440764 tench") == "n01440764_tench"

