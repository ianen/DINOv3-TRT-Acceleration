from pathlib import Path
from typing import cast

import numpy as np
import pytest
from PIL import Image

from dinov3_trt.infer.compare import compare_arrays
from dinov3_trt.infer.image_eval import (
    OutputMetricAccumulator,
    chunk_paths,
    list_image_paths,
    load_image_batch,
    read_image_manifest,
    stratified_eval_calib_split,
    write_image_manifest,
)


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (8, 8), color)
    image.save(path)


def test_list_image_paths_is_stable_and_limited(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    _write_image(nested / "b.PNG", (0, 0, 0))
    _write_image(tmp_path / "a.jpg", (255, 0, 0))
    (tmp_path / "ignore.txt").write_text("not image", encoding="utf-8")

    paths = list_image_paths(tmp_path, limit=2)

    assert [path.name for path in paths] == ["a.jpg", "b.PNG"]


def test_load_image_batch_normalizes_to_nchw_float32(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _write_image(first, (255, 0, 0))
    _write_image(second, (0, 255, 0))

    batch = load_image_batch((first, second), image_size=4)

    assert batch.paths == (first, second)
    assert batch.tensor.shape == (2, 3, 4, 4)
    assert batch.tensor.dtype == np.float32


def test_chunk_paths_rejects_invalid_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        chunk_paths((Path("a.jpg"),), 0)


def test_output_metric_accumulator_aggregates_batches() -> None:
    accumulator = OutputMetricAccumulator(name="feat")
    first = compare_arrays(
        "feat",
        np.array([1.0, 2.0], dtype=np.float32),
        np.array([1.0, 2.5], dtype=np.float32),
    )
    second = compare_arrays(
        "feat",
        np.array([1.0, 1.0], dtype=np.float32),
        np.array([1.5, 1.5], dtype=np.float32),
    )

    accumulator.update(first)
    accumulator.update(second)
    payload = accumulator.to_json()

    assert payload["batches"] == 2
    assert payload["tensor_values"] == 4
    assert payload["max_abs_error"] == pytest.approx(0.5)
    assert payload["mean_abs_error"] == pytest.approx(0.375)
    assert payload["reference_l2_norm_mean"] is not None
    assert payload["candidate_l2_norm_mean"] is not None
    assert payload["candidate_l2_norm_min"] is not None
    cosine_min = cast(float, payload["cosine_similarity_min"])
    cosine_mean = cast(float, payload["cosine_similarity_mean"])
    assert cosine_min <= cosine_mean


def test_manifest_roundtrip_uses_relative_paths(tmp_path: Path) -> None:
    image = tmp_path / "class-a" / "image.jpg"
    image.parent.mkdir()
    _write_image(image, (1, 2, 3))
    manifest = tmp_path / "manifest.json"

    write_image_manifest(
        manifest,
        image_root=tmp_path,
        images=(image,),
        seed=123,
        split="eval",
    )

    assert read_image_manifest(manifest) == (image,)


def test_stratified_eval_calib_split_is_disjoint(tmp_path: Path) -> None:
    paths: list[Path] = []
    for class_index in range(2):
        class_dir = tmp_path / f"class-{class_index}"
        class_dir.mkdir()
        for image_index in range(4):
            image = class_dir / f"{image_index}.jpg"
            _write_image(image, (class_index, image_index, 0))
            paths.append(image)

    eval_paths, calib_paths = stratified_eval_calib_split(
        tuple(paths),
        image_root=tmp_path,
        eval_count=3,
        calib_count=3,
        seed=7,
    )

    assert len(eval_paths) == 3
    assert len(calib_paths) == 3
    assert set(eval_paths).isdisjoint(calib_paths)
