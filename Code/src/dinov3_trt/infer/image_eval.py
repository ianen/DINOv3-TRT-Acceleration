"""Image-directory evaluation helpers for TensorRT engine pairs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Iterable

import numpy as np
from PIL import Image
from numpy.typing import NDArray

from dinov3_trt.infer.compare import OutputComparison
from dinov3_trt.utils.preprocess import hwc_uint8_to_nchw_float32

SUPPORTED_IMAGE_EXTENSIONS = frozenset((".jpg", ".jpeg", ".png", ".bmp", ".webp"))


@dataclass(frozen=True)
class ImageBatch:
    """Preprocessed image batch and source paths."""

    paths: tuple[Path, ...]
    tensor: NDArray[np.float32]


@dataclass
class OutputMetricAccumulator:
    """Streaming aggregate for per-output comparison metrics."""

    name: str
    batches: int = 0
    tensor_values: int = 0
    max_abs_error: float = 0.0
    mean_abs_error_weighted_sum: float = 0.0
    rmse_squared_weighted_sum: float = 0.0
    cosine_similarity_sum: float = 0.0
    cosine_similarity_min: float = 1.0
    reference_l2_norm_sum: float = 0.0
    candidate_l2_norm_sum: float = 0.0
    candidate_l2_norm_min: float | None = None

    def update(self, comparison: OutputComparison) -> None:
        if comparison.name != self.name:
            raise ValueError(f"comparison name mismatch: {comparison.name!r} != {self.name!r}")
        value_count = int(np.prod(comparison.shape))
        self.batches += 1
        self.tensor_values += value_count
        self.max_abs_error = max(self.max_abs_error, comparison.max_abs_error)
        self.mean_abs_error_weighted_sum += comparison.mean_abs_error * value_count
        self.rmse_squared_weighted_sum += (
            comparison.root_mean_square_error**2 * value_count
        )
        self.cosine_similarity_sum += comparison.cosine_similarity
        self.cosine_similarity_min = min(
            self.cosine_similarity_min,
            comparison.cosine_similarity,
        )
        self.reference_l2_norm_sum += comparison.reference_l2_norm
        self.candidate_l2_norm_sum += comparison.candidate_l2_norm
        self.candidate_l2_norm_min = (
            comparison.candidate_l2_norm
            if self.candidate_l2_norm_min is None
            else min(self.candidate_l2_norm_min, comparison.candidate_l2_norm)
        )

    def to_json(self) -> dict[str, object]:
        if self.batches < 1 or self.tensor_values < 1:
            payload = asdict(self)
            payload["mean_abs_error"] = None
            payload["root_mean_square_error"] = None
            payload["cosine_similarity_mean"] = None
            payload["reference_l2_norm_mean"] = None
            payload["candidate_l2_norm_mean"] = None
            payload["candidate_l2_norm_min"] = None
            return payload
        return {
            "name": self.name,
            "batches": self.batches,
            "tensor_values": self.tensor_values,
            "max_abs_error": self.max_abs_error,
            "mean_abs_error": self.mean_abs_error_weighted_sum / self.tensor_values,
            "root_mean_square_error": float(
                np.sqrt(self.rmse_squared_weighted_sum / self.tensor_values)
            ),
            "cosine_similarity_min": self.cosine_similarity_min,
            "cosine_similarity_mean": self.cosine_similarity_sum / self.batches,
            "reference_l2_norm_mean": self.reference_l2_norm_sum / self.batches,
            "candidate_l2_norm_mean": self.candidate_l2_norm_sum / self.batches,
            "candidate_l2_norm_min": self.candidate_l2_norm_min,
        }


def list_image_paths(
    root: Path,
    *,
    recursive: bool = True,
    limit: int | None = None,
) -> tuple[Path, ...]:
    """Return supported image files below `root` in stable order."""

    if not root.exists():
        raise FileNotFoundError(f"image root does not exist: {root}")
    paths: tuple[Path, ...]
    if root.is_file():
        paths = (root,) if root.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS else ()
    else:
        iterator: Iterable[Path] = root.rglob("*") if recursive else root.iterdir()
        paths = tuple(
            sorted(
                path
                for path in iterator
                if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            )
        )
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        paths = tuple(paths[:limit])
    if not paths:
        raise ValueError(f"no supported image files found under: {root}")
    return paths


def load_image_batch(paths: tuple[Path, ...], *, image_size: int = 224) -> ImageBatch:
    """Load RGB images and return one normalized NCHW float32 batch."""

    if not paths:
        raise ValueError("at least one image path is required")
    tensors: list[NDArray[np.float32]] = []
    for path in paths:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            array = np.asarray(rgb, dtype=np.uint8)
        tensors.append(hwc_uint8_to_nchw_float32(array, image_size=image_size))
    return ImageBatch(paths=paths, tensor=np.concatenate(tensors, axis=0))


def chunk_paths(paths: tuple[Path, ...], batch_size: int) -> tuple[tuple[Path, ...], ...]:
    """Split paths into stable non-empty chunks."""

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    return tuple(paths[index : index + batch_size] for index in range(0, len(paths), batch_size))


def write_image_manifest(
    path: Path,
    *,
    image_root: Path,
    images: tuple[Path, ...],
    seed: int,
    split: str,
) -> None:
    """Write a reproducible JSON manifest with paths relative to `image_root` when possible."""

    payload = {
        "image_root": str(image_root),
        "split": split,
        "seed": seed,
        "image_count": len(images),
        "images": [
            str(image.relative_to(image_root) if image.is_relative_to(image_root) else image)
            for image in images
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_image_manifest(path: Path) -> tuple[Path, ...]:
    """Read a JSON image manifest produced by `write_image_manifest`."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    image_root = Path(str(payload["image_root"]))
    images = payload["images"]
    if not isinstance(images, list):
        raise ValueError("manifest images must be a list")
    resolved: list[Path] = []
    for value in images:
        image_path = Path(str(value))
        resolved.append(image_path if image_path.is_absolute() else image_root / image_path)
    if not resolved:
        raise ValueError(f"manifest contains no images: {path}")
    return tuple(resolved)


def stratified_eval_calib_split(
    paths: tuple[Path, ...],
    *,
    image_root: Path,
    eval_count: int,
    calib_count: int,
    seed: int,
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Return disjoint eval/calibration image paths using parent-directory round-robin."""

    if eval_count < 1 or calib_count < 1:
        raise ValueError("eval_count and calib_count must be >= 1")
    if len(paths) < eval_count + calib_count:
        raise ValueError(
            f"need at least {eval_count + calib_count} images, found {len(paths)}"
        )
    rng = random.Random(seed)
    groups: dict[str, list[Path]] = {}
    for path in paths:
        try:
            relative_parent = path.parent.relative_to(image_root)
            key = str(relative_parent) if str(relative_parent) != "." else "__root__"
        except ValueError:
            key = str(path.parent)
        groups.setdefault(key, []).append(path)
    for values in groups.values():
        values.sort()
        rng.shuffle(values)

    def take(count: int) -> tuple[Path, ...]:
        selected: list[Path] = []
        keys = sorted(groups)
        while len(selected) < count:
            progressed = False
            rng.shuffle(keys)
            for key in keys:
                values = groups[key]
                if values:
                    selected.append(values.pop())
                    progressed = True
                    if len(selected) == count:
                        break
            if not progressed:
                raise ValueError("not enough images to complete split")
        return tuple(selected)

    return take(eval_count), take(calib_count)
