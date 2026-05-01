"""Shared preprocessing helpers for Python baseline and parity tests."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import numpy.typing as npt
from PIL import Image

IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def normalize_nchw(batch: npt.NDArray[np.generic]) -> npt.NDArray[np.float32]:
    """Normalize an NCHW image batch with ImageNet mean/std.

    Integer inputs are interpreted as 0-255 images. Floating-point inputs with a
    max value greater than 2 are also scaled by 255 to handle pre-converted image
    arrays without silently producing huge normalized values.
    """

    if batch.ndim != 4:
        raise ValueError(f"expected NCHW batch with 4 dims, got shape={batch.shape}")
    if batch.shape[1] != 3:
        raise ValueError(f"expected 3 channels in NCHW batch, got shape={batch.shape}")

    array = batch.astype(np.float32, copy=False)
    if np.issubdtype(batch.dtype, np.integer) or float(np.nanmax(array)) > 2.0:
        array = array / 255.0

    mean = IMAGENET_MEAN.reshape(1, 3, 1, 1)
    std = IMAGENET_STD.reshape(1, 3, 1, 1)
    return (array - mean) / std


def hwc_uint8_to_nchw_float32(
    image: npt.NDArray[np.generic],
    image_size: int = 224,
) -> npt.NDArray[np.float32]:
    """Convert one HWC uint8 RGB image to normalized NCHW float32 batch."""

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HWC RGB image, got shape={image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"expected uint8 image, got dtype={image.dtype}")

    pil_image = Image.fromarray(image, mode="RGB")
    if pil_image.size != (image_size, image_size):
        pil_image = pil_image.resize((image_size, image_size), Image.Resampling.BICUBIC)

    resized = np.asarray(pil_image, dtype=np.uint8)
    nchw = resized.transpose(2, 0, 1)[None, ...]
    return normalize_nchw(nchw)


def ensure_nchw_float32(
    value: npt.NDArray[np.generic],
    image_size: int = 224,
) -> npt.NDArray[np.float32]:
    """Accept HWC uint8 or NCHW numeric input and return normalized NCHW float32."""

    if value.ndim == 3:
        return hwc_uint8_to_nchw_float32(value, image_size=image_size)
    if value.ndim == 4:
        normalized = normalize_nchw(value)
        expected_hw: Tuple[int, int] = (image_size, image_size)
        if normalized.shape[2:] != expected_hw:
            raise ValueError(
                f"expected spatial shape {expected_hw} for NCHW input, got {normalized.shape[2:]}"
            )
        return normalized.astype(np.float32, copy=False)
    raise ValueError(f"expected HWC image or NCHW batch, got shape={value.shape}")
