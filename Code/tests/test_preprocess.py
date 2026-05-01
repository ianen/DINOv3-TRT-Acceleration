import numpy as np
import pytest

from dinov3_trt.utils.preprocess import ensure_nchw_float32, normalize_nchw


def test_normalize_nchw_uint8_returns_float32_batch() -> None:
    batch = np.zeros((1, 3, 224, 224), dtype=np.uint8)

    normalized = normalize_nchw(batch)

    assert normalized.dtype == np.float32
    assert normalized.shape == (1, 3, 224, 224)
    np.testing.assert_allclose(
        normalized[0, :, 0, 0],
        np.array([-2.117904, -2.035714, -1.804444], dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


def test_ensure_nchw_float32_accepts_hwc_uint8() -> None:
    image = np.zeros((224, 224, 3), dtype=np.uint8)

    normalized = ensure_nchw_float32(image)

    assert normalized.dtype == np.float32
    assert normalized.shape == (1, 3, 224, 224)


def test_ensure_nchw_float32_rejects_wrong_spatial_shape_for_nchw() -> None:
    batch = np.zeros((1, 3, 128, 128), dtype=np.float32)

    with pytest.raises(ValueError, match="spatial shape"):
        ensure_nchw_float32(batch)
