"""Tests for `dinov3_trt.sparsity.sparsify`."""

from __future__ import annotations

import numpy as np
import pytest

from dinov3_trt.sparsity import (
    SparsityPattern,
    apply_2to4_mask,
    compute_2to4_mask,
    is_2to4_compatible,
)


# =============================================================================
# is_2to4_compatible
# =============================================================================


def test_is_2to4_compatible_accepts_divisible_last_dim() -> None:
    assert is_2to4_compatible((1024, 1024)) is True
    assert is_2to4_compatible((4,)) is True
    assert is_2to4_compatible((3, 2, 8)) is True


def test_is_2to4_compatible_rejects_non_divisible_last_dim() -> None:
    assert is_2to4_compatible((1024, 1023)) is False
    assert is_2to4_compatible((5,)) is False
    assert is_2to4_compatible((1024, 6)) is False


def test_is_2to4_compatible_rejects_empty_or_zero_axis() -> None:
    assert is_2to4_compatible(()) is False
    assert is_2to4_compatible((0,)) is False


def test_is_2to4_compatible_handles_negative_axis() -> None:
    # axis -2 selects second-from-last
    assert is_2to4_compatible((8, 12), axis=-2) is True  # 8 % 4 == 0
    assert is_2to4_compatible((7, 12), axis=-2) is False  # 7 % 4 != 0


# =============================================================================
# SparsityPattern validation
# =============================================================================


def test_sparsity_pattern_default_is_2to4() -> None:
    pattern = SparsityPattern()
    assert pattern.group_size == 4
    assert pattern.nonzero_per_group == 2


def test_sparsity_pattern_rejects_unsupported_group_size() -> None:
    with pytest.raises(ValueError, match="group_size=4"):
        SparsityPattern(group_size=8)


def test_sparsity_pattern_rejects_unsupported_density() -> None:
    with pytest.raises(ValueError, match="nonzero_per_group=2"):
        SparsityPattern(nonzero_per_group=1)


# =============================================================================
# compute_2to4_mask
# =============================================================================


def test_mask_density_is_exactly_half() -> None:
    rng = np.random.default_rng(seed=42)
    weight = rng.normal(size=(64, 1024)).astype(np.float32)
    mask = compute_2to4_mask(weight)
    assert mask.shape == weight.shape
    assert mask.dtype == bool
    # Exactly 50% True overall and per group of 4
    assert mask.sum() == weight.size // 2
    grouped = mask.reshape(64, 1024 // 4, 4)
    assert (grouped.sum(axis=-1) == 2).all()


def test_mask_keeps_top_2_magnitude_in_simple_group() -> None:
    # 1×4 with abs magnitudes [1, 5, 3, 2]: top-2 are indices 1, 2.
    weight = np.array([[1.0, -5.0, 3.0, -2.0]], dtype=np.float32)
    mask = compute_2to4_mask(weight)
    expected = np.array([[False, True, True, False]], dtype=bool)
    np.testing.assert_array_equal(mask, expected)


def test_mask_keeps_top_2_magnitude_when_negative_dominates() -> None:
    weight = np.array([[0.1, -10.0, -8.0, 0.2]], dtype=np.float32)
    mask = compute_2to4_mask(weight)
    expected = np.array([[False, True, True, False]], dtype=bool)
    np.testing.assert_array_equal(mask, expected)


def test_mask_independent_groups() -> None:
    # Two groups: [-1, 2, -3, 4] -> keep 3,4 (abs 3,4); [5, -6, 7, 0] -> keep 6,7 (abs 6,7)
    weight = np.array([[-1.0, 2.0, -3.0, 4.0, 5.0, -6.0, 7.0, 0.0]], dtype=np.float32)
    mask = compute_2to4_mask(weight)
    expected = np.array(
        [[False, False, True, True, False, True, True, False]], dtype=bool
    )
    np.testing.assert_array_equal(mask, expected)


def test_mask_first_tie_breaker_keeps_lower_index() -> None:
    # All four equal magnitude → "first" stable argsort keeps indices 0, 1.
    weight = np.array([[1.0, -1.0, 1.0, -1.0]], dtype=np.float32)
    mask = compute_2to4_mask(weight, tie_breaker="first")
    expected = np.array([[True, True, False, False]], dtype=bool)
    np.testing.assert_array_equal(mask, expected)


def test_mask_last_tie_breaker_keeps_higher_index() -> None:
    weight = np.array([[1.0, -1.0, 1.0, -1.0]], dtype=np.float32)
    mask = compute_2to4_mask(weight, tie_breaker="last")
    expected = np.array([[False, False, True, True]], dtype=bool)
    np.testing.assert_array_equal(mask, expected)


def test_mask_invalid_tie_breaker_raises() -> None:
    weight = np.zeros((1, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="unknown tie_breaker"):
        compute_2to4_mask(weight, tie_breaker="middle")  # type: ignore[arg-type]


def test_mask_rejects_non_array_input() -> None:
    with pytest.raises(TypeError, match="numpy.ndarray"):
        compute_2to4_mask([1.0, 2.0, 3.0, 4.0])  # type: ignore[arg-type]


def test_mask_rejects_non_divisible_shape() -> None:
    weight = np.zeros((10, 7), dtype=np.float32)
    with pytest.raises(ValueError, match="not compatible with 2:4"):
        compute_2to4_mask(weight)


def test_mask_works_on_3d_weight() -> None:
    # e.g. attention QKV proj reshaped as [num_heads, head_dim, in_features]
    rng = np.random.default_rng(seed=123)
    weight = rng.normal(size=(16, 64, 1024)).astype(np.float32)
    mask = compute_2to4_mask(weight)
    assert mask.shape == weight.shape
    grouped = mask.reshape(16, 64, 1024 // 4, 4)
    assert (grouped.sum(axis=-1) == 2).all()


def test_mask_handles_bf16_input_via_float64_promotion() -> None:
    # We promote to float64 internally; bf16 is rare in numpy, simulate via fp32.
    weight = np.array([[0.1, -0.001, 0.05, 0.2]], dtype=np.float32)
    mask = compute_2to4_mask(weight)
    # top-2 abs = 0.2, 0.1 -> indices 3, 0.
    expected = np.array([[True, False, False, True]], dtype=bool)
    np.testing.assert_array_equal(mask, expected)


# =============================================================================
# apply_2to4_mask
# =============================================================================


def test_apply_zeroes_out_pruned_positions() -> None:
    weight = np.array([[1.0, -5.0, 3.0, -2.0]], dtype=np.float32)
    masked = apply_2to4_mask(weight)
    expected = np.array([[0.0, -5.0, 3.0, 0.0]], dtype=np.float32)
    np.testing.assert_array_equal(masked, expected)


def test_apply_does_not_modify_input() -> None:
    weight = np.array([[1.0, -5.0, 3.0, -2.0]], dtype=np.float32)
    snapshot = weight.copy()
    apply_2to4_mask(weight)
    np.testing.assert_array_equal(weight, snapshot)


def test_apply_preserves_dtype() -> None:
    weight = np.array([[1.0, -5.0, 3.0, -2.0]], dtype=np.float16)
    masked = apply_2to4_mask(weight)
    assert masked.dtype == np.float16


def test_apply_with_external_mask_uses_it() -> None:
    weight = np.array([[1.0, -5.0, 3.0, -2.0]], dtype=np.float32)
    custom_mask = np.array([[True, False, False, True]], dtype=bool)
    masked = apply_2to4_mask(weight, mask=custom_mask)
    expected = np.array([[1.0, 0.0, 0.0, -2.0]], dtype=np.float32)
    np.testing.assert_array_equal(masked, expected)


def test_apply_rejects_mask_shape_mismatch() -> None:
    weight = np.zeros((4,), dtype=np.float32)
    bad_mask = np.array([True, False], dtype=bool)
    with pytest.raises(ValueError, match="does not match"):
        apply_2to4_mask(weight, mask=bad_mask)


def test_apply_rejects_non_bool_mask() -> None:
    weight = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    bad_mask = np.array([1, 0, 1, 0], dtype=np.int8)
    with pytest.raises(TypeError, match="must be bool"):
        apply_2to4_mask(weight, mask=bad_mask)


# =============================================================================
# End-to-end: realistic ViT attention QKV weight matrix
# =============================================================================


def test_end_to_end_qkv_projection_weight_round_trip() -> None:
    """Smoke test on a tensor shaped like a real ViT-L attention QKV weight.

    DINOv3 ViT-L/16 attention QKV projection: [3 * 1024, 1024] (in BF16).
    We use FP32 here for numerical clarity; the algorithm is dtype-agnostic.
    """
    rng = np.random.default_rng(seed=2026)
    weight = rng.normal(scale=0.02, size=(3 * 1024, 1024)).astype(np.float32)
    masked = apply_2to4_mask(weight)
    # Density is exactly 50%
    nonzero_count = (masked != 0).sum()
    assert nonzero_count == weight.size // 2
    # Group constraint per row of 4-element groups
    grouped_nonzero = (masked != 0).reshape(3 * 1024, 1024 // 4, 4).sum(axis=-1)
    assert (grouped_nonzero == 2).all()
    # Frobenius norm shrinks but not catastrophically (top-half magnitude kept)
    energy_ratio = float(np.linalg.norm(masked) / np.linalg.norm(weight))
    assert energy_ratio > 0.85  # empirically ~0.93 for unit-variance Gaussian
