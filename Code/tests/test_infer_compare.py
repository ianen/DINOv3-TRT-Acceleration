import numpy as np
import pytest

from dinov3_trt.infer.compare import (
    compare_arrays,
    compare_output_tensors,
    make_input,
    make_random_input,
)


def test_make_random_input_is_deterministic() -> None:
    first = make_random_input(batch_size=2, image_size=8, seed=123)
    second = make_random_input(batch_size=2, image_size=8, seed=123)

    assert first.shape == (2, 3, 8, 8)
    assert first.dtype == np.float32
    np.testing.assert_array_equal(first, second)


def test_make_input_supports_deterministic_modes() -> None:
    zeros = make_input(batch_size=1, image_size=4, mode="zeros")
    ones = make_input(batch_size=1, image_size=4, mode="ones")
    uniform = make_input(batch_size=1, image_size=4, seed=7, mode="uniform-0-1")

    assert zeros.shape == (1, 3, 4, 4)
    assert zeros.dtype == np.float32
    assert uniform.dtype == np.float32
    np.testing.assert_array_equal(zeros, np.zeros((1, 3, 4, 4), dtype=np.float32))
    np.testing.assert_array_equal(ones, np.ones((1, 3, 4, 4), dtype=np.float32))
    assert float(uniform.min()) >= 0.0
    assert float(uniform.max()) < 1.0


def test_compare_arrays_reports_error_and_cosine() -> None:
    reference = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    candidate = np.array([[1.0, 2.25, 2.5]], dtype=np.float32)

    comparison = compare_arrays("feat", reference, candidate)

    assert comparison.name == "feat"
    assert comparison.shape == (1, 3)
    assert comparison.reference_dtype == "float32"
    assert comparison.candidate_dtype == "float32"
    assert comparison.max_abs_error == 0.5
    assert comparison.mean_abs_error == pytest.approx(0.25)
    assert comparison.root_mean_square_error == pytest.approx(0.3227486)
    assert 0.99 < comparison.cosine_similarity < 1.0
    assert comparison.reference_l2_norm > 0
    assert comparison.candidate_l2_norm > 0
    assert comparison.reference_abs_max == pytest.approx(3.0)
    assert comparison.candidate_abs_max == pytest.approx(2.5)


def test_compare_output_tensors_requires_matching_names() -> None:
    reference = {"a": np.array([1.0], dtype=np.float32)}
    candidate = {"b": np.array([1.0], dtype=np.float32)}

    with pytest.raises(ValueError, match="output names mismatch"):
        compare_output_tensors(reference, candidate)


def test_compare_output_tensors_preserves_reference_order() -> None:
    reference = {
        "feat_layer_4": np.array([1.0], dtype=np.float32),
        "feat_layer_12": np.array([1.0], dtype=np.float32),
    }
    candidate = {
        "feat_layer_12": np.array([1.0], dtype=np.float32),
        "feat_layer_4": np.array([1.0], dtype=np.float32),
    }

    comparisons = compare_output_tensors(reference, candidate)

    assert [comparison.name for comparison in comparisons] == ["feat_layer_4", "feat_layer_12"]


def test_compare_arrays_handles_matching_zero_vectors() -> None:
    reference = np.zeros((2,), dtype=np.float32)
    candidate = np.zeros((2,), dtype=np.float32)

    comparison = compare_arrays("zeros", reference, candidate)

    assert comparison.cosine_similarity == 1.0


def test_compare_arrays_rejects_shape_mismatch() -> None:
    reference = np.zeros((1, 2), dtype=np.float32)
    candidate = np.zeros((2, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="shape mismatch"):
        compare_arrays("feat", reference, candidate)


def test_compare_arrays_rejects_nonfinite_reference_values() -> None:
    reference = np.array([1.0, np.nan, np.inf], dtype=np.float32)
    candidate = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    with pytest.raises(
        ValueError,
        match=(
            r"output 'feat' reference contains non-finite values: "
            r"nan=1, inf=1, total=3"
        ),
    ):
        compare_arrays("feat", reference, candidate)


def test_compare_arrays_rejects_nonfinite_candidate_values() -> None:
    reference = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    candidate = np.array([1.0, np.nan, np.inf], dtype=np.float32)

    with pytest.raises(
        ValueError,
        match=(
            r"output 'feat' candidate contains non-finite values: "
            r"nan=1, inf=1, total=3"
        ),
    ):
        compare_arrays("feat", reference, candidate)
