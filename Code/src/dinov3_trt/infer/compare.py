"""Numerical metrics for comparing TensorRT outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Mapping

import numpy as np
from numpy.typing import NDArray


FloatTensor = NDArray[np.floating]
InputMode = Literal["random-normal", "uniform-0-1", "zeros", "ones"]


@dataclass(frozen=True)
class OutputComparison:
    """Per-output numerical difference metrics."""

    name: str
    shape: tuple[int, ...]
    reference_dtype: str
    candidate_dtype: str
    max_abs_error: float
    mean_abs_error: float
    root_mean_square_error: float
    cosine_similarity: float
    reference_l2_norm: float
    candidate_l2_norm: float
    reference_abs_max: float
    candidate_abs_max: float

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def make_random_input(
    batch_size: int,
    image_size: int = 224,
    seed: int = 0,
) -> NDArray[np.float32]:
    """Return a deterministic NCHW float32 input tensor."""

    return make_input(
        batch_size=batch_size,
        image_size=image_size,
        seed=seed,
        mode="random-normal",
    )


def make_input(
    batch_size: int,
    image_size: int = 224,
    seed: int = 0,
    mode: InputMode = "random-normal",
) -> NDArray[np.float32]:
    """Return a deterministic NCHW float32 input tensor for engine comparisons."""

    if batch_size < 1:
        raise ValueError("batch size must be >= 1")
    if image_size < 1:
        raise ValueError("image size must be >= 1")
    shape = (batch_size, 3, image_size, image_size)
    if mode == "zeros":
        return np.zeros(shape, dtype=np.float32)
    if mode == "ones":
        return np.ones(shape, dtype=np.float32)
    rng = np.random.default_rng(seed)
    if mode == "random-normal":
        return rng.standard_normal(shape, dtype=np.float32)
    if mode == "uniform-0-1":
        return rng.random(shape, dtype=np.float32)
    raise ValueError(f"unsupported input mode: {mode}")


def _raise_for_nonfinite_values(name: str, role: str, values: FloatTensor) -> None:
    if np.isfinite(values).all():
        return
    nan_count = int(np.isnan(values).sum())
    inf_count = int(np.isinf(values).sum())
    raise ValueError(
        f"output {name!r} {role} contains non-finite values: "
        f"nan={nan_count}, inf={inf_count}, total={values.size}"
    )


def compare_arrays(
    name: str,
    reference: FloatTensor,
    candidate: FloatTensor,
) -> OutputComparison:
    """Compare one output tensor using absolute error and cosine similarity."""

    if reference.shape != candidate.shape:
        raise ValueError(
            f"output {name!r} shape mismatch: {reference.shape} != {candidate.shape}"
        )
    _raise_for_nonfinite_values(name, "reference", reference)
    _raise_for_nonfinite_values(name, "candidate", candidate)

    reference64 = reference.astype(np.float64, copy=False)
    candidate64 = candidate.astype(np.float64, copy=False)
    diff = np.abs(reference64 - candidate64)
    reference_norm = float(np.linalg.norm(reference64.ravel()))
    candidate_norm = float(np.linalg.norm(candidate64.ravel()))
    denominator = reference_norm * candidate_norm
    if denominator == 0:
        cosine = 1.0 if np.array_equal(reference64, candidate64) else 0.0
    else:
        cosine = float(np.dot(reference64.ravel(), candidate64.ravel()) / denominator)
    return OutputComparison(
        name=name,
        shape=tuple(int(dim) for dim in reference.shape),
        reference_dtype=str(reference.dtype),
        candidate_dtype=str(candidate.dtype),
        max_abs_error=float(np.max(diff)),
        mean_abs_error=float(np.mean(diff)),
        root_mean_square_error=float(np.sqrt(np.mean(np.square(diff)))),
        cosine_similarity=cosine,
        reference_l2_norm=reference_norm,
        candidate_l2_norm=candidate_norm,
        reference_abs_max=float(np.max(np.abs(reference64))),
        candidate_abs_max=float(np.max(np.abs(candidate64))),
    )


def compare_output_tensors(
    reference_outputs: Mapping[str, FloatTensor],
    candidate_outputs: Mapping[str, FloatTensor],
) -> tuple[OutputComparison, ...]:
    """Compare two named output maps in stable output-name order."""

    reference_names = set(reference_outputs)
    candidate_names = set(candidate_outputs)
    if reference_names != candidate_names:
        missing = sorted(reference_names - candidate_names)
        unexpected = sorted(candidate_names - reference_names)
        raise ValueError(f"output names mismatch: missing={missing}, unexpected={unexpected}")
    return tuple(
        compare_arrays(name, reference_outputs[name], candidate_outputs[name])
        for name in reference_outputs
    )
