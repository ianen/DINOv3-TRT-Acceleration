"""2:4 structured sparsity mask generation (NVIDIA ASP-style).

The 2:4 sparsity contract requires that within every group of 4 consecutive
weights along the last (input-channel) dimension, at most 2 are non-zero.
NVIDIA Ampere+ sparse Tensor Core kernels exploit this pattern to halve
weight bandwidth and double compute throughput on the masked dimensions.

For DINOv3 ViT-L/16 we apply the mask post-training (no fine-tuning) to the
``in_features`` dimension of every ``Linear`` layer in attention QKV
projection / output projection / MLP fc1 / MLP fc2. ``in_features = 1024``
is divisible by 4, so the pattern fits cleanly.

This module is precision-agnostic — it operates on ``np.ndarray`` weight
tensors regardless of FP32/BF16/FP16 dtype.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

# Only NVIDIA-supported pattern is 2:4. We expose the constant for callers
# but reject other patterns at the API boundary.
SUPPORTED_GROUP_SIZE: int = 4
SUPPORTED_NONZERO: int = 2


@dataclass(frozen=True)
class SparsityPattern:
    """Describes a structured sparsity pattern.

    The 2:4 NVIDIA pattern is the only one supported by sparse Tensor Cores
    on Ampere/Ada/Blackwell; this dataclass exists to make call sites
    explicit and to document the contract uniformly across the project.
    """

    group_size: int = SUPPORTED_GROUP_SIZE
    nonzero_per_group: int = SUPPORTED_NONZERO
    axis: int = -1
    """Axis along which groups are formed. -1 means the last axis (input
    channel for matmul weights stored as ``[out_features, in_features]``)."""

    def __post_init__(self) -> None:
        if self.group_size != SUPPORTED_GROUP_SIZE:
            raise ValueError(
                f"only group_size={SUPPORTED_GROUP_SIZE} is supported (NVIDIA 2:4 pattern)"
            )
        if self.nonzero_per_group != SUPPORTED_NONZERO:
            raise ValueError(
                f"only nonzero_per_group={SUPPORTED_NONZERO} is supported"
            )


_DEFAULT_PATTERN: SparsityPattern = SparsityPattern()


def is_2to4_compatible(weight_shape: tuple[int, ...], axis: int = -1) -> bool:
    """Return whether ``weight_shape`` admits a 2:4 mask along ``axis``.

    The contract is ``shape[axis] % 4 == 0`` — every group of 4 along the
    target axis must be complete; partial trailing groups are rejected to
    keep the mask deterministic and tile-aligned.
    """

    if not weight_shape:
        return False
    last = weight_shape[axis if axis >= 0 else len(weight_shape) + axis]
    return last > 0 and last % SUPPORTED_GROUP_SIZE == 0


def compute_2to4_mask(
    weight: np.ndarray,
    pattern: SparsityPattern = _DEFAULT_PATTERN,
    tie_breaker: Literal["first", "last"] = "first",
) -> np.ndarray:
    """Compute a boolean 2:4 mask: keep top-2 magnitude per group of 4.

    Parameters
    ----------
    weight
        N-dimensional weight array. The axis given by ``pattern.axis`` must
        be divisible by 4.
    pattern
        Sparsity pattern descriptor (always 2:4 today).
    tie_breaker
        When two weights in a group of 4 have identical absolute values the
        ranker is ambiguous; ``"first"`` (numpy ``argsort`` default for
        stable kind) keeps the lower index, ``"last"`` reverses. Tests can
        pin either to verify deterministic output.

    Returns
    -------
    np.ndarray
        Boolean array with the same shape as ``weight``. Exactly half the
        entries along ``pattern.axis`` are ``True`` per group of 4.
    """

    if not isinstance(weight, np.ndarray):
        raise TypeError("weight must be a numpy.ndarray")
    if not is_2to4_compatible(weight.shape, axis=pattern.axis):
        raise ValueError(
            f"weight shape {weight.shape} not compatible with 2:4 along axis {pattern.axis} "
            f"(last dim must be divisible by 4)"
        )

    axis = pattern.axis if pattern.axis >= 0 else weight.ndim + pattern.axis
    # Move target axis to last to simplify reshape; restore at end.
    moved = np.moveaxis(weight, axis, -1)
    *batch_dims, last = moved.shape
    grouped = moved.reshape(*batch_dims, last // pattern.group_size, pattern.group_size)
    abs_w = np.abs(grouped).astype(np.float64, copy=False)

    # ``argsort`` with kind="stable" preserves index order on ties; descending
    # sort obtained via negation. ``tie_breaker="last"`` flips the group then
    # un-flips indices so the larger original index wins ties.
    if tie_breaker == "first":
        order = np.argsort(-abs_w, axis=-1, kind="stable")
    elif tie_breaker == "last":
        flipped = abs_w[..., ::-1]
        order_flipped = np.argsort(-flipped, axis=-1, kind="stable")
        order = pattern.group_size - 1 - order_flipped
    else:
        raise ValueError(f"unknown tie_breaker={tie_breaker!r}")

    top_indices = order[..., : pattern.nonzero_per_group]
    mask_grouped = np.zeros_like(grouped, dtype=bool)
    np.put_along_axis(mask_grouped, top_indices, True, axis=-1)
    mask = mask_grouped.reshape(*batch_dims, last)
    return np.moveaxis(mask, -1, axis)


def apply_2to4_mask(
    weight: np.ndarray,
    mask: np.ndarray | None = None,
    pattern: SparsityPattern = _DEFAULT_PATTERN,
) -> np.ndarray:
    """Return ``weight`` with the 2:4 mask applied; computes mask if absent.

    The masked weight is a copy — ``weight`` itself is never modified. The
    output dtype matches the input, so callers using BF16/FP16 safetensors
    do not lose precision class.
    """

    if mask is None:
        mask = compute_2to4_mask(weight, pattern=pattern)
    if mask.shape != weight.shape:
        raise ValueError(
            f"mask shape {mask.shape} does not match weight shape {weight.shape}"
        )
    if mask.dtype != bool:
        raise TypeError(f"mask must be bool, got {mask.dtype}")
    return np.where(mask, weight, np.zeros_like(weight))
