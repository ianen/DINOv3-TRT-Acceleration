"""V1.0.2 ADR-016: 2:4 structured sparsity utilities for DINOv3 ViT-L/16.

Provides NVIDIA Automatic Sparsity (ASP) style magnitude-based 2:4 mask
generation and ONNX weight rewriting helpers. Targets attention QKV
projections + MLP fc1/fc2 weights — the weight-bound layers that benefit
from halved HBM bandwidth on Blackwell sparse Tensor Core kernels.

The library is precision-agnostic — masks are computed on FP32/BF16/FP16
weight arrays uniformly, and stored back into the ONNX initializer tensors
with the same dtype.
"""

from dinov3_trt.sparsity.sparsify import (
    SparsityPattern,
    apply_2to4_mask,
    compute_2to4_mask,
    is_2to4_compatible,
)

__all__ = [
    "SparsityPattern",
    "apply_2to4_mask",
    "compute_2to4_mask",
    "is_2to4_compatible",
]
