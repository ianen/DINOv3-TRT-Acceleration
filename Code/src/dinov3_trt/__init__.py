"""DINOv3 TensorRT acceleration utilities."""

from dinov3_trt.contracts import (
    DINO_VITL16_224_CONTRACT,
    ModelContract,
    expected_output_shape,
    expected_token_count,
    validate_output_shapes,
)

__all__ = [
    "DINO_VITL16_224_CONTRACT",
    "ModelContract",
    "expected_output_shape",
    "expected_token_count",
    "validate_output_shapes",
]
