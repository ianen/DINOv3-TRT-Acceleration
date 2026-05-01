"""Shared project contracts for DINOv3 ViT-L/16 acceleration."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple

DINO_VITL16_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DINO_VITL16_MODEL_NAME = "dinov3_vit_l16_lvd1689m"
DINO_VITL16_PATCH_SIZE = 16
DINO_VITL16_HIDDEN_SIZE = 1024
DINO_VITL16_REGISTER_TOKENS = 4
DINO_VITL16_NUM_BLOCKS = 24
DINO_VITL16_LAYER_INDICES = (3, 11, 15, 19)
DINO_VITL16_OUTPUT_NAMES = ("feat_layer_4", "feat_layer_12", "feat_layer_16", "feat_layer_20")

# V1.1 stretch goal #3 — 4-layer-combination ablation candidates (0-based block indices).
# Each tuple maps to 1-based "feat_layer_{i+1}" output names.
#   PROJECT  — current main path (layers 4/12/16/20), conservative early-mid layers.
#   DPT      — DPT paper recommended layout (layers 5/11/17/23), uniformly spread across 24 blocks.
#   LATE     — late-heavy variant (layers 6/12/18/24), pushes the deepest hook to the final block.
DINO_VITL16_LAYER_INDICES_PROJECT = (3, 11, 15, 19)
DINO_VITL16_LAYER_INDICES_DPT = (4, 10, 16, 22)
DINO_VITL16_LAYER_INDICES_LATE = (5, 11, 17, 23)
DINO_VITL16_LAYER_ABLATION_CANDIDATES: Mapping[str, Tuple[int, ...]] = MappingProxyType(
    {
        "project": DINO_VITL16_LAYER_INDICES_PROJECT,
        "dpt": DINO_VITL16_LAYER_INDICES_DPT,
        "late": DINO_VITL16_LAYER_INDICES_LATE,
    }
)


@dataclass(frozen=True)
class ModelContract:
    """Static invariants that must match PyTorch, ONNX, TensorRT, and C++ paths."""

    model_id: str
    model_name: str
    image_size: int
    patch_size: int
    hidden_size: int
    num_register_tokens: int
    layer_indices: Tuple[int, ...]
    output_names: Tuple[str, ...]

    @property
    def layer_numbers(self) -> Tuple[int, ...]:
        return tuple(index + 1 for index in self.layer_indices)

    @property
    def patch_grid(self) -> int:
        if self.image_size < self.patch_size:
            raise ValueError("image_size must be at least patch_size")
        return self.image_size // self.patch_size

    @property
    def patch_token_count(self) -> int:
        return self.patch_grid * self.patch_grid


def _normalize_layer_indices(layer_indices: Sequence[int]) -> Tuple[int, ...]:
    """Validate and normalize a 0-based layer index sequence for ViT-L/16."""

    if isinstance(layer_indices, (str, bytes)):
        raise TypeError("layer_indices must be a sequence of int, not str/bytes")
    try:
        indices = tuple(layer_indices)
    except TypeError as exc:
        raise TypeError("layer_indices must be iterable") from exc
    if not indices:
        raise ValueError("layer_indices must be non-empty")
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("layer_indices must contain only int values")
        if index < 0:
            raise ValueError("layer_indices must be non-negative (0-based)")
        if index >= DINO_VITL16_NUM_BLOCKS:
            raise ValueError(
                f"layer_indices must be < {DINO_VITL16_NUM_BLOCKS} for ViT-L/16"
            )
    if len(set(indices)) != len(indices):
        raise ValueError("layer_indices must be unique")
    if list(indices) != sorted(indices):
        raise ValueError("layer_indices must be sorted ascending")
    return indices


def derive_output_names(layer_indices: Sequence[int]) -> Tuple[str, ...]:
    """Map 0-based layer indices to 1-based `feat_layer_{n+1}` output names."""

    indices = _normalize_layer_indices(layer_indices)
    return tuple(f"feat_layer_{index + 1}" for index in indices)


def make_dinov3_vitl16_contract(
    image_size: int,
    *,
    layer_indices: Optional[Sequence[int]] = None,
) -> ModelContract:
    """Return the ViT-L/16 contract for one static export resolution.

    Pass `layer_indices` (0-based) to override the default 4-layer hook set
    `(3, 11, 15, 19)`. Output names are auto-derived as `feat_layer_{i+1}`
    so downstream PyTorch/ONNX/TensorRT/C++ layers stay in lockstep.
    """

    if not isinstance(image_size, int) or isinstance(image_size, bool):
        raise TypeError("image_size must be an integer")
    if image_size < DINO_VITL16_PATCH_SIZE:
        raise ValueError("image_size must be at least patch_size")
    indices: Tuple[int, ...]
    names: Tuple[str, ...]
    if layer_indices is None:
        indices = DINO_VITL16_LAYER_INDICES
        names = DINO_VITL16_OUTPUT_NAMES
    else:
        indices = _normalize_layer_indices(layer_indices)
        names = tuple(f"feat_layer_{index + 1}" for index in indices)
    return ModelContract(
        model_id=DINO_VITL16_MODEL_ID,
        model_name=DINO_VITL16_MODEL_NAME,
        image_size=image_size,
        patch_size=DINO_VITL16_PATCH_SIZE,
        hidden_size=DINO_VITL16_HIDDEN_SIZE,
        num_register_tokens=DINO_VITL16_REGISTER_TOKENS,
        layer_indices=indices,
        output_names=names,
    )


DINO_VITL16_224_CONTRACT = make_dinov3_vitl16_contract(224)


def expected_token_count(
    contract: ModelContract = DINO_VITL16_224_CONTRACT,
    *,
    include_register_tokens: bool = False,
) -> int:
    """Return expected sequence length for the project output contract."""

    register_tokens = contract.num_register_tokens if include_register_tokens else 0
    return 1 + register_tokens + contract.patch_token_count


def expected_output_shape(
    batch_size: int,
    contract: ModelContract = DINO_VITL16_224_CONTRACT,
    *,
    include_register_tokens: bool = False,
) -> Tuple[int, int, int]:
    """Return `[B, tokens, hidden]` for one intermediate feature output."""

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    return (
        batch_size,
        expected_token_count(contract, include_register_tokens=include_register_tokens),
        contract.hidden_size,
    )


def _shape_tuple(value: Any) -> Optional[Tuple[int, ...]]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(dim) for dim in shape)
    except TypeError:
        return None


def validate_output_shapes(
    outputs: Mapping[str, Any],
    *,
    batch_size: Optional[int] = None,
    contract: ModelContract = DINO_VITL16_224_CONTRACT,
    include_register_tokens: bool = False,
) -> None:
    """Validate the 4-output feature map contract.

    `outputs` may contain NumPy arrays, Torch tensors, ONNX Runtime outputs wrapped
    in a mapping, or test doubles that expose a `.shape` attribute.
    """

    expected_names = set(contract.output_names)
    actual_names = set(outputs.keys())
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    if missing or extra:
        raise ValueError(f"output names mismatch: missing={missing}, extra={extra}")

    inferred_batch_size = batch_size
    if inferred_batch_size is None:
        first_shape = _shape_tuple(outputs[contract.output_names[0]])
        if first_shape is None or len(first_shape) != 3:
            raise ValueError(f"cannot infer batch size from {contract.output_names[0]}")
        inferred_batch_size = first_shape[0]

    expected = expected_output_shape(
        inferred_batch_size,
        contract,
        include_register_tokens=include_register_tokens,
    )
    bad_shapes = []
    for name in contract.output_names:
        shape = _shape_tuple(outputs[name])
        if shape != expected:
            bad_shapes.append((name, shape, expected))

    if bad_shapes:
        details = "; ".join(
            f"{name}: got {shape}, expected {expected_shape}"
            for name, shape, expected_shape in bad_shapes
        )
        raise ValueError(f"output shape mismatch: {details}")


def binding_names(contract: ModelContract = DINO_VITL16_224_CONTRACT) -> Sequence[str]:
    """Return stable TensorRT output binding names."""

    return contract.output_names
