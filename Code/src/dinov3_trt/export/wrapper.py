"""DINOv3 intermediate-layer wrapper for PyTorch and ONNX export."""

from __future__ import annotations

import importlib
from typing import Any, Iterable, Sequence

from dinov3_trt.contracts import DINO_VITL16_224_CONTRACT, ModelContract


class DinoV3IntermediateLayerWrapper:
    """Wrap a DINOv3-like model and return 4 ordered intermediate features.

    The official DINOv3 source returns patch tokens only by default. The project
    contract keeps the class token and drops only register/storage tokens, so the
    wrapper requests `return_class_token=True` when the upstream model supports it
    and prepends that class token to each patch-token output.
    """

    def __init__(self, model: Any, contract: ModelContract = DINO_VITL16_224_CONTRACT):
        if not hasattr(model, "get_intermediate_layers"):
            raise TypeError("model must expose get_intermediate_layers")
        self.model = model
        self.contract = contract

    @property
    def output_names(self) -> tuple[str, ...]:
        return self.contract.output_names

    @property
    def layer_indices(self) -> tuple[int, ...]:
        return self.contract.layer_indices

    def eval(self) -> "DinoV3IntermediateLayerWrapper":
        maybe_eval = getattr(self.model, "eval", None)
        if callable(maybe_eval):
            maybe_eval()
        return self

    def __call__(self, pixel_values: Any) -> tuple[Any, ...]:
        return self.forward(pixel_values)

    def forward(self, pixel_values: Any) -> tuple[Any, ...]:
        outputs = self._get_intermediate_layers(pixel_values)
        if len(outputs) != len(self.contract.output_names):
            raise ValueError(
                f"expected {len(self.contract.output_names)} intermediate outputs, got {len(outputs)}"
            )
        return tuple(outputs)

    def _get_intermediate_layers(self, pixel_values: Any) -> Sequence[Any]:
        getter = self.model.get_intermediate_layers
        try:
            outputs = tuple(
                getter(
                    pixel_values,
                    n=self.contract.layer_indices,
                    return_class_token=True,
                )
            )
        except TypeError:
            try:
                outputs = tuple(getter(pixel_values, n=self.contract.layer_indices))
            except TypeError:
                outputs = tuple(getter(pixel_values, self.contract.layer_indices))
        return tuple(_normalize_intermediate_output(output) for output in outputs)


def _normalize_intermediate_output(output: Any) -> Any:
    if not (isinstance(output, tuple) and len(output) == 2):
        return output
    patch_tokens, class_token = output
    return prepend_class_token(patch_tokens, class_token)


def prepend_class_token(patch_tokens: Any, class_token: Any) -> Any:
    """Return `[B, 1 + patch_tokens, C]` for Torch tensors or NumPy arrays."""

    class_sequence = _as_sequence_token(class_token)

    try:
        torch = importlib.import_module("torch")
    except ImportError:
        torch = None
    if torch is not None and isinstance(patch_tokens, torch.Tensor):
        return torch.cat((class_sequence, patch_tokens), dim=1)

    numpy = importlib.import_module("numpy")
    if isinstance(patch_tokens, numpy.ndarray):
        return numpy.concatenate((class_sequence, patch_tokens), axis=1)

    raise TypeError(f"unsupported token output type: {type(patch_tokens)!r}")


def _as_sequence_token(class_token: Any) -> Any:
    shape = getattr(class_token, "shape", None)
    if shape is not None and len(shape) == 3:
        return class_token
    if hasattr(class_token, "unsqueeze"):
        return class_token.unsqueeze(1)
    return class_token[:, None, :]


def ordered_output_mapping(
    names: Iterable[str],
    outputs: Sequence[Any],
) -> dict[str, Any]:
    """Pair output names and tensors while checking cardinality."""

    name_tuple = tuple(names)
    if len(name_tuple) != len(outputs):
        raise ValueError(f"output name count {len(name_tuple)} does not match outputs {len(outputs)}")
    return dict(zip(name_tuple, outputs))
