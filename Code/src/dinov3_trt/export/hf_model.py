"""Hugging Face DINOv3 loading and hidden-state adaptation helpers."""

from __future__ import annotations

import importlib
import math
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dinov3_trt.contracts import DINO_VITL16_224_CONTRACT, ModelContract


def create_hf_dinov3_model(
    model_name_or_path: str | Path,
    *,
    token: str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
    attn_implementation: str | None = None,
) -> Any:
    """Load a DINOv3 model through `transformers.AutoModel`."""

    try:
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise ImportError(
            "transformers is required for HF safetensors export. "
            'Install it with `python -m pip install -e ".[export]"`.'
        ) from exc

    kwargs: dict[str, Any] = {
        "local_files_only": local_files_only,
        "trust_remote_code": False,
    }
    if token is not None:
        kwargs["token"] = token
    if revision is not None:
        kwargs["revision"] = revision
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    return transformers.AutoModel.from_pretrained(str(model_name_or_path), **kwargs)


def patch_hf_dinov3_rope_for_onnx_export(model: Any) -> int:
    """Patch HF DINOv3 RoPE export to avoid ONNX `If` from `Tensor.tile(2)`."""

    rope_modules = _collect_hf_rope_modules(model)
    for rope_module in rope_modules:
        rope_module.forward = types.MethodType(_hf_rope_forward_without_tile_if, rope_module)
    return len(rope_modules)


def _collect_hf_rope_modules(model: Any) -> tuple[Any, ...]:
    direct_rope = getattr(model, "rope_embeddings", None)
    rope_modules: list[Any] = []
    if direct_rope is not None:
        rope_modules.append(direct_rope)

    maybe_modules = getattr(model, "modules", None)
    if callable(maybe_modules):
        for module in maybe_modules():
            if module is direct_rope:
                continue
            if module.__class__.__name__ == "DINOv3ViTRopePositionEmbedding":
                rope_modules.append(module)

    return tuple(rope_modules)


def _hf_rope_forward_without_tile_if(self: Any, pixel_values: Any) -> tuple[Any, Any]:
    torch = importlib.import_module("torch")
    modeling = importlib.import_module("transformers.models.dinov3_vit.modeling_dinov3_vit")
    get_patches_center_coordinates = modeling.get_patches_center_coordinates

    _, _, height, width = pixel_values.shape
    num_patches_h = height // self.config.patch_size
    num_patches_w = width // self.config.patch_size

    device = pixel_values.device
    device_type = device.type if isinstance(device.type, str) and device.type != "mps" else "cpu"

    if self.training:
        raise RuntimeError("HF DINOv3 RoPE ONNX export patch requires eval mode")

    with torch.autocast(device_type=device_type, enabled=False):
        patch_coords = get_patches_center_coordinates(
            num_patches_h,
            num_patches_w,
            dtype=torch.float32,
            device=device,
        )
        angles = 2 * math.pi * patch_coords[:, :, None] * self.inv_freq[None, None, :]
        angles = angles.flatten(1, 2)
        angles = torch.cat((angles, angles), dim=-1)

        cos = torch.cos(angles)
        sin = torch.sin(angles)

    dtype = pixel_values.dtype
    return cos.to(dtype=dtype), sin.to(dtype=dtype)


class HFDinoV3IntermediateLayerWrapper:
    """Select project intermediate layers from HF DINOv3 hidden states.

    Hugging Face DINOv3 ViT hidden states include the embedding output at index
    0, then one tensor per transformer layer. Each tensor includes `[CLS]`,
    register tokens, and patch tokens. The project contract keeps `[CLS]` and
    patch tokens, dropping only register tokens.
    """

    def __init__(
        self,
        model: Any,
        contract: ModelContract = DINO_VITL16_224_CONTRACT,
        num_register_tokens: int | None = None,
    ) -> None:
        self.model = model
        self.contract = contract
        self.num_register_tokens = _resolve_num_register_tokens(model, num_register_tokens)

    @property
    def output_names(self) -> tuple[str, ...]:
        return self.contract.output_names

    @property
    def layer_indices(self) -> tuple[int, ...]:
        return self.contract.layer_indices

    def eval(self) -> "HFDinoV3IntermediateLayerWrapper":
        maybe_eval = getattr(self.model, "eval", None)
        if callable(maybe_eval):
            maybe_eval()
        return self

    def __call__(self, pixel_values: Any) -> tuple[Any, ...]:
        return self.forward(pixel_values)

    def forward(self, pixel_values: Any) -> tuple[Any, ...]:
        outputs = self.model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = _extract_hidden_states(outputs)
        required_index = max(self.contract.layer_indices) + 1
        if len(hidden_states) <= required_index:
            raise ValueError(
                f"expected at least {required_index + 1} hidden states, got {len(hidden_states)}"
            )
        return tuple(
            drop_register_tokens(hidden_states[layer_index + 1], self.num_register_tokens)
            for layer_index in self.contract.layer_indices
        )


def make_hf_export_module(
    model: Any,
    *,
    contract: ModelContract = DINO_VITL16_224_CONTRACT,
    num_register_tokens: int | None = None,
) -> Any:
    """Wrap a HF DINOv3 model in an ``nn.Module`` exposing the 4-output contract.

    The returned object is suitable both for ``torch.onnx.export`` (the
    Transformers exporter sees a regular ``nn.Module``) and for
    ``modelopt.torch.quantization.quantize`` (which mutates the module tree
    in-place to insert input/weight quantizers).
    """

    torch = importlib.import_module("torch")

    class TorchHFDinoV3IntermediateLayerWrapper(torch.nn.Module):  # type: ignore[misc, name-defined]
        def __init__(self, wrapped_model: Any) -> None:
            super().__init__()
            self.model = wrapped_model
            self.inner_wrapper = HFDinoV3IntermediateLayerWrapper(
                self.model,
                contract=contract,
                num_register_tokens=num_register_tokens,
            )

        def forward(self, pixel_values: Any) -> tuple[Any, ...]:
            return self.inner_wrapper.forward(pixel_values)

    return TorchHFDinoV3IntermediateLayerWrapper(model)


def freeze_module_parameters(module: Any) -> Any:
    """Disable autograd on a module/parameter tree before tracing or quantization.

    Works with both ``nn.Module`` objects (calls ``requires_grad_`` if available)
    and bare iterables of parameters. Returns ``module`` for chaining.
    """

    maybe_requires_grad = getattr(module, "requires_grad_", None)
    if callable(maybe_requires_grad):
        maybe_requires_grad(False)
        return module

    maybe_parameters = getattr(module, "parameters", None)
    if callable(maybe_parameters):
        for parameter in maybe_parameters():
            maybe_parameter_requires_grad = getattr(parameter, "requires_grad_", None)
            if callable(maybe_parameter_requires_grad):
                maybe_parameter_requires_grad(False)
            elif hasattr(parameter, "requires_grad"):
                parameter.requires_grad = False
    return module


def drop_register_tokens(hidden_state: Any, num_register_tokens: int) -> Any:
    """Return `[CLS] + patch tokens` from `[CLS] + registers + patch tokens`."""

    if num_register_tokens < 0:
        raise ValueError("num_register_tokens must be >= 0")
    if num_register_tokens == 0:
        return hidden_state

    shape = getattr(hidden_state, "shape", None)
    if shape is None or len(shape) != 3:
        raise ValueError("hidden_state must have shape [batch, tokens, channels]")
    if int(shape[1]) <= 1 + num_register_tokens:
        raise ValueError(
            "hidden_state token dimension is too small for CLS plus register-token removal"
        )

    class_token = hidden_state[:, :1, :]
    patch_tokens = hidden_state[:, 1 + num_register_tokens :, :]

    try:
        torch = importlib.import_module("torch")
    except ImportError:
        torch = None
    if torch is not None and isinstance(hidden_state, torch.Tensor):
        return torch.cat((class_token, patch_tokens), dim=1)

    numpy = importlib.import_module("numpy")
    if isinstance(hidden_state, numpy.ndarray):
        return numpy.concatenate((class_token, patch_tokens), axis=1)

    raise TypeError(f"unsupported hidden_state type: {type(hidden_state)!r}")


def _resolve_num_register_tokens(model: Any, explicit: int | None) -> int:
    value = explicit
    if value is None:
        config = getattr(model, "config", None)
        value = getattr(config, "num_register_tokens", None)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("num_register_tokens must be supplied or present on model.config")
    if value < 0:
        raise ValueError("num_register_tokens must be >= 0")
    return value


def _extract_hidden_states(outputs: Any) -> Sequence[Any]:
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None and isinstance(outputs, Mapping):
        hidden_states = outputs.get("hidden_states")
    if hidden_states is None:
        raise ValueError("HF DINOv3 output did not include hidden_states")
    if not isinstance(hidden_states, Sequence):
        raise ValueError("hidden_states must be a sequence")
    return hidden_states
