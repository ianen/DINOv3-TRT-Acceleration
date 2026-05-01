"""Export-time DINOv3 patches for static eval-only RoPE and block execution."""

from __future__ import annotations

import math
import importlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PatchReport:
    applied: tuple[str, ...]
    already_applied: tuple[str, ...]

    def to_json(self) -> dict[str, list[str]]:
        return {
            "applied": list(self.applied),
            "already_applied": list(self.already_applied),
        }


def export_eval_rope_forward(self: Any, *, H: int, W: int) -> tuple[Any, Any]:
    """RoPE forward path with training-only augmentation branches removed."""

    torch = importlib.import_module("torch")

    device = self.periods.device
    dtype = self.dtype
    dd = {"device": device, "dtype": dtype}

    if self.normalize_coords == "max":
        max_hw = max(H, W)
        coords_h = _static_coordinate_tensor(torch, H, max_hw, dd)
        coords_w = _static_coordinate_tensor(torch, W, max_hw, dd)
    elif self.normalize_coords == "min":
        min_hw = min(H, W)
        coords_h = _static_coordinate_tensor(torch, H, min_hw, dd)
        coords_w = _static_coordinate_tensor(torch, W, min_hw, dd)
    elif self.normalize_coords == "separate":
        coords_h = _static_coordinate_tensor(torch, H, H, dd)
        coords_w = _static_coordinate_tensor(torch, W, W, dd)
    else:
        raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")

    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
    coords = coords.flatten(0, 1)
    coords = 2.0 * coords - 1.0

    angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
    angles = angles.flatten(1, 2)
    angles = torch.cat((angles, angles), dim=1)
    return torch.sin(angles), torch.cos(angles)


def _static_coordinate_tensor(torch: Any, length: int, denominator: int, dd: dict[str, Any]) -> Any:
    values = [(index + 0.5) / denominator for index in range(length)]
    return torch.tensor(values, **dd)


def _export_eval_block_forward_tensor(self: Any, x: Any, rope: Any = None) -> Any:
    x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
    return x_attn + self.ls2(self.mlp(self.norm2(x_attn)))


def export_eval_block_forward(self: Any, x_or_x_list: Any, rope_or_rope_list: Any = None) -> Any:
    """Block forward path with stochastic-depth training branches removed."""

    torch = importlib.import_module("torch")

    if isinstance(x_or_x_list, torch.Tensor):
        return _export_eval_block_forward_tensor(self, x_or_x_list, rope_or_rope_list)
    if isinstance(x_or_x_list, list):
        rope_list = rope_or_rope_list
        if rope_list is None:
            rope_list = [None for _ in x_or_x_list]
        return [
            _export_eval_block_forward_tensor(self, x, rope)
            for x, rope in zip(x_or_x_list, rope_list, strict=True)
        ]
    raise AssertionError(f"unsupported block input type: {type(x_or_x_list)!r}")


def _patch_class_method(cls: type[Any], method_name: str, replacement: Any) -> str:
    marker = f"_dinov3_trt_{method_name}_patched"
    original_name = f"_dinov3_trt_original_{method_name}"
    if getattr(cls, marker, False):
        return "already"
    setattr(cls, original_name, getattr(cls, method_name))
    setattr(cls, method_name, replacement)
    setattr(cls, marker, True)
    return "applied"


def _resolve_block_patch_target(block_module: Any) -> tuple[str, type[Any]]:
    block_cls = getattr(block_module, "Block", None)
    if block_cls is not None:
        return "Block.forward", block_cls
    self_attention_block_cls = getattr(block_module, "SelfAttentionBlock", None)
    if self_attention_block_cls is not None:
        return "SelfAttentionBlock.forward", self_attention_block_cls
    raise AttributeError("dinov3.layers.block exposes neither Block nor SelfAttentionBlock")


def apply_dinov3_export_patches() -> PatchReport:
    """Patch the official DINOv3 modules for ONNX/TensorRT export.

    The patch is intentionally narrow: it replaces only eval-time RoPE coordinate
    generation and transformer block execution. Weight loading and model
    structure stay in the official source tree.
    """

    block_module = importlib.import_module("dinov3.layers.block")
    rope_module = importlib.import_module("dinov3.layers.rope_position_encoding")
    block_label, block_cls = _resolve_block_patch_target(block_module)
    rope_cls = rope_module.RopePositionEmbedding

    applied: list[str] = []
    already_applied: list[str] = []
    targets = (
        (block_label, block_cls, "forward", export_eval_block_forward),
        ("RopePositionEmbedding.forward", rope_cls, "forward", export_eval_rope_forward),
    )
    for label, cls, method_name, replacement in targets:
        result = _patch_class_method(cls, method_name, replacement)
        if result == "applied":
            applied.append(label)
        else:
            already_applied.append(label)

    return PatchReport(applied=tuple(applied), already_applied=tuple(already_applied))
