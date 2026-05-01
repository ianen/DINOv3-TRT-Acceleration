import sys
import types
from typing import Any

import pytest

from dinov3_trt.export.rope_patch import (
    apply_dinov3_export_patches,
    export_eval_block_forward,
    export_eval_rope_forward,
)


class FakeSelfAttentionBlock:
    def forward(self, x_or_x_list: Any, rope_or_rope_list: Any = None) -> Any:
        return x_or_x_list, rope_or_rope_list


class FakeRopePositionEmbedding:
    def forward(self, *, H: int, W: int) -> tuple[int, int]:
        return H, W


def install_fake_dinov3_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    dinov3_module = types.ModuleType("dinov3")
    layers_module = types.ModuleType("dinov3.layers")
    block_module = types.ModuleType("dinov3.layers.block")
    rope_module = types.ModuleType("dinov3.layers.rope_position_encoding")
    setattr(block_module, "SelfAttentionBlock", FakeSelfAttentionBlock)
    setattr(rope_module, "RopePositionEmbedding", FakeRopePositionEmbedding)

    monkeypatch.setitem(sys.modules, "dinov3", dinov3_module)
    monkeypatch.setitem(sys.modules, "dinov3.layers", layers_module)
    monkeypatch.setitem(sys.modules, "dinov3.layers.block", block_module)
    monkeypatch.setitem(sys.modules, "dinov3.layers.rope_position_encoding", rope_module)


def test_apply_dinov3_export_patches_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_dinov3_modules(monkeypatch)

    first = apply_dinov3_export_patches()
    second = apply_dinov3_export_patches()

    assert first.applied == ("SelfAttentionBlock.forward", "RopePositionEmbedding.forward")
    assert first.already_applied == ()
    assert second.applied == ()
    assert second.already_applied == ("SelfAttentionBlock.forward", "RopePositionEmbedding.forward")
    assert FakeSelfAttentionBlock.forward is export_eval_block_forward
    assert FakeRopePositionEmbedding.forward is export_eval_rope_forward
    assert hasattr(FakeSelfAttentionBlock, "_dinov3_trt_original_forward")
    assert hasattr(FakeRopePositionEmbedding, "_dinov3_trt_original_forward")
