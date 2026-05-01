import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from dinov3_trt.export.hf_model import (
    HFDinoV3IntermediateLayerWrapper,
    create_hf_dinov3_model,
    drop_register_tokens,
    patch_hf_dinov3_rope_for_onnx_export,
)


class FakeHFDinoV3Model:
    def __init__(
        self,
        hidden_states: list[NDArray[np.float32]],
        num_register_tokens: int = 4,
    ) -> None:
        self.config = SimpleNamespace(num_register_tokens=num_register_tokens)
        self.hidden_states = hidden_states
        self.calls: list[dict[str, Any]] = []
        self.eval_called = False

    def eval(self) -> None:
        self.eval_called = True

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(hidden_states=self.hidden_states)


def _hidden_states(num_states: int = 25) -> list[NDArray[np.float32]]:
    return [
        np.full((2, 201, 1024), fill_value=index, dtype=np.float32)
        for index in range(num_states)
    ]


def test_hf_wrapper_selects_layers_and_drops_register_tokens() -> None:
    model = FakeHFDinoV3Model(_hidden_states())
    wrapper = HFDinoV3IntermediateLayerWrapper(model).eval()

    outputs = wrapper(np.zeros((2, 3, 224, 224), dtype=np.float32))

    assert model.eval_called is True
    assert len(model.calls) == 1
    assert model.calls[0]["pixel_values"].shape == (2, 3, 224, 224)
    assert model.calls[0]["output_hidden_states"] is True
    assert model.calls[0]["return_dict"] is True
    assert [output.shape for output in outputs] == [(2, 197, 1024)] * 4
    assert outputs[0][0, 0, 0] == 4
    assert outputs[0][0, 1, 0] == 4
    assert outputs[1][0, 0, 0] == 12
    assert outputs[2][0, 0, 0] == 16
    assert outputs[3][0, 0, 0] == 20


def test_hf_wrapper_requires_enough_hidden_states() -> None:
    model = FakeHFDinoV3Model(_hidden_states(num_states=20))
    wrapper = HFDinoV3IntermediateLayerWrapper(model)

    with pytest.raises(ValueError, match="expected at least 21 hidden states"):
        wrapper(np.zeros((1, 3, 224, 224), dtype=np.float32))


def test_hf_wrapper_requires_register_token_config() -> None:
    model = SimpleNamespace(config=SimpleNamespace(), eval=lambda: None)

    with pytest.raises(ValueError, match="num_register_tokens"):
        HFDinoV3IntermediateLayerWrapper(model)


def test_drop_register_tokens_preserves_cls_and_patch_tokens() -> None:
    hidden_state = np.arange(2 * 7 * 3, dtype=np.float32).reshape(2, 7, 3)

    output = drop_register_tokens(hidden_state, num_register_tokens=2)

    assert output.shape == (2, 5, 3)
    np.testing.assert_array_equal(output[:, 0, :], hidden_state[:, 0, :])
    np.testing.assert_array_equal(output[:, 1:, :], hidden_state[:, 3:, :])


def test_drop_register_tokens_rejects_too_few_tokens() -> None:
    hidden_state = np.zeros((1, 5, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="too small"):
        drop_register_tokens(hidden_state, num_register_tokens=4)


def test_create_hf_model_passes_export_attention_implementation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(path: str, **kwargs: Any) -> object:
            calls["path"] = path
            calls["kwargs"] = kwargs
            return object()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModel=FakeAutoModel))

    create_hf_dinov3_model(
        "local-model",
        token="token",
        revision="main",
        local_files_only=True,
        attn_implementation="eager",
    )

    assert calls == {
        "path": "local-model",
        "kwargs": {
            "local_files_only": True,
            "trust_remote_code": False,
            "token": "token",
            "revision": "main",
            "attn_implementation": "eager",
        },
    }


def test_hf_rope_export_patch_replaces_forward_with_no_if_path() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    class FakeRope(torch.nn.Module):  # type: ignore[name-defined, misc]
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(patch_size=16)
            self.inv_freq = torch.arange(0, 1, 4 / 64, dtype=torch.float32)

        def forward(self, pixel_values: Any) -> tuple[Any, Any]:
            raise AssertionError("original forward should be patched")

    model = SimpleNamespace(rope_embeddings=FakeRope().eval())

    patch_count = patch_hf_dinov3_rope_for_onnx_export(model)
    cos, sin = model.rope_embeddings(torch.zeros((1, 3, 224, 224), dtype=torch.float32))

    assert patch_count == 1
    assert cos.shape == (196, 64)
    assert sin.shape == (196, 64)
