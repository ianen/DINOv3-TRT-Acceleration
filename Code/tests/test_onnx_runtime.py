from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dinov3_trt.infer.onnx_runtime import (
    OnnxModelRunConfig,
    OnnxRuntimeExecutionError,
    parse_ort_providers,
    run_onnx_model,
)


class _FakeOutput:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSession:
    def __init__(self, path: str, providers: list[str]) -> None:
        self.path = path
        self.providers = providers

    def get_outputs(self) -> list[_FakeOutput]:
        return [_FakeOutput("feat_layer_4"), _FakeOutput("feat_layer_12")]

    def run(self, output_names: list[str], feeds: dict[str, np.ndarray[Any, Any]]) -> list[object]:
        assert output_names == ["feat_layer_4", "feat_layer_12"]
        input_tensor = feeds["pixel_values"]
        return [input_tensor + 1.0, input_tensor + 2.0]


def test_parse_ort_providers_rejects_empty() -> None:
    with pytest.raises(ValueError, match="execution provider"):
        parse_ort_providers(" , ")


def test_parse_ort_providers_strips_values() -> None:
    assert parse_ort_providers("CUDAExecutionProvider, CPUExecutionProvider") == (
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    )


def test_run_onnx_model_returns_named_outputs_with_fake_session(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def session_factory(path: str, providers: list[str]) -> _FakeSession:
        captured["path"] = path
        captured["providers"] = providers
        return _FakeSession(path, providers)

    model_path = tmp_path / "model.onnx"
    input_tensor = np.zeros((1, 3, 4, 4), dtype=np.float32)

    outputs = run_onnx_model(
        OnnxModelRunConfig(
            onnx_path=model_path,
            providers=("CPUExecutionProvider",),
        ),
        input_tensor,
        session_factory=session_factory,
    )

    assert captured["path"] == str(model_path)
    assert captured["providers"] == ["CPUExecutionProvider"]
    assert set(outputs) == {"feat_layer_4", "feat_layer_12"}
    np.testing.assert_array_equal(outputs["feat_layer_4"], input_tensor + 1.0)


def test_run_onnx_model_rejects_output_count_mismatch(tmp_path: Path) -> None:
    class MismatchSession(_FakeSession):
        def run(
            self,
            output_names: list[str],
            feeds: dict[str, np.ndarray[Any, Any]],
        ) -> list[object]:
            return [feeds["pixel_values"]]

    with pytest.raises(OnnxRuntimeExecutionError, match="returned 1 outputs"):
        run_onnx_model(
            OnnxModelRunConfig(onnx_path=tmp_path / "model.onnx"),
            np.zeros((1, 3, 4, 4), dtype=np.float32),
            session_factory=lambda path, providers: MismatchSession(path, providers),
        )
