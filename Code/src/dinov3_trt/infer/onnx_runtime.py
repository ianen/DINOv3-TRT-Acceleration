"""Minimal ONNX Runtime execution helper for ONNX-vs-ONNX comparisons."""

from __future__ import annotations

from collections.abc import Callable
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray


FloatTensor = NDArray[np.floating]
SessionFactory = Callable[[str, list[str]], Any]


class OnnxRuntimeExecutionError(RuntimeError):
    """Raised when ONNX Runtime execution cannot run or returns invalid data."""


@dataclass(frozen=True)
class OnnxModelRunConfig:
    """Configuration for one ONNX Runtime model execution."""

    onnx_path: Path
    input_name: str = "pixel_values"
    providers: tuple[str, ...] = ("CUDAExecutionProvider", "CPUExecutionProvider")

    def validate(self) -> None:
        if not self.providers:
            raise ValueError("at least one ONNX Runtime execution provider is required")


def parse_ort_providers(value: str) -> tuple[str, ...]:
    """Parse a comma-separated ONNX Runtime execution provider list."""

    providers = tuple(part.strip() for part in value.split(",") if part.strip())
    if not providers:
        raise ValueError("at least one ONNX Runtime execution provider is required")
    return providers


def run_onnx_model(
    config: OnnxModelRunConfig,
    input_tensor: NDArray[np.float32],
    *,
    session_factory: SessionFactory | None = None,
) -> dict[str, FloatTensor]:
    """Run one ONNX model and return named output tensors."""

    config.validate()
    input_array = np.ascontiguousarray(input_tensor.astype(np.float32, copy=False))
    session = (
        _create_inference_session(str(config.onnx_path), list(config.providers))
        if session_factory is None
        else session_factory(str(config.onnx_path), list(config.providers))
    )
    output_names = tuple(str(output.name) for output in session.get_outputs())
    if not output_names:
        raise OnnxRuntimeExecutionError(f"ONNX model has no outputs: {config.onnx_path}")
    outputs = session.run(list(output_names), {config.input_name: input_array})
    if len(outputs) != len(output_names):
        raise OnnxRuntimeExecutionError(
            f"ONNX Runtime returned {len(outputs)} outputs for {len(output_names)} names"
        )
    return {
        name: cast(FloatTensor, np.asarray(output))
        for name, output in zip(output_names, outputs, strict=True)
    }


def _create_inference_session(onnx_path: str, providers: list[str]) -> Any:
    ort = _import_onnxruntime()
    try:
        return ort.InferenceSession(onnx_path, providers=providers)
    except Exception as exc:  # pragma: no cover - depends on installed ORT build.
        raise OnnxRuntimeExecutionError(
            f"failed to create ONNX Runtime session for {onnx_path}: {exc}"
        ) from exc


def _import_onnxruntime() -> Any:
    try:
        return importlib.import_module("onnxruntime")
    except ImportError as exc:
        raise OnnxRuntimeExecutionError(
            "onnxruntime is required for ONNX execution. Install the `export` extra."
        ) from exc
