"""Helpers for Python-vs-C++ TensorRT runtime parity checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from dinov3_trt.infer.compare import compare_output_tensors
from dinov3_trt.infer.trt_runtime import TensorRTEngineRunConfig, run_engine


def make_cpp_parity_input(
    *,
    batch_size: int,
    image_size: int = 224,
) -> NDArray[np.float32]:
    """Return the deterministic sine input used by the C++ parity tools."""

    if batch_size < 1:
        raise ValueError("batch size must be >= 1")
    if image_size < 1:
        raise ValueError("image size must be >= 1")
    shape = (batch_size, 3, image_size, image_size)
    count = int(np.prod(shape))
    indices = np.arange(count, dtype=np.uint64)
    phases = ((indices % 1009) + 1).astype(np.float64) * 0.017
    return np.sin(phases).astype(np.float32).reshape(shape)


def read_cpp_output_dump(manifest_path: Path) -> dict[str, NDArray[np.float32]]:
    """Read a C++ output dump manifest and binary float32 tensors."""

    manifest = _load_manifest(manifest_path)
    outputs = _required_list(manifest, "outputs")
    tensors: dict[str, NDArray[np.float32]] = {}
    for output in outputs:
        output_map = _as_mapping(output, "output")
        name = _required_string(output_map, "name")
        dtype = _required_string(output_map, "dtype")
        if dtype != "float32":
            raise ValueError(f"unsupported C++ dump dtype for {name!r}: {dtype}")
        shape = tuple(_required_int_list(output_map, "shape"))
        relative_path = Path(_required_string(output_map, "path"))
        tensor_path = relative_path if relative_path.is_absolute() else manifest_path.parent / relative_path
        values = np.fromfile(tensor_path, dtype=np.float32)
        expected_values = int(np.prod(shape))
        if values.size != expected_values:
            raise ValueError(
                f"C++ dump {name!r} has {values.size} values, expected {expected_values}"
            )
        tensors[name] = values.reshape(shape)
    if not tensors:
        raise ValueError(f"C++ dump manifest contains no outputs: {manifest_path}")
    return tensors


def build_cpp_python_parity_report(
    *,
    engine_path: Path,
    cpp_manifest_path: Path,
    input_name: str = "pixel_values",
) -> dict[str, Any]:
    """Compare Python TensorRT runtime outputs against a C++ output dump."""

    manifest = _load_manifest(cpp_manifest_path)
    batch_size = _required_int(manifest, "batch_size")
    input_shape = tuple(_required_int_list(manifest, "input_shape"))
    if len(input_shape) != 4:
        raise ValueError(f"expected NCHW input shape, got: {input_shape}")
    image_size = input_shape[2]
    if input_shape[0] != batch_size or input_shape[1] != 3 or input_shape[2] != input_shape[3]:
        raise ValueError(f"unsupported C++ parity input shape: {input_shape}")

    input_tensor = make_cpp_parity_input(batch_size=batch_size, image_size=image_size)
    python_outputs = run_engine(
        TensorRTEngineRunConfig(engine_path=engine_path, input_name=input_name),
        input_tensor,
    )
    cpp_outputs = read_cpp_output_dump(cpp_manifest_path)
    comparisons = compare_output_tensors(python_outputs, cpp_outputs)
    return {
        "engine_path": str(engine_path),
        "cpp_manifest_path": str(cpp_manifest_path),
        "batch_size": batch_size,
        "image_size": image_size,
        "input_mode": "deterministic-sine",
        "reference_runtime": "python",
        "candidate_runtime": "cpp",
        "outputs": [comparison.to_json() for comparison in comparisons],
    }


def _load_manifest(path: Path) -> Mapping[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, Mapping):
        raise ValueError(f"manifest must be a JSON object: {path}")
    return data


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _required_list(mapping: Mapping[str, Any], key: str) -> list[Any]:
    value = mapping.get(key)
    if not isinstance(value, list):
        raise ValueError(f"missing list field: {key}")
    return value


def _required_string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str):
        raise ValueError(f"missing string field: {key}")
    return value


def _required_int(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"missing integer field: {key}")
    return value


def _required_int_list(mapping: Mapping[str, Any], key: str) -> list[int]:
    values = _required_list(mapping, key)
    parsed: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"field {key!r} must contain integers")
        parsed.append(value)
    return parsed

