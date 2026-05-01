"""ModelOpt ONNX PTQ helpers for explicit Q/DQ INT8 models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
import importlib
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from dinov3_trt.infer.image_eval import chunk_paths, load_image_batch, read_image_manifest

CalibrationMethod = Literal["max", "minmax", "entropy"]
HighPrecisionDType = Literal["fp16", "fp32", "bf16"]
QuantizeMode = Literal["int8", "fp8"]


@dataclass(frozen=True)
class ModelOptOnnxPtqConfig:
    """Configuration for one ModelOpt ONNX PTQ run."""

    onnx_path: Path
    output_path: Path
    calibration_manifest: Path
    input_name: str = "pixel_values"
    image_size: int = 224
    max_calibration_images: int | None = None
    load_batch_size: int = 16
    quantize_mode: QuantizeMode = "int8"
    calibration_method: CalibrationMethod = "max"
    calibration_eps: tuple[str, ...] = ("cuda:0", "cpu")
    high_precision_dtype: HighPrecisionDType = "fp32"
    mha_accumulation_dtype: HighPrecisionDType = "fp32"
    op_types_to_quantize: tuple[str, ...] | None = None
    op_types_to_exclude: tuple[str, ...] | None = None
    nodes_to_quantize: tuple[str, ...] | None = None
    nodes_to_exclude: tuple[str, ...] | None = None
    disable_mha_qdq: bool = False
    dq_only: bool = False
    simplify: bool = False
    use_external_data_format: bool = False
    log_level: str = "INFO"

    def validate(self) -> None:
        if self.image_size < 1:
            raise ValueError("image_size must be >= 1")
        if self.max_calibration_images is not None and self.max_calibration_images < 1:
            raise ValueError("max_calibration_images must be >= 1")
        if self.load_batch_size < 1:
            raise ValueError("load_batch_size must be >= 1")
        if not self.calibration_eps:
            raise ValueError("at least one calibration execution provider is required")
        if self.quantize_mode not in ("int8", "fp8"):
            raise ValueError(
                "quantize_mode must be one of {'int8', 'fp8'}; ModelOpt 0.43 also "
                "supports 'int4' but the project has not validated that path."
            )

    def to_json(self) -> dict[str, object]:
        payload = asdict(self)
        payload["onnx_path"] = str(self.onnx_path)
        payload["output_path"] = str(self.output_path)
        payload["calibration_manifest"] = str(self.calibration_manifest)
        payload["calibration_eps"] = list(self.calibration_eps)
        for key in (
            "op_types_to_quantize",
            "op_types_to_exclude",
            "nodes_to_quantize",
            "nodes_to_exclude",
        ):
            values = payload[key]
            if values is not None:
                payload[key] = list(values)
        return payload


def load_calibration_array(config: ModelOptOnnxPtqConfig) -> NDArray[np.float32]:
    """Load calibration images from a manifest into one NCHW float32 array."""

    config.validate()
    image_paths = read_image_manifest(config.calibration_manifest)
    if config.max_calibration_images is not None:
        image_paths = image_paths[: config.max_calibration_images]
    batches = [
        load_image_batch(path_batch, image_size=config.image_size).tensor
        for path_batch in chunk_paths(image_paths, config.load_batch_size)
    ]
    if not batches:
        raise ValueError("calibration manifest produced no images")
    return np.concatenate(batches, axis=0).astype(np.float32, copy=False)


def run_modelopt_onnx_ptq(
    config: ModelOptOnnxPtqConfig,
    *,
    quantize_func: Callable[..., object] | None = None,
) -> dict[str, object]:
    """Run ModelOpt ONNX PTQ and return a serializable execution report."""

    config.validate()
    calibration_array = load_calibration_array(config)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    if quantize_func is None:
        quantize_func = _load_modelopt_onnx_quantize()

    quantize_func(
        str(config.onnx_path),
        quantize_mode=config.quantize_mode,
        calibration_data={config.input_name: calibration_array},
        calibration_method=config.calibration_method,
        calibration_eps=list(config.calibration_eps),
        output_path=str(config.output_path),
        high_precision_dtype=config.high_precision_dtype,
        mha_accumulation_dtype=config.mha_accumulation_dtype,
        op_types_to_quantize=_optional_list(config.op_types_to_quantize),
        op_types_to_exclude=_optional_list(config.op_types_to_exclude),
        nodes_to_quantize=_optional_list(config.nodes_to_quantize),
        nodes_to_exclude=_optional_list(config.nodes_to_exclude),
        disable_mha_qdq=config.disable_mha_qdq,
        dq_only=config.dq_only,
        simplify=config.simplify,
        use_external_data_format=config.use_external_data_format,
        log_level=config.log_level,
    )

    return {
        "config": config.to_json(),
        "calibration": {
            "shape": list(calibration_array.shape),
            "dtype": str(calibration_array.dtype),
            "image_count": int(calibration_array.shape[0]),
        },
        "output_path": str(config.output_path),
        "output_exists": config.output_path.exists(),
        "output_size_bytes": config.output_path.stat().st_size if config.output_path.exists() else None,
    }


def parse_execution_providers(value: str) -> tuple[str, ...]:
    """Parse a comma-separated calibration execution provider list."""

    providers = tuple(part.strip() for part in value.split(",") if part.strip())
    if not providers:
        raise ValueError("at least one execution provider is required")
    return providers


def parse_optional_csv(value: str | None) -> tuple[str, ...] | None:
    """Parse an optional comma-separated string tuple."""

    if value is None:
        return None
    parts = tuple(part.strip() for part in value.split(",") if part.strip())
    return parts or None


def _optional_list(values: tuple[str, ...] | None) -> list[str] | None:
    return list(values) if values is not None else None


def _load_modelopt_onnx_quantize() -> Callable[..., object]:
    module: Any = importlib.import_module("modelopt.onnx.quantization")
    quantize = getattr(module, "quantize")
    if not callable(quantize):
        raise TypeError("modelopt.onnx.quantization.quantize is not callable")
    return cast(Callable[..., object], quantize)
