from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from dinov3_trt.infer.image_eval import write_image_manifest
from dinov3_trt.quantization.modelopt_onnx import (
    ModelOptOnnxPtqConfig,
    load_calibration_array,
    parse_execution_providers,
    parse_optional_csv,
    run_modelopt_onnx_ptq,
)


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (8, 8), color)
    image.save(path)


def test_load_calibration_array_from_manifest(tmp_path: Path) -> None:
    first = tmp_path / "a.jpg"
    second = tmp_path / "b.jpg"
    _write_image(first, (255, 0, 0))
    _write_image(second, (0, 255, 0))
    manifest = tmp_path / "calib.json"
    write_image_manifest(
        manifest,
        image_root=tmp_path,
        images=(first, second),
        seed=1,
        split="calib",
    )

    array = load_calibration_array(
        ModelOptOnnxPtqConfig(
            onnx_path=tmp_path / "model.onnx",
            output_path=tmp_path / "model.int8.onnx",
            calibration_manifest=manifest,
            image_size=4,
            load_batch_size=1,
        )
    )

    assert array.shape == (2, 3, 4, 4)
    assert array.dtype == np.float32


def test_parse_execution_providers_rejects_empty() -> None:
    try:
        parse_execution_providers(" , ")
    except ValueError as exc:
        assert "execution provider" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_parse_optional_csv_returns_none_for_empty_values() -> None:
    assert parse_optional_csv(None) is None
    assert parse_optional_csv(" , ") is None
    assert parse_optional_csv("LayerNormalization, Softmax") == (
        "LayerNormalization",
        "Softmax",
    )


def test_run_modelopt_onnx_ptq_passes_expected_arguments(tmp_path: Path) -> None:
    image = tmp_path / "a.jpg"
    _write_image(image, (1, 2, 3))
    manifest = tmp_path / "calib.json"
    onnx_path = tmp_path / "model.onnx"
    output_path = tmp_path / "model.int8.onnx"
    onnx_path.write_bytes(b"onnx")
    write_image_manifest(
        manifest,
        image_root=tmp_path,
        images=(image,),
        seed=1,
        split="calib",
    )
    captured: dict[str, Any] = {}

    def fake_quantize(onnx: str, **kwargs: Any) -> None:
        captured["onnx"] = onnx
        captured.update(kwargs)
        Path(str(kwargs["output_path"])).write_bytes(b"quantized")

    report = run_modelopt_onnx_ptq(
        ModelOptOnnxPtqConfig(
            onnx_path=onnx_path,
            output_path=output_path,
            calibration_manifest=manifest,
            image_size=4,
            mha_accumulation_dtype="fp32",
            op_types_to_exclude=("LayerNormalization", "Softmax"),
            nodes_to_exclude=("model/layer.0/attention",),
            disable_mha_qdq=True,
            simplify=True,
        ),
        quantize_func=fake_quantize,
    )

    assert captured["onnx"] == str(onnx_path)
    assert captured["quantize_mode"] == "int8"
    assert captured["calibration_method"] == "max"
    assert captured["high_precision_dtype"] == "fp32"
    assert captured["mha_accumulation_dtype"] == "fp32"
    assert captured["op_types_to_exclude"] == ["LayerNormalization", "Softmax"]
    assert captured["nodes_to_exclude"] == ["model/layer.0/attention"]
    assert captured["disable_mha_qdq"] is True
    assert captured["dq_only"] is False
    assert captured["simplify"] is True
    assert captured["calibration_data"]["pixel_values"].shape == (1, 3, 4, 4)
    assert report["output_exists"]


def test_run_modelopt_onnx_ptq_passes_fp8_quantize_mode_through(tmp_path: Path) -> None:
    image = tmp_path / "a.jpg"
    _write_image(image, (1, 2, 3))
    manifest = tmp_path / "calib_fp8.json"
    onnx_path = tmp_path / "model.onnx"
    output_path = tmp_path / "model.fp8.onnx"
    onnx_path.write_bytes(b"fake-onnx")
    write_image_manifest(
        manifest,
        image_root=tmp_path,
        images=(image,),
        seed=2,
        split="calib_fp8",
    )
    captured: dict[str, Any] = {}

    def fake_quantize(onnx: str, **kwargs: Any) -> None:
        captured["onnx"] = onnx
        captured.update(kwargs)
        Path(str(kwargs["output_path"])).write_bytes(b"fp8-quantized")

    report = run_modelopt_onnx_ptq(
        ModelOptOnnxPtqConfig(
            onnx_path=onnx_path,
            output_path=output_path,
            calibration_manifest=manifest,
            image_size=4,
            quantize_mode="fp8",
            calibration_method="entropy",
        ),
        quantize_func=fake_quantize,
    )

    assert captured["quantize_mode"] == "fp8"
    assert captured["calibration_method"] == "entropy"
    assert report["output_exists"]
    config_payload = report["config"]
    assert isinstance(config_payload, dict)
    assert config_payload["quantize_mode"] == "fp8"


def test_modelopt_onnx_ptq_config_rejects_unsupported_quantize_mode(tmp_path: Path) -> None:
    image = tmp_path / "a.jpg"
    _write_image(image, (1, 2, 3))
    manifest = tmp_path / "calib_int4.json"
    onnx_path = tmp_path / "model.onnx"
    output_path = tmp_path / "model.int4.onnx"
    onnx_path.write_bytes(b"onnx")
    write_image_manifest(
        manifest,
        image_root=tmp_path,
        images=(image,),
        seed=3,
        split="calib_int4",
    )

    import pytest

    with pytest.raises(ValueError, match="quantize_mode must be one of"):
        ModelOptOnnxPtqConfig(
            onnx_path=onnx_path,
            output_path=output_path,
            calibration_manifest=manifest,
            image_size=4,
            quantize_mode="int4",  # type: ignore[arg-type]
        ).validate()
