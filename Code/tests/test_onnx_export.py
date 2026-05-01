from pathlib import Path

import pytest

from dinov3_trt.contracts import DINO_VITL16_224_CONTRACT, make_dinov3_vitl16_contract
from dinov3_trt.export.onnx_export import OnnxExportConfig, build_dynamic_axes, make_dummy_input


def test_onnx_export_config_rejects_opset_below_18() -> None:
    config = OnnxExportConfig(output_path=Path("model.onnx"), opset=17)

    with pytest.raises(ValueError, match="opset >= 18"):
        config.validate()


def test_build_dynamic_axes_uses_project_output_names() -> None:
    config = OnnxExportConfig(output_path=Path("model.onnx"))

    axes = build_dynamic_axes(config)

    assert axes == {
        "pixel_values": {0: "batch"},
        **{name: {0: "batch"} for name in DINO_VITL16_224_CONTRACT.output_names},
    }


def test_build_dynamic_axes_can_be_disabled() -> None:
    config = OnnxExportConfig(output_path=Path("model.onnx"), dynamic_batch=False)

    assert build_dynamic_axes(config) is None


def test_make_dummy_input_uses_supplied_resolution_contract() -> None:
    torch = pytest.importorskip("torch")
    config = OnnxExportConfig(output_path=Path("model.onnx"), device="cpu")
    dummy_input = make_dummy_input(config, make_dinov3_vitl16_contract(336))

    assert isinstance(dummy_input, torch.Tensor)
    assert tuple(dummy_input.shape) == (1, 3, 336, 336)
