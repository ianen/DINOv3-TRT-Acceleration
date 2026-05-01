from pathlib import Path

import pytest

from dinov3_trt.contracts import make_dinov3_vitl16_contract
from dinov3_trt.engine.trtexec import ShapeProfile, TrtExecConfig, build_trtexec_command


def _portable_parts(command: list[str]) -> list[str]:
    return [part.replace("\\", "/") for part in command]


def test_shape_profile_uses_dynamic_batch_static_resolution() -> None:
    profile = ShapeProfile(input_name="pixel_values", min_batch=1, opt_batch=8, max_batch=32)

    assert profile.min_shape == "pixel_values:1x3x224x224"
    assert profile.opt_shape == "pixel_values:8x3x224x224"
    assert profile.max_shape == "pixel_values:32x3x224x224"


def test_build_fp16_trtexec_command_has_project_flags() -> None:
    command = build_trtexec_command(
        TrtExecConfig(
            onnx_path=Path("Artifacts/onnx/dinov3.onnx"),
            engine_path=Path("Artifacts/engines/dinov3.fp16.engine"),
            precision="fp16",
            timing_cache_path=Path("Artifacts/engines/timing.cache"),
        )
    )

    assert "trtexec" == command[0]
    assert "--fp16" in command
    assert "--noTF32" in command
    assert not any(part.startswith("--precisionConstraints=") for part in command)
    assert "--skipInference" in command
    assert "--memPoolSize=workspace:4G" in command
    assert "--minShapes=pixel_values:1x3x224x224" in command
    assert "--optShapes=pixel_values:8x3x224x224" in command
    assert "--maxShapes=pixel_values:32x3x224x224" in command
    assert "--timingCacheFile=Artifacts/engines/timing.cache" in _portable_parts(command)


def test_build_int8_trtexec_command_has_project_flags() -> None:
    command = build_trtexec_command(
        TrtExecConfig(
            onnx_path=Path("Artifacts/onnx/dinov3.int8.modelopt.onnx"),
            engine_path=Path("Artifacts/engines/dinov3.int8.engine"),
            precision="int8",
        )
    )

    assert "--int8" in command
    assert "--fp16" not in command
    assert "--bf16" not in command
    assert "--noTF32" in command
    assert "--skipInference" in command


def test_build_fp8_trtexec_command_emits_fp8_flag_without_precision_constraints() -> None:
    """FP8 engines (Blackwell sm_120 5th-gen Tensor Core) take Q/DQ ONNX directly,
    so the project does not pass `--precisionConstraints` for them."""

    command = build_trtexec_command(
        TrtExecConfig(
            onnx_path=Path("Artifacts/onnx/dinov3.fp8.modelopt.onnx"),
            engine_path=Path("Artifacts/engines/dinov3.fp8.engine"),
            precision="fp8",
        )
    )

    assert "--fp8" in command
    assert "--int8" not in command
    assert "--fp16" not in command
    assert "--bf16" not in command
    assert "--noTF32" in command
    assert "--skipInference" in command
    assert not any(part.startswith("--precisionConstraints=") for part in command)


def test_fp8_precision_rejects_mixed_precision_constraints() -> None:
    with pytest.raises(ValueError, match="precision constraints"):
        TrtExecConfig(
            onnx_path=Path("Artifacts/onnx/dinov3.fp8.onnx"),
            engine_path=Path("Artifacts/engines/dinov3.fp8.engine"),
            precision="fp8",
            layer_precisions=("/model/layer.0/*:fp32",),
        )


def test_rejects_resolution_mismatch() -> None:
    with pytest.raises(ValueError, match="static model contract"):
        TrtExecConfig(
            onnx_path=Path("Artifacts/onnx/dinov3.onnx"),
            engine_path=Path("Artifacts/engines/dinov3.fp32.engine"),
            precision="fp32",
            profile=ShapeProfile(height=336, width=336),
        )


def test_build_trtexec_command_accepts_matching_non_224_contract() -> None:
    command = build_trtexec_command(
        TrtExecConfig(
            onnx_path=Path("Artifacts/onnx/dinov3_r336.onnx"),
            engine_path=Path("Artifacts/engines/dinov3_r336.fp32.engine"),
            precision="fp32",
            profile=ShapeProfile(height=336, width=336),
            contract=make_dinov3_vitl16_contract(336),
        )
    )

    assert "--minShapes=pixel_values:1x3x336x336" in command
    assert "--optShapes=pixel_values:8x3x336x336" in command
    assert "--maxShapes=pixel_values:32x3x336x336" in command


def test_build_fp16_trtexec_command_supports_mixed_precision_constraints() -> None:
    command = build_trtexec_command(
        TrtExecConfig(
            onnx_path=Path("Artifacts/onnx/dinov3.onnx"),
            engine_path=Path("Artifacts/engines/dinov3.fp16.blocks_fp32.engine"),
            precision="fp16",
            layer_precisions=("/model/layer.0/*:fp32",),
            layer_output_types=("/model/layer.0/*:fp32",),
        )
    )

    assert "--fp16" in command
    assert "--precisionConstraints=prefer" in command
    assert "--layerPrecisions=/model/layer.0/*:fp32" in command
    assert "--layerOutputTypes=/model/layer.0/*:fp32" in command


def test_build_bf16_trtexec_command_has_project_flags() -> None:
    command = build_trtexec_command(
        TrtExecConfig(
            onnx_path=Path("Artifacts/onnx/dinov3.onnx"),
            engine_path=Path("Artifacts/engines/dinov3.bf16.engine"),
            precision="bf16",
        )
    )

    assert "--bf16" in command
    assert "--fp16" not in command
    assert "--noTF32" in command
    assert "--skipInference" in command


def test_rejects_precision_constraints_for_fp32_engine() -> None:
    with pytest.raises(ValueError, match="only supported for FP16/BF16/INT8"):
        TrtExecConfig(
            onnx_path=Path("Artifacts/onnx/dinov3.onnx"),
            engine_path=Path("Artifacts/engines/dinov3.fp32.engine"),
            precision="fp32",
            layer_precisions=("/model/layer.0/*:fp32",),
        )


def test_build_int8_trtexec_command_supports_layer_precisions_override() -> None:
    command = build_trtexec_command(
        TrtExecConfig(
            onnx_path=Path("Artifacts/onnx/dinov3.smoothquant.onnx"),
            engine_path=Path("Artifacts/engines/dinov3.smoothquant.mixed.engine"),
            precision="int8",
            precision_constraints="obey",
            layer_precisions=(
                "/model/layer.16/attention/q_proj/MatMul:bf16",
                "/model/layer.19/mlp/fc1/Add:bf16",
            ),
        )
    )

    assert "--int8" in command
    assert "--precisionConstraints=obey" in command
    assert (
        "--layerPrecisions=/model/layer.16/attention/q_proj/MatMul:bf16,"
        "/model/layer.19/mlp/fc1/Add:bf16"
    ) in command


def test_build_int8_trtexec_command_default_precision_constraints_when_layers_set() -> None:
    command = build_trtexec_command(
        TrtExecConfig(
            onnx_path=Path("Artifacts/onnx/dinov3.smoothquant.onnx"),
            engine_path=Path("Artifacts/engines/dinov3.smoothquant.mixed.engine"),
            precision="int8",
            layer_precisions=("/model/layer.16/MatMul:bf16",),
        )
    )

    assert "--int8" in command
    assert "--precisionConstraints=prefer" in command


def test_int8_precision_without_layer_overrides_emits_no_constraint_flag() -> None:
    command = build_trtexec_command(
        TrtExecConfig(
            onnx_path=Path("Artifacts/onnx/dinov3.smoothquant.onnx"),
            engine_path=Path("Artifacts/engines/dinov3.smoothquant.engine"),
            precision="int8",
        )
    )

    assert "--int8" in command
    assert not any(part.startswith("--precisionConstraints=") for part in command)
    assert not any(part.startswith("--layerPrecisions=") for part in command)
