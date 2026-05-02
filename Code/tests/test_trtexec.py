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


# =============================================================================
# V1.0.2 ADR-013: persistent timing cache + multi optimization profiles
# =============================================================================


def test_v102_default_has_no_extra_profiles() -> None:
    command = build_trtexec_command(
        TrtExecConfig(
            onnx_path=Path("Artifacts/onnx/dinov3.onnx"),
            engine_path=Path("Artifacts/engines/dinov3.engine"),
            precision="bf16",
        )
    )
    assert command.count("--profile") == 0
    assert sum(1 for p in command if p.startswith("--minShapes=")) == 1
    assert sum(1 for p in command if p.startswith("--optShapes=")) == 1
    assert sum(1 for p in command if p.startswith("--maxShapes=")) == 1


def test_v102_additional_profiles_emit_separators_and_extra_shape_blocks() -> None:
    command = build_trtexec_command(
        TrtExecConfig(
            onnx_path=Path("Artifacts/onnx/dinov3.onnx"),
            engine_path=Path("Artifacts/engines/dinov3.engine"),
            precision="bf16",
            profile=ShapeProfile(min_batch=1, opt_batch=1, max_batch=1),
            additional_profiles=(
                ShapeProfile(min_batch=4, opt_batch=8, max_batch=16),
                ShapeProfile(min_batch=16, opt_batch=32, max_batch=32),
            ),
        )
    )
    # 1 base profile + 2 additional profiles = 2 ``--profile`` separators
    assert command.count("--profile") == 2
    # 1 base + 2 additional = 3 of each shape flag
    assert sum(1 for p in command if p.startswith("--minShapes=")) == 3
    assert sum(1 for p in command if p.startswith("--optShapes=")) == 3
    assert sum(1 for p in command if p.startswith("--maxShapes=")) == 3
    # The base profile (b=1) appears before the first ``--profile`` separator
    first_profile_idx = command.index("--profile")
    base_min_idx = command.index("--minShapes=pixel_values:1x3x224x224")
    assert base_min_idx < first_profile_idx
    # The b=4-16 profile shapes appear after the first separator
    assert "--minShapes=pixel_values:4x3x224x224" in command[first_profile_idx:]
    assert "--minShapes=pixel_values:16x3x224x224" in command[first_profile_idx:]


def test_v102_additional_profile_must_match_contract_resolution() -> None:
    bad_profile = ShapeProfile(min_batch=1, opt_batch=4, max_batch=8, height=336, width=336)
    with pytest.raises(ValueError, match="additional profile"):
        TrtExecConfig(
            onnx_path=Path("Artifacts/onnx/dinov3.onnx"),
            engine_path=Path("Artifacts/engines/dinov3.engine"),
            precision="bf16",
            additional_profiles=(bad_profile,),
        )


def test_v102_builder_optimization_level_emitted_when_set() -> None:
    command = build_trtexec_command(
        TrtExecConfig(
            onnx_path=Path("Artifacts/onnx/dinov3.onnx"),
            engine_path=Path("Artifacts/engines/dinov3.engine"),
            precision="bf16",
            builder_optimization_level=5,
        )
    )
    assert "--builderOptimizationLevel=5" in command


def test_v102_builder_optimization_level_default_omitted() -> None:
    command = build_trtexec_command(
        TrtExecConfig(
            onnx_path=Path("Artifacts/onnx/dinov3.onnx"),
            engine_path=Path("Artifacts/engines/dinov3.engine"),
            precision="bf16",
        )
    )
    assert not any(p.startswith("--builderOptimizationLevel=") for p in command)


def test_v102_builder_optimization_level_must_be_in_range() -> None:
    with pytest.raises(ValueError, match=r"\[0, 5\]"):
        TrtExecConfig(
            onnx_path=Path("Artifacts/onnx/dinov3.onnx"),
            engine_path=Path("Artifacts/engines/dinov3.engine"),
            precision="bf16",
            builder_optimization_level=6,
        )
    with pytest.raises(ValueError, match=r"\[0, 5\]"):
        TrtExecConfig(
            onnx_path=Path("Artifacts/onnx/dinov3.onnx"),
            engine_path=Path("Artifacts/engines/dinov3.engine"),
            precision="bf16",
            builder_optimization_level=-1,
        )


def test_v102_persistent_cache_size_emitted_when_set() -> None:
    command = build_trtexec_command(
        TrtExecConfig(
            onnx_path=Path("Artifacts/onnx/dinov3.onnx"),
            engine_path=Path("Artifacts/engines/dinov3.engine"),
            precision="bf16",
            persistent_cache_size_mb=64,
        )
    )
    assert "--persistentCacheSize=64" in command


def test_v102_persistent_cache_size_default_omitted() -> None:
    command = build_trtexec_command(
        TrtExecConfig(
            onnx_path=Path("Artifacts/onnx/dinov3.onnx"),
            engine_path=Path("Artifacts/engines/dinov3.engine"),
            precision="bf16",
        )
    )
    assert not any(p.startswith("--persistentCacheSize=") for p in command)


def test_v102_persistent_cache_size_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        TrtExecConfig(
            onnx_path=Path("Artifacts/onnx/dinov3.onnx"),
            engine_path=Path("Artifacts/engines/dinov3.engine"),
            precision="bf16",
            persistent_cache_size_mb=-1,
        )


# =============================================================================
# V1.0.2 ADR-016: 2:4 structured sparsity flag
# =============================================================================


def test_v102_sparsity_flag_emitted_when_enabled() -> None:
    command = build_trtexec_command(
        TrtExecConfig(
            onnx_path=Path("Artifacts/onnx/dinov3.sparse.onnx"),
            engine_path=Path("Artifacts/engines/dinov3.sparse.engine"),
            precision="bf16",
            enable_sparsity=True,
        )
    )
    assert "--sparsity=enable" in command


def test_v102_sparsity_flag_omitted_by_default() -> None:
    command = build_trtexec_command(
        TrtExecConfig(
            onnx_path=Path("Artifacts/onnx/dinov3.onnx"),
            engine_path=Path("Artifacts/engines/dinov3.engine"),
            precision="bf16",
        )
    )
    assert "--sparsity=enable" not in command


def test_v102_full_v102_config_command_ordering() -> None:
    """Smoke test: all V1.0.2 flags together produce a valid command structure."""
    command = build_trtexec_command(
        TrtExecConfig(
            onnx_path=Path("Artifacts/onnx/dinov3.onnx"),
            engine_path=Path("Artifacts/engines/dinov3.bf16.v102.engine"),
            precision="bf16",
            profile=ShapeProfile(min_batch=1, opt_batch=1, max_batch=1),
            additional_profiles=(
                ShapeProfile(min_batch=4, opt_batch=8, max_batch=16),
                ShapeProfile(min_batch=16, opt_batch=32, max_batch=32),
            ),
            timing_cache_path=Path("Artifacts/timing_cache/shared_v102.cache"),
            builder_optimization_level=5,
            persistent_cache_size_mb=64,
            enable_sparsity=True,
        )
    )
    # Sanity: exactly 1 trtexec, 1 onnx, 1 saveEngine, 3 of each shape flag,
    # 2 ``--profile`` separators, BF16 selected, all V1.0.2 flags present.
    assert command[0] == "trtexec"
    assert sum(1 for p in command if p.startswith("--onnx=")) == 1
    assert sum(1 for p in command if p.startswith("--saveEngine=")) == 1
    assert command.count("--profile") == 2
    assert sum(1 for p in command if p.startswith("--minShapes=")) == 3
    assert "--bf16" in command
    assert "--builderOptimizationLevel=5" in command
    assert "--persistentCacheSize=64" in command
    assert "--sparsity=enable" in command
    assert any(p.startswith("--timingCacheFile=") for p in command)
