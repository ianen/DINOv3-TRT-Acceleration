from pathlib import Path

import pytest

from dinov3_trt.quantization.matmul_sweep import (
    MatMulSweepVariant,
    format_block_label,
    make_sweep_paths,
    matmul_nodes_for_blocks,
    parse_block_spec,
    parse_node_group,
    parse_variant_spec,
    parse_variant_specs,
)


def test_parse_block_spec_accepts_ranges_and_deduplicates() -> None:
    assert parse_block_spec("19, 17-18, 18") == (17, 18, 19)


def test_parse_block_spec_rejects_invalid_block() -> None:
    with pytest.raises(ValueError, match="0..19"):
        parse_block_spec("20")


def test_format_block_label_handles_contiguous_and_sparse_blocks() -> None:
    assert format_block_label((19,)) == "layer19"
    assert format_block_label((17, 18, 19)) == "layers17_19"
    assert format_block_label((16, 18, 19)) == "layers16_18_19"


def test_matmul_nodes_for_blocks_matches_exported_node_names() -> None:
    nodes = matmul_nodes_for_blocks((18,))

    assert nodes == (
        "/model/layer.18/attention/q_proj/MatMul",
        "/model/layer.18/attention/k_proj/MatMul",
        "/model/layer.18/attention/v_proj/MatMul",
        "/model/layer.18/attention/MatMul",
        "/model/layer.18/attention/MatMul_1",
        "/model/layer.18/attention/o_proj/MatMul",
        "/model/layer.18/mlp/up_proj/MatMul",
        "/model/layer.18/mlp/down_proj/MatMul",
    )


def test_matmul_nodes_for_blocks_can_filter_suffixes() -> None:
    nodes = matmul_nodes_for_blocks(
        (19,),
        suffixes=("mlp/up_proj/MatMul", "mlp/down_proj/MatMul"),
    )

    assert nodes == (
        "/model/layer.19/mlp/up_proj/MatMul",
        "/model/layer.19/mlp/down_proj/MatMul",
    )


def test_parse_variant_spec_accepts_inline_node_group() -> None:
    blocks, group = parse_variant_spec("18-19:mlp")

    assert blocks == (18, 19)
    assert group == "mlp"


def test_parse_node_group_rejects_unknown_group() -> None:
    with pytest.raises(ValueError, match="unknown MatMul node group"):
        parse_node_group("residual")


def test_parse_variant_specs_removes_duplicates() -> None:
    variants = parse_variant_specs(("18-19", "19,18"))

    assert variants == (MatMulSweepVariant(blocks=(18, 19)),)


def test_parse_variant_specs_crosses_plain_variants_with_node_groups() -> None:
    variants = parse_variant_specs(("19",), node_groups=("attention", "mlp"))

    assert variants == (
        MatMulSweepVariant(blocks=(19,), node_group="attention"),
        MatMulSweepVariant(blocks=(19,), node_group="mlp"),
    )
    assert variants[0].label == "layer19_attention"
    assert variants[1].nodes_to_quantize == (
        "/model/layer.19/mlp/up_proj/MatMul",
        "/model/layer.19/mlp/down_proj/MatMul",
    )


def test_inline_group_overrides_node_group_cross_product() -> None:
    variants = parse_variant_specs(("19:mlp",), node_groups=("attention",))

    assert variants == (MatMulSweepVariant(blocks=(19,), node_group="mlp"),)


def test_make_sweep_paths_uses_consistent_names() -> None:
    paths = make_sweep_paths(
        onnx_dir=Path("onnx"),
        reports_dir=Path("reports"),
        model_stem="dinov3",
        prefix="imagenette64_matmul",
        variant=MatMulSweepVariant(blocks=(18, 19)),
    )

    assert paths.quantized_onnx == Path(
        "onnx/dinov3.int8.modelopt.imagenette64_matmul_layers18_19.onnx"
    )
    assert paths.image_eval == Path(
        "reports/eval_onnx_imagenette32_fp32_vs_int8_modelopt_"
        "imagenette64_matmul_layers18_19.json"
    )


def test_make_sweep_paths_quantize_mode_distinguishes_fp8_and_int8() -> None:
    int8_paths = make_sweep_paths(
        onnx_dir=Path("onnx"),
        reports_dir=Path("reports"),
        model_stem="dinov3",
        prefix="imagenette64_matmul",
        variant=MatMulSweepVariant(blocks=(19,)),
        quantize_mode="int8",
    )
    fp8_paths = make_sweep_paths(
        onnx_dir=Path("onnx"),
        reports_dir=Path("reports"),
        model_stem="dinov3",
        prefix="imagenette64_matmul",
        variant=MatMulSweepVariant(blocks=(19,)),
        quantize_mode="fp8",
    )

    # FP8 sweeps must not collide with INT8 sweep artefacts.
    assert int8_paths.quantized_onnx != fp8_paths.quantized_onnx
    assert int8_paths.image_eval != fp8_paths.image_eval
    assert "fp8.modelopt" in fp8_paths.quantized_onnx.as_posix()
    assert "int8.modelopt" in int8_paths.quantized_onnx.as_posix()
    assert "fp32_vs_fp8_modelopt" in fp8_paths.image_eval.as_posix()
    assert "fp32_vs_int8_modelopt" in int8_paths.image_eval.as_posix()
    # Stdout sidecars must also stay separated so reruns don't clobber each other.
    assert "_fp8.stdout.txt" in fp8_paths.quantize_stdout.as_posix()
    assert "_int8.stdout.txt" in int8_paths.quantize_stdout.as_posix()
