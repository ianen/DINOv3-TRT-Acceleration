"""Tests for `dinov3_trt.quantization.layer_precision`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dinov3_trt.quantization.layer_precision import (
    DEFAULT_COMPUTE_OP_TYPES,
    DINOV3_NUM_BLOCKS,
    OnnxNodeInfo,
    SUPPORTED_PRECISIONS,
    build_layer_precisions_arg,
    parse_block_index,
    select_block_node_names,
    write_layer_precisions_file,
)


@pytest.mark.parametrize(
    "node_name, expected",
    [
        ("/model/layer.0/norm1/LayerNormalization", 0),
        ("/model/layer.5/attention/q_proj/MatMul", 5),
        ("/model/layer.16/mlp/fc1/MatMul", 16),
        ("/model/layer.23/norm2/LayerNormalization", 23),
    ],
)
def test_parse_block_index_extracts_zero_based_block(node_name: str, expected: int) -> None:
    assert parse_block_index(node_name) == expected


@pytest.mark.parametrize(
    "node_name",
    [
        "",
        "/model/embeddings/Conv",
        "/model/norm/LayerNormalization",
        "/some/other/path",
    ],
)
def test_parse_block_index_returns_none_for_non_block_nodes(node_name: str) -> None:
    assert parse_block_index(node_name) is None


def test_parse_block_index_handles_non_string_input() -> None:
    assert parse_block_index(None) is None  # type: ignore[arg-type]
    assert parse_block_index(123) is None  # type: ignore[arg-type]


def _make_nodes() -> list[OnnxNodeInfo]:
    return [
        OnnxNodeInfo("/model/embeddings/Conv", "Conv"),
        OnnxNodeInfo("/model/layer.0/attention/q_proj/MatMul", "MatMul"),
        OnnxNodeInfo("/model/layer.5/mlp/fc1/MatMul", "MatMul"),
        OnnxNodeInfo("/model/layer.5/mlp/fc1/Add", "Add"),
        OnnxNodeInfo("/model/layer.5/Constant", "Constant"),
        OnnxNodeInfo("/model/layer.16/norm1/LayerNormalization", "LayerNormalization"),
        OnnxNodeInfo("/model/layer.16/attention/q_proj/MatMul", "MatMul"),
        OnnxNodeInfo("/model/layer.17/attention/q_proj/MatMul", "MatMul"),
        OnnxNodeInfo("/model/layer.18/attention/q_proj/MatMul", "MatMul"),
        OnnxNodeInfo("/model/layer.19/mlp/fc1/MatMul", "MatMul"),
        OnnxNodeInfo("/model/layer.19/mlp/fc1/Add", "Add"),
        OnnxNodeInfo("/model/layer.19/mlp/fc1/Constant", "Constant"),
        OnnxNodeInfo("/model/norm/LayerNormalization", "LayerNormalization"),
    ]


def test_select_block_node_names_returns_only_matching_blocks() -> None:
    nodes = _make_nodes()

    selected = select_block_node_names(nodes, block_indices=[16, 17, 18, 19])

    assert selected == (
        "/model/layer.16/norm1/LayerNormalization",
        "/model/layer.16/attention/q_proj/MatMul",
        "/model/layer.17/attention/q_proj/MatMul",
        "/model/layer.18/attention/q_proj/MatMul",
        "/model/layer.19/mlp/fc1/MatMul",
        "/model/layer.19/mlp/fc1/Add",
        "/model/layer.19/mlp/fc1/Constant",
    )


def test_select_block_node_names_filters_by_op_types() -> None:
    nodes = _make_nodes()

    selected = select_block_node_names(
        nodes,
        block_indices=[16, 17, 18, 19],
        op_types=["MatMul", "Add"],
    )

    assert selected == (
        "/model/layer.16/attention/q_proj/MatMul",
        "/model/layer.17/attention/q_proj/MatMul",
        "/model/layer.18/attention/q_proj/MatMul",
        "/model/layer.19/mlp/fc1/MatMul",
        "/model/layer.19/mlp/fc1/Add",
    )


def test_select_block_node_names_default_compute_op_types_excludes_constant() -> None:
    nodes = _make_nodes()

    selected = select_block_node_names(
        nodes,
        block_indices=list(range(DINOV3_NUM_BLOCKS)),
        op_types=DEFAULT_COMPUTE_OP_TYPES,
    )

    assert "/model/layer.5/Constant" not in selected
    assert "/model/layer.19/mlp/fc1/Constant" not in selected
    assert "/model/layer.5/mlp/fc1/MatMul" in selected
    assert "/model/layer.19/mlp/fc1/Add" in selected


def test_select_block_node_names_rejects_empty_block_indices() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        select_block_node_names(_make_nodes(), block_indices=[])


def test_select_block_node_names_rejects_out_of_range_block_index() -> None:
    with pytest.raises(ValueError, match=r"\[0, 24\)"):
        select_block_node_names(_make_nodes(), block_indices=[24])


def test_select_block_node_names_rejects_negative_block_index() -> None:
    with pytest.raises(ValueError, match=r"\[0, 24\)"):
        select_block_node_names(_make_nodes(), block_indices=[-1])


def test_select_block_node_names_rejects_non_int_block_index() -> None:
    with pytest.raises(TypeError, match="ints"):
        select_block_node_names(
            _make_nodes(), block_indices=[16.0]  # type: ignore[list-item]
        )


def test_select_block_node_names_rejects_empty_op_types() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        select_block_node_names(_make_nodes(), block_indices=[0], op_types=[])


def test_build_layer_precisions_arg_emits_comma_separated_pairs() -> None:
    arg = build_layer_precisions_arg(
        ["/model/layer.16/MatMul", "/model/layer.17/MatMul"],
        "bf16",
    )

    assert arg == "/model/layer.16/MatMul:bf16,/model/layer.17/MatMul:bf16"


def test_build_layer_precisions_arg_supports_all_supported_precisions() -> None:
    for precision in SUPPORTED_PRECISIONS:
        arg = build_layer_precisions_arg(["/model/layer.0/MatMul"], precision)
        assert arg == f"/model/layer.0/MatMul:{precision}"


def test_build_layer_precisions_arg_rejects_unsupported_precision() -> None:
    with pytest.raises(ValueError, match="unsupported precision"):
        build_layer_precisions_arg(["a"], "tf32")


def test_build_layer_precisions_arg_rejects_empty_node_names() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        build_layer_precisions_arg([], "bf16")


def test_build_layer_precisions_arg_rejects_duplicate_node_names() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        build_layer_precisions_arg(["a", "a"], "bf16")


def test_build_layer_precisions_arg_rejects_separator_chars_in_name() -> None:
    with pytest.raises(ValueError, match="ambiguous"):
        build_layer_precisions_arg(["bad,name"], "bf16")
    with pytest.raises(ValueError, match="ambiguous"):
        build_layer_precisions_arg(["bad:name"], "bf16")


def test_build_layer_precisions_arg_rejects_empty_string_name() -> None:
    with pytest.raises(ValueError, match="non-empty strings"):
        build_layer_precisions_arg([""], "bf16")


def test_write_layer_precisions_file_persists_value_and_metadata(tmp_path: Path) -> None:
    target = tmp_path / "trtexec_layer_precisions_blocks_16-19.txt"
    arg_value = build_layer_precisions_arg(
        [
            "/model/layer.16/MatMul",
            "/model/layer.17/MatMul",
            "/model/layer.18/MatMul",
            "/model/layer.19/MatMul",
        ],
        "bf16",
    )

    metadata = write_layer_precisions_file(
        target,
        arg_value=arg_value,
        block_indices=[16, 17, 18, 19],
        precision="bf16",
        op_types=["MatMul"],
    )

    assert target.read_text(encoding="utf-8") == arg_value
    sidecar = target.with_suffix(target.suffix + ".meta.json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload == {
        "arg_value_path": str(target),
        "arg_value_chars": len(arg_value),
        "node_count": 4,
        "precision": "bf16",
        "block_indices": [16, 17, 18, 19],
        "op_types": ["MatMul"],
    }
    assert metadata == payload


def test_write_layer_precisions_file_handles_unsorted_block_indices(tmp_path: Path) -> None:
    target = tmp_path / "v.txt"
    arg_value = build_layer_precisions_arg(["/m/layer.5/MatMul"], "int8")

    metadata = write_layer_precisions_file(
        target,
        arg_value=arg_value,
        block_indices=[19, 16, 17, 18, 16],
        precision="int8",
    )

    assert metadata["block_indices"] == [16, 17, 18, 19]


def test_write_layer_precisions_file_rejects_invalid_inputs(tmp_path: Path) -> None:
    target = tmp_path / "v.txt"

    with pytest.raises(ValueError, match="non-empty"):
        write_layer_precisions_file(target, arg_value="", block_indices=[0], precision="bf16")
    with pytest.raises(ValueError, match="unsupported precision"):
        write_layer_precisions_file(
            target, arg_value="a:bf16", block_indices=[0], precision="tf32"
        )
