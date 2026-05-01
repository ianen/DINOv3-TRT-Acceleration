"""Tests for `dinov3_trt.quantization.onnx_qdq_stripper` (ADR-010 V1.2 prep)."""

from __future__ import annotations

import pytest

from dinov3_trt.quantization.onnx_qdq_stripper import (
    OnnxNodeWithEdges,
    QDQPair,
    classify_pair,
    find_block_qdq_pairs,
    find_qdq_pairs,
    parse_block_index,
    split_strippable_and_preserved,
    summarise_pairs,
)


@pytest.mark.parametrize(
    "node_name, expected",
    [
        ("/model/layer.0/attention/q_proj/MatMul", 0),
        ("/model/layer.5/Mul", 5),
        ("/model/layer.16/norm1/LayerNormalization", 16),
        ("/model/layer.23/mlp/fc2/MatMul", 23),
    ],
)
def test_parse_block_index_extracts_index(node_name: str, expected: int) -> None:
    assert parse_block_index(node_name) == expected


def test_parse_block_index_returns_none_outside_blocks() -> None:
    assert parse_block_index("") is None
    assert parse_block_index(None) is None  # type: ignore[arg-type]
    assert parse_block_index("/model/embeddings/Conv") is None


def _block_qdq_pair_nodes(block: int, suffix: str) -> tuple[OnnxNodeWithEdges, OnnxNodeWithEdges]:
    """Helper: produce a Q->DQ pair entirely inside one block."""

    q_name = f"/model/layer.{block}/{suffix}/QuantizeLinear"
    dq_name = f"/model/layer.{block}/{suffix}/DequantizeLinear"
    tensor = f"/model/layer.{block}/{suffix}/quant_out"
    q_node = OnnxNodeWithEdges(
        name=q_name,
        op_type="QuantizeLinear",
        inputs=(f"/model/layer.{block}/{suffix}/input",),
        outputs=(tensor,),
    )
    dq_node = OnnxNodeWithEdges(
        name=dq_name,
        op_type="DequantizeLinear",
        inputs=(tensor,),
        outputs=(f"/model/layer.{block}/{suffix}/dequant_out",),
    )
    return q_node, dq_node


def test_find_qdq_pairs_detects_adjacent_q_dq() -> None:
    q1, dq1 = _block_qdq_pair_nodes(5, "attention/input_quantizer")
    q2, dq2 = _block_qdq_pair_nodes(16, "attention/q_proj/weight_quantizer")
    other = OnnxNodeWithEdges(
        name="/model/layer.5/attention/q_proj/MatMul",
        op_type="MatMul",
        inputs=("a", "b"),
        outputs=("c",),
    )

    pairs = find_qdq_pairs([q1, other, dq1, q2, dq2])

    assert pairs == [
        (q1.name, dq1.name),
        (q2.name, dq2.name),
    ]


def test_find_qdq_pairs_ignores_q_without_consumer() -> None:
    q_alone = OnnxNodeWithEdges(
        name="/model/layer.0/dangling/QuantizeLinear",
        op_type="QuantizeLinear",
        inputs=("x",),
        outputs=("y",),
    )

    pairs = find_qdq_pairs([q_alone])

    assert pairs == []


def test_find_qdq_pairs_ignores_dq_consuming_non_quantize_output() -> None:
    matmul = OnnxNodeWithEdges(
        name="/model/layer.0/MatMul",
        op_type="MatMul",
        inputs=("x", "y"),
        outputs=("matmul_out",),
    )
    dq = OnnxNodeWithEdges(
        name="/model/layer.0/DequantizeLinear",
        op_type="DequantizeLinear",
        inputs=("matmul_out",),
        outputs=("z",),
    )

    pairs = find_qdq_pairs([matmul, dq])

    assert pairs == []


def test_classify_pair_internal_when_both_in_range() -> None:
    q_name = "/model/layer.16/attention/q_proj/QuantizeLinear"
    dq_name = "/model/layer.16/attention/q_proj/DequantizeLinear"

    pair = classify_pair(q_name, dq_name, block_indices=[16, 17, 18, 19])

    assert pair is not None
    assert pair.location == "internal"
    assert pair.quantize_block == 16
    assert pair.dequantize_block == 16


def test_classify_pair_boundary_input_when_only_dq_in_range() -> None:
    q_name = "/model/layer.15/attention/output/QuantizeLinear"
    dq_name = "/model/layer.16/norm1/DequantizeLinear"

    pair = classify_pair(q_name, dq_name, block_indices=[16, 17, 18, 19])

    assert pair is not None
    assert pair.location == "boundary_input"
    assert pair.quantize_block == 15
    assert pair.dequantize_block == 16


def test_classify_pair_boundary_output_when_only_q_in_range() -> None:
    q_name = "/model/layer.19/mlp/fc2/QuantizeLinear"
    dq_name = "/model/layer.20/norm1/DequantizeLinear"

    pair = classify_pair(q_name, dq_name, block_indices=[16, 17, 18, 19])

    assert pair is not None
    assert pair.location == "boundary_output"
    assert pair.quantize_block == 19
    assert pair.dequantize_block == 20


def test_classify_pair_returns_none_when_neither_in_range() -> None:
    q_name = "/model/layer.0/attention/QuantizeLinear"
    dq_name = "/model/layer.0/attention/DequantizeLinear"

    pair = classify_pair(q_name, dq_name, block_indices=[16, 17, 18, 19])

    assert pair is None


def test_classify_pair_handles_pair_outside_any_block() -> None:
    q_name = "/model/embeddings/QuantizeLinear"
    dq_name = "/model/embeddings/DequantizeLinear"

    pair = classify_pair(q_name, dq_name, block_indices=[16, 17, 18, 19])

    assert pair is None


def test_classify_pair_rejects_empty_block_indices() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        classify_pair("a", "b", block_indices=[])


def test_classify_pair_rejects_negative_block_indices() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        classify_pair("a", "b", block_indices=[-1, 0])


def test_find_block_qdq_pairs_returns_internal_and_boundary_pairs() -> None:
    boundary_in_q = OnnxNodeWithEdges(
        name="/model/layer.15/output_quantizer/QuantizeLinear",
        op_type="QuantizeLinear",
        inputs=("a",),
        outputs=("t1",),
    )
    boundary_in_dq = OnnxNodeWithEdges(
        name="/model/layer.16/norm1/DequantizeLinear",
        op_type="DequantizeLinear",
        inputs=("t1",),
        outputs=("t2",),
    )
    internal_q, internal_dq = _block_qdq_pair_nodes(17, "attention/q_proj/input_quantizer")
    boundary_out_q = OnnxNodeWithEdges(
        name="/model/layer.19/output_quantizer/QuantizeLinear",
        op_type="QuantizeLinear",
        inputs=("a3",),
        outputs=("t3",),
    )
    boundary_out_dq = OnnxNodeWithEdges(
        name="/model/layer.20/norm1/DequantizeLinear",
        op_type="DequantizeLinear",
        inputs=("t3",),
        outputs=("t4",),
    )
    unrelated_q, unrelated_dq = _block_qdq_pair_nodes(0, "attention/q_proj/input_quantizer")

    nodes = [
        unrelated_q,
        unrelated_dq,
        boundary_in_q,
        boundary_in_dq,
        internal_q,
        internal_dq,
        boundary_out_q,
        boundary_out_dq,
    ]

    pairs = find_block_qdq_pairs(nodes, block_indices=[16, 17, 18, 19])

    assert len(pairs) == 3
    locations = [p.location for p in pairs]
    assert locations == ["boundary_input", "internal", "boundary_output"]


def test_split_strippable_and_preserved_partitions_correctly() -> None:
    pairs = [
        QDQPair("q_internal_1", "dq_internal_1", 17, 17, "internal"),
        QDQPair("q_boundary_in", "dq_boundary_in", 15, 16, "boundary_input"),
        QDQPair("q_internal_2", "dq_internal_2", 18, 18, "internal"),
        QDQPair("q_boundary_out", "dq_boundary_out", 19, 20, "boundary_output"),
    ]

    strippable, preserved = split_strippable_and_preserved(pairs)

    assert [p.quantize_node for p in strippable] == ["q_internal_1", "q_internal_2"]
    assert [p.location for p in preserved] == ["boundary_input", "boundary_output"]


def test_summarise_pairs_counts_each_location() -> None:
    pairs = [
        QDQPair("q1", "dq1", 17, 17, "internal"),
        QDQPair("q2", "dq2", 18, 18, "internal"),
        QDQPair("q3", "dq3", 18, 18, "internal"),
        QDQPair("q4", "dq4", 15, 16, "boundary_input"),
        QDQPair("q5", "dq5", 19, 20, "boundary_output"),
    ]

    counts = summarise_pairs(pairs)

    assert counts == {
        "internal": 3,
        "boundary_input": 1,
        "boundary_output": 1,
        "total": 5,
    }


def test_summarise_pairs_handles_empty_input() -> None:
    counts = summarise_pairs([])

    assert counts["total"] == 0
    assert counts["internal"] == 0
    assert counts["boundary_input"] == 0
    assert counts["boundary_output"] == 0
