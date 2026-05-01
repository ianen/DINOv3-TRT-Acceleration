"""Tests for `dinov3_trt.quantization.onnx_qdq_strip_planner`."""

from __future__ import annotations

import pytest

from dinov3_trt.quantization.onnx_qdq_stripper import OnnxNodeWithEdges, QDQPair
from dinov3_trt.quantization.onnx_qdq_strip_planner import (
    StripPlan,
    apply_plan_to_node_list,
    plan_strip_operations,
)


def _make_block_qdq_around_matmul(
    block: int, suffix: str
) -> list[OnnxNodeWithEdges]:
    """Synthesize a Q→DQ→MatMul chain inside one transformer block.

    Topology:
        upstream (block.{N-1} output) → Q → mid → DQ → activation → MatMul → downstream
    """

    upstream = f"block_{block - 1}_output"
    mid = f"/model/layer.{block}/{suffix}/quant_mid"
    activation = f"/model/layer.{block}/{suffix}/dequant_out"
    downstream = f"/model/layer.{block}/{suffix}/matmul_out"
    q_node = OnnxNodeWithEdges(
        name=f"/model/layer.{block}/{suffix}/QuantizeLinear",
        op_type="QuantizeLinear",
        inputs=(upstream,),
        outputs=(mid,),
    )
    dq_node = OnnxNodeWithEdges(
        name=f"/model/layer.{block}/{suffix}/DequantizeLinear",
        op_type="DequantizeLinear",
        inputs=(mid,),
        outputs=(activation,),
    )
    matmul_node = OnnxNodeWithEdges(
        name=f"/model/layer.{block}/{suffix}/MatMul",
        op_type="MatMul",
        inputs=(activation, "weight_tensor"),
        outputs=(downstream,),
    )
    return [q_node, dq_node, matmul_node]


def test_plan_marks_internal_pair_for_deletion_and_rewires_consumer() -> None:
    nodes = _make_block_qdq_around_matmul(17, "attention/q_proj/input_quantizer")

    plan = plan_strip_operations(nodes, block_indices=[16, 17, 18, 19])

    assert plan.stripped_pair_count == 1
    assert plan.preserved_pairs == ()
    assert plan.nodes_to_delete == {
        "/model/layer.17/attention/q_proj/input_quantizer/QuantizeLinear",
        "/model/layer.17/attention/q_proj/input_quantizer/DequantizeLinear",
    }
    assert plan.tensor_rewires == {
        "/model/layer.17/attention/q_proj/input_quantizer/dequant_out": "block_16_output",
    }


def test_plan_strips_multiple_internal_pairs_in_range() -> None:
    nodes = (
        _make_block_qdq_around_matmul(16, "attention/q_proj/input_quantizer")
        + _make_block_qdq_around_matmul(17, "attention/q_proj/input_quantizer")
        + _make_block_qdq_around_matmul(19, "mlp/down_proj/input_quantizer")
    )

    plan = plan_strip_operations(nodes, block_indices=[16, 17, 18, 19])

    assert plan.stripped_pair_count == 3
    assert len(plan.nodes_to_delete) == 6  # 3 pairs * 2 nodes
    assert len(plan.tensor_rewires) == 3


def test_plan_skips_pairs_outside_block_range() -> None:
    in_range = _make_block_qdq_around_matmul(17, "attention/q_proj/input_quantizer")
    out_of_range = _make_block_qdq_around_matmul(0, "attention/q_proj/input_quantizer")

    plan = plan_strip_operations(in_range + out_of_range, block_indices=[16, 17, 18, 19])

    assert plan.stripped_pair_count == 1
    assert all("layer.0" not in name for name in plan.nodes_to_delete)


def test_plan_records_boundary_pairs_as_preserved() -> None:
    boundary_in_q = OnnxNodeWithEdges(
        name="/model/layer.15/output/QuantizeLinear",
        op_type="QuantizeLinear",
        inputs=("layer_15_output",),
        outputs=("boundary_mid",),
    )
    boundary_in_dq = OnnxNodeWithEdges(
        name="/model/layer.16/norm1/DequantizeLinear",
        op_type="DequantizeLinear",
        inputs=("boundary_mid",),
        outputs=("layer_16_norm_input",),
    )

    plan = plan_strip_operations(
        [boundary_in_q, boundary_in_dq],
        block_indices=[16, 17, 18, 19],
    )

    assert plan.stripped_pair_count == 0
    assert plan.nodes_to_delete == frozenset()
    assert plan.tensor_rewires == {}
    assert len(plan.preserved_pairs) == 1
    assert plan.preserved_pairs[0].location == "boundary_input"


def test_plan_rejects_empty_block_indices() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        plan_strip_operations(
            _make_block_qdq_around_matmul(17, "x"),
            block_indices=[],
        )


def test_plan_dict_serialisation_round_trip() -> None:
    nodes = _make_block_qdq_around_matmul(17, "attention/q_proj/input_quantizer")

    plan = plan_strip_operations(nodes, block_indices=[16, 17, 18, 19])

    payload = plan.to_dict()
    assert payload["stripped_pair_count"] == 1
    assert isinstance(payload["nodes_to_delete"], list)
    assert sorted(payload["nodes_to_delete"]) == payload["nodes_to_delete"]
    assert isinstance(payload["tensor_rewires"], dict)
    assert payload["preserved_pairs"] == []


def test_apply_plan_removes_q_dq_and_rewires_matmul_input() -> None:
    nodes = _make_block_qdq_around_matmul(17, "attention/q_proj/input_quantizer")
    plan = plan_strip_operations(nodes, block_indices=[16, 17, 18, 19])

    new_nodes = apply_plan_to_node_list(nodes, plan)

    assert len(new_nodes) == 1
    matmul = new_nodes[0]
    assert matmul.op_type == "MatMul"
    assert matmul.inputs == ("block_16_output", "weight_tensor")
    assert matmul.outputs == ("/model/layer.17/attention/q_proj/input_quantizer/matmul_out",)


def test_apply_plan_preserves_unrelated_nodes_unchanged() -> None:
    nodes_in = _make_block_qdq_around_matmul(17, "attention/q_proj/input_quantizer")
    unrelated = OnnxNodeWithEdges(
        name="/model/layer.0/something/Add",
        op_type="Add",
        inputs=("x", "y"),
        outputs=("z",),
    )

    plan = plan_strip_operations(nodes_in + [unrelated], block_indices=[16, 17, 18, 19])
    new_nodes = apply_plan_to_node_list(nodes_in + [unrelated], plan)

    assert any(node is unrelated for node in new_nodes), (
        "unrelated node must be passed through unchanged"
    )


def test_apply_plan_rejects_conflicting_rewires_indirectly() -> None:
    # Construct two pairs that share a downstream tensor; this should never
    # happen in real ONNX but the planner must raise rather than silently
    # produce an inconsistent plan.
    shared_downstream = "/model/layer.17/shared_output"
    q1 = OnnxNodeWithEdges(
        name="/model/layer.17/q1/QuantizeLinear",
        op_type="QuantizeLinear",
        inputs=("up1",),
        outputs=("mid1",),
    )
    dq1 = OnnxNodeWithEdges(
        name="/model/layer.17/q1/DequantizeLinear",
        op_type="DequantizeLinear",
        inputs=("mid1",),
        outputs=(shared_downstream,),
    )
    q2 = OnnxNodeWithEdges(
        name="/model/layer.17/q2/QuantizeLinear",
        op_type="QuantizeLinear",
        inputs=("up2",),
        outputs=("mid2",),
    )
    dq2 = OnnxNodeWithEdges(
        name="/model/layer.17/q2/DequantizeLinear",
        op_type="DequantizeLinear",
        inputs=("mid2",),
        outputs=(shared_downstream,),
    )

    with pytest.raises(ValueError, match="conflicting rewire"):
        plan_strip_operations(
            [q1, dq1, q2, dq2],
            block_indices=[16, 17, 18, 19],
        )


def test_strip_plan_to_dict_lists_preserved_pairs() -> None:
    plan = StripPlan(
        nodes_to_delete=frozenset({"a", "b"}),
        tensor_rewires={"x": "y"},
        preserved_pairs=(
            QDQPair(
                quantize_node="/model/layer.15/q",
                dequantize_node="/model/layer.16/dq",
                quantize_block=15,
                dequantize_block=16,
                location="boundary_input",
            ),
        ),
        stripped_pair_count=1,
    )

    payload = plan.to_dict()

    assert payload["preserved_pairs"] == [
        {
            "quantize_node": "/model/layer.15/q",
            "dequantize_node": "/model/layer.16/dq",
            "location": "boundary_input",
        }
    ]
