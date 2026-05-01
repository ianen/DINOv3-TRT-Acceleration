"""Tests for `scripts/build_layer_precisions_arg.py`."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_layer_precisions_arg.py"


def _load_script_module() -> Any:
    spec = importlib.util.spec_from_file_location("build_layer_precisions_arg", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load build_layer_precisions_arg script")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


SCRIPT = _load_script_module()


@pytest.mark.parametrize(
    "value, expected",
    [
        ("16-19", (16, 17, 18, 19)),
        ("16,17,18,19", (16, 17, 18, 19)),
        ("16-19,22", (16, 17, 18, 19, 22)),
        ("19,16-18,16", (16, 17, 18, 19)),
        ("0", (0,)),
    ],
)
def test_parse_blocks_spec_supports_ranges_csv_and_dedup(
    value: str, expected: tuple[int, ...]
) -> None:
    assert SCRIPT.parse_blocks_spec(value) == expected


def test_parse_blocks_spec_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        SCRIPT.parse_blocks_spec("")


def test_parse_blocks_spec_rejects_reversed_range() -> None:
    with pytest.raises(ValueError, match="reversed"):
        SCRIPT.parse_blocks_spec("19-16")


def test_parse_op_types_all_returns_none() -> None:
    assert SCRIPT.parse_op_types("all") is None
    assert SCRIPT.parse_op_types(" ALL ") is None


def test_parse_op_types_returns_csv_tuple() -> None:
    assert SCRIPT.parse_op_types("MatMul,Add") == ("MatMul", "Add")
    assert SCRIPT.parse_op_types(" MatMul , Add ") == ("MatMul", "Add")


def test_parse_op_types_rejects_empty_csv() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        SCRIPT.parse_op_types("  ,  ")


def test_main_writes_file_and_sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_nodes = (
        SCRIPT.OnnxNodeInfo("/model/embeddings/Conv", "Conv"),
        SCRIPT.OnnxNodeInfo("/model/layer.16/attention/q_proj/MatMul", "MatMul"),
        SCRIPT.OnnxNodeInfo("/model/layer.16/Constant", "Constant"),
        SCRIPT.OnnxNodeInfo("/model/layer.17/mlp/fc1/MatMul", "MatMul"),
        SCRIPT.OnnxNodeInfo("/model/layer.18/attention/q_proj/MatMul", "MatMul"),
        SCRIPT.OnnxNodeInfo("/model/layer.19/mlp/fc1/Add", "Add"),
    )
    monkeypatch.setattr(
        SCRIPT,
        "load_onnx_node_info",
        lambda path, *, load_external_data: fake_nodes,
    )

    out = tmp_path / "trtexec_layer_precisions.txt"
    rc = SCRIPT.main(
        [
            "--onnx",
            str(tmp_path / "fake.onnx"),
            "--blocks",
            "16-19",
            "--precision",
            "bf16",
            "--output",
            str(out),
            "--op-types",
            "MatMul,Add",
        ]
    )

    assert rc == 0
    text = out.read_text(encoding="utf-8")
    expected_pairs = [
        "/model/layer.16/attention/q_proj/MatMul:bf16",
        "/model/layer.17/mlp/fc1/MatMul:bf16",
        "/model/layer.18/attention/q_proj/MatMul:bf16",
        "/model/layer.19/mlp/fc1/Add:bf16",
    ]
    assert text == ",".join(expected_pairs)
    sidecar = out.with_suffix(out.suffix + ".meta.json")
    assert sidecar.exists()


def test_main_returns_error_when_no_nodes_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        SCRIPT,
        "load_onnx_node_info",
        lambda path, *, load_external_data: (
            SCRIPT.OnnxNodeInfo("/model/embeddings/Conv", "Conv"),
        ),
    )

    out = tmp_path / "trtexec_layer_precisions.txt"
    rc = SCRIPT.main(
        [
            "--onnx",
            str(tmp_path / "fake.onnx"),
            "--blocks",
            "16-19",
            "--precision",
            "bf16",
            "--output",
            str(out),
        ]
    )

    assert rc == 2
    assert not out.exists()
