"""Tests for `scripts/sparsify_onnx_weights.py`.

The script's pure-Python helpers (``find_matching_initializers``,
``parse_args``) are testable without onnx installed; the ``main`` flow
that touches actual ONNX is mock-tested via ``unittest.mock``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "sparsify_onnx_weights.py"


def _load_script_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "sparsify_onnx_weights", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load sparsify_onnx_weights script")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


SCRIPT = _load_script_module()


def test_parse_args_requires_input_output_and_at_least_one_pattern() -> None:
    with pytest.raises(SystemExit):
        SCRIPT.parse_args([])
    with pytest.raises(SystemExit):
        SCRIPT.parse_args(["--input", "x.onnx"])
    args = SCRIPT.parse_args(
        [
            "--input",
            "in.onnx",
            "--output",
            "out.onnx",
            "--layer-pattern",
            "blocks\\.\\d+\\.attn\\.qkv\\.weight",
        ]
    )
    assert args.input == Path("in.onnx")
    assert args.output == Path("out.onnx")
    assert args.layer_pattern == ["blocks\\.\\d+\\.attn\\.qkv\\.weight"]
    assert args.axis == -1
    assert args.dry_run is False


def test_parse_args_layer_pattern_repeatable() -> None:
    args = SCRIPT.parse_args(
        [
            "--input",
            "in.onnx",
            "--output",
            "out.onnx",
            "--layer-pattern",
            "blocks\\.\\d+\\.attn\\.qkv\\.weight",
            "--layer-pattern",
            "blocks\\.\\d+\\.mlp\\.fc1\\.weight",
        ]
    )
    assert len(args.layer_pattern) == 2


def test_parse_args_dry_run_flag() -> None:
    args = SCRIPT.parse_args(
        [
            "--input",
            "in.onnx",
            "--output",
            "out.onnx",
            "--layer-pattern",
            ".*",
            "--dry-run",
        ]
    )
    assert args.dry_run is True


def test_find_matching_initializers_anchors_via_search_not_match() -> None:
    """Patterns are matched with re.search so partial matches succeed."""
    names = [
        "blocks.0.attn.qkv.weight",
        "blocks.0.attn.qkv.bias",  # should NOT match weight pattern
        "blocks.12.attn.proj.weight",
        "blocks.0.mlp.fc1.weight",
        "patch_embed.proj.weight",
    ]
    patterns = [
        r"blocks\.\d+\.attn\.qkv\.weight",
        r"blocks\.\d+\.attn\.proj\.weight",
        r"blocks\.\d+\.mlp\.fc1\.weight",
    ]
    matched = SCRIPT.find_matching_initializers(names, patterns)
    assert "blocks.0.attn.qkv.weight" in matched
    assert "blocks.0.attn.qkv.bias" not in matched
    assert "blocks.12.attn.proj.weight" in matched
    assert "blocks.0.mlp.fc1.weight" in matched
    assert "patch_embed.proj.weight" not in matched
    # No duplicates
    assert len(matched) == len(set(matched))


def test_find_matching_initializers_empty_when_no_pattern_matches() -> None:
    names = ["x", "y", "z"]
    patterns = [r"blocks\.\d+\..*\.weight"]
    assert SCRIPT.find_matching_initializers(names, patterns) == []


def test_find_matching_initializers_handles_multi_pattern_overlap() -> None:
    """A name matched by two patterns is included exactly once."""
    names = ["blocks.0.attn.qkv.weight"]
    patterns = [r"blocks\.0", r".*qkv.*", r".*\.weight"]
    matched = SCRIPT.find_matching_initializers(names, patterns)
    assert matched == ["blocks.0.attn.qkv.weight"]


# =============================================================================
# main() flow: mock-test the onnx-dependent path
# =============================================================================


def _make_fake_onnx_model_with_weights(
    weights: dict[str, "tuple[tuple[int, ...], list[float]]"],
) -> Any:
    """Construct a fake ONNX model whose .graph.initializer mimics real onnx."""
    fake_initializers = []
    for name, (shape, flat_values) in weights.items():
        fake_initializer = MagicMock()
        fake_initializer.name = name
        # ``np_array`` is what the (mocked) ``numpy_helper.to_array`` should return.
        import numpy as np

        fake_initializer._np_array = np.array(flat_values, dtype=np.float32).reshape(
            shape
        )
        fake_initializers.append(fake_initializer)

    fake_graph = MagicMock()
    fake_graph.initializer = fake_initializers
    fake_model = MagicMock()
    fake_model.graph = fake_graph
    return fake_model


def test_main_dry_run_with_one_match_writes_no_output(tmp_path: Path) -> None:
    weights = {
        "blocks.0.attn.qkv.weight": (
            (4, 4),
            [1.0, 2.0, 3.0, 4.0,
             5.0, 6.0, 7.0, 8.0,
             9.0, 10.0, 11.0, 12.0,
             13.0, 14.0, 15.0, 16.0],
        ),
        "blocks.0.attn.qkv.bias": ((4,), [0.1, 0.2, 0.3, 0.4]),
    }
    fake_model = _make_fake_onnx_model_with_weights(weights)

    fake_onnx = MagicMock()
    fake_onnx.load = MagicMock(return_value=fake_model)
    fake_onnx.save = MagicMock()

    fake_numpy_helper = MagicMock()
    fake_numpy_helper.to_array = lambda tensor: tensor._np_array
    fake_numpy_helper.from_array = MagicMock(return_value=MagicMock())

    in_path = tmp_path / "in.onnx"
    in_path.write_bytes(b"fake")
    out_path = tmp_path / "out.onnx"

    with patch.dict(
        sys.modules,
        {
            "onnx": fake_onnx,
            "onnx.numpy_helper": fake_numpy_helper,
        },
    ):
        # Inside main(), ``from onnx import numpy_helper`` triggers import of
        # the ``onnx.numpy_helper`` submodule from sys.modules (already present
        # via patch.dict), so we additionally make ``fake_onnx.numpy_helper``
        # resolve to ``fake_numpy_helper``.
        fake_onnx.numpy_helper = fake_numpy_helper
        rc = SCRIPT.main(
            [
                "--input",
                str(in_path),
                "--output",
                str(out_path),
                "--layer-pattern",
                r"blocks\.\d+\.attn\.qkv\.weight",
                "--dry-run",
            ]
        )
    assert rc == 0
    fake_onnx.save.assert_not_called()


def test_main_writes_output_when_not_dry_run(tmp_path: Path) -> None:
    weights = {
        "blocks.0.attn.qkv.weight": (
            (1, 4),
            [1.0, 2.0, 3.0, 4.0],
        ),
    }
    fake_model = _make_fake_onnx_model_with_weights(weights)

    fake_onnx = MagicMock()
    fake_onnx.load = MagicMock(return_value=fake_model)
    fake_onnx.save = MagicMock()

    fake_numpy_helper = MagicMock()
    fake_numpy_helper.to_array = lambda tensor: tensor._np_array
    fake_numpy_helper.from_array = MagicMock(return_value=MagicMock())

    in_path = tmp_path / "in.onnx"
    in_path.write_bytes(b"fake")
    out_path = tmp_path / "out.onnx"
    report_path = tmp_path / "report.json"

    with patch.dict(
        sys.modules,
        {
            "onnx": fake_onnx,
            "onnx.numpy_helper": fake_numpy_helper,
        },
    ):
        fake_onnx.numpy_helper = fake_numpy_helper
        rc = SCRIPT.main(
            [
                "--input",
                str(in_path),
                "--output",
                str(out_path),
                "--layer-pattern",
                r"blocks\.\d+\.attn\.qkv\.weight",
                "--report",
                str(report_path),
            ]
        )
    assert rc == 0
    fake_onnx.save.assert_called_once()
    assert report_path.exists()
    import json as _json

    report = _json.loads(report_path.read_text(encoding="utf-8"))
    assert report["modified_count"] == 1
    assert report["tensors"][0]["name"] == "blocks.0.attn.qkv.weight"
    assert report["tensors"][0]["density"] == 0.5  # 2:4 → 50% density


def test_main_skips_incompatible_shapes(tmp_path: Path) -> None:
    """A weight whose last dim is not divisible by 4 is reported and skipped."""
    weights = {
        "blocks.0.attn.qkv.weight": (
            (1, 5),  # 5 is not divisible by 4
            [1.0, 2.0, 3.0, 4.0, 5.0],
        ),
    }
    fake_model = _make_fake_onnx_model_with_weights(weights)

    fake_onnx = MagicMock()
    fake_onnx.load = MagicMock(return_value=fake_model)
    fake_onnx.save = MagicMock()

    fake_numpy_helper = MagicMock()
    fake_numpy_helper.to_array = lambda tensor: tensor._np_array
    fake_numpy_helper.from_array = MagicMock(return_value=MagicMock())

    in_path = tmp_path / "in.onnx"
    in_path.write_bytes(b"fake")
    out_path = tmp_path / "out.onnx"

    with patch.dict(
        sys.modules,
        {
            "onnx": fake_onnx,
            "onnx.numpy_helper": fake_numpy_helper,
        },
    ):
        fake_onnx.numpy_helper = fake_numpy_helper
        rc = SCRIPT.main(
            [
                "--input",
                str(in_path),
                "--output",
                str(out_path),
                "--layer-pattern",
                r"blocks\.\d+\.attn\.qkv\.weight",
            ]
        )
    assert rc == 0
    # Output ONNX still saved (with no modifications), but no tensor was masked.
    fake_onnx.save.assert_called_once()


def test_main_raises_clear_error_when_onnx_missing() -> None:
    """Without onnx installed, main() should give an actionable SystemExit."""
    with patch.dict(sys.modules, {"onnx": None}):
        # Setting onnx=None forces ``import onnx`` to raise ImportError.
        with pytest.raises(SystemExit, match="onnx package is required"):
            SCRIPT.main(
                [
                    "--input",
                    "/tmp/x.onnx",
                    "--output",
                    "/tmp/y.onnx",
                    "--layer-pattern",
                    ".*",
                ]
            )
