"""Tests for `scripts/run_layer_ablation_pytorch.py`."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "run_layer_ablation_pytorch.py"


def _load_script_module() -> Any:
    spec = importlib.util.spec_from_file_location("run_layer_ablation_pytorch", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load run_layer_ablation_pytorch script")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


SCRIPT = _load_script_module()


def test_select_candidates_returns_subset_in_input_order() -> None:
    selected = SCRIPT.select_candidates("late,project")

    assert list(selected.keys()) == ["late", "project"]
    assert selected["project"] == (3, 11, 15, 19)
    assert selected["late"] == (5, 11, 17, 23)


def test_select_candidates_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown candidate"):
        SCRIPT.select_candidates("project,unknown")


def test_select_candidates_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        SCRIPT.select_candidates("")


def test_select_candidates_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        SCRIPT.select_candidates("project,project")


def test_pair_labels_uses_one_based_layer_numbers() -> None:
    labels = SCRIPT.pair_labels((4, 12, 16, 20))

    assert labels == ["L4-L12", "L4-L16", "L4-L20", "L12-L16", "L12-L20", "L16-L20"]


def test_pairwise_cosine_per_batch_orthogonal_yields_zero() -> None:
    rng = np.random.default_rng(0)
    base = rng.standard_normal((1, 3, 8))
    raw = rng.standard_normal((1, 3, 8))
    base_dot = (base * base).sum(axis=-1, keepdims=True)
    proj_coeff = (raw * base).sum(axis=-1, keepdims=True) / base_dot
    orthogonal = raw - proj_coeff * base
    features = np.concatenate([base, orthogonal], axis=0)

    pair_cos = SCRIPT.pairwise_cosine_per_batch(features)

    assert pair_cos.shape == (1,)
    assert abs(pair_cos[0]) < 1e-6


def test_pairwise_cosine_per_batch_identical_yields_one() -> None:
    rng = np.random.default_rng(1)
    feature = rng.standard_normal((1, 4, 16))
    features = np.concatenate([feature, feature, feature], axis=0)

    pair_cos = SCRIPT.pairwise_cosine_per_batch(features)

    assert pair_cos.shape == (3,)
    np.testing.assert_allclose(pair_cos, np.ones(3), atol=1e-6)


def test_per_output_magnitude_returns_l2_per_output() -> None:
    arr = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[3.0, 4.0], [3.0, 4.0]],
        ],
        dtype=np.float32,
    )

    magnitudes = SCRIPT.per_output_magnitude(arr)

    assert magnitudes.shape == (2,)
    assert abs(float(magnitudes[0]) - 1.0) < 1e-6
    assert abs(float(magnitudes[1]) - 5.0) < 1e-6


def test_aggregate_summary_uses_contract_output_names() -> None:
    candidates = {
        "project": (3, 11, 15, 19),
        "dpt": (4, 10, 16, 22),
    }
    cosine_pair_lists = {
        name: [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]] for name in candidates
    }
    magnitude_lists = {name: [[1.0, 2.0, 3.0, 4.0]] for name in candidates}

    summary = SCRIPT.aggregate_summary(candidates, cosine_pair_lists, magnitude_lists, 224)

    assert summary["project"]["output_names"] == [
        "feat_layer_4",
        "feat_layer_12",
        "feat_layer_16",
        "feat_layer_20",
    ]
    assert summary["dpt"]["output_names"] == [
        "feat_layer_5",
        "feat_layer_11",
        "feat_layer_17",
        "feat_layer_23",
    ]
    assert summary["project"]["pairwise_cosine_overall_mean"] == 0.5
    assert summary["dpt"]["per_output_magnitude_mean"] == [1.0, 2.0, 3.0, 4.0]


def test_dry_run_writes_plan_with_one_based_layer_numbers(tmp_path: Path) -> None:
    output = tmp_path / "plan.json"
    args = SCRIPT.parse_args(
        [
            "--model-name-or-path",
            "facebook/dinov3-vitl16-pretrain-lvd1689m",
            "--output",
            str(output),
            "--candidates",
            "project,dpt",
            "--dry-run",
        ]
    )
    candidates = SCRIPT.select_candidates(args.candidates)

    SCRIPT.write_dry_run_plan(args, candidates)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["dry_run"] is True
    assert payload["candidates_one_based"]["project"] == [4, 12, 16, 20]
    assert payload["candidates_one_based"]["dpt"] == [5, 11, 17, 23]
    assert payload["num_blocks"] == 24


def test_main_dry_run_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "ablation_dry.json"

    rc = SCRIPT.main(
        [
            "--model-name-or-path",
            "facebook/dinov3-vitl16-pretrain-lvd1689m",
            "--output",
            str(output),
            "--candidates",
            "project,dpt,late",
            "--dry-run",
        ]
    )

    assert rc == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert sorted(payload["candidates"]) == ["dpt", "late", "project"]


def test_write_markdown_report_renders_table_and_ranking(tmp_path: Path) -> None:
    payload = {
        "model_name_or_path": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 224,
        "batch_size": 4,
        "image_count": 100,
        "candidates": {
            "project": {
                "layer_numbers_1based": [4, 12, 16, 20],
                "pairwise_cosine_overall_mean": 0.65,
                "pairwise_cosine_overall_min": 0.6,
                "pairwise_cosine_overall_max": 0.7,
                "per_output_magnitude_mean": [10.0, 20.0, 30.0, 40.0],
                "per_output_magnitude_std": [1.0, 2.0, 3.0, 4.0],
            },
            "dpt": {
                "layer_numbers_1based": [5, 11, 17, 23],
                "pairwise_cosine_overall_mean": 0.55,
                "pairwise_cosine_overall_min": 0.5,
                "pairwise_cosine_overall_max": 0.6,
                "per_output_magnitude_mean": [11.0, 21.0, 31.0, 41.0],
                "per_output_magnitude_std": [1.5, 2.5, 3.5, 4.5],
            },
        },
        "diversity_ranking_low_to_high_cosine": ["dpt", "project"],
    }
    md_path = tmp_path / "ablation.md"

    SCRIPT.write_markdown_report(md_path, payload)

    text = md_path.read_text(encoding="utf-8")
    assert "| project | 4/12/16/20 | 0.6500 | 0.6000 | 0.7000 |" in text
    assert "| dpt | 5/11/17/23 | 0.5500 | 0.5000 | 0.6000 |" in text
    assert "| project | L4 | 10.00 | 1.00 |" in text
    assert "1. **dpt**" in text
