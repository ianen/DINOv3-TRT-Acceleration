"""Tests for the unified `scripts/build_all_figures.py` driver."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_all_figures.py"


def _load_script_module() -> Any:
    spec = importlib.util.spec_from_file_location("build_all_figures", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load build_all_figures script")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


SCRIPT = _load_script_module()


def _write_minimal_matrix_csv(target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=("runtime", "candidate", "reference", "batch_size", "latency_speedup"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "runtime": "trtexec",
                "candidate": "BF16-prefer",
                "reference": "FP32",
                "batch_size": "1",
                "latency_speedup": "2.45",
            }
        )


def _write_layer_ablation_report(target: Path) -> None:
    payload = {
        "candidates": {
            "project": {
                "layer_numbers_1based": [4, 12, 16, 20],
                "pairwise_cosine_overall_mean": 0.383,
                "per_output_magnitude_mean": [362.0, 972.0, 1753.0, 4560.0],
            },
        },
        "diversity_ranking_low_to_high_cosine": ["project"],
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload), encoding="utf-8")


def test_main_writes_figures_index_with_all_four_subsystems(tmp_path: Path) -> None:
    matrix_csv = tmp_path / "matrix.csv"
    reports_dir = tmp_path / "reports"
    output_dir = tmp_path / "figures"
    _write_minimal_matrix_csv(matrix_csv)
    _write_layer_ablation_report(reports_dir / "layer_ablation_pytorch_eval1000_r224.json")

    rc = SCRIPT.main(
        [
            "--matrix-csv",
            str(matrix_csv),
            "--reports-dir",
            str(reports_dir),
            "--output-dir",
            str(output_dir),
            "--allow-missing",
        ]
    )

    assert rc == 0
    index_path = output_dir / "figures_index.json"
    assert index_path.exists()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert set(payload["subsystems"].keys()) == {"speedup", "cosine", "tradeoff", "layer_ablation"}
    assert payload["allow_missing"] is True
    assert payload["matrix_csv"] == str(matrix_csv)
    assert payload["subsystems"]["layer_ablation"]["figure_count"] == 1


def test_main_uses_custom_index_output_path(tmp_path: Path) -> None:
    matrix_csv = tmp_path / "matrix.csv"
    reports_dir = tmp_path / "reports"
    output_dir = tmp_path / "figures"
    custom_index = tmp_path / "elsewhere" / "index.json"
    _write_minimal_matrix_csv(matrix_csv)

    rc = SCRIPT.main(
        [
            "--matrix-csv",
            str(matrix_csv),
            "--reports-dir",
            str(reports_dir),
            "--output-dir",
            str(output_dir),
            "--index-output",
            str(custom_index),
            "--allow-missing",
        ]
    )

    assert rc == 0
    assert custom_index.exists()
    assert not (output_dir / "figures_index.json").exists()


def test_main_propagates_allow_missing_to_subsystems(tmp_path: Path) -> None:
    matrix_csv = tmp_path / "matrix.csv"
    reports_dir = tmp_path / "reports"
    output_dir = tmp_path / "figures"
    _write_minimal_matrix_csv(matrix_csv)
    reports_dir.mkdir(parents=True, exist_ok=True)

    rc = SCRIPT.main(
        [
            "--matrix-csv",
            str(matrix_csv),
            "--reports-dir",
            str(reports_dir),
            "--output-dir",
            str(output_dir),
            "--allow-missing",
        ]
    )

    assert rc == 0
    payload = json.loads((output_dir / "figures_index.json").read_text(encoding="utf-8"))
    assert payload["subsystems"]["cosine"]["figure_count"] >= 1
    assert payload["subsystems"]["tradeoff"]["figure_count"] >= 1
    assert payload["subsystems"]["layer_ablation"]["figure_count"] >= 1


def test_main_without_allow_missing_propagates_failure(tmp_path: Path) -> None:
    matrix_csv = tmp_path / "matrix.csv"
    reports_dir = tmp_path / "reports"
    output_dir = tmp_path / "figures"
    _write_minimal_matrix_csv(matrix_csv)
    reports_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises((FileNotFoundError, ValueError)):
        SCRIPT.main(
            [
                "--matrix-csv",
                str(matrix_csv),
                "--reports-dir",
                str(reports_dir),
                "--output-dir",
                str(output_dir),
            ]
        )


def test_summarise_manifest_extracts_row_counts() -> None:
    manifest = {
        "figures": [
            {
                "name": "cosine-r224",
                "row_count": 4,
                "output": "/tmp/x.svg",
                "missing_reports": [],
            },
            {
                "name": "cosine-r518",
                "row_count": 0,
                "missing_report": "/tmp/missing.json",
            },
        ]
    }

    summary = SCRIPT._summarise_manifest(manifest)

    assert summary["figure_count"] == 2
    assert summary["figures"][0]["name"] == "cosine-r224"
    assert summary["figures"][0]["row_count"] == 4
    assert "missing_reports" not in summary["figures"][0]
    assert summary["figures"][1]["missing_report"] == "/tmp/missing.json"
