"""Tests for `scripts/run_imagenet_val_post_download.py`."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "run_imagenet_val_post_download.py"


def _load_script_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "run_imagenet_val_post_download", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load script")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


SCRIPT = _load_script_module()


def test_parse_args_defaults() -> None:
    args = SCRIPT.parse_args([])
    assert args.eval_count == 1000
    assert args.calib_count == 500
    assert args.seed == 42
    assert args.batch_size == 8
    assert args.image_size == 224
    assert args.dry_run is False
    assert args.skip_pair == []


def test_parse_args_skip_pair_repeatable() -> None:
    args = SCRIPT.parse_args(
        [
            "--skip-pair",
            "bf16_prefer",
            "--skip-pair",
            "int8_smoothquant_a080",
        ]
    )
    assert set(args.skip_pair) == {"bf16_prefer", "int8_smoothquant_a080"}


def test_resolve_image_root_uses_cli_when_provided(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    resolved = SCRIPT.resolve_image_root(image_dir)
    assert resolved == image_dir.resolve()


def test_resolve_image_root_rejects_nonexistent_cli(tmp_path: Path) -> None:
    bogus = tmp_path / "missing"
    with pytest.raises(SystemExit):
        SCRIPT.resolve_image_root(bogus)


def test_resolve_image_root_reads_success_marker(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (target / "ILSVRC2012_val_00000001.JPEG").write_bytes(b"\xff\xd8")

    success = tmp_path / "success_marker"
    success.write_text(str(target), encoding="utf-8")

    resolved = SCRIPT.resolve_image_root(None, success_marker=success)
    assert resolved == target.resolve()


def test_resolve_image_root_descends_one_level_when_marker_root_lacks_images(
    tmp_path: Path,
) -> None:
    """kagglehub returns a versions/ root; images may be one level deeper."""
    versions_root = tmp_path / "versions" / "1"
    nested = versions_root / "imagenet1k_val"
    nested.mkdir(parents=True)
    (nested / "ILSVRC2012_val_00000001.JPEG").write_bytes(b"\xff\xd8")

    success = tmp_path / "success_marker"
    success.write_text(str(versions_root), encoding="utf-8")

    resolved = SCRIPT.resolve_image_root(None, success_marker=success)
    assert resolved == nested.resolve()


def test_resolve_image_root_missing_marker_raises(tmp_path: Path) -> None:
    success = tmp_path / "no_such_marker"
    with pytest.raises(SystemExit):
        SCRIPT.resolve_image_root(None, success_marker=success)


def test_build_manifest_paths_uses_eval_count(tmp_path: Path) -> None:
    eval_p, calib_p = SCRIPT.build_manifest_paths(tmp_path, 1000)
    assert eval_p.name == "imagenet_val_50k_eval_1000.json"
    assert calib_p.name == "imagenet_val_50k_calib_1000.json"


def _build_canonical_report(
    outputs: list[dict[str, float]],
) -> dict[str, object]:
    """Match the schema produced by ``evaluate_engine_pair_on_images.py``."""
    return {
        "outputs": [
            {
                "name": o["name"],
                "cosine_similarity_min": o["cos_min"],
                "cosine_similarity_mean": o["cos_mean"],
                "max_abs_error": 0.0,
                "mean_abs_error": 0.0,
                "root_mean_square_error": 0.0,
            }
            for o in outputs
        ],
        "batches": [],
    }


def test_summarize_pair_r1_strict_pass(tmp_path: Path) -> None:
    pair = SCRIPT.EnginePair(
        label="bf16_prefer",
        reference="ref.engine",
        candidate="cand.engine",
    )
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            _build_canonical_report(
                [
                    {"name": "feat_layer_4", "cos_min": 0.998, "cos_mean": 0.9995},
                    {"name": "feat_layer_12", "cos_min": 0.997, "cos_mean": 0.9992},
                    {"name": "feat_layer_16", "cos_min": 0.996, "cos_mean": 0.9990},
                    {"name": "feat_layer_20", "cos_min": 0.995, "cos_mean": 0.9988},
                ]
            )
        ),
        encoding="utf-8",
    )
    s = SCRIPT.summarize_pair(pair, report)
    assert s["verdict"] == "R1_PASS_strict"
    assert pytest.approx(s["cos_min_overall"], rel=1e-6) == 0.995


def test_summarize_pair_r2_emergency_pass(tmp_path: Path) -> None:
    pair = SCRIPT.EnginePair(
        label="int8_smoothquant_a080",
        reference="ref.engine",
        candidate="cand.engine",
    )
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            _build_canonical_report(
                [
                    {"name": "feat_layer_4", "cos_min": 0.985, "cos_mean": 0.992},
                    {"name": "feat_layer_20", "cos_min": 0.972, "cos_mean": 0.985},
                ]
            )
        ),
        encoding="utf-8",
    )
    s = SCRIPT.summarize_pair(pair, report)
    assert s["verdict"] == "R2_PASS_emergency"
    assert pytest.approx(s["cos_min_overall"], rel=1e-6) == 0.972


def test_summarize_pair_fail_below_r2(tmp_path: Path) -> None:
    pair = SCRIPT.EnginePair(
        label="int8_smoothquant_a080",
        reference="ref.engine",
        candidate="cand.engine",
    )
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            _build_canonical_report(
                [{"name": "feat_layer_20", "cos_min": 0.94, "cos_mean": 0.97}]
            )
        ),
        encoding="utf-8",
    )
    s = SCRIPT.summarize_pair(pair, report)
    assert s["verdict"] == "FAIL"
    assert pytest.approx(s["cos_min_overall"], rel=1e-6) == 0.94


def test_summarize_pair_handles_missing_report(tmp_path: Path) -> None:
    pair = SCRIPT.EnginePair(
        label="bf16_prefer",
        reference="ref.engine",
        candidate="cand.engine",
    )
    s = SCRIPT.summarize_pair(pair, tmp_path / "absent.json")
    assert s["verdict"] == "REPORT_MISSING"


def test_summarize_pair_accepts_legacy_dict_layout(tmp_path: Path) -> None:
    """Older fixtures used per_output_metrics as a {name: metrics} mapping."""
    pair = SCRIPT.EnginePair(
        label="bf16_prefer",
        reference="ref.engine",
        candidate="cand.engine",
    )
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "per_output_metrics": {
                    "feat_layer_20": {"cos_min": 0.9985, "cos_mean": 0.9995},
                }
            }
        ),
        encoding="utf-8",
    )
    s = SCRIPT.summarize_pair(pair, report)
    assert s["verdict"] == "R1_PASS_strict"


def test_summarize_pair_uses_canonical_evaluator_schema(tmp_path: Path) -> None:
    """Regression: real evaluate_engine_pair_on_images.py output format.

    The canonical schema places per-output aggregates inside
    ``outputs: list[dict]`` with field name ``cosine_similarity_min`` and
    ``cosine_similarity_mean`` — older field-name aliases must NOT take
    precedence over canonical names.
    """
    pair = SCRIPT.EnginePair(
        label="bf16_prefer",
        reference="ref.engine",
        candidate="cand.engine",
    )
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "outputs": [
                    {
                        "name": "feat_layer_4",
                        "cosine_similarity_min": 0.99994,
                        "cosine_similarity_mean": 0.99995,
                        "max_abs_error": 0.83,
                    },
                    {
                        "name": "feat_layer_20",
                        "cosine_similarity_min": 0.99924,
                        "cosine_similarity_mean": 0.99956,
                        "max_abs_error": 16.28,
                    },
                ],
                "batches": [],
            }
        ),
        encoding="utf-8",
    )
    s = SCRIPT.summarize_pair(pair, report)
    assert s["verdict"] == "R1_PASS_strict"
    assert pytest.approx(s["cos_min_overall"], rel=1e-6) == 0.99924


def test_run_subprocess_dry_run_skips_execution() -> None:
    with patch("subprocess.run") as run:
        rc = SCRIPT.run_subprocess(["echo", "hi"], dry_run=True)
        assert rc == 0
        run.assert_not_called()


def test_default_pairs_include_bf16_and_int8() -> None:
    labels = {p.label for p in SCRIPT.DEFAULT_PAIRS}
    assert {"bf16_prefer", "int8_smoothquant_a080"} <= labels


def test_default_pairs_thresholds_match_v101_doctrine() -> None:
    """V1.0.1 §12.1: R1 strict cos_min ≥ 0.99, R2 emergency cos_min ≥ 0.97."""
    for pair in SCRIPT.DEFAULT_PAIRS:
        assert pair.cos_min_threshold_r1 == 0.99
        assert pair.cos_min_threshold_r2 == 0.97


def test_main_dry_run_with_one_pair_skipped(tmp_path: Path) -> None:
    image_dir = tmp_path / "imgs"
    image_dir.mkdir()
    (image_dir / "x.jpg").write_bytes(b"\xff\xd8")

    engines_dir = tmp_path / "engines"
    engines_dir.mkdir()
    # Touch the BF16 prefer pair's engines so existence checks pass even in
    # dry-run mode (run_cosine_eval validates files before short-circuiting).
    for engine_filename in (
        "dinov3_vitl16_4out.fp32.engine",
        "dinov3_vitl16_4out.bf16.prefer.engine",
    ):
        (engines_dir / engine_filename).write_bytes(b"")

    rc = SCRIPT.main(
        [
            "--image-root",
            str(image_dir),
            "--manifest-dir",
            str(tmp_path / "manifests"),
            "--report-dir",
            str(tmp_path / "reports"),
            "--engines-dir",
            str(engines_dir),
            "--skip-pair",
            "int8_smoothquant_a080",
            "--dry-run",
        ]
    )
    assert rc == 0


def test_main_dry_run_skip_all_pairs_raises(tmp_path: Path) -> None:
    image_dir = tmp_path / "imgs"
    image_dir.mkdir()
    (image_dir / "x.jpg").write_bytes(b"\xff\xd8")
    with pytest.raises(SystemExit):
        SCRIPT.main(
            [
                "--image-root",
                str(image_dir),
                "--manifest-dir",
                str(tmp_path / "manifests"),
                "--report-dir",
                str(tmp_path / "reports"),
                "--engines-dir",
                str(tmp_path / "engines"),
                "--skip-pair",
                "bf16_prefer",
                "--skip-pair",
                "int8_smoothquant_a080",
                "--dry-run",
            ]
        )
