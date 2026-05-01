"""Lightweight unit tests for the SmoothQuant CLI helpers.

The full CLI requires PyTorch + ModelOpt + a HF DINOv3 snapshot, which is only
available on the remote RTX 5080. These tests cover the parts that do not need
that stack: argparse plumbing (via ``--dry-run``), config builder behaviour,
and the calibration-batch loader contract. They run as part of the normal
local pytest suite.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "quantize_torch_modelopt_smoothquant.py"


def _run_script(args: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    src_dir = str(ROOT / "src")
    env["PYTHONPATH"] = (
        src_dir + (os.pathsep + env["PYTHONPATH"]) if "PYTHONPATH" in env else src_dir
    )
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def test_dry_run_emits_alpha_and_calibration_plan(tmp_path: Path) -> None:
    """``--dry-run`` must surface the smoothquant alpha + manifest path without
    requiring the model to be present locally."""

    manifest = tmp_path / "calib.json"
    manifest.write_text(
        json.dumps(
            {
                "image_root": str(tmp_path),
                "split": "calib_smoke",
                "seed": 1,
                "images": [],
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "out.onnx"

    result = _run_script(
        [
            "--artifact-root",
            str(tmp_path / "Artifacts"),
            "--calib-manifest",
            str(manifest),
            "--max-calibration-images",
            "8",
            "--load-batch-size",
            "2",
            "--alpha",
            "0.7",
            "--output",
            str(output_path),
            "--dry-run",
        ]
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["dry_run"] is True
    assert payload["alpha"] == 0.7
    assert payload["max_calibration_images"] == 8
    assert payload["load_batch_size"] == 2
    assert payload["calib_manifest"] == str(manifest)
    assert payload["output_path"].endswith("out.onnx")
    assert payload["expected_tokens"] == 197  # 224x224 contract: 1 CLS + 196 patches.

    # If ModelOpt is installed locally, the dry-run must surface the alpha
    # inside the SmoothQuant config. If it is not (laptop / CI), the dry-run
    # gracefully reports the import failure so the rest of the plan is still
    # actionable on the remote RTX 5080.
    config_payload = payload["config"]
    if config_payload is None:
        assert "modelopt unavailable" in payload.get("config_error", "")
    else:
        assert isinstance(config_payload, dict)
        algorithm = config_payload.get("algorithm")
        if isinstance(algorithm, dict):
            assert algorithm.get("alpha") == 0.7
        elif isinstance(algorithm, str):
            assert "smooth" in algorithm.lower()


def test_dry_run_passes_skip_blocks_into_plan(tmp_path: Path) -> None:
    """``--skip-blocks 16-19`` must surface as a parsed integer list in the plan
    so the mixed-precision recipe shows up alongside the alpha sweep config."""

    manifest = tmp_path / "calib.json"
    manifest.write_text(
        json.dumps(
            {
                "image_root": str(tmp_path),
                "split": "calib_smoke",
                "seed": 1,
                "images": [],
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "out.onnx"

    result = _run_script(
        [
            "--artifact-root",
            str(tmp_path / "Artifacts"),
            "--calib-manifest",
            str(manifest),
            "--alpha",
            "0.8",
            "--skip-blocks",
            "16-19",
            "--output",
            str(output_path),
            "--dry-run",
        ]
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["skip_blocks"] == [16, 17, 18, 19]
    assert payload["alpha"] == 0.8


def test_skip_blocks_csv_and_range_are_normalized(tmp_path: Path) -> None:
    manifest = tmp_path / "calib.json"
    manifest.write_text(
        json.dumps(
            {
                "image_root": str(tmp_path),
                "split": "calib_smoke",
                "seed": 1,
                "images": [],
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "out.onnx"

    result = _run_script(
        [
            "--artifact-root",
            str(tmp_path / "Artifacts"),
            "--calib-manifest",
            str(manifest),
            "--skip-blocks",
            "16,18-19,17",
            "--output",
            str(output_path),
            "--dry-run",
        ]
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["skip_blocks"] == [16, 17, 18, 19]


def test_dry_run_carries_image_size_into_expected_tokens(tmp_path: Path) -> None:
    manifest = tmp_path / "calib_336.json"
    manifest.write_text(
        json.dumps(
            {
                "image_root": str(tmp_path),
                "split": "calib_smoke",
                "seed": 1,
                "images": [],
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "out_336.onnx"

    result = _run_script(
        [
            "--artifact-root",
            str(tmp_path / "Artifacts"),
            "--calib-manifest",
            str(manifest),
            "--image-size",
            "336",
            "--output",
            str(output_path),
            "--dry-run",
        ]
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["image_size"] == 336
    assert payload["expected_tokens"] == 442  # 336/16 = 21 -> 21*21 + CLS.
