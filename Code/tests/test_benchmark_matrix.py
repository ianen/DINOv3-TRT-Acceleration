import csv
import json
from pathlib import Path

import pytest

from dinov3_trt.reports.benchmark_matrix import (
    DEFAULT_BENCHMARK_MATRIX_SPECS,
    BenchmarkMatrixSpec,
    build_benchmark_matrix,
    render_benchmark_matrix_markdown,
    write_benchmark_matrix,
)


def test_default_matrix_specs_include_formal_bf16_and_int8_followups() -> None:
    filenames = {spec.filename for spec in DEFAULT_BENCHMARK_MATRIX_SPECS}

    assert "trtexec_formal_fp32_vs_bf16_prefer_locked2752_spinwait_speedup.json" in filenames
    assert "trtexec_formal_r336_fp32_vs_bf16_prefer_locked2752_spinwait_speedup.json" in filenames
    assert "trtexec_formal_r518_fp32_vs_bf16_prefer_locked2752_spinwait_speedup.json" in filenames
    assert (
        "trtexec_formal_fp32_vs_int8_modelopt_imagenette64_"
        "matmul_fine_layer19_attention_locked2752_spinwait_speedup.json"
        in filenames
    )
    assert "cpp_runtime_formal_fp32_vs_bf16_prefer_speedup.json" in filenames


def test_default_matrix_specs_include_smoothquant_mixed_layer_precisions_followup() -> None:
    candidates = {spec.candidate for spec in DEFAULT_BENCHMARK_MATRIX_SPECS}

    assert "INT8 SmoothQuant alpha=0.8" in candidates
    assert "INT8 SmoothQuant alpha=0.8 skip16-19" in candidates
    assert "INT8 SmoothQuant alpha=0.8 mixed l16-19:fp32" in candidates

    mixed_spec = next(
        spec
        for spec in DEFAULT_BENCHMARK_MATRIX_SPECS
        if spec.candidate == "INT8 SmoothQuant alpha=0.8 mixed l16-19:fp32"
    )
    assert mixed_spec.filename == (
        "trtexec_formal_fp32_vs_int8_smoothquant_alpha080_mixed_l16-19_fp32_"
        "locked2752_spinwait_speedup.json"
    )
    assert mixed_spec.runtime == "trtexec"
    assert mixed_spec.precision == "int8"
    assert "mixed-l16-19-fp32" in mixed_spec.quant_path


def test_default_matrix_specs_include_v1_2_onnx_stripped_followup() -> None:
    candidates = {spec.candidate for spec in DEFAULT_BENCHMARK_MATRIX_SPECS}

    assert "INT8 SmoothQuant alpha=0.8 ONNX-stripped l16-19" in candidates

    stripped_spec = next(
        spec
        for spec in DEFAULT_BENCHMARK_MATRIX_SPECS
        if spec.candidate == "INT8 SmoothQuant alpha=0.8 ONNX-stripped l16-19"
    )
    assert stripped_spec.filename == (
        "trtexec_formal_fp32_vs_int8_smoothquant_a080_stripped_l16-19_"
        "locked2752_spinwait_speedup.json"
    )
    assert stripped_spec.runtime == "trtexec"
    assert stripped_spec.precision == "int8"
    assert "onnx-stripped-l16-19" in stripped_spec.quant_path


def test_default_matrix_specs_track_resolution_for_each_bf16_run() -> None:
    bf16_trtexec = [
        spec
        for spec in DEFAULT_BENCHMARK_MATRIX_SPECS
        if spec.candidate == "BF16-prefer" and spec.runtime == "trtexec"
    ]
    assert {spec.resolution for spec in bf16_trtexec} == {224, 336, 518}

    bf16_cpp = [
        spec
        for spec in DEFAULT_BENCHMARK_MATRIX_SPECS
        if spec.candidate == "BF16-prefer" and spec.runtime == "cpp"
    ]
    assert {spec.resolution for spec in bf16_cpp} == {224, 336, 518}


def test_build_benchmark_matrix_extracts_speedup_rows(tmp_path: Path) -> None:
    report_path = tmp_path / "bf16_speedup.json"
    report_path.write_text(
        json.dumps(
            {
                "metric_name": "gpu_compute_time",
                "rows": [
                    {
                        "batch_size": 8,
                        "reference_median_ms": 28.0,
                        "candidate_median_ms": 10.0,
                        "latency_speedup": 2.8,
                        "reference_throughput_qps": 34.0,
                        "candidate_throughput_qps": 90.0,
                        "throughput_speedup": 2.65,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    matrix = build_benchmark_matrix(
        tmp_path,
        specs=(
            BenchmarkMatrixSpec(
                label="BF16",
                filename="bf16_speedup.json",
                runtime="trtexec",
                precision="bf16",
                quant_path="bf16-prefer",
                candidate="BF16-prefer",
            ),
        ),
    )

    assert matrix["missing_reports"] == []
    assert len(matrix["rows"]) == 1
    row = matrix["rows"][0]
    assert row["run_id"] == "formal-trtexec-bf16-prefer-vs-fp32-b8-r224"
    assert row["runtime"] == "trtexec"
    assert row["precision"] == "bf16"
    assert row["latency_p50_ms"] == 10.0
    assert row["reference_latency_p50_ms"] == 28.0
    assert row["throughput_imgs_s"] == 90.0


def test_build_benchmark_matrix_tracks_missing_reports(tmp_path: Path) -> None:
    matrix = build_benchmark_matrix(
        tmp_path,
        specs=(
            BenchmarkMatrixSpec(
                label="missing",
                filename="missing.json",
                runtime="cpp",
                precision="int8",
                quant_path="modelopt",
                candidate="INT8",
            ),
        ),
        allow_missing=True,
    )

    assert matrix["missing_reports"] == ["missing.json"]
    assert matrix["rows"] == []


def test_build_benchmark_matrix_raises_for_missing_reports(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing.json"):
        build_benchmark_matrix(
            tmp_path,
            specs=(
                BenchmarkMatrixSpec(
                    label="missing",
                    filename="missing.json",
                    runtime="cpp",
                    precision="int8",
                    quant_path="modelopt",
                    candidate="INT8",
                ),
            ),
        )


def test_write_benchmark_matrix_outputs_csv_and_markdown(tmp_path: Path) -> None:
    matrix = {
        "schema_version": 1,
        "reports_dir": str(tmp_path),
        "missing_reports": [],
        "rows": [
            {
                "run_id": "formal-cpp-bf16-vs-fp32-b1-r224",
                "source_report": "speedup.json",
                "source": "formal",
                "runtime": "cpp",
                "metric_name": "cpp_runtime_end_to_end_latency",
                "model": "dinov3-vitl16-pretrain-lvd1689m",
                "precision": "bf16",
                "quant_path": "bf16-prefer",
                "candidate": "BF16-prefer",
                "reference": "FP32",
                "batch_size": 1,
                "resolution": 224,
                "gpu_arch": "RTX 5080 / Blackwell sm_120",
                "trt_version": "10.13.2.6",
                "latency_p50_ms": 3.2,
                "reference_latency_p50_ms": 7.4,
                "latency_speedup": 2.31,
                "throughput_imgs_s": 302.0,
                "reference_throughput_imgs_s": 133.0,
                "throughput_speedup": 2.27,
            }
        ],
    }

    json_output = tmp_path / "matrix.json"
    csv_output = tmp_path / "matrix.csv"
    markdown_output = tmp_path / "matrix.md"
    write_benchmark_matrix(
        matrix,
        json_output=json_output,
        csv_output=csv_output,
        markdown_output=markdown_output,
    )

    csv_rows = list(csv.DictReader(csv_output.open(encoding="utf-8")))
    markdown = render_benchmark_matrix_markdown(matrix)

    assert json.loads(json_output.read_text(encoding="utf-8"))["rows"][0]["runtime"] == "cpp"
    assert csv_rows[0]["run_id"] == "formal-cpp-bf16-vs-fp32-b1-r224"
    assert csv_rows[0]["latency_p50_ms"] == "3.2"
    assert "| cpp | BF16-prefer vs FP32 | 1 | 3.2 | 2.31x | 302 | 2.27x |" in markdown
    assert markdown_output.read_text(encoding="utf-8") == markdown
