from pathlib import Path

from dinov3_trt.reports.formal_summary import (
    DEFAULT_IMAGE_EVAL_REPORTS,
    DEFAULT_PARITY_REPORTS,
    DEFAULT_SPEEDUP_REPORTS,
    ReportSpec,
    build_formal_summary,
    render_formal_summary_markdown,
    summarize_image_eval_report,
    summarize_parity_report,
    summarize_speedup_report,
)


def test_default_report_specs_include_layer19_int8_followup() -> None:
    image_eval_filenames = {spec.filename for spec in DEFAULT_IMAGE_EVAL_REPORTS}
    speedup_filenames = {spec.filename for spec in DEFAULT_SPEEDUP_REPORTS}

    assert (
        "eval_imagenette1000_fp32_vs_int8_modelopt_imagenette64_"
        "matmul_layer19.json"
        in image_eval_filenames
    )
    assert (
        "eval_imagenette1000_fp32_vs_int8_modelopt_imagenette64_"
        "matmul_fine_layer19_attention.json"
        in image_eval_filenames
    )
    assert (
        "trtexec_formal_fp32_vs_int8_modelopt_imagenette64_"
        "matmul_layer19_locked2752_spinwait_speedup.json"
        in speedup_filenames
    )
    assert (
        "trtexec_formal_bf16_prefer_vs_int8_modelopt_imagenette64_"
        "matmul_layer19_locked2752_spinwait_speedup.json"
        in speedup_filenames
    )
    assert (
        "cpp_runtime_fp32_vs_int8_modelopt_imagenette64_matmul_layer19_"
        "speedup.json"
        in speedup_filenames
    )
    assert (
        "trtexec_formal_fp32_vs_int8_modelopt_imagenette64_"
        "matmul_fine_layer19_attention_locked2752_spinwait_speedup.json"
        in speedup_filenames
    )


def test_summarize_image_eval_report_extracts_per_output_metrics() -> None:
    summary = summarize_image_eval_report(
        "bf16",
        {
            "reference_engine": "fp32.engine",
            "candidate_engine": "bf16.engine",
            "image_root": "manifest.json",
            "image_count": 1000,
            "batch_size": 32,
            "outputs": [
                {
                    "name": "feat_layer_4",
                    "cosine_similarity_mean": 0.9999,
                    "cosine_similarity_min": 0.9998,
                    "max_abs_error": 0.5,
                    "root_mean_square_error": 0.01,
                }
            ],
        },
    )

    assert summary["label"] == "bf16"
    assert summary["image_count"] == 1000
    assert summary["outputs"][0]["name"] == "feat_layer_4"
    assert summary["outputs"][0]["cosine_similarity_mean"] == 0.9999


def test_summarize_speedup_report_extracts_rows() -> None:
    summary = summarize_speedup_report(
        "speed",
        {
            "metric_name": "gpu_compute_time",
            "reference": {"label": "fp32", "engine": "fp32.engine"},
            "candidate": {"label": "bf16", "engine": "bf16.engine"},
            "rows": [
                {
                    "batch_size": 1,
                    "reference_median_ms": 8.0,
                    "candidate_median_ms": 4.0,
                    "latency_speedup": 2.0,
                    "throughput_speedup": 1.8,
                }
            ],
        },
    )

    assert summary["label"] == "speed"
    assert summary["rows"][0]["batch_size"] == 1
    assert summary["rows"][0]["latency_speedup"] == 2.0


def test_summarize_parity_report_extracts_per_output_metrics() -> None:
    summary = summarize_parity_report(
        "Python vs C++ FP32",
        {
            "engine_path": "fp32.engine",
            "batch_size": 1,
            "reference_runtime": "python",
            "candidate_runtime": "cpp",
            "outputs": [
                {
                    "name": "feat_layer_4",
                    "max_abs_error": 0.0,
                    "root_mean_square_error": 0.0,
                    "cosine_similarity": 1.0,
                }
            ],
        },
    )

    assert summary["label"] == "Python vs C++ FP32"
    assert summary["batch_size"] == 1
    assert summary["outputs"][0]["cosine_similarity"] == 1.0


def test_default_report_specs_include_layer19_cpp_parity_followup() -> None:
    parity_filenames = {spec.filename for spec in DEFAULT_PARITY_REPORTS}

    assert (
        "cpp_python_parity_int8_modelopt_imagenette64_matmul_layer19_b1.json"
        in parity_filenames
    )


def test_build_formal_summary_tracks_missing_reports(tmp_path: Path) -> None:
    summary = build_formal_summary(
        tmp_path,
        image_eval_reports=(ReportSpec("missing eval", "missing_eval.json"),),
        speedup_reports=(ReportSpec("missing speed", "missing_speed.json"),),
        parity_reports=(ReportSpec("missing parity", "missing_parity.json"),),
        allow_missing=True,
    )

    assert summary["missing_reports"] == [
        "missing_eval.json",
        "missing_speed.json",
        "missing_parity.json",
    ]
    assert summary["image_evals"] == []
    assert summary["speedups"] == []
    assert summary["parity"] == []


def test_render_formal_summary_markdown_contains_tables() -> None:
    markdown = render_formal_summary_markdown(
        {
            "missing_reports": [],
            "decision": {
                "current_candidate": "BF16 prefer",
                "rationale": "stable",
            },
            "image_evals": [
                {
                    "label": "BF16",
                    "image_count": 1000,
                    "batch_size": 32,
                    "outputs": [
                        {
                            "name": "feat_layer_20",
                            "cosine_similarity_mean": 0.9991,
                            "cosine_similarity_min": 0.9987,
                            "max_abs_error": 116.0,
                            "root_mean_square_error": 0.42,
                        }
                    ],
                }
            ],
            "speedups": [
                {
                    "label": "C++ BF16",
                    "metric_name": "cpp_runtime_end_to_end_latency",
                    "rows": [
                        {
                            "batch_size": 32,
                            "reference_median_ms": 126.8,
                            "candidate_median_ms": 44.794,
                            "latency_speedup": 2.8307,
                            "throughput_speedup": 2.831,
                        }
                    ],
                }
            ],
            "parity": [
                {
                    "label": "Python vs C++ BF16",
                    "batch_size": 1,
                    "outputs": [
                        {
                            "name": "feat_layer_4",
                            "max_abs_error": 0.0,
                            "root_mean_square_error": 0.0,
                            "cosine_similarity": 1.0,
                        }
                    ],
                }
            ],
        }
    )

    assert "# DINOv3 TensorRT Formal Result Summary" in markdown
    assert "| `feat_layer_20` | 0.9991 | 0.9987 | 116 | 0.42 |" in markdown
    assert "| 32 | 126.8 | 44.794 | 2.83x | 2.83x |" in markdown
    assert "## Python/C++ Runtime Parity" in markdown
    assert "| `feat_layer_4` | 0 | 0 | 1 |" in markdown
