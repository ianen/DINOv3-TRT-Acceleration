import pytest

from dinov3_trt.benchmarks.summary import (
    render_speedup_markdown,
    summarize_cpp_runtime_pair,
    summarize_trtexec_pair,
)


def _report(engine: str, rows: list[tuple[int, float, float | None]]) -> dict[str, object]:
    return {
        "engine": engine,
        "results": [
            {
                "batch_size": batch_size,
                "summary": {
                    "throughput_qps": throughput_qps,
                    "metrics": {
                        "gpu_compute_time": {
                            "median_ms": median_ms,
                        }
                    },
                },
            }
            for batch_size, median_ms, throughput_qps in rows
        ],
    }


def test_summarize_trtexec_pair_reports_latency_and_throughput_speedups() -> None:
    summary = summarize_trtexec_pair(
        _report("fp32.engine", [(1, 8.0, 100.0), (8, 32.0, 25.0)]),
        _report("fp16.engine", [(1, 2.0, 300.0), (8, 8.0, 100.0)]),
        reference_label="FP32",
        candidate_label="FP16",
    )

    assert summary["shared_batches"] == [1, 8]
    assert summary["reference"] == {"label": "FP32", "engine": "fp32.engine"}
    assert summary["candidate"] == {"label": "FP16", "engine": "fp16.engine"}
    assert summary["rows"][0] == {
        "batch_size": 1,
        "reference_median_ms": 8.0,
        "candidate_median_ms": 2.0,
        "latency_speedup": 4.0,
        "reference_throughput_qps": 100.0,
        "candidate_throughput_qps": 300.0,
        "throughput_speedup": 3.0,
    }
    assert summary["rows"][1]["latency_speedup"] == 4.0
    assert summary["rows"][1]["throughput_speedup"] == 4.0


def test_summarize_trtexec_pair_tracks_unmatched_batches() -> None:
    summary = summarize_trtexec_pair(
        _report("fp32.engine", [(1, 8.0, None), (32, 144.0, 7.0)]),
        _report("fp16.engine", [(1, 2.0, None), (8, 7.0, 114.0)]),
    )

    assert summary["shared_batches"] == [1]
    assert summary["reference_only_batches"] == [32]
    assert summary["candidate_only_batches"] == [8]
    assert summary["rows"][0]["throughput_speedup"] is None


def test_summarize_cpp_runtime_pair_uses_latency_median_and_throughput() -> None:
    reference = {
        "engine_path": "fp32.engine",
        "results": [
            {"batch_size": 1, "latency_ms": {"median": 9.0}, "throughput_qps": 100.0},
            {"batch_size": 8, "latency_ms": {"median": 36.0}, "throughput_qps": 25.0},
        ],
    }
    candidate = {
        "engine_path": "fp16.engine",
        "results": [
            {"batch_size": 1, "latency_ms": {"median": 3.0}, "throughput_qps": 300.0},
            {"batch_size": 8, "latency_ms": {"median": 9.0}, "throughput_qps": 100.0},
        ],
    }

    summary = summarize_cpp_runtime_pair(
        reference,
        candidate,
        reference_label="FP32 C++",
        candidate_label="FP16 C++",
    )

    assert summary["metric_name"] == "cpp_runtime_end_to_end_latency"
    assert summary["reference"] == {"label": "FP32 C++", "engine": "fp32.engine"}
    assert summary["candidate"] == {"label": "FP16 C++", "engine": "fp16.engine"}
    assert summary["rows"][0] == {
        "batch_size": 1,
        "reference_median_ms": 9.0,
        "candidate_median_ms": 3.0,
        "latency_speedup": 3.0,
        "reference_throughput_qps": 100.0,
        "candidate_throughput_qps": 300.0,
        "throughput_speedup": 3.0,
    }
    assert summary["rows"][1]["latency_speedup"] == 4.0


def test_render_speedup_markdown_formats_table() -> None:
    summary = summarize_trtexec_pair(
        _report("fp32.engine", [(1, 7.99707, 112.062)]),
        _report("fp16.engine", [(1, 2.36206, 318.024)]),
        reference_label="FP32",
        candidate_label="FP16",
    )

    markdown = render_speedup_markdown(summary)

    assert "Reference: `FP32`" in markdown
    assert "Candidate: `FP16`" in markdown
    assert "| 1 | 7.9971 | 2.3621 | 3.39x | 112.06 | 318.02 | 2.84x |" in markdown


def test_summarize_trtexec_pair_rejects_missing_metric() -> None:
    with pytest.raises(ValueError, match="missing mapping field: gpu_compute_time"):
        summarize_trtexec_pair(
            {"results": [{"batch_size": 1, "summary": {"metrics": {}}}]},
            _report("fp16.engine", [(1, 2.0, 300.0)]),
        )
