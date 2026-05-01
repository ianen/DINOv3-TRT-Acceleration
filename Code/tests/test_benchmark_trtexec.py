from pathlib import Path

from dinov3_trt.benchmarks.trtexec import (
    TrtExecLoadConfig,
    build_trtexec_load_command,
    parse_trtexec_summary,
)


def _portable_parts(command: list[str]) -> list[str]:
    return [part.replace("\\", "/") for part in command]


def test_build_trtexec_load_command_uses_project_shape() -> None:
    command = build_trtexec_load_command(
        TrtExecLoadConfig(
            engine_path=Path("Artifacts/engines/dinov3.fp16.engine"),
            batch_size=8,
            duration_seconds=3,
            warmup_ms=200,
        )
    )

    assert _portable_parts(command) == [
        "trtexec",
        "--loadEngine=Artifacts/engines/dinov3.fp16.engine",
        "--shapes=pixel_values:8x3x224x224",
        "--duration=3",
        "--warmUp=200",
        "--noTF32",
    ]


def test_parse_trtexec_summary_extracts_metrics_and_bindings() -> None:
    latency = (
        "[I] Latency: min = 2.33557 ms, max = 6.16833 ms, "
        "mean = 2.82042 ms, median = 2.35593 ms, "
        "percentile(90%) = 4.43231 ms, percentile(95%) = 5.40356 ms, "
        "percentile(99%) = 5.71826 ms"
    )
    gpu_compute = (
        "[I] GPU Compute Time: min = 2.22644 ms, max = 6.05981 ms, "
        "mean = 2.71121 ms, median = 2.24707 ms, "
        "percentile(90%) = 4.32117 ms, percentile(95%) = 5.29272 ms, "
        "percentile(99%) = 5.60913 ms"
    )
    output = "\n".join(
        (
            "[I] Input binding for pixel_values with dimensions 1x3x224x224 is created.",
            "[I] Output binding for feat_layer_4 with dimensions 1x197x1024 is created.",
            "[I] Output binding for feat_layer_20 with dimensions 1x197x1024 is created.",
            "[I] Throughput: 329.305 qps",
            latency,
            gpu_compute,
        )
    )

    summary = parse_trtexec_summary(output)

    assert summary["throughput_qps"] == 329.305
    assert summary["bindings"]["inputs"] == {"pixel_values": "1x3x224x224"}
    assert summary["bindings"]["outputs"] == {
        "feat_layer_4": "1x197x1024",
        "feat_layer_20": "1x197x1024",
    }
    assert summary["metrics"]["latency"]["median_ms"] == 2.35593
    assert summary["metrics"]["gpu_compute_time"]["p95_ms"] == 5.29272
