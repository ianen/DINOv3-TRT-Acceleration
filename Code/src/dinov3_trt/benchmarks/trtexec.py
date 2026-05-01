"""`trtexec --loadEngine` benchmark command and output parsing helpers."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


METRIC_LABELS = {
    "Latency": "latency",
    "Enqueue Time": "enqueue_time",
    "H2D Latency": "h2d_latency",
    "GPU Compute Time": "gpu_compute_time",
    "D2H Latency": "d2h_latency",
}

_THROUGHPUT_RE = re.compile(r"Throughput:\s+(?P<value>[0-9.]+)\s+qps")
_METRIC_RE = re.compile(
    r"(?P<label>Latency|Enqueue Time|H2D Latency|GPU Compute Time|D2H Latency): "
    r"min = (?P<min>[0-9.]+) ms, "
    r"max = (?P<max>[0-9.]+) ms, "
    r"mean = (?P<mean>[0-9.]+) ms, "
    r"median = (?P<median>[0-9.]+) ms, "
    r"percentile\(90%\) = (?P<p90>[0-9.]+) ms, "
    r"percentile\(95%\) = (?P<p95>[0-9.]+) ms, "
    r"percentile\(99%\) = (?P<p99>[0-9.]+) ms"
)
_BINDING_RE = re.compile(
    r"(?P<kind>Input|Output) binding for (?P<name>\S+) "
    r"with dimensions (?P<shape>[0-9x]+) is created\."
)


@dataclass(frozen=True)
class TrtExecLoadConfig:
    engine_path: Path
    batch_size: int
    input_name: str = "pixel_values"
    channels: int = 3
    height: int = 224
    width: int = 224
    duration_seconds: int = 10
    warmup_ms: int = 200
    trtexec: str = "trtexec"
    no_tf32: bool = True
    use_spin_wait: bool = False

    @property
    def input_shape(self) -> str:
        return (
            f"{self.input_name}:"
            f"{self.batch_size}x{self.channels}x{self.height}x{self.width}"
        )


def build_trtexec_load_command(config: TrtExecLoadConfig) -> list[str]:
    if config.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if config.duration_seconds < 1:
        raise ValueError("duration_seconds must be >= 1")
    if config.warmup_ms < 0:
        raise ValueError("warmup_ms must be >= 0")

    command = [
        config.trtexec,
        f"--loadEngine={config.engine_path}",
        f"--shapes={config.input_shape}",
        f"--duration={config.duration_seconds}",
        f"--warmUp={config.warmup_ms}",
    ]
    if config.no_tf32:
        command.append("--noTF32")
    if config.use_spin_wait:
        command.append("--useSpinWait")
    return command


def parse_trtexec_summary(output: str) -> dict[str, Any]:
    throughput_match = _THROUGHPUT_RE.search(output)
    metrics: dict[str, dict[str, float]] = {}
    inputs: dict[str, str] = {}
    outputs: dict[str, str] = {}

    for match in _METRIC_RE.finditer(output):
        label = METRIC_LABELS[match.group("label")]
        metrics[label] = {
            "min_ms": float(match.group("min")),
            "max_ms": float(match.group("max")),
            "mean_ms": float(match.group("mean")),
            "median_ms": float(match.group("median")),
            "p90_ms": float(match.group("p90")),
            "p95_ms": float(match.group("p95")),
            "p99_ms": float(match.group("p99")),
        }

    for match in _BINDING_RE.finditer(output):
        target = inputs if match.group("kind") == "Input" else outputs
        target[match.group("name")] = match.group("shape")

    return {
        "throughput_qps": None if throughput_match is None else float(throughput_match.group("value")),
        "metrics": metrics,
        "bindings": {
            "inputs": inputs,
            "outputs": outputs,
        },
    }


def run_trtexec_load(config: TrtExecLoadConfig) -> dict[str, Any]:
    command = build_trtexec_load_command(config)
    result = subprocess.run(
        command,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    combined_output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    return {
        "batch_size": config.batch_size,
        "command": command,
        "returncode": result.returncode,
        "summary": parse_trtexec_summary(combined_output),
        "stdout_tail": _tail_lines(combined_output),
    }


def run_trtexec_benchmarks(
    engine_path: Path,
    batch_sizes: Sequence[int],
    *,
    duration_seconds: int = 10,
    warmup_ms: int = 200,
    trtexec: str = "trtexec",
    input_name: str = "pixel_values",
    image_size: int = 224,
    use_spin_wait: bool = False,
) -> dict[str, Any]:
    results = [
        run_trtexec_load(
            TrtExecLoadConfig(
                engine_path=engine_path,
                batch_size=batch_size,
                input_name=input_name,
                height=image_size,
                width=image_size,
                duration_seconds=duration_seconds,
                warmup_ms=warmup_ms,
                trtexec=trtexec,
                use_spin_wait=use_spin_wait,
            )
        )
        for batch_size in batch_sizes
    ]
    return {
        "engine": str(engine_path),
        "batches": list(batch_sizes),
        "duration_seconds": duration_seconds,
        "warmup_ms": warmup_ms,
        "results": results,
    }


def _tail_lines(output: str, *, max_lines: int = 40) -> list[str]:
    lines = output.splitlines()
    return lines[-max_lines:]
