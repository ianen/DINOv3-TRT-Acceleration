"""Summarize paired TensorRT benchmark reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class BenchmarkSpeedupRow:
    batch_size: int
    reference_median_ms: float
    candidate_median_ms: float
    latency_speedup: float
    reference_throughput_qps: float | None
    candidate_throughput_qps: float | None
    throughput_speedup: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_trtexec_pair(
    reference_report: Mapping[str, Any],
    candidate_report: Mapping[str, Any],
    *,
    reference_label: str = "reference",
    candidate_label: str = "candidate",
    metric_name: str = "gpu_compute_time",
) -> dict[str, Any]:
    """Return speedup rows for batches present in both benchmark reports."""

    reference_results = _index_results_by_batch(reference_report)
    candidate_results = _index_results_by_batch(candidate_report)
    common_batches = sorted(set(reference_results).intersection(candidate_results))
    if not common_batches:
        raise ValueError("reports do not share any batch sizes")

    rows = [
        _summarize_batch(
            batch_size=batch_size,
            reference_result=reference_results[batch_size],
            candidate_result=candidate_results[batch_size],
            metric_name=metric_name,
        ).to_dict()
        for batch_size in common_batches
    ]

    return {
        "reference": {
            "label": reference_label,
            "engine": str(reference_report.get("engine", "")),
        },
        "candidate": {
            "label": candidate_label,
            "engine": str(candidate_report.get("engine", "")),
        },
        "metric_name": metric_name,
        "shared_batches": common_batches,
        "reference_only_batches": sorted(set(reference_results) - set(candidate_results)),
        "candidate_only_batches": sorted(set(candidate_results) - set(reference_results)),
        "rows": rows,
    }


def summarize_cpp_runtime_pair(
    reference_report: Mapping[str, Any],
    candidate_report: Mapping[str, Any],
    *,
    reference_label: str = "reference",
    candidate_label: str = "candidate",
) -> dict[str, Any]:
    """Return speedup rows for C++ runtime benchmark reports."""

    reference_results = _index_results_by_batch(reference_report)
    candidate_results = _index_results_by_batch(candidate_report)
    common_batches = sorted(set(reference_results).intersection(candidate_results))
    if not common_batches:
        raise ValueError("reports do not share any batch sizes")

    rows = [
        _summarize_cpp_runtime_batch(
            batch_size=batch_size,
            reference_result=reference_results[batch_size],
            candidate_result=candidate_results[batch_size],
        ).to_dict()
        for batch_size in common_batches
    ]

    return {
        "reference": {
            "label": reference_label,
            "engine": str(reference_report.get("engine_path", "")),
        },
        "candidate": {
            "label": candidate_label,
            "engine": str(candidate_report.get("engine_path", "")),
        },
        "metric_name": "cpp_runtime_end_to_end_latency",
        "shared_batches": common_batches,
        "reference_only_batches": sorted(set(reference_results) - set(candidate_results)),
        "candidate_only_batches": sorted(set(candidate_results) - set(reference_results)),
        "rows": rows,
    }


def render_speedup_markdown(summary: Mapping[str, Any]) -> str:
    """Render a compact Markdown table for a speedup summary."""

    reference = _required_mapping(summary, "reference")
    candidate = _required_mapping(summary, "candidate")
    reference_label = str(reference.get("label", "reference"))
    candidate_label = str(candidate.get("label", "candidate"))
    metric_name = str(summary.get("metric_name", "gpu_compute_time"))
    rows = _required_list(summary, "rows")

    lines = [
        f"# TensorRT Benchmark Speedup ({metric_name})",
        "",
        f"Reference: `{reference_label}`",
        f"Candidate: `{candidate_label}`",
        "",
        "| batch | reference median ms | candidate median ms | latency speedup | reference qps | candidate qps | throughput speedup |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("summary rows must be mappings")
        lines.append(
            "| "
            f"{_required_int(row, 'batch_size')} | "
            f"{_format_float(_required_number(row, 'reference_median_ms'))} | "
            f"{_format_float(_required_number(row, 'candidate_median_ms'))} | "
            f"{_format_ratio(_required_number(row, 'latency_speedup'))} | "
            f"{_format_optional_float(_optional_number(row, 'reference_throughput_qps'))} | "
            f"{_format_optional_float(_optional_number(row, 'candidate_throughput_qps'))} | "
            f"{_format_optional_ratio(_optional_number(row, 'throughput_speedup'))} |"
        )
    return "\n".join(lines)


def _summarize_batch(
    *,
    batch_size: int,
    reference_result: Mapping[str, Any],
    candidate_result: Mapping[str, Any],
    metric_name: str,
) -> BenchmarkSpeedupRow:
    reference_median = _metric_median_ms(reference_result, metric_name)
    candidate_median = _metric_median_ms(candidate_result, metric_name)
    if candidate_median <= 0:
        raise ValueError(f"candidate median must be > 0 for batch {batch_size}")

    reference_throughput = _throughput_qps(reference_result)
    candidate_throughput = _throughput_qps(candidate_result)
    throughput_speedup = (
        None
        if reference_throughput is None or candidate_throughput is None or reference_throughput <= 0
        else candidate_throughput / reference_throughput
    )

    return BenchmarkSpeedupRow(
        batch_size=batch_size,
        reference_median_ms=reference_median,
        candidate_median_ms=candidate_median,
        latency_speedup=reference_median / candidate_median,
        reference_throughput_qps=reference_throughput,
        candidate_throughput_qps=candidate_throughput,
        throughput_speedup=throughput_speedup,
    )


def _summarize_cpp_runtime_batch(
    *,
    batch_size: int,
    reference_result: Mapping[str, Any],
    candidate_result: Mapping[str, Any],
) -> BenchmarkSpeedupRow:
    reference_median = _cpp_runtime_median_ms(reference_result)
    candidate_median = _cpp_runtime_median_ms(candidate_result)
    if candidate_median <= 0:
        raise ValueError(f"candidate median must be > 0 for batch {batch_size}")

    reference_throughput = _optional_number(reference_result, "throughput_qps")
    candidate_throughput = _optional_number(candidate_result, "throughput_qps")
    throughput_speedup = (
        None
        if reference_throughput is None or candidate_throughput is None or reference_throughput <= 0
        else candidate_throughput / reference_throughput
    )

    return BenchmarkSpeedupRow(
        batch_size=batch_size,
        reference_median_ms=reference_median,
        candidate_median_ms=candidate_median,
        latency_speedup=reference_median / candidate_median,
        reference_throughput_qps=reference_throughput,
        candidate_throughput_qps=candidate_throughput,
        throughput_speedup=throughput_speedup,
    )


def _index_results_by_batch(report: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    raw_results = _required_list(report, "results")
    results: dict[int, Mapping[str, Any]] = {}
    for raw_result in raw_results:
        if not isinstance(raw_result, Mapping):
            raise ValueError("each benchmark result must be a mapping")
        batch_size = _required_int(raw_result, "batch_size")
        results[batch_size] = raw_result
    return results


def _metric_median_ms(result: Mapping[str, Any], metric_name: str) -> float:
    summary = _required_mapping(result, "summary")
    metrics = _required_mapping(summary, "metrics")
    metric = _required_mapping(metrics, metric_name)
    return _required_number(metric, "median_ms")


def _cpp_runtime_median_ms(result: Mapping[str, Any]) -> float:
    latency_ms = _required_mapping(result, "latency_ms")
    return _required_number(latency_ms, "median")


def _throughput_qps(result: Mapping[str, Any]) -> float | None:
    summary = _required_mapping(result, "summary")
    return _optional_number(summary, "throughput_qps")


def _required_mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"missing mapping field: {key}")
    return value


def _required_list(mapping: Mapping[str, Any], key: str) -> list[Any]:
    value = mapping.get(key)
    if not isinstance(value, list):
        raise ValueError(f"missing list field: {key}")
    return value


def _required_int(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"missing integer field: {key}")
    return value


def _required_number(mapping: Mapping[str, Any], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"missing numeric field: {key}")
    return float(value)


def _optional_number(mapping: Mapping[str, Any], key: str) -> float | None:
    value = mapping.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"field must be numeric or null: {key}")
    return float(value)


def _format_float(value: float) -> str:
    return f"{value:.5g}"


def _format_optional_float(value: float | None) -> str:
    return "-" if value is None else _format_float(value)


def _format_ratio(value: float) -> str:
    return f"{value:.2f}x"


def _format_optional_ratio(value: float | None) -> str:
    return "-" if value is None else _format_ratio(value)
