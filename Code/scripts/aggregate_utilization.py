"""Aggregate nvidia-smi raw timeline into V1.0.3 §10.3 benchmark CSV row.

V1.0.3 G7 utilization acceptance gate: SM ≥ 95% (saturation regime),
Tensor Core ∈ [65%, 75%] (BF16 dense ceiling), HBM ≥ 90% (b≥16 memory-bound),
power < 300W (TDP).

Input: ``timeline.csv`` produced by ``utilization_monitor.ps1`` (nvidia-smi
``--format=csv,nounits`` with fields ``timestamp,utilization.gpu,
utilization.memory,memory.used,power.draw,temperature.gpu,clocks.sm``).

Output: a single aggregated row appended to a target benchmark CSV with
columns ``sm_pct``, ``tensor_core_pct``, ``hbm_pct``, ``avg_power_w``
matching the V1.0.3 §10.3 schema. ``tensor_core_pct`` is sourced from the
optional ``ncu_metrics.txt`` (Nsight Compute) when available, since
nvidia-smi does not expose Tensor Core utilization directly. When ncu is
absent we emit ``NaN`` for that field and let downstream gates flag it.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class TimelineSample:
    """A single 100ms cadence row from nvidia-smi --query-gpu output."""

    timestamp: str
    sm_pct: float
    mem_io_pct: float
    memory_used_mib: float
    power_w: float
    temperature_c: float
    sm_clock_mhz: float


@dataclass
class AggregatedMetrics:
    """V1.0.3 §10.3 aggregated row."""

    run_id: str
    sample_count: int
    duration_s: float
    sm_pct_mean: float
    sm_pct_p50: float
    sm_pct_p95: float
    sm_pct_max: float
    hbm_pct_mean: float
    hbm_pct_p50: float
    hbm_pct_p95: float
    hbm_pct_max: float
    avg_power_w: float
    peak_power_w: float
    avg_temperature_c: float
    peak_temperature_c: float
    tensor_core_pct: float = field(default=float("nan"))
    notes: str = ""


def _parse_float(value: str) -> float:
    """Parse a stringified float from nvidia-smi output, tolerant of '[N/A]'."""

    cleaned = value.strip()
    if not cleaned or cleaned in {"[N/A]", "N/A", "-"}:
        return float("nan")
    # Strip trailing units that nvidia-smi sometimes leaves even with --format=nounits.
    cleaned = re.sub(r"[^\d\.\-eE+]", "", cleaned)
    if not cleaned:
        return float("nan")
    return float(cleaned)


def load_timeline(path: Path) -> list[TimelineSample]:
    samples: list[TimelineSample] = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return samples
        normalized = [h.strip().lower() for h in header]
        if not any("utilization.gpu" in h for h in normalized):
            raise ValueError(
                f"timeline.csv header missing utilization.gpu column: {header!r}"
            )
        for row in reader:
            if len(row) < 7:
                continue
            try:
                samples.append(
                    TimelineSample(
                        timestamp=row[0].strip(),
                        sm_pct=_parse_float(row[1]),
                        mem_io_pct=_parse_float(row[2]),
                        memory_used_mib=_parse_float(row[3]),
                        power_w=_parse_float(row[4]),
                        temperature_c=_parse_float(row[5]),
                        sm_clock_mhz=_parse_float(row[6]),
                    )
                )
            except ValueError:
                continue
    return samples


def _drop_warmup(
    samples: list[TimelineSample], warmup_seconds: float, interval_ms: int
) -> list[TimelineSample]:
    if warmup_seconds <= 0 or not samples:
        return samples
    skip = int((warmup_seconds * 1000) / interval_ms)
    if skip >= len(samples):
        return []
    return samples[skip:]


def _percentile(values: list[float], pct: float) -> float:
    cleaned = [v for v in values if not math.isnan(v)]
    if not cleaned:
        return float("nan")
    cleaned.sort()
    if pct <= 0:
        return cleaned[0]
    if pct >= 100:
        return cleaned[-1]
    rank = (pct / 100.0) * (len(cleaned) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return cleaned[lo]
    frac = rank - lo
    return cleaned[lo] * (1 - frac) + cleaned[hi] * frac


def _safe_mean(values: Iterable[float]) -> float:
    cleaned = [v for v in values if not math.isnan(v)]
    if not cleaned:
        return float("nan")
    return statistics.fmean(cleaned)


def _safe_max(values: Iterable[float]) -> float:
    cleaned = [v for v in values if not math.isnan(v)]
    if not cleaned:
        return float("nan")
    return max(cleaned)


_NCU_TC_PATTERN = re.compile(
    r"sm__pipe_tensor_op_hmma_cycles_active\.avg\.pct_of_peak_sustained_elapsed"
)


def parse_ncu_tensor_core_pct(ncu_path: Path) -> float:
    """Extract Tensor Core utilization mean from ncu raw page output.

    ncu emits a CSV table with one row per kernel; we average the
    ``sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed``
    column across kernels. This tracks the fraction of cycles in which Tensor
    Core HMMA pipes were active.
    """

    if not ncu_path.exists():
        return float("nan")

    try:
        text = ncu_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return float("nan")

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return float("nan")

    header_idx = None
    header_cols: list[str] = []
    for idx, line in enumerate(lines):
        if "," in line and _NCU_TC_PATTERN.search(line):
            header_idx = idx
            header_cols = [c.strip().strip('"') for c in line.split(",")]
            break

    if header_idx is None:
        return float("nan")

    target_col = None
    for col_idx, name in enumerate(header_cols):
        if _NCU_TC_PATTERN.search(name):
            target_col = col_idx
            break
    if target_col is None:
        return float("nan")

    values: list[float] = []
    for line in lines[header_idx + 1 :]:
        if "," not in line:
            continue
        parts = [c.strip().strip('"') for c in line.split(",")]
        if len(parts) <= target_col:
            continue
        try:
            values.append(_parse_float(parts[target_col]))
        except ValueError:
            continue

    return _safe_mean(values)


def aggregate(run_dir: Path) -> AggregatedMetrics:
    timeline_path = run_dir / "timeline.csv"
    if not timeline_path.exists():
        raise FileNotFoundError(f"timeline.csv not found in {run_dir}")

    meta_path = run_dir / "meta.json"
    interval_ms = 100
    warmup_skip = 5.0
    run_id = run_dir.name
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            interval_ms = int(meta.get("interval_ms", interval_ms))
            warmup_skip = float(meta.get("warmup_skip_seconds", warmup_skip))
            run_id = str(meta.get("run_id", run_id))
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass

    samples = load_timeline(timeline_path)
    samples = _drop_warmup(samples, warmup_skip, interval_ms)

    if not samples:
        raise ValueError(
            f"No samples remain after warmup skip ({warmup_skip}s @ {interval_ms}ms cadence) in {timeline_path}"
        )

    sm_values = [s.sm_pct for s in samples]
    hbm_values = [s.mem_io_pct for s in samples]
    power_values = [s.power_w for s in samples]
    temp_values = [s.temperature_c for s in samples]

    duration_s = (len(samples) * interval_ms) / 1000.0

    ncu_path = run_dir / "ncu_metrics.txt"
    tc_pct = parse_ncu_tensor_core_pct(ncu_path)

    notes_parts: list[str] = []
    if math.isnan(tc_pct):
        notes_parts.append("tensor_core_pct=NaN (ncu_metrics.txt missing or unparseable)")
    if math.isnan(_safe_mean(sm_values)):
        notes_parts.append("sm_pct=NaN (timeline read failed)")

    return AggregatedMetrics(
        run_id=run_id,
        sample_count=len(samples),
        duration_s=duration_s,
        sm_pct_mean=_safe_mean(sm_values),
        sm_pct_p50=_percentile(sm_values, 50),
        sm_pct_p95=_percentile(sm_values, 95),
        sm_pct_max=_safe_max(sm_values),
        hbm_pct_mean=_safe_mean(hbm_values),
        hbm_pct_p50=_percentile(hbm_values, 50),
        hbm_pct_p95=_percentile(hbm_values, 95),
        hbm_pct_max=_safe_max(hbm_values),
        avg_power_w=_safe_mean(power_values),
        peak_power_w=_safe_max(power_values),
        avg_temperature_c=_safe_mean(temp_values),
        peak_temperature_c=_safe_max(temp_values),
        tensor_core_pct=tc_pct,
        notes="; ".join(notes_parts),
    )


def append_to_benchmark_csv(
    metrics: AggregatedMetrics,
    benchmark_csv: Path,
    extra_row_fields: dict[str, str] | None = None,
) -> None:
    """Append the V1.0.3 §10.3 sm_pct/tensor_core_pct/hbm_pct/avg_power_w columns.

    If ``benchmark_csv`` does not exist, create it with V1.0.3 §10.3 schema.
    Otherwise append a row. ``extra_row_fields`` lets the caller supply
    run_id, framework, precision, resolution, batch_size, n_instances,
    n_clients, preferred_batch_size, max_queue_delay_us, aggregate_qps,
    p50_latency_ms, p99_latency_ms.
    """

    schema = [
        "run_id",
        "source",
        "framework",
        "precision",
        "resolution",
        "batch_size",
        "n_instances",
        "n_clients",
        "preferred_batch_size",
        "max_queue_delay_us",
        "aggregate_qps",
        "p50_latency_ms",
        "p99_latency_ms",
        "sm_pct",
        "tensor_core_pct",
        "hbm_pct",
        "avg_power_w",
    ]

    row = {key: "" for key in schema}
    row["run_id"] = metrics.run_id
    row["sm_pct"] = f"{metrics.sm_pct_mean:.2f}" if not math.isnan(metrics.sm_pct_mean) else ""
    row["tensor_core_pct"] = (
        f"{metrics.tensor_core_pct:.2f}" if not math.isnan(metrics.tensor_core_pct) else ""
    )
    row["hbm_pct"] = f"{metrics.hbm_pct_mean:.2f}" if not math.isnan(metrics.hbm_pct_mean) else ""
    row["avg_power_w"] = f"{metrics.avg_power_w:.2f}" if not math.isnan(metrics.avg_power_w) else ""

    if extra_row_fields:
        for key, val in extra_row_fields.items():
            if key in row:
                row[key] = val

    write_header = not benchmark_csv.exists()
    benchmark_csv.parent.mkdir(parents=True, exist_ok=True)
    with benchmark_csv.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=schema)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Directory produced by utilization_monitor.ps1 (contains timeline.csv + meta.json)",
    )
    parser.add_argument(
        "--benchmark-csv",
        type=Path,
        default=None,
        help="Optional benchmark CSV to append the aggregated row to (V1.0.3 §10.3 schema)",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to write the AggregatedMetrics as JSON",
    )
    parser.add_argument(
        "--row-field",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra benchmark CSV field (repeatable), e.g. --row-field framework=cpp_pool",
    )
    return parser


def _parse_row_fields(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--row-field expects KEY=VALUE, got: {item!r}")
        key, _, val = item.partition("=")
        out[key.strip()] = val.strip()
    return out


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    metrics = aggregate(args.run_dir)

    print(json.dumps(metrics.__dict__, indent=2, ensure_ascii=False))

    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(metrics.__dict__, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    if args.benchmark_csv:
        extras = _parse_row_fields(args.row_field)
        append_to_benchmark_csv(metrics, args.benchmark_csv, extras)
        print(f"appended row to {args.benchmark_csv}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
