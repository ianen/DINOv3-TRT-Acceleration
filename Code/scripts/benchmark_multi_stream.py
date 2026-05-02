#!/usr/bin/env python
"""V1.0.2 ADR-015 first-cut multi-stream throughput benchmark.

Spawns N concurrent worker threads, each running TensorRT inference on its
own engine + context + CUDA stream + buffers. Measures aggregate images/sec
relative to the single-thread baseline to validate whether GPU SM concurrency
yields any throughput gain on the RTX 5080 84-SM Blackwell architecture.

This is a first-cut validation: every worker loads its own engine (4× weight
memory at N=4). A V1.0.3 follow-up will share one ICudaEngine across contexts
to eliminate the duplication cost (per ADR-015 §4.2).

The reference baseline is N=1 single-stream throughput; targets:
- N=2 aggregate ≥ 1.7× single-stream (low contention)
- N=4 aggregate ≥ 3.0× single-stream (V1.0.2 G4 minimum)
- N=4 aggregate ≥ 3.5× single-stream (V1.0.2 G4 stretch)

Usage
=====
::

    .venv/bin/python scripts/benchmark_multi_stream.py \\
        --engine Artifacts/engines/dinov3_vitl16_4out.bf16.prefer.engine \\
        --batch-size 1 --image-size 224 \\
        --num-streams 1,2,4,8 --warmup 30 --iters 200 \\
        --output Artifacts/reports/v102_multistream_bench.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--num-streams",
        type=str,
        default="1,2,4",
        help="Comma-separated stream counts to benchmark, e.g. '1,2,4,8'",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=30,
        help="Warmup iterations per worker before timing begins.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=200,
        help="Timed iterations per worker.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report path.",
    )
    parser.add_argument(
        "--input-name",
        default="pixel_values",
    )
    return parser.parse_args(argv)


@dataclass
class WorkerStats:
    worker_id: int
    iterations: int
    elapsed_seconds: float
    throughput_qps: float
    median_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float


def _build_inference_state(
    engine_path: Path, input_name: str, batch_size: int, image_size: int
) -> tuple[Any, Any, Any, dict[str, Any], dict[str, int]]:
    """Set up an isolated TRT engine + context + buffers for one worker.

    Returns (engine, context, cuda_runtime_ctx, host_buffers, device_buffers).
    Each worker holds its own copy of all of the above so concurrent threads
    do not share mutable TRT state.
    """
    import importlib

    trt = importlib.import_module("tensorrt")
    try:
        cudart = importlib.import_module("cuda.bindings.runtime")
    except ImportError:
        cudart = importlib.import_module("cuda.cudart")

    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    with engine_path.open("rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"failed to deserialize engine: {engine_path}")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("failed to create execution context")

    input_shape = (batch_size, 3, image_size, image_size)
    if not context.set_input_shape(input_name, input_shape):
        raise RuntimeError(f"failed to set input shape: {input_shape}")

    # Allocate per-tensor host + device buffers.
    host_buffers: dict[str, np.ndarray] = {}
    device_buffers: dict[str, int] = {}
    n_tensors = engine.num_io_tensors
    for i in range(n_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        dtype_trt = engine.get_tensor_dtype(name)
        # tensorrt → numpy dtype
        np_dtype = np.dtype(trt.nptype(dtype_trt))
        if mode == trt.TensorIOMode.INPUT:
            host = np.ascontiguousarray(
                np.random.RandomState(0xBEEF + i).standard_normal(input_shape).astype(np_dtype)
            )
        else:
            shape = tuple(int(d) for d in context.get_tensor_shape(name))
            host = np.empty(shape, dtype=np_dtype)
        nbytes = int(host.nbytes)
        err, dev_ptr = cudart.cudaMalloc(nbytes)
        if int(err) != 0:
            raise RuntimeError(f"cudaMalloc failed for {name!r}: {err}")
        if not context.set_tensor_address(name, int(dev_ptr)):
            raise RuntimeError(f"failed to set tensor address for {name!r}")
        host_buffers[name] = host
        device_buffers[name] = int(dev_ptr)

    err, stream = cudart.cudaStreamCreate()
    if int(err) != 0:
        raise RuntimeError(f"cudaStreamCreate failed: {err}")

    return (engine, context, (cudart, stream), host_buffers, device_buffers)


def _free_inference_state(state: tuple[Any, Any, Any, dict[str, Any], dict[str, int]]) -> None:
    _, _, (cudart, stream), _, device_buffers = state
    for name, ptr in device_buffers.items():
        cudart.cudaFree(int(ptr))
    cudart.cudaStreamDestroy(stream)


def run_worker(
    worker_id: int,
    engine_path: Path,
    input_name: str,
    batch_size: int,
    image_size: int,
    warmup: int,
    iters: int,
    barrier: threading.Barrier,
) -> WorkerStats:
    """One-thread inference loop. All workers wait at the barrier before
    timing begins so the aggregate measurement reflects truly concurrent work."""
    state = _build_inference_state(engine_path, input_name, batch_size, image_size)
    try:
        engine, context, (cudart, stream), host_buffers, device_buffers = state

        # H2D once for input (random data; reused across all iterations).
        host_input = host_buffers[input_name]
        cudart.cudaMemcpyAsync(
            int(device_buffers[input_name]),
            int(host_input.ctypes.data),
            int(host_input.nbytes),
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            stream,
        )
        cudart.cudaStreamSynchronize(stream)

        # Warmup
        for _ in range(warmup):
            context.execute_async_v3(stream_handle=int(stream))
        cudart.cudaStreamSynchronize(stream)

        # Synchronize all workers at the start of the timed region.
        barrier.wait()

        latencies: list[float] = []
        t_start_global = time.perf_counter()
        for _ in range(iters):
            t0 = time.perf_counter()
            context.execute_async_v3(stream_handle=int(stream))
            cudart.cudaStreamSynchronize(stream)
            latencies.append((time.perf_counter() - t0) * 1000.0)
        elapsed = time.perf_counter() - t_start_global

        latencies_arr = np.array(latencies)
        return WorkerStats(
            worker_id=worker_id,
            iterations=iters,
            elapsed_seconds=elapsed,
            throughput_qps=iters * batch_size / elapsed,
            median_latency_ms=float(np.median(latencies_arr)),
            min_latency_ms=float(np.min(latencies_arr)),
            max_latency_ms=float(np.max(latencies_arr)),
        )
    finally:
        _free_inference_state(state)


def benchmark_n_streams(
    engine_path: Path,
    input_name: str,
    batch_size: int,
    image_size: int,
    n_streams: int,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    """Run N concurrent workers, return aggregate + per-worker stats."""
    barrier = threading.Barrier(n_streams)
    with ThreadPoolExecutor(max_workers=n_streams) as ex:
        futures = [
            ex.submit(
                run_worker,
                worker_id=i,
                engine_path=engine_path,
                input_name=input_name,
                batch_size=batch_size,
                image_size=image_size,
                warmup=warmup,
                iters=iters,
                barrier=barrier,
            )
            for i in range(n_streams)
        ]
        worker_stats = [f.result() for f in futures]

    total_imgs = sum(w.iterations * batch_size for w in worker_stats)
    # Aggregate elapsed = max worker elapsed (true concurrent wall time).
    max_elapsed = max(w.elapsed_seconds for w in worker_stats)
    aggregate_qps = total_imgs / max_elapsed
    return {
        "n_streams": n_streams,
        "aggregate_qps": aggregate_qps,
        "max_worker_elapsed_seconds": max_elapsed,
        "total_iterations": total_imgs,
        "median_worker_qps": float(np.median([w.throughput_qps for w in worker_stats])),
        "median_latency_ms_p50_across_workers": float(
            np.median([w.median_latency_ms for w in worker_stats])
        ),
        "per_worker": [
            {
                "worker_id": w.worker_id,
                "throughput_qps": w.throughput_qps,
                "median_latency_ms": w.median_latency_ms,
                "min_latency_ms": w.min_latency_ms,
                "max_latency_ms": w.max_latency_ms,
            }
            for w in worker_stats
        ],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    logger = logging.getLogger("multi-stream")

    n_stream_values = [int(s.strip()) for s in args.num_streams.split(",") if s.strip()]
    results: list[dict[str, Any]] = []

    for n in n_stream_values:
        logger.info(f"Running N={n} streams...")
        result = benchmark_n_streams(
            engine_path=args.engine,
            input_name=args.input_name,
            batch_size=args.batch_size,
            image_size=args.image_size,
            n_streams=n,
            warmup=args.warmup,
            iters=args.iters,
        )
        results.append(result)
        logger.info(
            f"N={n}: aggregate {result['aggregate_qps']:.2f} qps "
            f"(median worker {result['median_worker_qps']:.2f} qps, "
            f"p50 latency {result['median_latency_ms_p50_across_workers']:.2f} ms)"
        )

    # Compute speedup vs N=1 baseline.
    if results and results[0]["n_streams"] == 1:
        baseline_qps = results[0]["aggregate_qps"]
        for r in results:
            r["speedup_vs_n1"] = r["aggregate_qps"] / baseline_qps

    payload = {
        "engine": str(args.engine),
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "warmup": args.warmup,
        "iters": args.iters,
        "results": results,
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info(f"Report saved to {args.output}")

    print()
    print("=== V1.0.2 ADR-015 Multi-Stream Throughput Summary ===")
    print(f"{'N':>4} {'aggregate_qps':>15} {'speedup_vs_n1':>15} {'p50_lat_ms':>12}")
    for r in results:
        sp = r.get("speedup_vs_n1", 0)
        print(
            f"{r['n_streams']:>4} {r['aggregate_qps']:>15.2f} "
            f"{sp:>15.3f}  {r['median_latency_ms_p50_across_workers']:>12.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
