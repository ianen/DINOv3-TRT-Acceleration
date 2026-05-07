#!/usr/bin/env python
"""V1.0.4 ADR-026 — Python production-environment inference benchmark.

模拟生产环境推理流水线，对 r=512 印花布数据集（``Artifacts/datasets/good_r512/``）
做 N 次推理并**分阶段独立计时**。

6 段独立计时:

1. ``disk_read``    — 从磁盘 read jpg bytes
2. ``jpg_decode``   — JPEG decode 到 RGB ndarray (PIL)
3. ``preprocess``   — float32 normalize + ImageNet mean/std + HWC→NCHW
4. ``h2d``          — host → device memcpy (cudaMemcpyAsync H2D)
5. ``enqueueV3``    — TRT context.execute_async_v3 + stream sync
6. ``d2h``          — device → host memcpy (4 outputs)

resize 不在 6 段内 — 数据集已离线 resize 为 r=512（ADR-024）。

Usage
=====
::

    python Code/scripts/production_benchmark.py \\
        --engine Artifacts/engines/dinov3_vitl16_4out.r512.bf16.prefer.engine \\
        --dataset Artifacts/datasets/good_r512 \\
        --batch-size 8 \\
        --image-size 512 \\
        --warmup 10 \\
        --iters 100 \\
        --output Artifacts/reports/v104_runs/r512_bf16_b8_py.json
"""

from __future__ import annotations

import argparse
import importlib
import io
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ImageNet normalization stats (DINOv3 follows DINOv2 convention)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class StageTimings:
    """Per-iteration timings for the 6 production stages, ms."""

    disk_read: list[float] = field(default_factory=list)
    jpg_decode: list[float] = field(default_factory=list)
    preprocess: list[float] = field(default_factory=list)
    h2d: list[float] = field(default_factory=list)
    enqueueV3: list[float] = field(default_factory=list)
    d2h: list[float] = field(default_factory=list)
    total: list[float] = field(default_factory=list)

    def append(
        self,
        disk_read: float,
        jpg_decode: float,
        preprocess: float,
        h2d: float,
        enqueueV3: float,
        d2h: float,
    ) -> None:
        self.disk_read.append(disk_read)
        self.jpg_decode.append(jpg_decode)
        self.preprocess.append(preprocess)
        self.h2d.append(h2d)
        self.enqueueV3.append(enqueueV3)
        self.d2h.append(d2h)
        self.total.append(disk_read + jpg_decode + preprocess + h2d + enqueueV3 + d2h)

    def summary(self) -> dict[str, dict[str, float]]:
        """Return p50 / p95 / mean / max per stage."""

        def stat(values: list[float]) -> dict[str, float]:
            if not values:
                return {"p50": 0.0, "p95": 0.0, "mean": 0.0, "max": 0.0, "count": 0}
            sorted_values = sorted(values)
            n = len(sorted_values)
            p50 = sorted_values[n // 2]
            p95 = sorted_values[int(n * 0.95)] if n >= 20 else sorted_values[-1]
            return {
                "p50": p50,
                "p95": p95,
                "mean": statistics.fmean(sorted_values),
                "max": sorted_values[-1],
                "count": n,
            }

        return {
            "disk_read": stat(self.disk_read),
            "jpg_decode": stat(self.jpg_decode),
            "preprocess": stat(self.preprocess),
            "h2d": stat(self.h2d),
            "enqueueV3": stat(self.enqueueV3),
            "d2h": stat(self.d2h),
            "total": stat(self.total),
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True, help="TRT engine path")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Dataset directory containing *.jpg (V1.0.4: Artifacts/datasets/good_r512)",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Image side (default 512). Must match engine input shape.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations (excluded from timings)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Timed iterations",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path (per-image trace + summary)",
    )
    parser.add_argument(
        "--input-name",
        default="pixel_values",
        help="TRT engine input tensor name",
    )
    return parser.parse_args(argv)


def list_dataset_jpgs(dataset: Path) -> list[Path]:
    """List jpg files in dataset, sorted by name (deterministic order)."""

    candidates: list[Path] = []
    for ext in (".jpg", ".jpeg", ".JPG", ".JPEG"):
        candidates.extend(dataset.glob(f"*{ext}"))
    if not candidates:
        raise FileNotFoundError(f"No .jpg files in {dataset}")
    candidates = list(set(candidates))
    candidates.sort(key=lambda p: p.name)
    return candidates


def setup_trt_engine(engine_path: Path, input_name: str) -> tuple[Any, Any, Any, Any, Any, dict, dict]:
    """Deserialize engine + create context + stream + buffers.

    Returns (engine, context, stream, runtime, cudart, host_buffers, device_buffers).
    """

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

    err, stream = cudart.cudaStreamCreate()
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaStreamCreate failed: {err}")

    return engine, context, stream, runtime, cudart, {}, {}


def allocate_buffers(
    engine: Any,
    context: Any,
    cudart: Any,
    batch_size: int,
    image_size: int,
    input_name: str,
) -> tuple[dict[str, np.ndarray], dict[str, int], list[str]]:
    """Allocate pinned host buffers + device buffers for input + outputs.

    Returns (host_buffers, device_buffers_ptrs, output_names).
    """

    # Set input shape
    input_shape = (batch_size, 3, image_size, image_size)
    context.set_input_shape(input_name, input_shape)

    host_buffers: dict[str, np.ndarray] = {}
    device_ptrs: dict[str, int] = {}
    output_names: list[str] = []

    # Iterate all tensors (input + outputs)
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        shape = tuple(context.get_tensor_shape(name))
        dtype = engine.get_tensor_dtype(name)

        # Map TRT dtype → numpy dtype (assume float32 outputs unless engine declares otherwise)
        import tensorrt as trt
        if dtype == trt.float32:
            np_dtype = np.float32
        elif dtype == trt.float16:
            np_dtype = np.float16
        elif dtype == trt.bfloat16:
            np_dtype = np.float32  # numpy lacks bf16; use fp32 buffer
        else:
            raise RuntimeError(f"unsupported tensor dtype {dtype} for {name}")

        elements = int(np.prod(shape))
        nbytes = elements * np.dtype(np_dtype).itemsize

        host = np.zeros(elements, dtype=np_dtype)
        host_buffers[name] = host

        err, ptr = cudart.cudaMalloc(nbytes)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaMalloc failed for {name}: {err}")
        device_ptrs[name] = int(ptr)
        context.set_tensor_address(name, int(ptr))

        if mode == trt.TensorIOMode.OUTPUT:
            output_names.append(name)

    return host_buffers, device_ptrs, output_names


def preprocess_image(img_array: np.ndarray, image_size: int) -> np.ndarray:
    """ImageNet normalize + HWC→NCHW. Input HxWxC uint8, output 1xCxHxW float32."""

    if img_array.shape != (image_size, image_size, 3):
        raise ValueError(
            f"expected input shape ({image_size}, {image_size}, 3), got {img_array.shape}"
        )
    arr = img_array.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)  # HWC → CHW
    return arr  # (3, H, W)


def run_single_iteration(
    img_paths: list[Path],
    iter_idx: int,
    batch_size: int,
    image_size: int,
    context: Any,
    cudart: Any,
    stream: Any,
    host_input: np.ndarray,
    host_outputs: dict[str, np.ndarray],
    device_ptrs: dict[str, int],
    input_name: str,
    output_names: list[str],
    timings: StageTimings | None,
) -> None:
    """Run one inference iteration, optionally record timings."""

    PIL = importlib.import_module("PIL.Image")

    # ===== Stage 1: disk_read (batch_size jpg bytes) =====
    t0 = time.perf_counter()
    image_bytes_batch: list[bytes] = []
    for b in range(batch_size):
        path = img_paths[(iter_idx * batch_size + b) % len(img_paths)]
        image_bytes_batch.append(path.read_bytes())
    t1 = time.perf_counter()

    # ===== Stage 2: jpg_decode =====
    decoded_batch: list[np.ndarray] = []
    for raw in image_bytes_batch:
        img = PIL.open(io.BytesIO(raw)).convert("RGB")
        decoded_batch.append(np.asarray(img, dtype=np.uint8))
    t2 = time.perf_counter()

    # ===== Stage 3: preprocess =====
    preprocessed = np.stack(
        [preprocess_image(arr, image_size) for arr in decoded_batch], axis=0
    )  # (B, 3, H, W) float32
    np.copyto(host_input.reshape(batch_size, 3, image_size, image_size), preprocessed)
    t3 = time.perf_counter()

    # ===== Stage 4: H2D =====
    err = cudart.cudaMemcpyAsync(
        device_ptrs[input_name],
        host_input.ctypes.data,
        host_input.nbytes,
        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        stream,
    )
    if err[0] != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"H2D memcpy failed: {err}")
    err = cudart.cudaStreamSynchronize(stream)
    if err[0] != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"H2D sync failed: {err}")
    t4 = time.perf_counter()

    # ===== Stage 5: enqueueV3 + sync =====
    if not context.execute_async_v3(stream):
        raise RuntimeError("execute_async_v3 returned False")
    err = cudart.cudaStreamSynchronize(stream)
    if err[0] != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"compute sync failed: {err}")
    t5 = time.perf_counter()

    # ===== Stage 6: D2H (4 outputs) =====
    for name in output_names:
        host = host_outputs[name]
        err = cudart.cudaMemcpyAsync(
            host.ctypes.data,
            device_ptrs[name],
            host.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            stream,
        )
        if err[0] != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"D2H memcpy failed for {name}: {err}")
    err = cudart.cudaStreamSynchronize(stream)
    if err[0] != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"D2H sync failed: {err}")
    t6 = time.perf_counter()

    if timings is not None:
        timings.append(
            disk_read=(t1 - t0) * 1000.0,
            jpg_decode=(t2 - t1) * 1000.0,
            preprocess=(t3 - t2) * 1000.0,
            h2d=(t4 - t3) * 1000.0,
            enqueueV3=(t5 - t4) * 1000.0,
            d2h=(t6 - t5) * 1000.0,
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.engine.exists():
        print(f"error: engine not found: {args.engine}", file=sys.stderr)
        return 1
    if not args.dataset.is_dir():
        print(f"error: dataset directory not found: {args.dataset}", file=sys.stderr)
        return 1

    print(f"[production_benchmark] engine = {args.engine}")
    print(f"[production_benchmark] dataset = {args.dataset}")
    print(f"[production_benchmark] batch_size = {args.batch_size}")
    print(f"[production_benchmark] image_size = {args.image_size}")
    print(f"[production_benchmark] warmup = {args.warmup}, iters = {args.iters}")

    img_paths = list_dataset_jpgs(args.dataset)
    print(f"[production_benchmark] dataset images = {len(img_paths)}")

    engine, context, stream, runtime, cudart, _, _ = setup_trt_engine(
        args.engine, args.input_name
    )
    host_buffers, device_ptrs, output_names = allocate_buffers(
        engine, context, cudart, args.batch_size, args.image_size, args.input_name
    )
    host_input = host_buffers[args.input_name]
    host_outputs = {name: host_buffers[name] for name in output_names}
    print(f"[production_benchmark] outputs = {output_names}")

    # Warmup (no timings recorded)
    print(f"[production_benchmark] warmup × {args.warmup}...")
    for w in range(args.warmup):
        run_single_iteration(
            img_paths,
            w,
            args.batch_size,
            args.image_size,
            context,
            cudart,
            stream,
            host_input,
            host_outputs,
            device_ptrs,
            args.input_name,
            output_names,
            None,
        )

    # Timed iterations
    timings = StageTimings()
    print(f"[production_benchmark] timed × {args.iters}...")
    wall_start = time.perf_counter()
    for i in range(args.iters):
        run_single_iteration(
            img_paths,
            i,
            args.batch_size,
            args.image_size,
            context,
            cudart,
            stream,
            host_input,
            host_outputs,
            device_ptrs,
            args.input_name,
            output_names,
            timings,
        )
    wall_elapsed = time.perf_counter() - wall_start

    summary = timings.summary()
    total_images = args.iters * args.batch_size
    aggregate_qps = total_images / wall_elapsed if wall_elapsed > 0 else 0.0

    payload = {
        "engine": str(args.engine),
        "dataset": str(args.dataset),
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "warmup": args.warmup,
        "iters": args.iters,
        "language": "python",
        "total_inferences": args.iters,
        "total_images": total_images,
        "wall_elapsed_s": wall_elapsed,
        "aggregate_qps_imgs_per_sec": aggregate_qps,
        "stages_p50_ms": {k: v["p50"] for k, v in summary.items()},
        "stages_p95_ms": {k: v["p95"] for k, v in summary.items()},
        "stages_mean_ms": {k: v["mean"] for k, v in summary.items()},
        "stages_max_ms": {k: v["max"] for k, v in summary.items()},
        "per_image_trace": {
            "disk_read_ms": timings.disk_read,
            "jpg_decode_ms": timings.jpg_decode,
            "preprocess_ms": timings.preprocess,
            "h2d_ms": timings.h2d,
            "enqueueV3_ms": timings.enqueueV3,
            "d2h_ms": timings.d2h,
            "total_ms": timings.total,
        },
        "output_names": output_names,
    }

    print()
    print("=== V1.0.4 Production Benchmark Summary (p50 ms per stage) ===")
    print(f"  disk_read   : {summary['disk_read']['p50']:8.3f} ms")
    print(f"  jpg_decode  : {summary['jpg_decode']['p50']:8.3f} ms")
    print(f"  preprocess  : {summary['preprocess']['p50']:8.3f} ms")
    print(f"  h2d         : {summary['h2d']['p50']:8.3f} ms")
    print(f"  enqueueV3   : {summary['enqueueV3']['p50']:8.3f} ms")
    print(f"  d2h         : {summary['d2h']['p50']:8.3f} ms")
    print(f"  ---")
    print(f"  total       : {summary['total']['p50']:8.3f} ms")
    print(f"  aggregate   : {aggregate_qps:8.2f} imgs/sec")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[production_benchmark] report → {args.output}")

    # Cleanup
    for name, ptr in device_ptrs.items():
        cudart.cudaFree(ptr)
    cudart.cudaStreamDestroy(stream)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
