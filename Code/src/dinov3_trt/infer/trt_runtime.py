"""Minimal TensorRT Python runtime backed by cuda-python."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray


FloatTensor = NDArray[np.floating]


class TensorRTRuntimeError(RuntimeError):
    """Raised when TensorRT or CUDA runtime execution fails."""


@dataclass(frozen=True)
class TensorRTEngineRunConfig:
    """Configuration for one TensorRT engine execution."""

    engine_path: Path
    input_name: str = "pixel_values"


def _import_tensorrt() -> Any:
    try:
        return importlib.import_module("tensorrt")
    except ImportError as exc:
        raise TensorRTRuntimeError(
            "TensorRT Python is required for engine execution. Install the `trt` extra."
        ) from exc


def _import_cudart() -> Any:
    try:
        return importlib.import_module("cuda.bindings.runtime")
    except ImportError:
        try:
            return importlib.import_module("cuda.cudart")
        except ImportError as exc:
            raise TensorRTRuntimeError(
                "cuda-python is required for engine execution. Install the `trt` extra."
            ) from exc


def _error_code(error: Any) -> int:
    value = getattr(error, "value", error)
    return int(value)


def _first_success_value(result: Any, call_name: str) -> Any:
    if not isinstance(result, tuple):
        if _error_code(result) != 0:
            raise TensorRTRuntimeError(f"{call_name} failed with CUDA error {result}")
        return None
    error = result[0]
    if _error_code(error) != 0:
        raise TensorRTRuntimeError(f"{call_name} failed with CUDA error {error}")
    if len(result) == 1:
        return None
    return result[1]


def _check_cuda(result: Any, call_name: str) -> None:
    _first_success_value(result, call_name)


def _tensor_names(engine: Any) -> tuple[str, ...]:
    return tuple(str(engine.get_tensor_name(index)) for index in range(engine.num_io_tensors))


def _shape_tuple(shape: Any, name: str) -> tuple[int, ...]:
    dims = tuple(int(dim) for dim in shape)
    if any(dim < 1 for dim in dims):
        raise TensorRTRuntimeError(f"tensor {name!r} has unresolved shape {dims}")
    return dims


def _tensor_numpy_dtype(trt: Any, dtype: Any) -> np.dtype[np.generic]:
    return cast("np.dtype[np.generic]", np.dtype(trt.nptype(dtype)))


class _CudaRuntime:
    def __init__(self, cudart: Any) -> None:
        self._cudart = cudart
        self._stream = _first_success_value(cudart.cudaStreamCreate(), "cudaStreamCreate")
        self._device_allocations: list[int] = []

    @property
    def stream(self) -> int:
        return int(self._stream)

    def malloc(self, nbytes: int) -> int:
        ptr = int(_first_success_value(self._cudart.cudaMalloc(nbytes), "cudaMalloc"))
        self._device_allocations.append(ptr)
        return ptr

    def memcpy_host_to_device(self, device_ptr: int, array: np.ndarray[Any, Any]) -> None:
        _check_cuda(
            self._cudart.cudaMemcpyAsync(
                device_ptr,
                int(array.ctypes.data),
                array.nbytes,
                self._cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self.stream,
            ),
            "cudaMemcpyAsync(H2D)",
        )

    def memcpy_device_to_host(self, array: np.ndarray[Any, Any], device_ptr: int) -> None:
        _check_cuda(
            self._cudart.cudaMemcpyAsync(
                int(array.ctypes.data),
                device_ptr,
                array.nbytes,
                self._cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self.stream,
            ),
            "cudaMemcpyAsync(D2H)",
        )

    def synchronize(self) -> None:
        _check_cuda(self._cudart.cudaStreamSynchronize(self.stream), "cudaStreamSynchronize")

    def close(self) -> None:
        for ptr in reversed(self._device_allocations):
            _check_cuda(self._cudart.cudaFree(ptr), "cudaFree")
        self._device_allocations.clear()
        _check_cuda(self._cudart.cudaStreamDestroy(self.stream), "cudaStreamDestroy")

    def __enter__(self) -> _CudaRuntime:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def run_engine(
    config: TensorRTEngineRunConfig,
    input_tensor: NDArray[np.float32],
) -> dict[str, FloatTensor]:
    """Run one TensorRT engine and return named output tensors."""

    trt = _import_tensorrt()
    cudart = _import_cudart()
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    with config.engine_path.open("rb") as engine_file:
        engine = runtime.deserialize_cuda_engine(engine_file.read())
    if engine is None:
        raise TensorRTRuntimeError(f"failed to deserialize engine: {config.engine_path}")
    context = engine.create_execution_context()
    if context is None:
        raise TensorRTRuntimeError("failed to create TensorRT execution context")
    if not context.set_input_shape(config.input_name, tuple(int(dim) for dim in input_tensor.shape)):
        raise TensorRTRuntimeError(
            f"failed to set input shape for {config.input_name!r}: {input_tensor.shape}"
        )

    host_buffers: dict[str, np.ndarray[Any, Any]] = {}
    device_buffers: dict[str, int] = {}
    output_names: list[str] = []
    with _CudaRuntime(cudart) as cuda:
        for name in _tensor_names(engine):
            mode = engine.get_tensor_mode(name)
            dtype = _tensor_numpy_dtype(trt, engine.get_tensor_dtype(name))
            if mode == trt.TensorIOMode.INPUT:
                if name != config.input_name:
                    raise TensorRTRuntimeError(f"unexpected input tensor {name!r}")
                host = np.ascontiguousarray(input_tensor.astype(dtype, copy=False))
            else:
                shape = _shape_tuple(context.get_tensor_shape(name), name)
                host = np.empty(shape, dtype=dtype)
                output_names.append(name)
            ptr = cuda.malloc(host.nbytes)
            host_buffers[name] = host
            device_buffers[name] = ptr
            if not context.set_tensor_address(name, ptr):
                raise TensorRTRuntimeError(f"failed to set tensor address for {name!r}")

        cuda.memcpy_host_to_device(device_buffers[config.input_name], host_buffers[config.input_name])
        if not context.execute_async_v3(stream_handle=cuda.stream):
            raise TensorRTRuntimeError("TensorRT execute_async_v3 failed")
        for name in output_names:
            cuda.memcpy_device_to_host(host_buffers[name], device_buffers[name])
        cuda.synchronize()

    return {name: host_buffers[name] for name in output_names}
