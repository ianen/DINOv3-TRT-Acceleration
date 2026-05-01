"""Mock-based unit tests for `infer/trt_runtime.py` helpers (no GPU/TRT required)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dinov3_trt.infer.trt_runtime import (
    TensorRTEngineRunConfig,
    TensorRTRuntimeError,
    _check_cuda,
    _CudaRuntime,
    _error_code,
    _first_success_value,
    _import_cudart,
    _import_tensorrt,
    _shape_tuple,
    _tensor_names,
    _tensor_numpy_dtype,
)


# --- _error_code ---------------------------------------------------------


def test_error_code_extracts_value_attr() -> None:
    assert _error_code(SimpleNamespace(value=42)) == 42


def test_error_code_falls_back_to_int_cast() -> None:
    assert _error_code(7) == 7


# --- _first_success_value -----------------------------------------------


def test_first_success_value_returns_payload_for_zero_error_tuple() -> None:
    error = SimpleNamespace(value=0)
    assert _first_success_value((error, "payload"), "fakeCall") == "payload"


def test_first_success_value_returns_none_for_zero_error_single() -> None:
    error = SimpleNamespace(value=0)
    assert _first_success_value((error,), "fakeCall") is None


def test_first_success_value_returns_none_when_not_tuple_zero_error() -> None:
    assert _first_success_value(SimpleNamespace(value=0), "fakeCall") is None


def test_first_success_value_raises_on_nonzero_error() -> None:
    error = SimpleNamespace(value=99)
    with pytest.raises(TensorRTRuntimeError, match="failed with CUDA error"):
        _first_success_value((error, "ignored"), "cudaMalloc")


def test_first_success_value_raises_when_not_tuple_nonzero() -> None:
    with pytest.raises(TensorRTRuntimeError, match="failed with CUDA error"):
        _first_success_value(SimpleNamespace(value=1), "cudaMalloc")


# --- _check_cuda ---------------------------------------------------------


def test_check_cuda_success_returns_none() -> None:
    error = SimpleNamespace(value=0)
    # _check_cuda returns None on success (no return value)
    _check_cuda((error,), "test")


def test_check_cuda_raises_on_failure() -> None:
    error = SimpleNamespace(value=2)
    with pytest.raises(TensorRTRuntimeError):
        _check_cuda((error, "x"), "test")


# --- _tensor_names -------------------------------------------------------


def test_tensor_names_returns_tuple_of_strings() -> None:
    engine = MagicMock()
    engine.num_io_tensors = 3
    engine.get_tensor_name.side_effect = ["pixel_values", "feat_layer_4", "feat_layer_12"]

    names = _tensor_names(engine)

    assert names == ("pixel_values", "feat_layer_4", "feat_layer_12")
    assert all(isinstance(n, str) for n in names)


def test_tensor_names_empty_engine_returns_empty_tuple() -> None:
    engine = MagicMock()
    engine.num_io_tensors = 0

    assert _tensor_names(engine) == ()


# --- _shape_tuple --------------------------------------------------------


def test_shape_tuple_converts_to_int_tuple() -> None:
    assert _shape_tuple([1, 197, 1024], "feat_layer_4") == (1, 197, 1024)


def test_shape_tuple_accepts_iterables() -> None:
    assert _shape_tuple((8, 442, 1024), "feat_layer_12") == (8, 442, 1024)


def test_shape_tuple_raises_on_unresolved_dim() -> None:
    with pytest.raises(TensorRTRuntimeError, match="unresolved shape"):
        _shape_tuple([-1, 197, 1024], "pixel_values")


def test_shape_tuple_raises_on_zero_dim() -> None:
    with pytest.raises(TensorRTRuntimeError, match="unresolved shape"):
        _shape_tuple([0, 197, 1024], "pixel_values")


# --- _tensor_numpy_dtype -------------------------------------------------


def test_tensor_numpy_dtype_uses_trt_nptype() -> None:
    trt = MagicMock()
    trt.nptype.return_value = np.float32

    dtype = _tensor_numpy_dtype(trt, "fake_trt_float")

    assert dtype == np.dtype(np.float32)
    trt.nptype.assert_called_once_with("fake_trt_float")


# --- _import_tensorrt / _import_cudart ----------------------------------


def test_import_tensorrt_returns_module_when_available() -> None:
    fake_trt = SimpleNamespace(__name__="tensorrt", Logger="fake")

    with patch("importlib.import_module", return_value=fake_trt) as mock_import:
        module = _import_tensorrt()

    assert module is fake_trt
    mock_import.assert_called_once_with("tensorrt")


def test_import_tensorrt_raises_when_missing() -> None:
    with patch("importlib.import_module", side_effect=ImportError("no tensorrt")):
        with pytest.raises(TensorRTRuntimeError, match="TensorRT Python is required"):
            _import_tensorrt()


def test_import_cudart_falls_back_to_legacy_path() -> None:
    fake_legacy = SimpleNamespace(__name__="cuda.cudart")

    def side_effect(name: str) -> object:
        if name == "cuda.bindings.runtime":
            raise ImportError("missing new path")
        if name == "cuda.cudart":
            return fake_legacy
        raise ImportError(f"unexpected import {name}")

    with patch("importlib.import_module", side_effect=side_effect):
        module = _import_cudart()

    assert module is fake_legacy


def test_import_cudart_raises_when_both_paths_missing() -> None:
    with patch("importlib.import_module", side_effect=ImportError("missing")):
        with pytest.raises(TensorRTRuntimeError, match="cuda-python is required"):
            _import_cudart()


def test_import_cudart_returns_new_path_when_available() -> None:
    fake_new = SimpleNamespace(__name__="cuda.bindings.runtime")

    with patch("importlib.import_module", return_value=fake_new):
        module = _import_cudart()

    assert module is fake_new


# --- TensorRTEngineRunConfig dataclass -----------------------------------


def test_engine_run_config_has_default_input_name() -> None:
    cfg = TensorRTEngineRunConfig(engine_path=Path("/tmp/x.engine"))
    assert cfg.input_name == "pixel_values"


def test_engine_run_config_accepts_custom_input_name() -> None:
    cfg = TensorRTEngineRunConfig(engine_path=Path("/tmp/x.engine"), input_name="custom_input")
    assert cfg.input_name == "custom_input"


def test_engine_run_config_is_frozen() -> None:
    cfg = TensorRTEngineRunConfig(engine_path=Path("/tmp/x.engine"))
    with pytest.raises((AttributeError, Exception)):
        cfg.input_name = "modified"  # type: ignore[misc]


# --- TensorRTRuntimeError ------------------------------------------------


def test_tensor_rt_runtime_error_is_runtime_error_subclass() -> None:
    assert issubclass(TensorRTRuntimeError, RuntimeError)


def test_tensor_rt_runtime_error_message_round_trip() -> None:
    err = TensorRTRuntimeError("specific failure")
    assert "specific failure" in str(err)


# --- _CudaRuntime context manager (mock-based) ---------------------------


def _make_mock_cudart() -> MagicMock:
    """Return a cudart mock where every call returns success-coded tuples."""

    cudart = MagicMock()
    success = SimpleNamespace(value=0)
    cudart.cudaStreamCreate.return_value = (success, 12345)
    cudart.cudaMalloc.return_value = (success, 0xDEADBEEF)
    cudart.cudaMemcpyAsync.return_value = (success,)
    cudart.cudaStreamSynchronize.return_value = (success,)
    cudart.cudaFree.return_value = (success,)
    cudart.cudaStreamDestroy.return_value = (success,)
    cudart.cudaMemcpyKind = SimpleNamespace(
        cudaMemcpyHostToDevice="H2D",
        cudaMemcpyDeviceToHost="D2H",
    )
    return cudart


def test_cuda_runtime_creates_stream_on_init() -> None:
    cudart = _make_mock_cudart()

    with _CudaRuntime(cudart) as runtime:
        assert runtime.stream == 12345

    cudart.cudaStreamCreate.assert_called_once()


def test_cuda_runtime_malloc_tracks_pointers() -> None:
    cudart = _make_mock_cudart()

    with _CudaRuntime(cudart) as runtime:
        ptr = runtime.malloc(1024)
        assert ptr == 0xDEADBEEF
        cudart.cudaMalloc.assert_called_once_with(1024)


def test_cuda_runtime_close_frees_allocations_in_reverse_order() -> None:
    cudart = _make_mock_cudart()
    success = SimpleNamespace(value=0)
    cudart.cudaMalloc.side_effect = [(success, 1), (success, 2), (success, 3)]

    runtime = _CudaRuntime(cudart)
    runtime.malloc(128)
    runtime.malloc(256)
    runtime.malloc(512)
    runtime.close()

    free_calls = [call.args[0] for call in cudart.cudaFree.call_args_list]
    assert free_calls == [3, 2, 1]


def test_cuda_runtime_h2d_uses_correct_kind() -> None:
    cudart = _make_mock_cudart()
    arr = np.zeros(4, dtype=np.float32)

    with _CudaRuntime(cudart) as runtime:
        runtime.memcpy_host_to_device(0xCAFEBABE, arr)

    h2d_calls = cudart.cudaMemcpyAsync.call_args_list
    assert any("H2D" in str(call) for call in h2d_calls)


def test_cuda_runtime_d2h_uses_correct_kind() -> None:
    cudart = _make_mock_cudart()
    arr = np.zeros(4, dtype=np.float32)

    with _CudaRuntime(cudart) as runtime:
        runtime.memcpy_device_to_host(arr, 0xCAFEBABE)

    d2h_calls = cudart.cudaMemcpyAsync.call_args_list
    assert any("D2H" in str(call) for call in d2h_calls)


def test_cuda_runtime_synchronize_calls_stream_sync() -> None:
    cudart = _make_mock_cudart()

    with _CudaRuntime(cudart) as runtime:
        runtime.synchronize()

    cudart.cudaStreamSynchronize.assert_called_once_with(12345)


def test_cuda_runtime_handles_create_stream_failure() -> None:
    cudart = MagicMock()
    error = SimpleNamespace(value=99)
    cudart.cudaStreamCreate.return_value = (error,)

    with pytest.raises(TensorRTRuntimeError, match="cudaStreamCreate"):
        _CudaRuntime(cudart)


# --- module-level import surface ----------------------------------------


def test_module_exposes_public_runtime_api() -> None:
    import dinov3_trt.infer.trt_runtime as mod

    public_names = {name for name in dir(mod) if not name.startswith("_")}
    assert {"TensorRTEngineRunConfig", "TensorRTRuntimeError", "run_engine"}.issubset(public_names)


# --- run_engine() mock happy path + error paths --------------------------


class _FakeTrt:
    """Minimal mock of the tensorrt module."""

    class Logger:
        ERROR = "ERROR"

        def __init__(self, level: object) -> None:
            self._level = level

    class TensorIOMode:
        INPUT = "INPUT"
        OUTPUT = "OUTPUT"

    @staticmethod
    def nptype(dtype: object) -> object:
        # Map any TRT dtype to float32 for the mock.
        return np.float32

    class Runtime:
        def __init__(self, logger: object) -> None:
            self.logger = logger
            self._engine: object = None

        def deserialize_cuda_engine(self, blob: bytes) -> object:
            return self._engine


def _build_fake_engine_and_context(
    *, deserialize_returns_none: bool = False, set_input_shape_ok: bool = True,
    set_tensor_address_ok: bool = True, execute_ok: bool = True,
) -> tuple[MagicMock, MagicMock]:
    """Create a (fake_engine, fake_context) pair that mimics the TRT API surface."""

    if deserialize_returns_none:
        return MagicMock(name="engine_none"), MagicMock(name="ctx_unused")

    context = MagicMock()
    context.set_input_shape.return_value = set_input_shape_ok
    context.set_tensor_address.return_value = set_tensor_address_ok
    context.execute_async_v3.return_value = execute_ok
    context.get_tensor_shape.side_effect = lambda name: (1, 197, 1024)

    engine = MagicMock()
    engine.create_execution_context.return_value = context
    engine.num_io_tensors = 2
    engine.get_tensor_name.side_effect = ["pixel_values", "feat_layer_4"]
    engine.get_tensor_mode.side_effect = lambda n: (
        _FakeTrt.TensorIOMode.INPUT if n == "pixel_values" else _FakeTrt.TensorIOMode.OUTPUT
    )
    engine.get_tensor_dtype.return_value = "fake_dtype"
    return engine, context


def _patch_run_engine(
    *, engine: MagicMock | None, deserialize_returns_none: bool = False
) -> tuple[object, object]:
    """Return patched (trt_module, cudart_module) helpers for run_engine."""

    fake_trt = _FakeTrt()
    runtime_obj = MagicMock()
    runtime_obj._engine = None if deserialize_returns_none else engine
    runtime_obj.deserialize_cuda_engine = lambda blob: runtime_obj._engine
    fake_trt.Runtime = lambda logger: runtime_obj  # type: ignore[assignment]

    cudart = _make_mock_cudart()
    return fake_trt, cudart


def test_run_engine_happy_path(tmp_path: Path) -> None:
    from dinov3_trt.infer.trt_runtime import run_engine

    engine_path = tmp_path / "fake.engine"
    engine_path.write_bytes(b"engine-blob")
    engine, context = _build_fake_engine_and_context()
    fake_trt, cudart = _patch_run_engine(engine=engine)

    with patch("dinov3_trt.infer.trt_runtime._import_tensorrt", return_value=fake_trt), patch(
        "dinov3_trt.infer.trt_runtime._import_cudart", return_value=cudart
    ):
        outputs = run_engine(
            TensorRTEngineRunConfig(engine_path=engine_path),
            np.zeros((1, 3, 224, 224), dtype=np.float32),
        )

    assert "feat_layer_4" in outputs
    assert outputs["feat_layer_4"].shape == (1, 197, 1024)
    assert outputs["feat_layer_4"].dtype == np.float32
    context.set_input_shape.assert_called_once_with("pixel_values", (1, 3, 224, 224))
    context.execute_async_v3.assert_called_once()


def test_run_engine_raises_when_deserialize_returns_none(tmp_path: Path) -> None:
    from dinov3_trt.infer.trt_runtime import run_engine

    engine_path = tmp_path / "broken.engine"
    engine_path.write_bytes(b"")
    fake_trt, cudart = _patch_run_engine(engine=None, deserialize_returns_none=True)

    with patch("dinov3_trt.infer.trt_runtime._import_tensorrt", return_value=fake_trt), patch(
        "dinov3_trt.infer.trt_runtime._import_cudart", return_value=cudart
    ):
        with pytest.raises(TensorRTRuntimeError, match="failed to deserialize engine"):
            run_engine(
                TensorRTEngineRunConfig(engine_path=engine_path),
                np.zeros((1, 3, 224, 224), dtype=np.float32),
            )


def test_run_engine_raises_when_create_context_returns_none(tmp_path: Path) -> None:
    from dinov3_trt.infer.trt_runtime import run_engine

    engine_path = tmp_path / "fake.engine"
    engine_path.write_bytes(b"engine-blob")
    engine = MagicMock()
    engine.create_execution_context.return_value = None
    fake_trt, cudart = _patch_run_engine(engine=engine)

    with patch("dinov3_trt.infer.trt_runtime._import_tensorrt", return_value=fake_trt), patch(
        "dinov3_trt.infer.trt_runtime._import_cudart", return_value=cudart
    ):
        with pytest.raises(TensorRTRuntimeError, match="execution context"):
            run_engine(
                TensorRTEngineRunConfig(engine_path=engine_path),
                np.zeros((1, 3, 224, 224), dtype=np.float32),
            )


def test_run_engine_raises_when_set_input_shape_fails(tmp_path: Path) -> None:
    from dinov3_trt.infer.trt_runtime import run_engine

    engine_path = tmp_path / "fake.engine"
    engine_path.write_bytes(b"engine-blob")
    engine, _ = _build_fake_engine_and_context(set_input_shape_ok=False)
    fake_trt, cudart = _patch_run_engine(engine=engine)

    with patch("dinov3_trt.infer.trt_runtime._import_tensorrt", return_value=fake_trt), patch(
        "dinov3_trt.infer.trt_runtime._import_cudart", return_value=cudart
    ):
        with pytest.raises(TensorRTRuntimeError, match="set input shape"):
            run_engine(
                TensorRTEngineRunConfig(engine_path=engine_path),
                np.zeros((1, 3, 224, 224), dtype=np.float32),
            )


def test_run_engine_raises_when_execute_returns_false(tmp_path: Path) -> None:
    from dinov3_trt.infer.trt_runtime import run_engine

    engine_path = tmp_path / "fake.engine"
    engine_path.write_bytes(b"engine-blob")
    engine, _ = _build_fake_engine_and_context(execute_ok=False)
    fake_trt, cudart = _patch_run_engine(engine=engine)

    with patch("dinov3_trt.infer.trt_runtime._import_tensorrt", return_value=fake_trt), patch(
        "dinov3_trt.infer.trt_runtime._import_cudart", return_value=cudart
    ):
        with pytest.raises(TensorRTRuntimeError, match="execute_async_v3 failed"):
            run_engine(
                TensorRTEngineRunConfig(engine_path=engine_path),
                np.zeros((1, 3, 224, 224), dtype=np.float32),
            )


def test_run_engine_raises_when_set_tensor_address_fails(tmp_path: Path) -> None:
    from dinov3_trt.infer.trt_runtime import run_engine

    engine_path = tmp_path / "fake.engine"
    engine_path.write_bytes(b"engine-blob")
    engine, _ = _build_fake_engine_and_context(set_tensor_address_ok=False)
    fake_trt, cudart = _patch_run_engine(engine=engine)

    with patch("dinov3_trt.infer.trt_runtime._import_tensorrt", return_value=fake_trt), patch(
        "dinov3_trt.infer.trt_runtime._import_cudart", return_value=cudart
    ):
        with pytest.raises(TensorRTRuntimeError, match="set tensor address"):
            run_engine(
                TensorRTEngineRunConfig(engine_path=engine_path),
                np.zeros((1, 3, 224, 224), dtype=np.float32),
            )
