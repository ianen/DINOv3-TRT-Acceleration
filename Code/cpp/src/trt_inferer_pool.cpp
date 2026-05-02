#include "dinov3_trt/trt_inferer_pool.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <atomic>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <semaphore>
#include <stdexcept>
#include <utility>
#include <vector>

// V1.0.3 ADR-020 Phase 2 — shared engine + N independent contexts.
//
// Phase 2 (this implementation, since 2026-05-02) deserializes the engine
// ONCE at pool construction and shares the resulting `nvinfer1::ICudaEngine`
// across all slots. Each slot owns its own `IExecutionContext`,
// `cudaStream_t`, device input + output buffers, and per-call mutex (defense
// in depth). This matches TensorRT's documented "single engine, multiple
// contexts, multiple streams = thread-safe" architecture.
//
// Phase 1 (per-slot independent TRTInferer / engine) failed at concurrent
// N=2 with TRT 10.13 Myelin runner.cpp:778 errors — the root cause was
// duplicate runtime/engine deserialization in the same process, NOT a
// fundamental TRT thread-safety issue. Phase 2 prototype
// (`tools/test_concurrent_contexts.cpp`) confirmed shared engine works:
// r224 b1 N=2 521 qps (1.51×), r224 b8 N=2 921.94 qps (2.67×) — the latter
// passes V1.0.3 G1 ≥ 800 qps SMART target.
//
// G7 utilization saturation regimes (r336 b8 SM 96%, r518 b8 SM 99%) leave
// no headroom for multi-context scaling — pool throughput at those configs
// matches single-context. This is documented behavior, not a defect.

namespace dinov3_trt {

namespace {

constexpr const char* kInputTensorName = "pixel_values";

class TrtLogger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << "[TensorRT] " << message << '\n';
    }
  }
};

[[nodiscard]] std::vector<char> read_binary_file(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    return {};
  }
  const std::streamsize size = file.tellg();
  if (size <= 0) {
    return {};
  }
  std::vector<char> data(static_cast<std::size_t>(size));
  file.seekg(0, std::ios::beg);
  if (!file.read(data.data(), size)) {
    return {};
  }
  return data;
}

[[nodiscard]] Status cuda_status(cudaError_t error, const std::string& call_name) {
  if (error == cudaSuccess) {
    return Status::ok();
  }
  return Status::runtime_error(call_name + " failed: " + cudaGetErrorString(error));
}

[[nodiscard]] Status validate_input(const TensorView& input) {
  if (input.is_empty()) {
    return Status::invalid_argument("input tensor is empty");
  }
  if (input.data_type != DataType::kFloat32) {
    return Status::invalid_argument("input tensor must be float32");
  }
  if (input.shape.rank != 4) {
    return Status::invalid_argument("input tensor must have 4 dims (NCHW)");
  }
  return Status::ok();
}

class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  ~DeviceBuffer() {
    if (ptr_ != nullptr) {
      static_cast<void>(cudaFree(ptr_));
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), bytes_(other.bytes_) {
    other.ptr_ = nullptr;
    other.bytes_ = 0;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      if (ptr_ != nullptr) {
        static_cast<void>(cudaFree(ptr_));
      }
      ptr_ = other.ptr_;
      bytes_ = other.bytes_;
      other.ptr_ = nullptr;
      other.bytes_ = 0;
    }
    return *this;
  }

  [[nodiscard]] Status ensure_allocated(std::size_t bytes) noexcept {
    if (bytes == 0) {
      return Status::invalid_argument("cannot allocate zero-byte CUDA buffer");
    }
    if (ptr_ != nullptr && bytes_ == bytes) {
      return Status::ok();
    }
    release();
    void* p = nullptr;
    Status status = cuda_status(cudaMalloc(&p, bytes), "cudaMalloc");
    if (!status.is_ok()) return status;
    ptr_ = p;
    bytes_ = bytes;
    return Status::ok();
  }

  void release() noexcept {
    if (ptr_ != nullptr) {
      static_cast<void>(cudaFree(ptr_));
      ptr_ = nullptr;
      bytes_ = 0;
    }
  }

  [[nodiscard]] void* get() const noexcept { return ptr_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

 private:
  void* ptr_{nullptr};
  std::size_t bytes_{0};
};

}  // namespace

class TRTInfererPool::Impl {
 public:
  explicit Impl(Config cfg) : config_(std::move(cfg)), available_slots_(0) {
    if (config_.num_streams < 1) {
      throw std::invalid_argument("TRTInfererPool: num_streams must be >= 1");
    }
    if (config_.engine_path.empty()) {
      throw std::invalid_argument("TRTInfererPool: engine_path is empty");
    }

    // Step 1: deserialize engine ONCE — Phase 2 architecture.
    const auto blob = read_binary_file(config_.engine_path);
    if (blob.empty()) {
      throw std::runtime_error("TRTInfererPool: failed to read engine file: " + config_.engine_path);
    }
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) {
      throw std::runtime_error("TRTInfererPool: createInferRuntime failed");
    }
    engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
    if (!engine_) {
      throw std::runtime_error("TRTInfererPool: deserializeCudaEngine failed: " + config_.engine_path);
    }

    // Step 2: create N independent execution contexts on the shared engine.
    slots_.reserve(static_cast<std::size_t>(config_.num_streams));
    for (int slot_index = 0; slot_index < config_.num_streams; ++slot_index) {
      auto slot = std::make_unique<Slot>();
      slot->ctx.reset(engine_->createExecutionContext());
      if (!slot->ctx) {
        throw std::runtime_error("TRTInfererPool: createExecutionContext failed for slot");
      }
      const cudaError_t err = cudaStreamCreate(&slot->stream);
      if (err != cudaSuccess) {
        throw std::runtime_error(std::string("TRTInfererPool: cudaStreamCreate failed: ") +
                                  cudaGetErrorString(err));
      }
      slots_.push_back(std::move(slot));
    }

    // Step 3: prime semaphore with all slots free.
    available_slots_.release(config_.num_streams);
  }

  ~Impl() {
    for (auto& slot : slots_) {
      if (slot && slot->stream) {
        static_cast<void>(cudaStreamDestroy(slot->stream));
        slot->stream = nullptr;
      }
    }
  }

  [[nodiscard]] Status infer(
      const TensorView& input,
      std::array<TensorView, kOutputCount>& outputs) noexcept {
    Status status = validate_input(input);
    if (!status.is_ok()) {
      return status;
    }

    available_slots_.acquire();
    const std::size_t slot_index = next_slot_index();

    Status result = Status::ok();
    {
      Slot& slot = *slots_[slot_index];
      std::lock_guard<std::mutex> lock(slot.mu);
      result = run_one(input, outputs, slot);
    }

    available_slots_.release();
    return result;
  }

  [[nodiscard]] int num_streams() const noexcept { return config_.num_streams; }

 private:
  struct Slot {
    std::unique_ptr<nvinfer1::IExecutionContext> ctx;
    cudaStream_t stream{nullptr};
    DeviceBuffer d_input;
    std::array<DeviceBuffer, kOutputCount> d_outputs;
    std::mutex mu;
  };

  [[nodiscard]] std::size_t next_slot_index() noexcept {
    const std::size_t idx = dispatch_counter_.fetch_add(1, std::memory_order_relaxed);
    return idx % static_cast<std::size_t>(config_.num_streams);
  }

  [[nodiscard]] Status run_one(
      const TensorView& input,
      std::array<TensorView, kOutputCount>& outputs,
      Slot& slot) noexcept {
    const int batch = static_cast<int>(input.shape.dims[0]);
    const int image_size = static_cast<int>(input.shape.dims[2]);

    // Set dynamic input shape on this slot's context.
    nvinfer1::Dims4 dims{batch, static_cast<int32_t>(input.shape.dims[1]),
                         image_size, static_cast<int32_t>(input.shape.dims[3])};
    if (!slot.ctx->setInputShape(kInputTensorName, dims)) {
      return Status::runtime_error("setInputShape failed");
    }

    Status status = slot.d_input.ensure_allocated(input.byte_size());
    if (!status.is_ok()) return status;
    if (!slot.ctx->setTensorAddress(kInputTensorName, slot.d_input.get())) {
      return Status::runtime_error("setTensorAddress(input) failed");
    }

    for (std::size_t i = 0; i < kOutputCount; ++i) {
      status = slot.d_outputs[i].ensure_allocated(outputs[i].byte_size());
      if (!status.is_ok()) return status;
      if (!slot.ctx->setTensorAddress(kOutputTensorNames[i], slot.d_outputs[i].get())) {
        return Status::runtime_error(std::string("setTensorAddress(output ") +
                                      kOutputTensorNames[i] + ") failed");
      }
    }

    status = cuda_status(
        cudaMemcpyAsync(slot.d_input.get(), input.data, input.byte_size(),
                        cudaMemcpyHostToDevice, slot.stream),
        "cudaMemcpyAsync(H2D)");
    if (!status.is_ok()) return status;

    if (!slot.ctx->enqueueV3(slot.stream)) {
      return Status::runtime_error("enqueueV3 failed");
    }

    for (std::size_t i = 0; i < kOutputCount; ++i) {
      status = cuda_status(
          cudaMemcpyAsync(outputs[i].data, slot.d_outputs[i].get(), outputs[i].byte_size(),
                          cudaMemcpyDeviceToHost, slot.stream),
          "cudaMemcpyAsync(D2H)");
      if (!status.is_ok()) return status;
    }

    status = cuda_status(cudaStreamSynchronize(slot.stream), "cudaStreamSynchronize");
    return status;
  }

  Config config_;
  TrtLogger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::vector<std::unique_ptr<Slot>> slots_;
  std::counting_semaphore<128> available_slots_;
  std::atomic<std::size_t> dispatch_counter_{0};
};

TRTInfererPool::TRTInfererPool(Config cfg) : impl_(std::make_unique<Impl>(std::move(cfg))) {}

TRTInfererPool::~TRTInfererPool() = default;

TRTInfererPool::TRTInfererPool(TRTInfererPool&&) noexcept = default;
TRTInfererPool& TRTInfererPool::operator=(TRTInfererPool&&) noexcept = default;

Status TRTInfererPool::infer(
    const TensorView& input,
    std::array<TensorView, kOutputCount>& outputs) noexcept {
  return impl_->infer(input, outputs);
}

int TRTInfererPool::num_streams() const noexcept { return impl_->num_streams(); }

}  // namespace dinov3_trt
