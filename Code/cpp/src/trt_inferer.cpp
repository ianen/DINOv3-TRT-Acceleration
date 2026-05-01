#include "dinov3_trt/trt_inferer.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <array>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace dinov3_trt {
namespace {

inline constexpr const char* kInputTensorName = "pixel_values";

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

[[nodiscard]] bool same_shape(const TensorShape& left, const TensorShape& right) noexcept {
  if (left.rank != right.rank) {
    return false;
  }
  for (std::size_t index = 0; index < left.rank; ++index) {
    if (left.dims[index] != right.dims[index]) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] Status validate_input(const TensorView& input) {
  if (input.is_empty()) {
    return Status::invalid_argument("input tensor is empty");
  }
  if (input.data_type != DataType::kFloat32) {
    return Status::invalid_argument("input tensor must be float32");
  }
  if (input.shape.rank != 4) {
    return Status::invalid_argument("input tensor must be NCHW rank-4");
  }
  if (input.shape.dims[0] < 1) {
    return Status::invalid_argument("input batch size must be >= 1");
  }
  if (input.shape.dims[0] > std::numeric_limits<int32_t>::max()) {
    return Status::invalid_argument("input batch size exceeds TensorRT int32 shape limit");
  }
  if (input.shape.dims[1] != kInputChannels) {
    return Status::invalid_argument("input channel count must match kInputChannels (3)");
  }
  const std::int64_t height = input.shape.dims[2];
  const std::int64_t width = input.shape.dims[3];
  if (height <= 0 || width <= 0 || height != width) {
    return Status::invalid_argument("input spatial dims must be positive and square");
  }
  if (height < kPatchSize) {
    return Status::invalid_argument("input image size must be at least the patch size (16)");
  }
  // DINOv3 ViT applies floor(H / patch_size) when forming patch tokens (518 maps to a 32x32
  // grid by dropping 6 trailing pixels), so we do not require strict divisibility here.
  return Status::ok();
}

[[nodiscard]] Status validate_outputs(
    const std::array<TensorView, kOutputCount>& outputs,
    std::int64_t batch_size,
    int image_size) {
  const TensorShape expected = output_shape_for(batch_size, image_size);
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    if (outputs[index].is_empty()) {
      return Status::invalid_argument(
          std::string("output tensor is empty: ") + kOutputTensorNames[index]);
    }
    if (outputs[index].data_type != DataType::kFloat32) {
      return Status::invalid_argument(
          std::string("output tensor must be float32: ") + kOutputTensorNames[index]);
    }
    if (!same_shape(outputs[index].shape, expected)) {
      return Status::invalid_argument(
          std::string("output tensor shape must match [B,") +
          std::to_string(output_tokens_for(image_size)) + ",1024]: " +
          kOutputTensorNames[index]);
    }
  }
  return Status::ok();
}

[[nodiscard]] Status set_tensor_address(
    nvinfer1::IExecutionContext& context,
    const char* name,
    void* address) {
  if (!context.setTensorAddress(name, address)) {
    return Status::runtime_error(std::string("failed to set TensorRT tensor address: ") + name);
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

    void* ptr = nullptr;
    const Status status = cuda_status(cudaMalloc(&ptr, bytes), "cudaMalloc");
    if (!status.is_ok()) {
      return status;
    }
    ptr_ = ptr;
    bytes_ = bytes;
    return Status::ok();
  }

  [[nodiscard]] void* get() const noexcept { return ptr_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

 private:
  void release() noexcept {
    if (ptr_ != nullptr) {
      static_cast<void>(cudaFree(ptr_));
      ptr_ = nullptr;
      bytes_ = 0;
    }
  }

  void* ptr_{nullptr};
  std::size_t bytes_{0};
};

}  // namespace

class TRTInferer::Impl {
 public:
  Impl(std::string engine_path, int device_id)
      : engine_path_(std::move(engine_path)), device_id_(device_id) {
    init_status_ = initialize();
  }

  ~Impl() {
    if (stream_ != nullptr) {
      static_cast<void>(cudaStreamDestroy(stream_));
    }
  }

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;

  [[nodiscard]] Status infer(
      const TensorView& input,
      std::array<TensorView, kOutputCount>& outputs) noexcept {
    try {
      if (!init_status_.is_ok()) {
        return init_status_;
      }

      Status status = validate_input(input);
      if (!status.is_ok()) {
        return status;
      }
      const int input_image_size = static_cast<int>(input.shape.dims[2]);
      status = validate_outputs(outputs, input.shape.dims[0], input_image_size);
      if (!status.is_ok()) {
        return status;
      }

      nvinfer1::Dims4 input_dims{
          static_cast<int32_t>(input.shape.dims[0]),
          static_cast<int32_t>(input.shape.dims[1]),
          static_cast<int32_t>(input.shape.dims[2]),
          static_cast<int32_t>(input.shape.dims[3]),
      };
      if (!context_->setInputShape(kInputTensorName, input_dims)) {
        return Status::runtime_error("failed to set TensorRT input shape");
      }

      status = input_buffer_.ensure_allocated(input.byte_size());
      if (!status.is_ok()) {
        return status;
      }
      status = set_tensor_address(*context_, kInputTensorName, input_buffer_.get());
      if (!status.is_ok()) {
        return status;
      }

      for (std::size_t index = 0; index < output_buffers_.size(); ++index) {
        status = output_buffers_[index].ensure_allocated(outputs[index].byte_size());
        if (!status.is_ok()) {
          return status;
        }
        status =
            set_tensor_address(*context_, kOutputTensorNames[index], output_buffers_[index].get());
        if (!status.is_ok()) {
          return status;
        }
      }

      status = cuda_status(
          cudaMemcpyAsync(
              input_buffer_.get(),
              input.data,
              input.byte_size(),
              cudaMemcpyHostToDevice,
              stream_),
          "cudaMemcpyAsync(H2D)");
      if (!status.is_ok()) {
        return status;
      }

      if (!context_->enqueueV3(stream_)) {
        return Status::runtime_error("TensorRT enqueueV3 failed");
      }

      for (std::size_t index = 0; index < output_buffers_.size(); ++index) {
        status = cuda_status(
            cudaMemcpyAsync(
                outputs[index].data,
                output_buffers_[index].get(),
                outputs[index].byte_size(),
                cudaMemcpyDeviceToHost,
                stream_),
            "cudaMemcpyAsync(D2H)");
        if (!status.is_ok()) {
          return status;
        }
      }

      return cuda_status(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
    } catch (const std::exception& exc) {
      return Status::runtime_error(std::string("TRTInferer::infer exception: ") + exc.what());
    } catch (...) {
      return Status::runtime_error("TRTInferer::infer unknown exception");
    }
  }

 private:
  [[nodiscard]] Status initialize() {
    Status status = cuda_status(cudaSetDevice(device_id_), "cudaSetDevice");
    if (!status.is_ok()) {
      return status;
    }

    const std::vector<char> engine_data = read_binary_file(engine_path_);
    if (engine_data.empty()) {
      return Status::not_found("failed to read engine file: " + engine_path_);
    }

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) {
      return Status::runtime_error("failed to create TensorRT runtime");
    }

    engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    if (!engine_) {
      return Status::runtime_error("failed to deserialize TensorRT engine: " + engine_path_);
    }

    context_.reset(engine_->createExecutionContext());
    if (!context_) {
      return Status::runtime_error("failed to create TensorRT execution context");
    }

    status = cuda_status(cudaStreamCreate(&stream_), "cudaStreamCreate");
    if (!status.is_ok()) {
      return status;
    }

    return Status::ok();
  }

  std::string engine_path_;
  int device_id_{0};
  TrtLogger logger_;
  Status init_status_{Status::runtime_error("TRTInferer has not been initialized")};
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  DeviceBuffer input_buffer_;
  std::array<DeviceBuffer, kOutputCount> output_buffers_;
  cudaStream_t stream_{nullptr};
};

TRTInferer::TRTInferer(std::string engine_path, int device_id)
    : impl_(std::make_unique<Impl>(std::move(engine_path), device_id)) {}

TRTInferer::~TRTInferer() = default;

TRTInferer::TRTInferer(TRTInferer&&) noexcept = default;

TRTInferer& TRTInferer::operator=(TRTInferer&&) noexcept = default;

Status TRTInferer::infer(
    const TensorView& input,
    std::array<TensorView, kOutputCount>& outputs) noexcept {
  if (!impl_) {
    return Status::runtime_error("TRTInferer moved-from instance cannot run inference");
  }
  return impl_->infer(input, outputs);
}

}  // namespace dinov3_trt
