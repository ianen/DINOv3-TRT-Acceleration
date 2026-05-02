// V1.0.2 ADR-012 · Pinned host memory implementation.
//
// Mirrors the in-anonymous-namespace DeviceBuffer pattern from trt_inferer.cpp
// but allocates page-locked host memory via cudaMallocHost. Pinned buffers
// allow truly async cudaMemcpyAsync(H2D/D2H) overlap with kernel execution
// and roughly halve the staging cost (driver no longer copies through an
// intermediate pinned scratch).

#include "dinov3_trt/pinned_buffer.h"

#include <cuda_runtime.h>

#include <utility>

namespace dinov3_trt {

namespace {

// Translate a cudaError_t to a project Status using the same conventions as
// trt_inferer.cpp's cuda_status helper. Local copy to keep this header-light
// and avoid pulling that anonymous helper into a public header.
[[nodiscard]] Status cuda_to_status(cudaError_t err, const char* context) noexcept {
  if (err == cudaSuccess) {
    return Status::ok();
  }
  return Status::runtime_error(std::string{context} + ": " + cudaGetErrorString(err));
}

}  // namespace

PinnedBuffer::PinnedBuffer(PinnedBuffer&& other) noexcept
    : host_ptr_(other.host_ptr_), bytes_(other.bytes_) {
  other.host_ptr_ = nullptr;
  other.bytes_ = 0;
}

PinnedBuffer& PinnedBuffer::operator=(PinnedBuffer&& other) noexcept {
  if (this != &other) {
    release();
    host_ptr_ = other.host_ptr_;
    bytes_ = other.bytes_;
    other.host_ptr_ = nullptr;
    other.bytes_ = 0;
  }
  return *this;
}

Status PinnedBuffer::ensure_allocated(std::size_t bytes) noexcept {
  if (bytes == 0) {
    return Status::invalid_argument("cannot allocate zero-byte pinned buffer");
  }
  if (host_ptr_ != nullptr && bytes_ == bytes) {
    return Status::ok();
  }
  release();

  void* ptr = nullptr;
  // cudaHostAllocDefault gives portable, async-capable pinned memory on the
  // current device. Use cudaHostAllocPortable if the buffer is shared across
  // devices in the future.
  const Status status = cuda_to_status(cudaMallocHost(&ptr, bytes), "cudaMallocHost");
  if (!status.is_ok()) {
    return status;
  }
  host_ptr_ = ptr;
  bytes_ = bytes;
  return Status::ok();
}

void PinnedBuffer::release() noexcept {
  if (host_ptr_ != nullptr) {
    static_cast<void>(cudaFreeHost(host_ptr_));
    host_ptr_ = nullptr;
    bytes_ = 0;
  }
}

}  // namespace dinov3_trt
