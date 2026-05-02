#pragma once

// V1.0.2 ADR-012 · Pinned (page-locked) host memory RAII wrapper.
//
// Replaces malloc-backed pageable host buffers with cudaMallocHost-backed
// pinned buffers. Pinned memory enables true async H2D/D2H overlap with GPU
// compute and roughly halves PCIe DMA staging cost.
//
// Lifetime: identical to the project's existing DeviceBuffer pattern in
// trt_inferer.cpp anonymous namespace (RAII, no copy, move-only). The
// allocation is lazy via ensure_allocated().

#include <cstddef>

#include "dinov3_trt/status.h"

namespace dinov3_trt {

class PinnedBuffer {
 public:
  PinnedBuffer() = default;
  ~PinnedBuffer() { release(); }

  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;

  PinnedBuffer(PinnedBuffer&& other) noexcept;
  PinnedBuffer& operator=(PinnedBuffer&& other) noexcept;

  // Allocate (or reuse) a pinned host buffer of the requested size.
  // No-op when ptr_ already matches bytes. Reallocates when size differs.
  [[nodiscard]] Status ensure_allocated(std::size_t bytes) noexcept;

  // Free the pinned allocation. Idempotent.
  void release() noexcept;

  [[nodiscard]] void* host_ptr() const noexcept { return host_ptr_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }
  [[nodiscard]] bool empty() const noexcept { return host_ptr_ == nullptr; }

 private:
  void* host_ptr_{nullptr};
  std::size_t bytes_{0};
};

}  // namespace dinov3_trt
