#pragma once

// V1.0.2 ADR-012 · CUDA Graph capture / replay cache.
//
// Eliminates per-inference kernel-launch overhead (~50–200 μs at low batch)
// by capturing the full H2D + enqueueV3 + D2H sequence once and replaying
// the recorded graph on subsequent calls. The pool keys captures by
// (batch_size, resolution) because graph capture binds to the exact shapes
// of the captured calls — different batch or resolution requires a fresh
// capture.
//
// Capacity is bounded (default 6 = 2 resolutions × 3 batch sizes) and the
// pool evicts least-recently-used entries when full.

#include <cstddef>
#include <cuda_runtime.h>
#include <functional>
#include <list>
#include <unordered_map>

#include "dinov3_trt/status.h"

namespace dinov3_trt {

struct GraphKey {
  int batch_size{0};
  int resolution{0};

  bool operator==(const GraphKey& other) const noexcept {
    return batch_size == other.batch_size && resolution == other.resolution;
  }
};

struct GraphKeyHash {
  std::size_t operator()(const GraphKey& key) const noexcept {
    // Deterministic combine; resolution rarely exceeds 1024, batch rarely
    // exceeds 64, so a 32-bit shifted xor is collision-free in practice.
    return (static_cast<std::size_t>(key.resolution) << 16) ^
           static_cast<std::size_t>(key.batch_size);
  }
};

// Function signature for the capture body: callee performs H2D + enqueueV3
// + D2H operations on the supplied stream. The pool wraps this in a
// cudaStreamBeginCapture / cudaStreamEndCapture pair.
using GraphCaptureFn = std::function<Status(cudaStream_t)>;

class CudaGraphPool {
 public:
  explicit CudaGraphPool(std::size_t max_cached = 6) noexcept
      : max_cached_(max_cached) {}
  ~CudaGraphPool() { clear(); }

  CudaGraphPool(const CudaGraphPool&) = delete;
  CudaGraphPool& operator=(const CudaGraphPool&) = delete;

  // Returns a cached cudaGraphExec_t for the given key, or nullptr if not
  // cached. On hit, the entry is moved to the front of the LRU list.
  [[nodiscard]] cudaGraphExec_t get(const GraphKey& key) noexcept;

  // Capture a new graph by invoking ``capture_fn`` between
  // cudaStreamBeginCapture / cudaStreamEndCapture, instantiate it, and
  // insert into the cache. On error returns a non-ok Status; the pool is
  // unchanged. On success ``out_exec`` is set and the entry occupies the
  // most-recently-used slot (evicting the LRU entry if at capacity).
  [[nodiscard]] Status capture_and_insert(
      const GraphKey& key,
      cudaStream_t capture_stream,
      const GraphCaptureFn& capture_fn,
      cudaGraphExec_t& out_exec) noexcept;

  // Drop all cached graph executables; idempotent.
  void clear() noexcept;

  [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }
  [[nodiscard]] std::size_t max_cached() const noexcept { return max_cached_; }

 private:
  struct Entry {
    GraphKey key;
    cudaGraphExec_t exec{nullptr};
  };

  void evict_lru() noexcept;

  std::size_t max_cached_;
  // LRU list: front = most recent, back = oldest (eviction target).
  std::list<Entry> lru_;
  // Index into lru_ for O(1) lookup.
  std::unordered_map<GraphKey, std::list<Entry>::iterator, GraphKeyHash>
      entries_;
};

}  // namespace dinov3_trt
