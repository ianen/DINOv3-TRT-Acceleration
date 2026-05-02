// V1.0.2 ADR-012 · CUDA Graph cache implementation.

#include "dinov3_trt/cuda_graph_pool.h"

#include <string>

namespace dinov3_trt {

namespace {

[[nodiscard]] Status cuda_to_status(cudaError_t err, const char* context) noexcept {
  if (err == cudaSuccess) {
    return Status::ok();
  }
  return Status::runtime_error(std::string{context} + ": " + cudaGetErrorString(err));
}

}  // namespace

cudaGraphExec_t CudaGraphPool::get(const GraphKey& key) noexcept {
  const auto it = entries_.find(key);
  if (it == entries_.end()) {
    return nullptr;
  }
  // Move the entry to the front (MRU).
  lru_.splice(lru_.begin(), lru_, it->second);
  // it->second is still valid after splice (iterators stable for std::list).
  return it->second->exec;
}

Status CudaGraphPool::capture_and_insert(
    const GraphKey& key,
    cudaStream_t capture_stream,
    const GraphCaptureFn& capture_fn,
    cudaGraphExec_t& out_exec) noexcept {
  if (capture_stream == nullptr) {
    return Status::invalid_argument("capture_stream must not be null");
  }
  if (!capture_fn) {
    return Status::invalid_argument("capture_fn must not be empty");
  }

  Status status = cuda_to_status(
      cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal),
      "cudaStreamBeginCapture");
  if (!status.is_ok()) {
    return status;
  }

  // Run user-supplied work on the captured stream. If user code returns an
  // error we still must end capture to leave the stream in a usable state.
  Status capture_status = capture_fn(capture_stream);

  cudaGraph_t graph{nullptr};
  Status end_status = cuda_to_status(
      cudaStreamEndCapture(capture_stream, &graph), "cudaStreamEndCapture");

  if (!capture_status.is_ok()) {
    if (graph != nullptr) {
      static_cast<void>(cudaGraphDestroy(graph));
    }
    return capture_status;
  }
  if (!end_status.is_ok()) {
    if (graph != nullptr) {
      static_cast<void>(cudaGraphDestroy(graph));
    }
    return end_status;
  }

  cudaGraphExec_t exec{nullptr};
  Status inst_status = cuda_to_status(
      cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0),
      "cudaGraphInstantiate");
  // The graph template is no longer needed after instantiation.
  static_cast<void>(cudaGraphDestroy(graph));
  if (!inst_status.is_ok()) {
    return inst_status;
  }

  // Evict if at capacity.
  if (entries_.size() >= max_cached_) {
    evict_lru();
  }

  lru_.push_front(Entry{key, exec});
  entries_.emplace(key, lru_.begin());
  out_exec = exec;
  return Status::ok();
}

void CudaGraphPool::evict_lru() noexcept {
  if (lru_.empty()) {
    return;
  }
  Entry& back = lru_.back();
  if (back.exec != nullptr) {
    static_cast<void>(cudaGraphExecDestroy(back.exec));
    back.exec = nullptr;
  }
  entries_.erase(back.key);
  lru_.pop_back();
}

void CudaGraphPool::clear() noexcept {
  for (auto& entry : lru_) {
    if (entry.exec != nullptr) {
      static_cast<void>(cudaGraphExecDestroy(entry.exec));
      entry.exec = nullptr;
    }
  }
  lru_.clear();
  entries_.clear();
}

}  // namespace dinov3_trt
