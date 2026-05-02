#include "dinov3_trt/trt_inferer_pool.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <semaphore>
#include <stdexcept>
#include <utility>
#include <vector>

#include "dinov3_trt/trt_inferer.h"

// V1.0.3 ADR-020 implementation — Phase 1: per-slot independent TRTInferer.
//
// This phase trades engine-memory deduplication for implementation
// simplicity and ships the public API + correctness gate first. Each slot
// owns a private TRTInferer (which deserializes its own ICudaEngine).
// Throughput-wise this still removes Python's GIL contention and provides
// the round-robin dispatch surface that ADR-021 dynamic batching will
// build on. Phase 2 (planned, not in this commit) replaces the per-slot
// TRTInferers with a single shared ICudaEngine + N IExecutionContexts to
// recover the (N-1) × engine_size_bytes of host RAM at large N.

namespace dinov3_trt {

class TRTInfererPool::Impl {
 public:
  explicit Impl(Config cfg) : config_(std::move(cfg)), available_slots_(0) {
    if (config_.num_streams < 1) {
      throw std::invalid_argument("TRTInfererPool: num_streams must be >= 1");
    }
    if (config_.engine_path.empty()) {
      throw std::invalid_argument("TRTInfererPool: engine_path is empty");
    }

    slots_.reserve(static_cast<std::size_t>(config_.num_streams));
    for (int slot_index = 0; slot_index < config_.num_streams; ++slot_index) {
      auto slot = std::make_unique<Slot>();
      slot->inferer = std::make_unique<TRTInferer>(config_.engine_path, config_.device_id);
      slots_.push_back(std::move(slot));
    }

    // Initialize semaphore with all slots free. Constructed empty above
    // because counting_semaphore template arg is the *least max value* and
    // we want a runtime-sized count.
    available_slots_.release(config_.num_streams);
  }

  [[nodiscard]] Status infer(
      const TensorView& input,
      std::array<TensorView, kOutputCount>& outputs) noexcept {
    available_slots_.acquire();
    const std::size_t slot_index = next_slot_index();

    Status status = Status::ok();
    {
      Slot& slot = *slots_[slot_index];
      std::lock_guard<std::mutex> lock(slot.mu);
      status = slot.inferer->infer(input, outputs);
    }

    available_slots_.release();
    return status;
  }

  [[nodiscard]] int num_streams() const noexcept { return config_.num_streams; }

 private:
  struct Slot {
    std::unique_ptr<TRTInferer> inferer;
    std::mutex mu;
  };

  [[nodiscard]] std::size_t next_slot_index() noexcept {
    // Round-robin via atomic counter. Modulo per-call is fine; the
    // semaphore-gated capacity ensures we won't overflow. We accept the
    // theoretical case where two threads pick the same index between
    // acquire and lock — the per-slot mutex serializes them; throughput
    // cost of that rare collision is small relative to the cost of a
    // global queue.
    const std::size_t index = dispatch_counter_.fetch_add(1, std::memory_order_relaxed);
    return index % static_cast<std::size_t>(config_.num_streams);
  }

  Config config_;
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
