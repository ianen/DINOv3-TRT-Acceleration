#pragma once

// V1.0.3 ADR-020 · TensorRT multi-context engine pool.
//
// Reuses one shared `nvinfer1::ICudaEngine` (deserialize once, share across
// N concurrent execution contexts), with one `cudaStream_t`,
// one `IExecutionContext`, one `PinnedBuffer` set, and one `CudaGraphPool`
// per slot. Permit-gated capacity (`std::counting_semaphore`) ensures we
// never over-subscribe the pool; per-slot mutex provides defense-in-depth
// against concurrent `enqueueV3` despite TRT documenting non-modifying
// engine ops as thread-safe.
//
// Use this pool to scale request throughput in low-load regimes (small
// batch / small resolution) where single-context inference leaves SM idle
// time between launches. Empirically (V1.0.3 G7 datapoint 2026-05-02) the
// r224 b1 BF16 prefer regime has ~12 percentage points of SM headroom; the
// pool exists to convert that headroom into concurrent QPS.
//
// At saturation regimes (r336 b8 96.39% SM, r518 b8 99.08% SM) the pool
// will not produce additional throughput — the SMs are already 100% busy.
// This is documented behavior, not a defect.

#include <array>
#include <cstddef>
#include <memory>
#include <string>

#include "dinov3_trt/status.h"
#include "dinov3_trt/tensor.h"

namespace dinov3_trt {

class TRTInfererPool {
 public:
  struct Config {
    // Path to the serialized TRT engine. Must be openable for read.
    std::string engine_path;

    // Number of concurrent execution contexts (= cuda streams = pool size).
    // V1.0.2 ADR-015 + V1.0.3 G7 evidence: 2 is the sweet spot at r224 b1.
    // 1 degenerates to single-context; 4+ tends to regress (engine memory
    // pressure + scheduling overhead exceeds the gains from filling
    // remaining SM idle time).
    int num_streams{2};

    // CUDA device id for engine + contexts. All slots share one device.
    int device_id{0};

    // Toggle per-slot CudaGraphPool. False matches V1.0.1 baseline behavior;
    // true matches V1.0.2 ADR-012 (1.135× r224 b1, bit-exact). Independent
    // of the global DINOV3_USE_CUDA_GRAPH env var so callers can A/B test
    // without environment juggling.
    bool enable_cuda_graphs{true};

    // Maximum graphs cached per slot. (resolutions × batch sizes the pool
    // expects to see). Default 6 = 2 res × 3 batch matches V1.0.2 ADR-012.
    std::size_t graph_cache_size{6};
  };

  explicit TRTInfererPool(Config cfg);
  ~TRTInfererPool();

  TRTInfererPool(const TRTInfererPool&) = delete;
  TRTInfererPool& operator=(const TRTInfererPool&) = delete;

  TRTInfererPool(TRTInfererPool&&) noexcept;
  TRTInfererPool& operator=(TRTInfererPool&&) noexcept;

  // Submit one inference. Blocks until a slot is free, copies input H2D,
  // captures or replays the per-slot CUDA graph (if enabled), copies
  // outputs D2H, and releases the slot. Thread-safe: any number of caller
  // threads may invoke `infer` concurrently — submissions are dispatched
  // round-robin across the pool.
  //
  // Returns non-ok Status on the first failing CUDA / TRT call. The slot
  // is released even on error. Output `TensorView`s must be pre-allocated
  // by the caller with byte sizes matching this engine's contract (4 outputs
  // of `[B, output_tokens_for(image_size), kHiddenSize]` BF16 / FP32).
  [[nodiscard]] Status infer(
      const TensorView& input,
      std::array<TensorView, kOutputCount>& outputs) noexcept;

  // Configured pool size. Constant for the lifetime of this pool.
  [[nodiscard]] int num_streams() const noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace dinov3_trt
