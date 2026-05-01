#pragma once

#include <array>
#include <memory>
#include <string>

#include "dinov3_trt/status.h"
#include "dinov3_trt/tensor.h"

namespace dinov3_trt {

class TRTInferer {
 public:
  explicit TRTInferer(std::string engine_path, int device_id = 0);
  ~TRTInferer();

  TRTInferer(const TRTInferer&) = delete;
  TRTInferer& operator=(const TRTInferer&) = delete;
  TRTInferer(TRTInferer&&) noexcept;
  TRTInferer& operator=(TRTInferer&&) noexcept;

  [[nodiscard]] Status infer(
      const TensorView& input,
      std::array<TensorView, kOutputCount>& outputs) noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace dinov3_trt
