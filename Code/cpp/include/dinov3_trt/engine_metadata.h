#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "dinov3_trt/status.h"

namespace dinov3_trt {

enum class TensorIOMode {
  kInput,
  kOutput,
};

struct TensorBindingMetadata {
  std::string name;
  TensorIOMode mode;
  std::string data_type;
  std::vector<std::int64_t> dims;
};

struct EngineMetadata {
  std::string engine_path;
  std::vector<TensorBindingMetadata> bindings;
};

struct EngineInspectionResult {
  Status status;
  EngineMetadata metadata;
};

[[nodiscard]] EngineInspectionResult inspect_engine_file(const std::string& engine_path);

}  // namespace dinov3_trt
