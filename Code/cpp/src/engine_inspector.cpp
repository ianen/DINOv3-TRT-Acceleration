#include "dinov3_trt/engine_metadata.h"

#include <NvInfer.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace dinov3_trt {
namespace {

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

[[nodiscard]] std::string data_type_name(nvinfer1::DataType data_type) {
  switch (data_type) {
    case nvinfer1::DataType::kFLOAT:
      return "float32";
    case nvinfer1::DataType::kHALF:
      return "float16";
    case nvinfer1::DataType::kINT8:
      return "int8";
    case nvinfer1::DataType::kINT32:
      return "int32";
    case nvinfer1::DataType::kBOOL:
      return "bool";
    case nvinfer1::DataType::kUINT8:
      return "uint8";
    case nvinfer1::DataType::kFP8:
      return "fp8";
    case nvinfer1::DataType::kBF16:
      return "bf16";
    case nvinfer1::DataType::kINT64:
      return "int64";
    case nvinfer1::DataType::kINT4:
      return "int4";
    case nvinfer1::DataType::kFP4:
      return "fp4";
    case nvinfer1::DataType::kE8M0:
      return "e8m0";
  }
  return "unknown";
}

[[nodiscard]] std::vector<std::int64_t> dims_to_vector(const nvinfer1::Dims& dims) {
  std::vector<std::int64_t> values;
  if (dims.nbDims < 0) {
    return values;
  }

  values.reserve(static_cast<std::size_t>(dims.nbDims));
  for (int32_t index = 0; index < dims.nbDims; ++index) {
    values.push_back(dims.d[index]);
  }
  return values;
}

}  // namespace

EngineInspectionResult inspect_engine_file(const std::string& engine_path) {
  EngineInspectionResult result{
      Status::runtime_error("engine inspection did not run"),
      EngineMetadata{engine_path, {}},
  };

  const std::vector<char> engine_data = read_binary_file(engine_path);
  if (engine_data.empty()) {
    result.status = Status::not_found("failed to read engine file: " + engine_path);
    return result;
  }

  TrtLogger logger;
  std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
  if (!runtime) {
    result.status = Status::runtime_error("failed to create TensorRT runtime");
    return result;
  }

  std::unique_ptr<nvinfer1::ICudaEngine> engine(
      runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
  if (!engine) {
    result.status = Status::runtime_error("failed to deserialize TensorRT engine: " + engine_path);
    return result;
  }

  const int32_t tensor_count = engine->getNbIOTensors();
  result.metadata.bindings.reserve(static_cast<std::size_t>(tensor_count));
  for (int32_t index = 0; index < tensor_count; ++index) {
    const char* name = engine->getIOTensorName(index);
    const nvinfer1::TensorIOMode mode = engine->getTensorIOMode(name);
    result.metadata.bindings.push_back(TensorBindingMetadata{
        name,
        mode == nvinfer1::TensorIOMode::kINPUT ? TensorIOMode::kInput : TensorIOMode::kOutput,
        data_type_name(engine->getTensorDataType(name)),
        dims_to_vector(engine->getTensorShape(name)),
    });
  }

  result.status = Status::ok();
  return result;
}

}  // namespace dinov3_trt
