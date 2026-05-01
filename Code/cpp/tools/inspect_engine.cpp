#include "dinov3_trt/engine_metadata.h"

#include <iostream>
#include <string>
#include <string_view>

namespace {

[[nodiscard]] std::string json_escape(std::string_view value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (const char character : value) {
    switch (character) {
      case '\\':
        escaped += "\\\\";
        break;
      case '"':
        escaped += "\\\"";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        escaped += character;
        break;
    }
  }
  return escaped;
}

void print_json(const dinov3_trt::EngineMetadata& metadata) {
  std::cout << "{\n";
  std::cout << "  \"engine_path\": \"" << json_escape(metadata.engine_path) << "\",\n";
  std::cout << "  \"bindings\": [\n";
  for (std::size_t binding_index = 0; binding_index < metadata.bindings.size(); ++binding_index) {
    const auto& binding = metadata.bindings[binding_index];
    std::cout << "    {\n";
    std::cout << "      \"name\": \"" << json_escape(binding.name) << "\",\n";
    std::cout << "      \"mode\": \""
              << (binding.mode == dinov3_trt::TensorIOMode::kInput ? "input" : "output")
              << "\",\n";
    std::cout << "      \"data_type\": \"" << json_escape(binding.data_type) << "\",\n";
    std::cout << "      \"dims\": [";
    for (std::size_t dim_index = 0; dim_index < binding.dims.size(); ++dim_index) {
      if (dim_index != 0) {
        std::cout << ", ";
      }
      std::cout << binding.dims[dim_index];
    }
    std::cout << "]\n";
    std::cout << "    }" << (binding_index + 1 == metadata.bindings.size() ? "\n" : ",\n");
  }
  std::cout << "  ]\n";
  std::cout << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: dinov3_trt_inspect_engine <engine_path>\n";
    return 2;
  }

  const dinov3_trt::EngineInspectionResult result = dinov3_trt::inspect_engine_file(argv[1]);
  if (!result.status.is_ok()) {
    std::cerr << result.status.message() << '\n';
    return 1;
  }

  print_json(result.metadata);
  return 0;
}
