#include "dinov3_trt/trt_inferer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct TensorStats {
  double min{0.0};
  double max{0.0};
  double mean{0.0};
  double rms{0.0};
  std::size_t element_count{0};
  std::size_t finite_count{0};
};

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

[[nodiscard]] bool parse_positive_int(const char* value, int* parsed) {
  char* end = nullptr;
  const long result = std::strtol(value, &end, 10);
  if (end == value || *end != '\0' || result < 1 ||
      result > std::numeric_limits<int>::max()) {
    return false;
  }
  *parsed = static_cast<int>(result);
  return true;
}

// Extract a `--image-size N` (or `--image-size=N`) flag from argv in-place.
// Returns true on success and rewrites argc/argv to drop the matched tokens.
// The flag is optional; absence leaves *image_size unchanged.
[[nodiscard]] bool extract_image_size_flag(int* argc_inout, char** argv, int* image_size) {
  int dst = 1;
  for (int src = 1; src < *argc_inout; ++src) {
    const std::string_view token{argv[src]};
    if (token == "--image-size") {
      if (src + 1 >= *argc_inout) {
        std::cerr << "--image-size requires a value\n";
        return false;
      }
      int parsed = 0;
      if (!parse_positive_int(argv[src + 1], &parsed)) {
        std::cerr << "--image-size value must be a positive integer\n";
        return false;
      }
      *image_size = parsed;
      ++src;  // Skip the value token as well.
      continue;
    }
    if (token.rfind("--image-size=", 0) == 0) {
      int parsed = 0;
      if (!parse_positive_int(argv[src] + std::string_view{"--image-size="}.size(), &parsed)) {
        std::cerr << "--image-size value must be a positive integer\n";
        return false;
      }
      *image_size = parsed;
      continue;
    }
    argv[dst++] = argv[src];
  }
  *argc_inout = dst;
  return true;
}

void fill_deterministic_input(std::vector<float>& input) {
  for (std::size_t index = 0; index < input.size(); ++index) {
    const double phase = static_cast<double>((index % 1009U) + 1U) * 0.017;
    input[index] = static_cast<float>(std::sin(phase));
  }
}

[[nodiscard]] TensorStats compute_stats(const std::vector<float>& values) {
  TensorStats stats;
  stats.element_count = values.size();
  if (values.empty()) {
    return stats;
  }

  double min_value = std::numeric_limits<double>::infinity();
  double max_value = -std::numeric_limits<double>::infinity();
  double sum = 0.0;
  double squared_sum = 0.0;
  for (const float value : values) {
    if (!std::isfinite(value)) {
      continue;
    }
    const double current = static_cast<double>(value);
    min_value = std::min(min_value, current);
    max_value = std::max(max_value, current);
    sum += current;
    squared_sum += current * current;
    ++stats.finite_count;
  }

  if (stats.finite_count == 0) {
    return stats;
  }
  stats.min = min_value;
  stats.max = max_value;
  stats.mean = sum / static_cast<double>(stats.finite_count);
  stats.rms = std::sqrt(squared_sum / static_cast<double>(stats.finite_count));
  return stats;
}

void print_shape(const dinov3_trt::TensorShape& shape) {
  std::cout << '[';
  for (std::size_t index = 0; index < shape.rank; ++index) {
    if (index != 0) {
      std::cout << ", ";
    }
    std::cout << shape.dims[index];
  }
  std::cout << ']';
}

void print_stats_json(
    const std::string& engine_path,
    int batch_size,
    int image_size,
    const std::array<std::vector<float>, dinov3_trt::kOutputCount>& outputs) {
  std::cout << std::setprecision(10);
  std::cout << "{\n";
  std::cout << "  \"engine_path\": \"" << json_escape(engine_path) << "\",\n";
  std::cout << "  \"batch_size\": " << batch_size << ",\n";
  std::cout << "  \"image_size\": " << image_size << ",\n";
  std::cout << "  \"input_shape\": ";
  print_shape(dinov3_trt::input_shape_for(batch_size, image_size));
  std::cout << ",\n";
  std::cout << "  \"outputs\": [\n";
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    const TensorStats stats = compute_stats(outputs[index]);
    std::cout << "    {\n";
    std::cout << "      \"name\": \"" << dinov3_trt::kOutputTensorNames[index] << "\",\n";
    std::cout << "      \"shape\": ";
    print_shape(dinov3_trt::output_shape_for(batch_size, image_size));
    std::cout << ",\n";
    std::cout << "      \"element_count\": " << stats.element_count << ",\n";
    std::cout << "      \"finite_count\": " << stats.finite_count << ",\n";
    std::cout << "      \"min\": " << stats.min << ",\n";
    std::cout << "      \"max\": " << stats.max << ",\n";
    std::cout << "      \"mean\": " << stats.mean << ",\n";
    std::cout << "      \"rms\": " << stats.rms << "\n";
    std::cout << "    }" << (index + 1 == outputs.size() ? "\n" : ",\n");
  }
  std::cout << "  ]\n";
  std::cout << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  int image_size = dinov3_trt::kImageSize;
  if (!extract_image_size_flag(&argc, argv, &image_size)) {
    return 2;
  }
  if (argc < 2 || argc > 3) {
    std::cerr
        << "usage: dinov3_trt_runtime_smoke <engine_path> [batch_size] "
        << "[--image-size N]\n";
    return 2;
  }

  int batch_size = 1;
  if (argc == 3 && !parse_positive_int(argv[2], &batch_size)) {
    std::cerr << "batch_size must be a positive integer\n";
    return 2;
  }

  const dinov3_trt::TensorShape input_shape =
      dinov3_trt::input_shape_for(batch_size, image_size);
  std::vector<float> input(static_cast<std::size_t>(input_shape.element_count()));
  fill_deterministic_input(input);

  std::array<std::vector<float>, dinov3_trt::kOutputCount> output_buffers;
  std::array<dinov3_trt::TensorView, dinov3_trt::kOutputCount> output_views;
  const dinov3_trt::TensorShape out_shape =
      dinov3_trt::output_shape_for(batch_size, image_size);
  const std::size_t output_elements = static_cast<std::size_t>(out_shape.element_count());
  for (std::size_t index = 0; index < output_buffers.size(); ++index) {
    output_buffers[index].resize(output_elements);
    output_views[index] = dinov3_trt::TensorView{
        output_buffers[index].data(),
        out_shape,
        dinov3_trt::DataType::kFloat32,
    };
  }

  dinov3_trt::TRTInferer inferer(argv[1]);
  dinov3_trt::TensorView input_view{
      input.data(),
      input_shape,
      dinov3_trt::DataType::kFloat32,
  };
  const dinov3_trt::Status status = inferer.infer(input_view, output_views);
  if (!status.is_ok()) {
    std::cerr << status.message() << '\n';
    return 1;
  }

  print_stats_json(argv[1], batch_size, image_size, output_buffers);
  return 0;
}
