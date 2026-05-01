#include "dinov3_trt/trt_inferer.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct LatencySummary {
  double min_ms{0.0};
  double mean_ms{0.0};
  double median_ms{0.0};
  double p90_ms{0.0};
  double p95_ms{0.0};
  double p99_ms{0.0};
  double max_ms{0.0};
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
      ++src;
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

[[nodiscard]] double percentile(const std::vector<double>& sorted_values, double p) {
  if (sorted_values.empty()) {
    return 0.0;
  }
  const double raw_index = (p / 100.0) * static_cast<double>(sorted_values.size() - 1);
  const std::size_t lower = static_cast<std::size_t>(std::floor(raw_index));
  const std::size_t upper = static_cast<std::size_t>(std::ceil(raw_index));
  if (lower == upper) {
    return sorted_values[lower];
  }
  const double fraction = raw_index - static_cast<double>(lower);
  return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction;
}

[[nodiscard]] LatencySummary summarize_latencies(std::vector<double> latencies_ms) {
  LatencySummary summary;
  if (latencies_ms.empty()) {
    return summary;
  }

  std::sort(latencies_ms.begin(), latencies_ms.end());
  summary.min_ms = latencies_ms.front();
  summary.max_ms = latencies_ms.back();
  summary.median_ms = percentile(latencies_ms, 50.0);
  summary.p90_ms = percentile(latencies_ms, 90.0);
  summary.p95_ms = percentile(latencies_ms, 95.0);
  summary.p99_ms = percentile(latencies_ms, 99.0);
  const double sum = std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0);
  summary.mean_ms = sum / static_cast<double>(latencies_ms.size());
  return summary;
}

[[nodiscard]] bool all_finite(const std::array<std::vector<float>, dinov3_trt::kOutputCount>& outputs) {
  for (const auto& output : outputs) {
    for (const float value : output) {
      if (!std::isfinite(value)) {
        return false;
      }
    }
  }
  return true;
}

void print_benchmark_json(
    const std::string& engine_path,
    int batch_size,
    int image_size,
    int warmup_iterations,
    int iterations,
    double total_wall_ms,
    const LatencySummary& latency,
    bool outputs_are_finite) {
  const double throughput_qps =
      total_wall_ms <= 0.0 ? 0.0 : (static_cast<double>(iterations) * 1000.0) / total_wall_ms;

  std::cout << std::setprecision(10);
  std::cout << "{\n";
  std::cout << "  \"engine_path\": \"" << json_escape(engine_path) << "\",\n";
  std::cout << "  \"batch_size\": " << batch_size << ",\n";
  std::cout << "  \"image_size\": " << image_size << ",\n";
  std::cout << "  \"warmup_iterations\": " << warmup_iterations << ",\n";
  std::cout << "  \"iterations\": " << iterations << ",\n";
  std::cout << "  \"operation\": \"TRTInferer::infer host_to_device_enqueue_device_to_host\",\n";
  std::cout << "  \"input_shape\": ";
  print_shape(dinov3_trt::input_shape_for(batch_size, image_size));
  std::cout << ",\n";
  std::cout << "  \"output_shape\": ";
  print_shape(dinov3_trt::output_shape_for(batch_size, image_size));
  std::cout << ",\n";
  std::cout << "  \"outputs_all_finite\": " << (outputs_are_finite ? "true" : "false") << ",\n";
  std::cout << "  \"total_wall_ms\": " << total_wall_ms << ",\n";
  std::cout << "  \"throughput_qps\": " << throughput_qps << ",\n";
  std::cout << "  \"latency_ms\": {\n";
  std::cout << "    \"min\": " << latency.min_ms << ",\n";
  std::cout << "    \"mean\": " << latency.mean_ms << ",\n";
  std::cout << "    \"median\": " << latency.median_ms << ",\n";
  std::cout << "    \"p90\": " << latency.p90_ms << ",\n";
  std::cout << "    \"p95\": " << latency.p95_ms << ",\n";
  std::cout << "    \"p99\": " << latency.p99_ms << ",\n";
  std::cout << "    \"max\": " << latency.max_ms << "\n";
  std::cout << "  }\n";
  std::cout << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  int image_size = dinov3_trt::kImageSize;
  if (!extract_image_size_flag(&argc, argv, &image_size)) {
    return 2;
  }
  if (argc < 2 || argc > 5) {
    std::cerr
        << "usage: dinov3_trt_runtime_benchmark <engine_path> [batch_size] "
        << "[warmup_iterations] [iterations] [--image-size N]\n";
    return 2;
  }

  int batch_size = 1;
  int warmup_iterations = 10;
  int iterations = 50;
  if (argc >= 3 && !parse_positive_int(argv[2], &batch_size)) {
    std::cerr << "batch_size must be a positive integer\n";
    return 2;
  }
  if (argc >= 4 && !parse_positive_int(argv[3], &warmup_iterations)) {
    std::cerr << "warmup_iterations must be a positive integer\n";
    return 2;
  }
  if (argc >= 5 && !parse_positive_int(argv[4], &iterations)) {
    std::cerr << "iterations must be a positive integer\n";
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

  for (int index = 0; index < warmup_iterations; ++index) {
    const dinov3_trt::Status status = inferer.infer(input_view, output_views);
    if (!status.is_ok()) {
      std::cerr << status.message() << '\n';
      return 1;
    }
  }

  std::vector<double> latencies_ms;
  latencies_ms.reserve(static_cast<std::size_t>(iterations));
  const auto total_start = std::chrono::steady_clock::now();
  for (int index = 0; index < iterations; ++index) {
    const auto start = std::chrono::steady_clock::now();
    const dinov3_trt::Status status = inferer.infer(input_view, output_views);
    const auto end = std::chrono::steady_clock::now();
    if (!status.is_ok()) {
      std::cerr << status.message() << '\n';
      return 1;
    }
    const std::chrono::duration<double, std::milli> elapsed = end - start;
    latencies_ms.push_back(elapsed.count());
  }
  const auto total_end = std::chrono::steady_clock::now();
  const std::chrono::duration<double, std::milli> total_elapsed = total_end - total_start;

  print_benchmark_json(
      argv[1],
      batch_size,
      image_size,
      warmup_iterations,
      iterations,
      total_elapsed.count(),
      summarize_latencies(latencies_ms),
      all_finite(output_buffers));
  return 0;
}
