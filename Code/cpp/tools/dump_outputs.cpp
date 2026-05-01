#include "dinov3_trt/trt_inferer.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

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

void write_binary_file(const std::filesystem::path& path, const std::vector<float>& values) {
  std::ofstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("failed to open output file: " + path.string());
  }
  stream.write(
      reinterpret_cast<const char*>(values.data()),
      static_cast<std::streamsize>(values.size() * sizeof(float)));
  if (!stream) {
    throw std::runtime_error("failed to write output file: " + path.string());
  }
}

void print_shape_json(std::ostream& stream, const dinov3_trt::TensorShape& shape) {
  stream << '[';
  for (std::size_t index = 0; index < shape.rank; ++index) {
    if (index != 0) {
      stream << ", ";
    }
    stream << shape.dims[index];
  }
  stream << ']';
}

void write_manifest_json(
    const std::filesystem::path& manifest_path,
    const std::string& engine_path,
    const std::filesystem::path& output_dir,
    int batch_size,
    int image_size,
    const std::array<std::filesystem::path, dinov3_trt::kOutputCount>& output_files) {
  std::ofstream stream(manifest_path);
  if (!stream) {
    throw std::runtime_error("failed to open manifest: " + manifest_path.string());
  }

  const dinov3_trt::TensorShape input_shape =
      dinov3_trt::input_shape_for(batch_size, image_size);
  const dinov3_trt::TensorShape output_shape =
      dinov3_trt::output_shape_for(batch_size, image_size);
  stream << std::setprecision(10);
  stream << "{\n";
  stream << "  \"engine_path\": \"" << json_escape(engine_path) << "\",\n";
  stream << "  \"batch_size\": " << batch_size << ",\n";
  stream << "  \"image_size\": " << image_size << ",\n";
  stream << "  \"input_mode\": \"deterministic-sine\",\n";
  stream << "  \"input_shape\": ";
  print_shape_json(stream, input_shape);
  stream << ",\n";
  stream << "  \"output_dir\": \"" << json_escape(output_dir.string()) << "\",\n";
  stream << "  \"outputs\": [\n";
  for (std::size_t index = 0; index < dinov3_trt::kOutputCount; ++index) {
    stream << "    {\n";
    stream << "      \"name\": \"" << dinov3_trt::kOutputTensorNames[index] << "\",\n";
    stream << "      \"dtype\": \"float32\",\n";
    stream << "      \"shape\": ";
    print_shape_json(stream, output_shape);
    stream << ",\n";
    stream << "      \"path\": \"" << json_escape(output_files[index].filename().string())
           << "\",\n";
    stream << "      \"byte_size\": "
           << static_cast<unsigned long long>(std::filesystem::file_size(output_files[index]))
           << "\n";
    stream << "    }" << (index + 1 == dinov3_trt::kOutputCount ? "\n" : ",\n");
  }
  stream << "  ]\n";
  stream << "}\n";
  if (!stream) {
    throw std::runtime_error("failed to write manifest: " + manifest_path.string());
  }
}

}  // namespace

int main(int argc, char** argv) {
  int image_size = dinov3_trt::kImageSize;
  if (!extract_image_size_flag(&argc, argv, &image_size)) {
    return 2;
  }
  if (argc < 3 || argc > 4) {
    std::cerr
        << "usage: dinov3_trt_dump_outputs <engine_path> <output_dir> "
        << "[batch_size] [--image-size N]\n";
    return 2;
  }

  int batch_size = 1;
  if (argc == 4 && !parse_positive_int(argv[3], &batch_size)) {
    std::cerr << "batch_size must be a positive integer\n";
    return 2;
  }

  try {
    const std::string engine_path = argv[1];
    const std::filesystem::path output_dir = argv[2];
    std::filesystem::create_directories(output_dir);

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

    dinov3_trt::TRTInferer inferer(engine_path);
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

    std::array<std::filesystem::path, dinov3_trt::kOutputCount> output_files;
    for (std::size_t index = 0; index < output_buffers.size(); ++index) {
      output_files[index] =
          output_dir / (std::string(dinov3_trt::kOutputTensorNames[index]) + ".float32.bin");
      write_binary_file(output_files[index], output_buffers[index]);
    }
    const std::filesystem::path manifest_path = output_dir / "manifest.json";
    write_manifest_json(
        manifest_path, engine_path, output_dir, batch_size, image_size, output_files);
    std::cout << manifest_path.string() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

