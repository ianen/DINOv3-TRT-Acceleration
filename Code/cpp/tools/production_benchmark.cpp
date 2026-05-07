// V1.0.4 ADR-027 — C++ production-environment inference benchmark.
//
// Mirrors `Code/scripts/production_benchmark.py` (ADR-026) for cross-language
// parity: same 6 independent timed stages, same JSON output schema, same
// dataset (Artifacts/datasets/good_r512). The 6 stages:
//
//   1. disk_read    — std::ifstream read raw jpg bytes
//   2. jpg_decode   — stb_image::stbi_load_from_memory (RGB)
//   3. preprocess   — float / 255 + ImageNet mean/std + HWC→NCHW
//   4. h2d          — cudaMemcpyAsync H2D + cudaStreamSynchronize
//   5. enqueueV3    — context->enqueueV3 + sync
//   6. d2h          — 4 outputs cudaMemcpyAsync D2H + sync
//
// G5 acceptance: 6-stage p50 ms diff vs Python ≤ 10%; bit-exact
// inference output (cosine = 1.0) at single-image deterministic input.

#define STB_IMAGE_IMPLEMENTATION
#include "../third_party/stb/stb_image.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

namespace {

// ImageNet normalization stats (DINOv3 follows DINOv2 convention).
constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};

class TrtLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << "[TRT] " << msg << '\n';
    }
  }
};

struct CliArgs {
  std::string engine_path;
  std::string dataset_path;
  std::string output_path;
  std::string input_name = "pixel_values";
  int batch_size = 1;
  int image_size = 512;
  int warmup = 10;
  int iters = 100;
};

void print_usage(const char* prog) {
  std::cerr
      << "usage: " << prog << " --engine PATH --dataset DIR\n"
      << "  [--batch-size N] [--image-size N] [--warmup N] [--iters N]\n"
      << "  [--output PATH] [--input-name NAME]\n";
}

bool parse_args(int argc, char** argv, CliArgs& out) {
  for (int i = 1; i < argc; ++i) {
    std::string_view tok(argv[i]);
    auto need = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::cerr << name << " requires a value\n";
        return nullptr;
      }
      return argv[++i];
    };
    if (tok == "--engine") {
      auto v = need("--engine"); if (!v) return false;
      out.engine_path = v;
    } else if (tok == "--dataset") {
      auto v = need("--dataset"); if (!v) return false;
      out.dataset_path = v;
    } else if (tok == "--output") {
      auto v = need("--output"); if (!v) return false;
      out.output_path = v;
    } else if (tok == "--input-name") {
      auto v = need("--input-name"); if (!v) return false;
      out.input_name = v;
    } else if (tok == "--batch-size") {
      auto v = need("--batch-size"); if (!v) return false;
      out.batch_size = std::atoi(v);
    } else if (tok == "--image-size") {
      auto v = need("--image-size"); if (!v) return false;
      out.image_size = std::atoi(v);
    } else if (tok == "--warmup") {
      auto v = need("--warmup"); if (!v) return false;
      out.warmup = std::atoi(v);
    } else if (tok == "--iters") {
      auto v = need("--iters"); if (!v) return false;
      out.iters = std::atoi(v);
    } else if (tok == "--help" || tok == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else {
      std::cerr << "unknown arg: " << tok << "\n";
      print_usage(argv[0]);
      return false;
    }
  }
  if (out.engine_path.empty() || out.dataset_path.empty()) {
    print_usage(argv[0]);
    return false;
  }
  return true;
}

[[nodiscard]] std::vector<char> read_binary_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return {};
  std::streamsize sz = f.tellg();
  if (sz <= 0) return {};
  std::vector<char> data(static_cast<std::size_t>(sz));
  f.seekg(0);
  if (!f.read(data.data(), sz)) return {};
  return data;
}

[[nodiscard]] std::vector<std::string> list_dataset_jpgs(const std::string& dir) {
  std::vector<std::string> out;
  for (auto& ent : fs::directory_iterator(dir)) {
    if (!ent.is_regular_file()) continue;
    auto ext = ent.path().extension().string();
    if (ext == ".jpg" || ext == ".jpeg" || ext == ".JPG" || ext == ".JPEG") {
      out.push_back(ent.path().string());
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

struct StageTimings {
  std::vector<double> disk_read, jpg_decode, preprocess, h2d, enqueue_v3, d2h, total;

  void append(double dr, double jd, double pp, double h, double e, double d) {
    disk_read.push_back(dr);
    jpg_decode.push_back(jd);
    preprocess.push_back(pp);
    h2d.push_back(h);
    enqueue_v3.push_back(e);
    d2h.push_back(d);
    total.push_back(dr + jd + pp + h + e + d);
  }
  void reserve(std::size_t n) {
    disk_read.reserve(n);
    jpg_decode.reserve(n);
    preprocess.reserve(n);
    h2d.reserve(n);
    enqueue_v3.reserve(n);
    d2h.reserve(n);
    total.reserve(n);
  }
};

[[nodiscard]] double percentile(std::vector<double> values, double p) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  if (p <= 0.0) return values.front();
  if (p >= 100.0) return values.back();
  const double r = (p / 100.0) * static_cast<double>(values.size() - 1);
  const std::size_t lo = static_cast<std::size_t>(std::floor(r));
  const std::size_t hi = static_cast<std::size_t>(std::ceil(r));
  if (lo == hi) return values[lo];
  return values[lo] + (values[hi] - values[lo]) * (r - static_cast<double>(lo));
}

[[nodiscard]] double meanv(const std::vector<double>& v) {
  if (v.empty()) return 0.0;
  return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

[[nodiscard]] double maxv(const std::vector<double>& v) {
  if (v.empty()) return 0.0;
  return *std::max_element(v.begin(), v.end());
}

void write_json_summary(const CliArgs& args, const StageTimings& T,
                        double wall_s, double agg_qps,
                        const std::vector<std::string>& output_names,
                        const std::string& path) {
  std::ofstream f(path);
  if (!f) {
    std::cerr << "failed to open output: " << path << "\n";
    return;
  }
  f << std::setprecision(10);
  f << "{\n";
  f << "  \"engine\": \"" << args.engine_path << "\",\n";
  f << "  \"dataset\": \"" << args.dataset_path << "\",\n";
  f << "  \"batch_size\": " << args.batch_size << ",\n";
  f << "  \"image_size\": " << args.image_size << ",\n";
  f << "  \"warmup\": " << args.warmup << ",\n";
  f << "  \"iters\": " << args.iters << ",\n";
  f << "  \"language\": \"cpp\",\n";
  f << "  \"total_inferences\": " << args.iters << ",\n";
  f << "  \"total_images\": " << (args.iters * args.batch_size) << ",\n";
  f << "  \"wall_elapsed_s\": " << wall_s << ",\n";
  f << "  \"aggregate_qps_imgs_per_sec\": " << agg_qps << ",\n";

  auto write_stage_block = [&](const char* tag, auto fn) {
    f << "  \"stages_" << tag << "_ms\": {\n";
    f << "    \"disk_read\": " << fn(T.disk_read) << ",\n";
    f << "    \"jpg_decode\": " << fn(T.jpg_decode) << ",\n";
    f << "    \"preprocess\": " << fn(T.preprocess) << ",\n";
    f << "    \"h2d\": " << fn(T.h2d) << ",\n";
    f << "    \"enqueueV3\": " << fn(T.enqueue_v3) << ",\n";
    f << "    \"d2h\": " << fn(T.d2h) << ",\n";
    f << "    \"total\": " << fn(T.total) << "\n";
    f << "  },\n";
  };
  write_stage_block("p50", [](const std::vector<double>& v) { return percentile(v, 50.0); });
  write_stage_block("p95", [](const std::vector<double>& v) { return percentile(v, 95.0); });
  write_stage_block("mean", [](const std::vector<double>& v) { return meanv(v); });
  write_stage_block("max", [](const std::vector<double>& v) { return maxv(v); });

  auto write_array = [&](const char* name, const std::vector<double>& v, bool last = false) {
    f << "    \"" << name << "\": [";
    for (std::size_t i = 0; i < v.size(); ++i) {
      if (i > 0) f << ", ";
      f << v[i];
    }
    f << "]" << (last ? "" : ",") << "\n";
  };
  f << "  \"per_image_trace\": {\n";
  write_array("disk_read_ms", T.disk_read);
  write_array("jpg_decode_ms", T.jpg_decode);
  write_array("preprocess_ms", T.preprocess);
  write_array("h2d_ms", T.h2d);
  write_array("enqueueV3_ms", T.enqueue_v3);
  write_array("d2h_ms", T.d2h);
  write_array("total_ms", T.total, true);
  f << "  },\n";

  f << "  \"output_names\": [";
  for (std::size_t i = 0; i < output_names.size(); ++i) {
    if (i > 0) f << ", ";
    f << "\"" << output_names[i] << "\"";
  }
  f << "]\n";
  f << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  CliArgs args;
  if (!parse_args(argc, argv, args)) return 2;

  std::cerr << "[production_benchmark] engine = " << args.engine_path << "\n";
  std::cerr << "[production_benchmark] dataset = " << args.dataset_path << "\n";
  std::cerr << "[production_benchmark] batch=" << args.batch_size
            << " image_size=" << args.image_size
            << " warmup=" << args.warmup << " iters=" << args.iters << "\n";

  // --- Engine load ---
  TrtLogger logger;
  auto engine_blob = read_binary_file(args.engine_path);
  if (engine_blob.empty()) {
    std::cerr << "failed to read engine: " << args.engine_path << "\n";
    return 1;
  }
  std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
  if (!runtime) { std::cerr << "createInferRuntime failed\n"; return 1; }
  std::unique_ptr<nvinfer1::ICudaEngine> engine{
      runtime->deserializeCudaEngine(engine_blob.data(), engine_blob.size())};
  if (!engine) { std::cerr << "deserializeCudaEngine failed\n"; return 1; }
  std::unique_ptr<nvinfer1::IExecutionContext> context{engine->createExecutionContext()};
  if (!context) { std::cerr << "createExecutionContext failed\n"; return 1; }

  // --- Set input shape + allocate buffers ---
  nvinfer1::Dims4 in_shape{args.batch_size, 3, args.image_size, args.image_size};
  if (!context->setInputShape(args.input_name.c_str(), in_shape)) {
    std::cerr << "setInputShape failed\n"; return 1;
  }

  cudaStream_t stream{};
  if (cudaStreamCreate(&stream) != cudaSuccess) {
    std::cerr << "cudaStreamCreate failed\n"; return 1;
  }

  std::vector<std::string> output_names;
  std::vector<void*> device_ptrs(static_cast<std::size_t>(engine->getNbIOTensors()), nullptr);
  std::vector<std::vector<float>> host_outputs;
  std::vector<std::size_t> output_byte_sizes;
  void* d_input = nullptr;
  std::size_t input_bytes = static_cast<std::size_t>(args.batch_size) * 3
                             * static_cast<std::size_t>(args.image_size)
                             * static_cast<std::size_t>(args.image_size) * sizeof(float);

  for (int i = 0; i < engine->getNbIOTensors(); ++i) {
    const char* name = engine->getIOTensorName(i);
    auto mode = engine->getTensorIOMode(name);
    auto dims = context->getTensorShape(name);
    auto dtype = engine->getTensorDataType(name);

    std::size_t elements = 1;
    for (int d = 0; d < dims.nbDims; ++d) elements *= static_cast<std::size_t>(dims.d[d]);

    // Map TRT dtype size (assume FP32 / FP16 / BF16 outputs).
    std::size_t elem_bytes = 4;  // default float32
    if (dtype == nvinfer1::DataType::kHALF) elem_bytes = 2;
    else if (dtype == nvinfer1::DataType::kBF16) elem_bytes = 2;
    std::size_t nbytes = elements * elem_bytes;

    void* dptr = nullptr;
    if (cudaMalloc(&dptr, nbytes) != cudaSuccess) {
      std::cerr << "cudaMalloc failed for " << name << "\n"; return 1;
    }
    device_ptrs[i] = dptr;
    if (!context->setTensorAddress(name, dptr)) {
      std::cerr << "setTensorAddress failed for " << name << "\n"; return 1;
    }

    if (mode == nvinfer1::TensorIOMode::kINPUT) {
      d_input = dptr;
    } else {
      output_names.emplace_back(name);
      // Allocate FP32 host buffer for D2H of any precision (BF16/FP16 outputs
      // would need conversion before use, but we just measure D2H here).
      host_outputs.emplace_back(elements);
      output_byte_sizes.push_back(nbytes);
    }
  }
  std::cerr << "[production_benchmark] outputs = ";
  for (auto& n : output_names) std::cerr << n << " "; std::cerr << "\n";

  std::vector<float> host_input(input_bytes / sizeof(float));

  // --- Dataset image list ---
  auto img_paths = list_dataset_jpgs(args.dataset_path);
  if (img_paths.empty()) {
    std::cerr << "no jpg files in " << args.dataset_path << "\n"; return 1;
  }
  std::cerr << "[production_benchmark] dataset images = " << img_paths.size() << "\n";

  // ------------------- Iteration loop --------------------------
  auto run_iteration = [&](int iter_idx, StageTimings* timings) -> bool {
    using Clock = std::chrono::steady_clock;
    using ms = std::chrono::duration<double, std::milli>;

    // Stage 1: disk_read — batch_size jpg bytes
    auto t0 = Clock::now();
    std::vector<std::vector<char>> raw_batch(static_cast<std::size_t>(args.batch_size));
    for (int b = 0; b < args.batch_size; ++b) {
      const auto& path = img_paths[(static_cast<std::size_t>(iter_idx) * args.batch_size + b)
                                    % img_paths.size()];
      raw_batch[b] = read_binary_file(path);
      if (raw_batch[b].empty()) {
        std::cerr << "failed reading " << path << "\n"; return false;
      }
    }
    auto t1 = Clock::now();

    // Stage 2: jpg_decode → uint8 RGB ndarray (HxWx3)
    std::vector<std::vector<unsigned char>> decoded_batch(args.batch_size);
    for (int b = 0; b < args.batch_size; ++b) {
      int w, h, ch;
      unsigned char* px = stbi_load_from_memory(
          reinterpret_cast<const stbi_uc*>(raw_batch[b].data()),
          static_cast<int>(raw_batch[b].size()),
          &w, &h, &ch, /*req_comp=*/3);
      if (!px) { std::cerr << "stbi decode failed\n"; return false; }
      if (w != args.image_size || h != args.image_size) {
        std::cerr << "image size mismatch: got " << w << "x" << h
                  << " expected " << args.image_size << "x" << args.image_size << "\n";
        stbi_image_free(px);
        return false;
      }
      decoded_batch[b].assign(px, px + (w * h * 3));
      stbi_image_free(px);
    }
    auto t2 = Clock::now();

    // Stage 3: preprocess — float/255 + ImageNet mean/std + HWC→NCHW
    const std::size_t pixels_per_img = static_cast<std::size_t>(args.image_size)
                                        * args.image_size;
    for (int b = 0; b < args.batch_size; ++b) {
      const auto& raw = decoded_batch[b];
      // Source HWC: pixel (y,x,c) at offset (y * W + x) * 3 + c
      // Destination NCHW: pixel (b,c,y,x) at offset b * 3 * H * W + c * H * W + y * W + x
      const std::size_t b_offset = static_cast<std::size_t>(b) * 3 * pixels_per_img;
      for (int y = 0; y < args.image_size; ++y) {
        for (int x = 0; x < args.image_size; ++x) {
          const std::size_t src_pix = (static_cast<std::size_t>(y) * args.image_size + x) * 3;
          for (int c = 0; c < 3; ++c) {
            const float val = raw[src_pix + c] / 255.0f;
            const float norm = (val - kMean[c]) / kStd[c];
            host_input[b_offset + static_cast<std::size_t>(c) * pixels_per_img
                        + static_cast<std::size_t>(y) * args.image_size + x] = norm;
          }
        }
      }
    }
    auto t3 = Clock::now();

    // Stage 4: H2D
    if (cudaMemcpyAsync(d_input, host_input.data(), input_bytes,
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
      std::cerr << "H2D memcpy failed\n"; return false;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
      std::cerr << "H2D sync failed\n"; return false;
    }
    auto t4 = Clock::now();

    // Stage 5: enqueueV3 + sync
    if (!context->enqueueV3(stream)) {
      std::cerr << "enqueueV3 failed\n"; return false;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
      std::cerr << "compute sync failed\n"; return false;
    }
    auto t5 = Clock::now();

    // Stage 6: D2H — 4 outputs
    std::size_t out_idx = 0;
    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
      const char* name = engine->getIOTensorName(i);
      auto mode = engine->getTensorIOMode(name);
      if (mode != nvinfer1::TensorIOMode::kOUTPUT) continue;
      if (cudaMemcpyAsync(host_outputs[out_idx].data(), device_ptrs[i],
                          output_byte_sizes[out_idx],
                          cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        std::cerr << "D2H memcpy failed for " << name << "\n"; return false;
      }
      ++out_idx;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
      std::cerr << "D2H sync failed\n"; return false;
    }
    auto t6 = Clock::now();

    if (timings) {
      timings->append(
          ms(t1 - t0).count(),
          ms(t2 - t1).count(),
          ms(t3 - t2).count(),
          ms(t4 - t3).count(),
          ms(t5 - t4).count(),
          ms(t6 - t5).count());
    }
    return true;
  };

  // Warmup
  std::cerr << "[production_benchmark] warmup × " << args.warmup << "...\n";
  for (int w = 0; w < args.warmup; ++w) {
    if (!run_iteration(w, nullptr)) {
      std::cerr << "warmup iter " << w << " failed\n"; return 1;
    }
  }

  // Timed
  StageTimings T; T.reserve(args.iters);
  std::cerr << "[production_benchmark] timed × " << args.iters << "...\n";
  auto wall0 = std::chrono::steady_clock::now();
  for (int i = 0; i < args.iters; ++i) {
    if (!run_iteration(i, &T)) {
      std::cerr << "timed iter " << i << " failed\n"; return 1;
    }
  }
  auto wall1 = std::chrono::steady_clock::now();
  double wall_s = std::chrono::duration<double>(wall1 - wall0).count();
  double total_images = static_cast<double>(args.iters) * args.batch_size;
  double agg_qps = (wall_s > 0.0) ? total_images / wall_s : 0.0;

  std::cout << std::setprecision(6);
  std::cout << "\n=== V1.0.4 C++ Production Benchmark Summary (p50 ms) ===\n";
  std::cout << "  disk_read   : " << percentile(T.disk_read, 50.0) << " ms\n";
  std::cout << "  jpg_decode  : " << percentile(T.jpg_decode, 50.0) << " ms\n";
  std::cout << "  preprocess  : " << percentile(T.preprocess, 50.0) << " ms\n";
  std::cout << "  h2d         : " << percentile(T.h2d, 50.0) << " ms\n";
  std::cout << "  enqueueV3   : " << percentile(T.enqueue_v3, 50.0) << " ms\n";
  std::cout << "  d2h         : " << percentile(T.d2h, 50.0) << " ms\n";
  std::cout << "  ---\n";
  std::cout << "  total       : " << percentile(T.total, 50.0) << " ms\n";
  std::cout << "  aggregate   : " << agg_qps << " imgs/sec\n";

  if (!args.output_path.empty()) {
    write_json_summary(args, T, wall_s, agg_qps, output_names, args.output_path);
    std::cerr << "[production_benchmark] report → " << args.output_path << "\n";
  }

  for (auto* p : device_ptrs) if (p) cudaFree(p);
  cudaStreamDestroy(stream);
  return 0;
}
