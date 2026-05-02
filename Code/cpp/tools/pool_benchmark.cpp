// V1.0.3 ADR-020 · Multi-context engine pool throughput benchmark.
//
// Drives `TRTInfererPool` with M concurrent caller threads × K iterations
// each, reports aggregate QPS + per-thread / aggregate latency CDF in
// JSON. Mirrors the existing `runtime_benchmark` argument shape so it can
// drop into the same orchestration scripts. Adds `--num-streams` (pool
// size), `--threads` (caller concurrency), and `--no-graphs` (disable
// V1.0.2 ADR-012 CudaGraphPool, for A/B testing) on top.

#include "dinov3_trt/trt_inferer_pool.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <string>
#include <string_view>
#include <thread>
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

struct CliArgs {
  std::string engine_path;
  int batch_size{1};
  int image_size{dinov3_trt::kImageSize};
  int num_streams{2};
  int threads{0};   // 0 -> default to num_streams
  int warmup{10};
  int iterations{200};
  bool enable_cuda_graphs{true};
  bool pinned_buffers{false};  // Use cudaMallocHost-pinned worker buffers — fast cudaMemcpyAsync path
};

[[nodiscard]] std::string json_escape(std::string_view value) {
  std::string out;
  out.reserve(value.size());
  for (const char ch : value) {
    switch (ch) {
      case '\\': out += "\\\\"; break;
      case '"':  out += "\\\""; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:   out += ch; break;
    }
  }
  return out;
}

[[nodiscard]] bool parse_int(const char* s, int* out, int min_val = 1) {
  char* end = nullptr;
  const long v = std::strtol(s, &end, 10);
  if (end == s || *end != '\0' || v < min_val ||
      v > std::numeric_limits<int>::max()) {
    return false;
  }
  *out = static_cast<int>(v);
  return true;
}

void print_usage(const char* prog) {
  std::cerr
      << "usage: " << prog << " --engine PATH [--batch-size N] [--image-size N]\n"
      << "       [--num-streams N] [--threads N] [--warmup N] [--iters N] [--no-graphs]\n"
      << "\n"
      << "  --engine        path to serialized TRT engine (required)\n"
      << "  --batch-size    inference batch size (default: 1)\n"
      << "  --image-size    spatial input size (default: 224)\n"
      << "  --num-streams   pool size (default: 2)\n"
      << "  --threads       caller thread count (default: equal to --num-streams)\n"
      << "  --warmup        warmup iterations PER thread (default: 10)\n"
      << "  --iters         timed iterations PER thread (default: 200)\n"
      << "  --no-graphs     disable per-slot CudaGraphPool (V1.0.2 ADR-012)\n"
      << "  --pinned        allocate worker thread input/output buffers via\n"
      << "                  cudaMallocHost so cudaMemcpyAsync inside the pool\n"
      << "                  uses the pinned-fast-path. Best end-to-end qps.\n";
}

[[nodiscard]] bool parse_args(int argc, char** argv, CliArgs& out) {
  for (int i = 1; i < argc; ++i) {
    const std::string_view tok{argv[i]};
    auto need_value = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::cerr << name << " requires a value\n";
        return nullptr;
      }
      return argv[++i];
    };
    if (tok == "--engine") {
      const char* v = need_value("--engine"); if (!v) return false;
      out.engine_path = v;
    } else if (tok == "--batch-size") {
      const char* v = need_value("--batch-size"); if (!v) return false;
      if (!parse_int(v, &out.batch_size)) { std::cerr << "--batch-size invalid\n"; return false; }
    } else if (tok == "--image-size") {
      const char* v = need_value("--image-size"); if (!v) return false;
      if (!parse_int(v, &out.image_size)) { std::cerr << "--image-size invalid\n"; return false; }
    } else if (tok == "--num-streams") {
      const char* v = need_value("--num-streams"); if (!v) return false;
      if (!parse_int(v, &out.num_streams)) { std::cerr << "--num-streams invalid\n"; return false; }
    } else if (tok == "--threads") {
      const char* v = need_value("--threads"); if (!v) return false;
      if (!parse_int(v, &out.threads)) { std::cerr << "--threads invalid\n"; return false; }
    } else if (tok == "--warmup") {
      const char* v = need_value("--warmup"); if (!v) return false;
      if (!parse_int(v, &out.warmup, 0)) { std::cerr << "--warmup invalid\n"; return false; }
    } else if (tok == "--iters") {
      const char* v = need_value("--iters"); if (!v) return false;
      if (!parse_int(v, &out.iterations)) { std::cerr << "--iters invalid\n"; return false; }
    } else if (tok == "--no-graphs") {
      out.enable_cuda_graphs = false;
    } else if (tok == "--pinned") {
      out.pinned_buffers = true;
    } else if (tok == "--help" || tok == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else {
      std::cerr << "unknown argument: " << tok << "\n";
      print_usage(argv[0]);
      return false;
    }
  }
  if (out.engine_path.empty()) {
    print_usage(argv[0]);
    return false;
  }
  if (out.threads == 0) {
    out.threads = out.num_streams;
  }
  return true;
}

[[nodiscard]] double percentile(const std::vector<double>& sorted, double p) {
  if (sorted.empty()) return 0.0;
  const double r = (p / 100.0) * static_cast<double>(sorted.size() - 1);
  const std::size_t lo = static_cast<std::size_t>(std::floor(r));
  const std::size_t hi = static_cast<std::size_t>(std::ceil(r));
  if (lo == hi) return sorted[lo];
  const double frac = r - static_cast<double>(lo);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * frac;
}

[[nodiscard]] LatencySummary summarize(std::vector<double> latencies) {
  LatencySummary s;
  if (latencies.empty()) return s;
  std::sort(latencies.begin(), latencies.end());
  s.min_ms = latencies.front();
  s.max_ms = latencies.back();
  s.median_ms = percentile(latencies, 50.0);
  s.p90_ms = percentile(latencies, 90.0);
  s.p95_ms = percentile(latencies, 95.0);
  s.p99_ms = percentile(latencies, 99.0);
  const double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
  s.mean_ms = sum / static_cast<double>(latencies.size());
  return s;
}

struct ThreadResult {
  std::vector<double> latencies_ms;
  bool ok{true};
  std::string error;
};

// Lightweight pinned-host buffer for the benchmark caller side. When --pinned
// is given, each worker allocates its input + 4 output buffers via
// cudaMallocHost so cudaMemcpyAsync inside the pool takes the pinned-DMA fast
// path. Without --pinned we fall back to std::vector<float> (pageable).
struct PinnedFloat {
  float* ptr{nullptr};
  std::size_t elements{0};
  ~PinnedFloat() { if (ptr) cudaFreeHost(ptr); }
  bool allocate(std::size_t n) {
    elements = n;
    return cudaMallocHost(reinterpret_cast<void**>(&ptr), n * sizeof(float)) == cudaSuccess;
  }
  PinnedFloat() = default;
  PinnedFloat(const PinnedFloat&) = delete;
  PinnedFloat& operator=(const PinnedFloat&) = delete;
};

void fill_deterministic_pinned(float* p, std::size_t n, int seed) {
  for (std::size_t i = 0; i < n; ++i) {
    const double phase = static_cast<double>((i % 1009U) + static_cast<unsigned>(seed) + 1U) * 0.017;
    p[i] = static_cast<float>(std::sin(phase));
  }
}

void run_thread(
    int thread_id,
    int warmup,
    int iters,
    int batch_size,
    int image_size,
    bool pinned,
    dinov3_trt::TRTInfererPool& pool,
    ThreadResult& result) {
  const dinov3_trt::TensorShape in_shape =
      dinov3_trt::input_shape_for(batch_size, image_size);
  const dinov3_trt::TensorShape out_shape =
      dinov3_trt::output_shape_for(batch_size, image_size);
  const std::size_t in_elems = static_cast<std::size_t>(in_shape.element_count());
  const std::size_t out_elems = static_cast<std::size_t>(out_shape.element_count());

  // Two-track allocation: pageable (std::vector) or pinned (cudaMallocHost).
  std::vector<float> v_input;
  std::array<std::vector<float>, dinov3_trt::kOutputCount> v_outs;
  PinnedFloat p_input;
  std::array<PinnedFloat, dinov3_trt::kOutputCount> p_outs;
  float* in_ptr = nullptr;
  std::array<float*, dinov3_trt::kOutputCount> out_ptrs{};

  if (pinned) {
    if (!p_input.allocate(in_elems)) {
      result.ok = false;
      result.error = "cudaMallocHost(input) failed";
      return;
    }
    in_ptr = p_input.ptr;
    for (std::size_t i = 0; i < dinov3_trt::kOutputCount; ++i) {
      if (!p_outs[i].allocate(out_elems)) {
        result.ok = false;
        result.error = "cudaMallocHost(output) failed";
        return;
      }
      out_ptrs[i] = p_outs[i].ptr;
    }
  } else {
    v_input.resize(in_elems);
    in_ptr = v_input.data();
    for (std::size_t i = 0; i < dinov3_trt::kOutputCount; ++i) {
      v_outs[i].resize(out_elems);
      out_ptrs[i] = v_outs[i].data();
    }
  }
  fill_deterministic_pinned(in_ptr, in_elems, thread_id);

  const dinov3_trt::TensorView input_view{in_ptr, in_shape, dinov3_trt::DataType::kFloat32};
  std::array<dinov3_trt::TensorView, dinov3_trt::kOutputCount> out_views;
  for (std::size_t i = 0; i < dinov3_trt::kOutputCount; ++i) {
    out_views[i] = dinov3_trt::TensorView{out_ptrs[i], out_shape, dinov3_trt::DataType::kFloat32};
  }

  for (int w = 0; w < warmup; ++w) {
    const auto status = pool.infer(input_view, out_views);
    if (!status.is_ok()) {
      result.ok = false;
      result.error = "warmup infer failed: " + status.message();
      return;
    }
  }

  result.latencies_ms.reserve(static_cast<std::size_t>(iters));
  for (int it = 0; it < iters; ++it) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto status = pool.infer(input_view, out_views);
    const auto t1 = std::chrono::steady_clock::now();
    if (!status.is_ok()) {
      result.ok = false;
      result.error = "timed infer failed: " + status.message();
      return;
    }
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.latencies_ms.push_back(ms);
  }
}

}  // namespace

int main(int argc, char** argv) {
  CliArgs args;
  if (!parse_args(argc, argv, args)) {
    return 2;
  }

  dinov3_trt::TRTInfererPool::Config cfg;
  cfg.engine_path = args.engine_path;
  cfg.num_streams = args.num_streams;
  cfg.enable_cuda_graphs = args.enable_cuda_graphs;
  dinov3_trt::TRTInfererPool pool(cfg);

  std::cerr << "[pool_benchmark] engine=" << args.engine_path
            << " batch=" << args.batch_size
            << " image=" << args.image_size
            << " num_streams=" << args.num_streams
            << " threads=" << args.threads
            << " warmup=" << args.warmup << "/thread"
            << " iters=" << args.iterations << "/thread"
            << " graphs=" << (args.enable_cuda_graphs ? "on" : "off") << "\n";

  std::vector<ThreadResult> results(static_cast<std::size_t>(args.threads));
  std::vector<std::thread> threads;
  threads.reserve(static_cast<std::size_t>(args.threads));

  const auto wall_start = std::chrono::steady_clock::now();
  for (int t = 0; t < args.threads; ++t) {
    threads.emplace_back(run_thread, t, args.warmup, args.iterations,
                         args.batch_size, args.image_size, args.pinned_buffers,
                         std::ref(pool), std::ref(results[t]));
  }
  for (auto& th : threads) th.join();
  const auto wall_end = std::chrono::steady_clock::now();
  const double total_wall_ms =
      std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

  std::vector<double> all_latencies;
  bool any_failed = false;
  std::string first_error;
  long long total_inferences = 0;
  for (const auto& r : results) {
    if (!r.ok) {
      any_failed = true;
      if (first_error.empty()) first_error = r.error;
      continue;
    }
    total_inferences += static_cast<long long>(r.latencies_ms.size());
    all_latencies.insert(all_latencies.end(), r.latencies_ms.begin(), r.latencies_ms.end());
  }

  const LatencySummary agg = summarize(std::move(all_latencies));
  const double aggregate_calls_per_sec =
      total_wall_ms <= 0.0
          ? 0.0
          : (static_cast<double>(total_inferences) * 1000.0) / total_wall_ms;
  // For V1.0.3 G1/G2/G3 SMART targets (which measure image-level throughput),
  // multiply by batch_size — each pool.infer() call processes a batch of
  // batch_size images. The "qps" in the V1.0.2 baseline 343.69 was already
  // image-level for batch=1, so this is the apples-to-apples metric.
  const double aggregate_qps = aggregate_calls_per_sec * static_cast<double>(args.batch_size);

  std::cout << std::setprecision(10);
  std::cout << "{\n";
  std::cout << "  \"engine_path\": \"" << json_escape(args.engine_path) << "\",\n";
  std::cout << "  \"batch_size\": " << args.batch_size << ",\n";
  std::cout << "  \"image_size\": " << args.image_size << ",\n";
  std::cout << "  \"num_streams\": " << args.num_streams << ",\n";
  std::cout << "  \"threads\": " << args.threads << ",\n";
  std::cout << "  \"warmup_per_thread\": " << args.warmup << ",\n";
  std::cout << "  \"iters_per_thread\": " << args.iterations << ",\n";
  std::cout << "  \"enable_cuda_graphs\": "
            << (args.enable_cuda_graphs ? "true" : "false") << ",\n";
  std::cout << "  \"pinned_buffers\": "
            << (args.pinned_buffers ? "true" : "false") << ",\n";
  std::cout << "  \"total_inferences\": " << total_inferences << ",\n";
  std::cout << "  \"total_wall_ms\": " << total_wall_ms << ",\n";
  std::cout << "  \"aggregate_calls_per_sec\": " << aggregate_calls_per_sec << ",\n";
  std::cout << "  \"aggregate_qps\": " << aggregate_qps << ",\n";
  std::cout << "  \"latency_ms\": {\n";
  std::cout << "    \"min\": " << agg.min_ms << ",\n";
  std::cout << "    \"mean\": " << agg.mean_ms << ",\n";
  std::cout << "    \"median\": " << agg.median_ms << ",\n";
  std::cout << "    \"p90\": " << agg.p90_ms << ",\n";
  std::cout << "    \"p95\": " << agg.p95_ms << ",\n";
  std::cout << "    \"p99\": " << agg.p99_ms << ",\n";
  std::cout << "    \"max\": " << agg.max_ms << "\n";
  std::cout << "  },\n";
  std::cout << "  \"any_thread_failed\": " << (any_failed ? "true" : "false");
  if (any_failed) {
    std::cout << ",\n  \"first_error\": \"" << json_escape(first_error) << "\"";
  }
  std::cout << "\n}\n";

  return any_failed ? 1 : 0;
}
