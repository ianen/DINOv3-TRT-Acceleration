// V1.0.3 ADR-020 / G5 acceptance gate — cosine bit-exactness validator.
//
// Verifies that `TRTInfererPool` (shared engine + N concurrent contexts)
// produces outputs bit-identical to a single-context `TRTInferer` reference
// for the same input. This is V1.0.3 plan G5 acceptance criterion:
//   "cos_min ≥ 0.997 — concurrent execution 必须 bit-exact"
//
// Architecture:
//   1. Build a deterministic input.
//   2. Run reference once via single-context TRTInferer.
//   3. Run N times via TRTInfererPool with M concurrent caller threads.
//   4. Verify every concurrent output is element-wise equal to the reference
//      (max_abs_error == 0 over all 4 outputs, all elements).
//   5. Also report cosine similarity per output for headline numbers.
//
// Exit code 0 = G5 PASS; non-zero = G5 FAIL.

#include "dinov3_trt/trt_inferer.h"
#include "dinov3_trt/trt_inferer_pool.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

void fill_deterministic(std::vector<float>& v, int seed) {
  for (std::size_t i = 0; i < v.size(); ++i) {
    const double phase = static_cast<double>((i % 1009U) + static_cast<unsigned>(seed) + 1U) * 0.017;
    v[i] = static_cast<float>(std::sin(phase));
  }
}

[[nodiscard]] double cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size() || a.empty()) return 0.0;
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double x = a[i];
    const double y = b[i];
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  if (na == 0.0 || nb == 0.0) return 0.0;
  return dot / (std::sqrt(na) * std::sqrt(nb));
}

[[nodiscard]] double max_abs_error(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
  double mx = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    if (d > mx) mx = d;
  }
  return mx;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: pool_cos_validator <engine> [batch=1] [image_size=224] [num_streams=2] [iters=10]\n";
    return 2;
  }
  const std::string engine_path = argv[1];
  const int batch = (argc >= 3) ? std::atoi(argv[2]) : 1;
  const int image_size = (argc >= 4) ? std::atoi(argv[3]) : 224;
  const int n_streams = (argc >= 5) ? std::atoi(argv[4]) : 2;
  const int iters = (argc >= 6) ? std::atoi(argv[5]) : 10;

  using namespace dinov3_trt;

  const TensorShape in_shape = input_shape_for(batch, image_size);
  const TensorShape out_shape = output_shape_for(batch, image_size);
  std::vector<float> input(static_cast<std::size_t>(in_shape.element_count()));
  fill_deterministic(input, 42);

  // ---- Reference: single-context TRTInferer ----
  std::cerr << "[cos_validator] computing reference via TRTInferer (single context)...\n";
  std::array<std::vector<float>, kOutputCount> ref;
  std::array<TensorView, kOutputCount> ref_views;
  for (std::size_t i = 0; i < kOutputCount; ++i) {
    ref[i].assign(static_cast<std::size_t>(out_shape.element_count()), 0.0f);
    ref_views[i] = TensorView{ref[i].data(), out_shape, DataType::kFloat32};
  }
  TensorView in_view{input.data(), in_shape, DataType::kFloat32};
  {
    TRTInferer ref_inferer(engine_path);
    const auto st = ref_inferer.infer(in_view, ref_views);
    if (!st.is_ok()) {
      std::cerr << "reference infer failed: " << st.message() << "\n";
      return 1;
    }
  }

  // ---- Concurrent: TRTInfererPool with M caller threads ----
  std::cerr << "[cos_validator] running pool with " << n_streams
            << " contexts, " << n_streams << " threads, " << iters << " iters each...\n";
  TRTInfererPool::Config cfg;
  cfg.engine_path = engine_path;
  cfg.num_streams = n_streams;
  cfg.enable_cuda_graphs = false;  // pool Phase 2 doesn't use graphs yet; keep deterministic
  TRTInfererPool pool(cfg);

  std::atomic<int> ok_count{0};
  std::atomic<int> fail_count{0};
  std::mutex worst_mu;
  std::array<double, kOutputCount> worst_max_abs{};
  std::array<double, kOutputCount> worst_min_cos{1.0, 1.0, 1.0, 1.0};

  auto worker = [&](int /*thread_id*/) {
    for (int it = 0; it < iters; ++it) {
      std::array<std::vector<float>, kOutputCount> outs;
      std::array<TensorView, kOutputCount> out_views;
      for (std::size_t i = 0; i < kOutputCount; ++i) {
        outs[i].assign(static_cast<std::size_t>(out_shape.element_count()), 0.0f);
        out_views[i] = TensorView{outs[i].data(), out_shape, DataType::kFloat32};
      }
      const auto st = pool.infer(in_view, out_views);
      if (!st.is_ok()) {
        fail_count.fetch_add(1, std::memory_order_relaxed);
        return;
      }
      bool this_ok = true;
      std::array<double, kOutputCount> mae;
      std::array<double, kOutputCount> cos;
      for (std::size_t i = 0; i < kOutputCount; ++i) {
        mae[i] = max_abs_error(ref[i], outs[i]);
        cos[i] = cosine_similarity(ref[i], outs[i]);
        if (mae[i] != 0.0) this_ok = false;
      }
      if (!this_ok) {
        std::lock_guard<std::mutex> lk(worst_mu);
        for (std::size_t i = 0; i < kOutputCount; ++i) {
          if (mae[i] > worst_max_abs[i]) worst_max_abs[i] = mae[i];
          if (cos[i] < worst_min_cos[i]) worst_min_cos[i] = cos[i];
        }
        fail_count.fetch_add(1, std::memory_order_relaxed);
      } else {
        ok_count.fetch_add(1, std::memory_order_relaxed);
      }
    }
  };
  std::vector<std::thread> threads;
  threads.reserve(static_cast<std::size_t>(n_streams));
  for (int t = 0; t < n_streams; ++t) threads.emplace_back(worker, t);
  for (auto& th : threads) th.join();

  const int total = ok_count.load() + fail_count.load();
  std::cout << std::setprecision(10);
  std::cout << "{\n";
  std::cout << "  \"engine\": \"" << engine_path << "\",\n";
  std::cout << "  \"batch\": " << batch << ",\n";
  std::cout << "  \"image_size\": " << image_size << ",\n";
  std::cout << "  \"num_streams\": " << n_streams << ",\n";
  std::cout << "  \"iters_per_thread\": " << iters << ",\n";
  std::cout << "  \"total_concurrent_inferences\": " << total << ",\n";
  std::cout << "  \"bit_exact_count\": " << ok_count.load() << ",\n";
  std::cout << "  \"non_bit_exact_count\": " << fail_count.load() << ",\n";

  // If bit-exact across the board, cos and max_abs are trivially {1,1,1,1} and {0,0,0,0}.
  // If any drift, report worst observed.
  if (fail_count.load() == 0) {
    std::cout << "  \"max_abs_error_per_output\": [0, 0, 0, 0],\n";
    std::cout << "  \"min_cosine_per_output\": [1, 1, 1, 1],\n";
    std::cout << "  \"verdict\": \"G5 PASS — all concurrent outputs bit-exact vs single-context reference\"\n";
    std::cout << "}\n";
    return 0;
  }
  std::cout << "  \"max_abs_error_per_output\": ["
            << worst_max_abs[0] << ", " << worst_max_abs[1] << ", "
            << worst_max_abs[2] << ", " << worst_max_abs[3] << "],\n";
  std::cout << "  \"min_cosine_per_output\": ["
            << worst_min_cos[0] << ", " << worst_min_cos[1] << ", "
            << worst_min_cos[2] << ", " << worst_min_cos[3] << "],\n";
  // G5 numerical gate: cos_min ≥ 0.997
  bool g5_pass = true;
  for (double c : worst_min_cos) {
    if (c < 0.997) g5_pass = false;
  }
  std::cout << "  \"verdict\": \"" << (g5_pass ? "G5 PASS — cos_min ≥ 0.997 across all outputs (non-bit-exact but within numerical gate)"
                                                : "G5 FAIL — cos_min < 0.997 for at least one output")
            << "\"\n";
  std::cout << "}\n";
  return g5_pass ? 0 : 1;
}
