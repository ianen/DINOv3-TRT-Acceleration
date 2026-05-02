// V1.0.3 ADR-020 Phase 2 prototype — tests TRT 10.13 thread-safety claim
// for "single engine, multiple IExecutionContext, multiple cuda streams,
// concurrent enqueueV3" architecture. If this works, Phase 2 shared-engine
// pool is viable; if this hits the same Myelin runner.cpp:778 error as
// Phase 1's per-slot-engine pool, the blocker is below the engine layer
// and only a TRT version upgrade can fix it.

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << "[TRT] " << msg << '\n';
    }
  }
};

[[nodiscard]] std::vector<char> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return {};
  const std::streamsize sz = f.tellg();
  if (sz <= 0) return {};
  std::vector<char> data(static_cast<std::size_t>(sz));
  f.seekg(0);
  if (!f.read(data.data(), sz)) return {};
  return data;
}

constexpr const char* kInputName = "pixel_values";
constexpr int kHidden = 1024;
constexpr int kPatchSize = 16;
constexpr int kOutputs = 4;
// kImageSize replaced with runtime-configurable image_size below.
constexpr const char* kOutputNames[kOutputs] = {
    "feat_layer_4", "feat_layer_12", "feat_layer_16", "feat_layer_20"};

struct ContextState {
  std::unique_ptr<nvinfer1::IExecutionContext> ctx;
  cudaStream_t stream{nullptr};
  void* d_input{nullptr};
  void* d_outputs[kOutputs]{};
  std::size_t input_bytes{0};
  std::size_t output_bytes{0};
};

bool init_context(ContextState& st, nvinfer1::ICudaEngine& engine, int batch, int image_size) {
  st.ctx.reset(engine.createExecutionContext());
  if (!st.ctx) {
    std::cerr << "createExecutionContext failed\n";
    return false;
  }
  if (cudaStreamCreate(&st.stream) != cudaSuccess) {
    std::cerr << "cudaStreamCreate failed\n";
    return false;
  }
  const int grid = image_size / kPatchSize;
  const int tokens = 1 + grid * grid;
  st.input_bytes = static_cast<std::size_t>(batch) * 3 * image_size * image_size * sizeof(float);
  st.output_bytes = static_cast<std::size_t>(batch) * tokens * kHidden * sizeof(float);
  if (cudaMalloc(&st.d_input, st.input_bytes) != cudaSuccess) return false;
  for (int i = 0; i < kOutputs; ++i) {
    if (cudaMalloc(&st.d_outputs[i], st.output_bytes) != cudaSuccess) return false;
  }
  // Set input shape + tensor addresses.
  nvinfer1::Dims4 dims{batch, 3, image_size, image_size};
  if (!st.ctx->setInputShape(kInputName, dims)) {
    std::cerr << "setInputShape failed\n";
    return false;
  }
  if (!st.ctx->setTensorAddress(kInputName, st.d_input)) return false;
  for (int i = 0; i < kOutputs; ++i) {
    if (!st.ctx->setTensorAddress(kOutputNames[i], st.d_outputs[i])) return false;
  }
  // Initialize input with zeros (Myelin happy with any valid float).
  return cudaMemsetAsync(st.d_input, 0, st.input_bytes, st.stream) == cudaSuccess;
}

void teardown_context(ContextState& st) {
  if (st.d_input) cudaFree(st.d_input);
  for (int i = 0; i < kOutputs; ++i) {
    if (st.d_outputs[i]) cudaFree(st.d_outputs[i]);
  }
  if (st.stream) cudaStreamDestroy(st.stream);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: test_concurrent_contexts <engine_path> [iterations=20] [n_contexts=2] [batch=1] [image_size=224]\n";
    return 2;
  }
  const std::string engine_path = argv[1];
  const int iters = (argc >= 3) ? std::atoi(argv[2]) : 20;
  const int n_ctx = (argc >= 4) ? std::atoi(argv[3]) : 2;
  const int batch = (argc >= 5) ? std::atoi(argv[4]) : 1;
  const int image_size_arg = (argc >= 6) ? std::atoi(argv[5]) : 224;
  if (n_ctx < 1 || n_ctx > 16) {
    std::cerr << "n_contexts must be in [1, 16]\n";
    return 2;
  }

  std::cerr << "[test] reading engine: " << engine_path << "\n";
  const auto blob = read_file(engine_path);
  if (blob.empty()) {
    std::cerr << "failed to read engine\n";
    return 1;
  }

  Logger logger;
  std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
  if (!runtime) {
    std::cerr << "createInferRuntime failed\n";
    return 1;
  }
  std::unique_ptr<nvinfer1::ICudaEngine> engine{
      runtime->deserializeCudaEngine(blob.data(), blob.size())};
  if (!engine) {
    std::cerr << "deserializeCudaEngine failed\n";
    return 1;
  }
  std::cerr << "[test] engine deserialized once, creating " << n_ctx << " contexts (batch=" << batch << ", image=" << image_size_arg << ")...\n";

  std::vector<ContextState> ctx(static_cast<std::size_t>(n_ctx));
  for (int i = 0; i < n_ctx; ++i) {
    if (!init_context(ctx[i], *engine, batch, image_size_arg)) {
      std::cerr << "init_context " << i << " failed\n";
      return 1;
    }
  }
  std::cerr << "[test] all " << n_ctx << " contexts initialized; warming up sequentially...\n";

  for (int i = 0; i < n_ctx; ++i) {
    if (!ctx[i].ctx->enqueueV3(ctx[i].stream)) {
      std::cerr << "warmup enqueueV3 ctx " << i << " failed\n";
      return 1;
    }
    cudaStreamSynchronize(ctx[i].stream);
  }
  std::cerr << "[test] sequential warmup OK; launching " << n_ctx << " concurrent threads...\n";

  std::atomic<int> errors{0};
  auto worker = [&](int idx) {
    for (int it = 0; it < iters; ++it) {
      if (!ctx[idx].ctx->enqueueV3(ctx[idx].stream)) {
        std::cerr << "thread " << idx << " iter " << it << " enqueueV3 failed\n";
        errors.fetch_add(1, std::memory_order_relaxed);
        return;
      }
      if (cudaStreamSynchronize(ctx[idx].stream) != cudaSuccess) {
        std::cerr << "thread " << idx << " iter " << it << " sync failed\n";
        errors.fetch_add(1, std::memory_order_relaxed);
        return;
      }
    }
  };
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<std::thread> threads;
  threads.reserve(static_cast<std::size_t>(n_ctx));
  for (int i = 0; i < n_ctx; ++i) threads.emplace_back(worker, i);
  for (auto& t : threads) t.join();
  const auto t1 = std::chrono::steady_clock::now();
  const double wall_s = std::chrono::duration<double>(t1 - t0).count();

  for (int i = 0; i < n_ctx; ++i) teardown_context(ctx[i]);

  if (errors.load() > 0) {
    std::cerr << "[test] FAIL: " << errors.load() << " concurrent enqueueV3 errors\n";
    return 1;
  }
  const double total_inferences = static_cast<double>(iters) * static_cast<double>(n_ctx) * static_cast<double>(batch);
  const double agg_qps = total_inferences / wall_s;
  std::cout << "{\n";
  std::cout << "  \"verdict\": \"PASS\",\n";
  std::cout << "  \"shared_engine_concurrent_works\": true,\n";
  std::cout << "  \"iters_per_thread\": " << iters << ",\n";
  std::cout << "  \"n_contexts\": " << n_ctx << ",\n";
  std::cout << "  \"batch_size\": " << batch << ",\n";
  std::cout << "  \"image_size\": " << image_size_arg << ",\n";
  std::cout << "  \"wall_seconds\": " << wall_s << ",\n";
  std::cout << "  \"aggregate_qps\": " << agg_qps << "\n";
  std::cout << "}\n";
  return 0;
}
