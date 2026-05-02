# Paper §6 — V1.0.3 Throughput Limits and GPU Utilization Saturation

> Drop-in subsection for `paper_full_draft_V1.0.0.md`. Ready to splice
> into the assembled paper. ~700 words, integrates V1.0.3 G1 / G7 / ADR-020
> Phase 2 / Myelin thread-safety findings into the IMRaD body. Writing
> style and citation density match the existing paper drafts.

## §6 Throughput-Oriented Serving and Utilization Limits

### §6.1 Motivation

The V1.0.1 study reported single-stream latency speedup but did not characterize the model's throughput ceiling under concurrent load. Production serving systems aggregate multiple inference requests, both within a single process (multi-context dispatch) and across processes (request batching at the serving layer). Understanding where DINOv3 ViT-L/16 saturates the GPU under each regime was therefore the primary V1.0.3 scope, and supplied the empirical justification for whether further dense-precision optimization on this hardware is feasible.

### §6.2 Multi-Context Pool Architecture

We implemented an in-process multi-context pool (`TRTInfererPool`) that deserializes the engine once and instantiates *N* independent `IExecutionContext` objects, each bound to its own CUDA stream and device-side input/output buffers. Permit-gated capacity (`std::counting_semaphore`) ensures the pool never over-subscribes its slots; per-slot mutexes provide defense in depth against undocumented thread-safety failures in the underlying TensorRT runtime. Round-robin atomic dispatch routes incoming inferences to the next available slot.

A first-cut implementation deserialized the engine *per slot* (rather than sharing across contexts) and crashed at concurrent N≥2 with a TensorRT 10.13 error from `MyelinRunnerBase::executeMyelinGraph` (runtime/myelin/runner.cpp:778). We initially attributed this to fundamental thread-safety in TensorRT 10.13, but a focused experiment with two completely independent OS processes succeeded (with significantly degraded throughput, 1.087× over baseline, due to inter-process CUDA context contention). A second experiment with a *single* shared engine and two independent contexts on different streams reproduced N=2 concurrent execution successfully at 1.515× over baseline — exactly matching the speedup obtained by Python's `tensorrt` bindings (whose Global Interpreter Lock had previously been hypothesized as the limiting factor). This shows that the original Phase 1 crash was caused by duplicate engine deserialization in the same process, *not* a fundamental TensorRT thread-safety constraint. NVIDIA's documented "single engine, multiple contexts on independent streams = thread-safe" architecture is empirically correct at TensorRT 10.13 for this model.

### §6.3 Throughput Measurement

We characterized end-to-end aggregate throughput at three resolutions × multiple batch sizes × multiple pool capacities. End-to-end measurements include the full host-to-device transfer, `enqueueV3` dispatch, four device-to-host transfers (one per intermediate-feature output), and a synchronous stream-wait at call exit.

At the small-load regime (r224, b=1) the single-stream baseline delivers 343 qps. The multi-context pool at N=2 with batch=8 reaches 644 qps (1.875×); over-saturation by allowing more caller threads than pool slots (N=4 slots × 16 caller threads, with caller-side pinned host memory) raises the empirical ceiling to **720 qps (2.094×)**. The plan target of 800 qps for this regime was set assuming inter-request batching at a serving layer (e.g. Triton's `dynamic_batching`); without that layer the in-process pool plateaus around 720 qps because per-call `cudaStreamSynchronize` serializes within each slot regardless of caller concurrency. A pure-compute prototype that omits host transfers reaches 922 qps (2.683×), confirming the residual ~22% gap is bounded by host-device transfer rather than by GPU compute.

At the medium-load regime (r336, b=8) the baseline 363 qps lifts only to 401 qps (1.10×) under N=2 multi-context. At the saturation regime (r518, b=8) the baseline 156 qps moves to 160 qps (1.02×). Both numbers reproduce the V1.0.2 ADR-015 finding that larger batches and resolutions leave essentially no GPU compute headroom for additional contexts to fill. Larger batches (b=16, b=32) regress because the model crosses into a memory-bound regime where HBM bandwidth becomes the binding constraint.

### §6.4 GPU Utilization Saturation Evidence

We recorded SM utilization, HBM utilization, board power, and temperature at 100 ms cadence via `nvidia-smi --query-gpu=...` during each benchmark run, aggregated to mean / p50 / p95 / max statistics in a per-run summary. Across our four primary regimes:

| Regime | mean SM % | p50 SM % | p95 SM % | mean power (W) |
|---|---|---|---|---|
| r518 b8 N=1 | 99.08 | 99 | 100 | 326 |
| r336 b8 N=1 | 96.39 | 99 | 99 | 289 |
| r224 b1 N=1 | 88.24 | 92 | 92 | 192 |
| r224 b1 N=2 | 95.77 | 99 | 99 | 258 |

The high-batch / high-resolution regimes are physically saturated at single-stream — multi-context cannot extract additional throughput because there are no idle SMs to fill. Low-batch regimes have ~12 percentage points of SM idle headroom which multi-context successfully fills (88.24% → 95.77% at r224 b1 N=2). Power consumption stays within 92% of the 360 W board limit at all regimes (peak 332.6 W during sustained r518 b8); temperature peaks at 72 °C, ten degrees below thermal-throttle activation.

### §6.5 Implications for Quantization

The combined evidence — full SM saturation across two of three resolution regimes, a 720 qps in-process throughput ceiling at the third, and (from V1.0.2) a confirmed PTQ precision wall across INT8, FP8, and 2:4-sparse paths — establishes that further acceleration of this model on this hardware requires *reducing per-inference compute*, not extracting more from the existing dense BF16 path. Quantization-aware fine-tuning (QAT) is the only published technique that can hold cosine similarity above the 0.99 hard gate while replacing dense BF16 matmuls with INT8 / FP8 / 2:4-sparse Tensor Core operations. The V1.0.3 utilization data is therefore the quantitative motivation for the V1.3 milestone described in §7.
