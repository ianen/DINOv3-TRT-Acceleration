# DINOv3 TensorRT Acceleration — Research Contributions V1.0.0

> Academic-tone summary of research contributions for the V1.0+V1.1+V1.2 closed
> phases. Companion document to `技术报告_V1.0.0.md` (engineering report) and
> `答辩问答预案_V1.0.0.md` (defense Q&A); written in论文/intro tone for direct
> reuse in publication or thesis introductions.

## Abstract

We accelerate the DINOv3 ViT-L/16 LVD-1689M visual self-supervised foundation
model on consumer-grade Blackwell hardware (RTX 5080, sm_120) by way of
NVIDIA TensorRT 10.13. Across resolutions {224, 336, 518} and batch sizes
{1…32}, BF16 prefer engines achieve up to **3.86× trtexec GPU-compute
median speedup** and **3.40× C++ end-to-end speedup** over FP32, with
worst-case feat_layer_20 cosine similarity ≥ 0.998 against FP32 baselines on
1000 real images. A complete sensitivity analysis of INT8 quantization paths
— spanning default ModelOpt PTQ, node-level partial quantization, full
SmoothQuant α-sweep, and three independent mixed-precision strategies (
PyTorch ModelOpt `disable_quantizer`, TensorRT `--layerPrecisions` per-layer
override, and ONNX-level Q/DQ stripping) — establishes that none of them
meet a stringent G2 cosine threshold of ≥ 0.99 on this model + hardware
combination. We trace the bottleneck empirically to **upstream cumulative
INT8 quantization noise** in blocks 0-15, which downstream mixed-precision
cannot recover, and propose a quantization-aware fine-tuning (QAT) path as
the only remaining avenue for closing the precision gap.

## Key Contributions

1. **Empirical hardware-precision compatibility map for ViT-L/16 on
   Blackwell + TensorRT 10.13.** We document that FP16 is a hard
   incompatibility on this combination (output collapses to NaN under formal
   weights despite passing trtexec), BF16 is the only viable low-precision
   primary candidate, and BF16 is incompatible with explicit Q/DQ ONNX in
   TRT 10.13's Myelin Fill kernel — a previously undocumented intersection
   between the two production constraints.

2. **Three-tool-chain equivalence proof for mixed-precision recovery.** We
   show via independent implementations across PyTorch ModelOpt, TensorRT
   command-line, and ONNX library layers that all three tool chains produce
   numerically equivalent results when forcing transformer blocks 16-19 onto
   higher precision while leaving blocks 0-15 in INT8: feat_layer_20
   cos_min within 0.0005 across paths, b8 latency within 0.02× across paths.
   The convergence eliminates "wrong tool chain" as an explanation for
   negative results and pinpoints upstream quantization noise as the root
   cause.

3. **Pure-Python testing infrastructure for ONNX graph manipulation.** We
   introduce a project pattern in which graph-level operations are split
   into (a) pure-Python data-projection helpers (no `onnx` dependency) for
   identification and planning, and (b) thin remote-only driver scripts for
   actual graph mutation. This enables full unit testing on developer
   workstations without GPU or native ONNX dependencies — covering 271
   pytest cases across 111 source files including ONNX Q/DQ pair
   classification, strip planning, and benchmark figure rendering.

4. **DPT-style 4-layer hook selection ablation.** We empirically validate the
   project's intermediate-layer choice `[4, 12, 16, 20]` (1-based) on 1000
   real images by comparing it against the DPT paper's recommendation
   `[5, 11, 17, 23]` and a late-heavy variant `[6, 12, 18, 24]` along two
   axes: inter-output cosine similarity (lower = more diverse, better for
   multi-scale fusion) and per-output magnitude balance (max/min ratio
   across the 4 outputs). The chosen layout achieves the **tightest
   magnitude balance (12.6×)** at the cost of slightly higher inter-output
   cosine (0.383 vs DPT's 0.299), avoiding the 84× magnitude blow-up of
   the late-heavy variant where the deepest hook would dominate fusion.

5. **Atomic SHA256 manifest with self-exclusion.** We identify and resolve a
   subtle bug in shell-redirected manifest generation where `>`
   pre-creates a 0-byte file that the manifest scanner then records as a
   stale entry. Our `check_assets.py --output PATH` performs an atomic
   write (tempfile + `os.replace`) and adds the target path to the
   exclude set, eliminating the self-containment loop. This pattern
   generalizes to any manifest-generates-itself situation.

6. **Multi-resolution static-spatial / dynamic-batch profile strategy.** We
   establish that for ViT-L/16 on 16 GB VRAM, separate engines per
   resolution with static spatial dimensions and dynamic batch profiles
   (e.g., r518 builds a `min=1, opt=4, max=8` engine plus a separate
   `max=8` engine with its own timing cache) achieves higher kernel
   selection quality than a single engine with both spatial and batch
   dynamism. C++ runtime parity confirms this scales bit-identically across
   all three resolutions.

## Detailed Findings

### V1.0.0 Main Path (rounds 1-13)

- **G1 acceleration**: BF16 prefer is the project's primary low-precision
  candidate after FP16 was demonstrated as a NaN failure case on formal
  weights. BF16 trtexec speedups vs FP32 (locked 2752 MHz + spin-wait):
  r224 b1/b8/b32 = 2.45× / 2.81× / 3.25×; r336 b1/b4/b8 = 2.80× / 2.96× /
  3.25×; r518 b1/b2/b4/b8 = 3.12× / 3.50× / 3.76× / 3.86×.
- **G3 cross-language parity**: 224 / 336 / 518 batch=1 FP32 + BF16-prefer
  bit-identical (`max_abs=0`, `cosine=1.0`) under deterministic sine
  inputs; identical engine binaries used by Python and C++ runtimes.
- **G4 benchmark matrix**: 56-row CSV covering trtexec + C++ runtime ×
  3 resolutions × multiple batches with locked-clock + spin-wait
  methodology that suppresses Windows WDDM 100ms jitter.
- **G5 reproducibility**: Atomic SHA256 manifest, one-shot PowerShell
  pipeline, license bundling, "Built with DINOv3" attribution.

### V1.1 Stretch Goals (rounds 14-21)

- **FP8 PTQ default** achieves the highest raw speedup of any candidate
  (5.05× vs FP32 at b32, 1.55× vs BF16 prefer at b32) but cosine_mean
  collapses to 0.20 under formal weights — a symmetric failure to default
  INT8.
- **FP8 partial layer19** recovers cosine to 0.99941 but the speedup
  collapses to 1.04×, mirroring INT8 partial layer19 behavior.
- **SmoothQuant α-sweep (0.5 / 0.7 / 0.8)** establishes 0.8 as the best
  PTQ candidate for this model: cos_mean 0.982, cos_min 0.968, b8 speedup
  3.48× vs FP32. Speed crosses G2 threshold (≥ 2.2×) but cos_min remains
  0.022 below the G2 cosine threshold.
- **4-layer ablation** (PyTorch hooks, 1000 real images) provides the
  empirical justification for the project's `[4, 12, 16, 20]` selection.

### V1.2 Mixed-Precision Investigation (rounds 22-26)

Three independent mixed-precision strategies tested in parallel, all yielding
numerically equivalent negative results:

| Strategy | Layer | cos_min | b8 speedup |
|---|---|---:|---:|
| ModelOpt `disable_quantizer` skip 16-19 | PyTorch | 0.971 (+0.003) | 2.41× (-30%) |
| trtexec `--layerPrecisions=l16-19:fp32` | TRT | 0.9683 (≈) | 3.43× (≈) |
| ONNX-level Q/DQ stripping (this work) | ONNX | 0.9705 (+0.003) | 2.39× (-31%) |

The convergence (cos_min within 0.003, speedup within 0.02× across all three
paths) validates the upstream-noise hypothesis: all three implementations are
"correct" in their own framing but operate on the same underlying TRT
fallback behavior, which cannot recover information already lost to INT8
quantization in earlier blocks.

### V1.3 Future Work (round 27)

- **QAT (quantization-aware fine-tuning)** is identified as the only
  remaining avenue for crossing the G2 cosine threshold while preserving
  ≥ 2.2× speedup. ADR-011 provides a complete design including dataset
  requirements (ImageNet val 50K minimum), framework selection (NVIDIA
  Model Optimizer QAT mode for tool-chain continuity), expected outcomes
  (cos_min ≥ 0.99 + speedup ≥ 3.0×), risks (5 enumerated), and a 4-condition
  launch threshold preventing premature implementation under unmet
  preconditions.

## Methodological Innovations

### Pure-Python testing for native-tool-chain operations

The conventional approach to testing ONNX graph manipulation requires either
(a) installing `onnx` + `onnx-graphsurgeon` on every developer workstation,
or (b) skipping unit tests on workstations without GPUs. We instead split
the work:

1. **Identification + planning** — pure-Python functions operating on
   `(name, op_type, inputs, outputs)` tuples projected from `onnx
   .GraphProto.node`. No `onnx` dependency. Runs on macOS in CI.
2. **Application** — thin driver scripts that load ONNX, invoke the
   pure-Python planner, and apply the resulting plan via direct
   `onnx.GraphProto` mutation.

This pattern produced the layer_precision (round 17), onnx_qdq_stripper
(round 24), and onnx_qdq_strip_planner (round 25) modules, each with full
unit-test coverage on synthetic graph inputs.

### Bidirectional remote-sync workflow

Heavy compute happens on a Windows + RTX 5080 node; documentation, planning,
and figure generation happen on macOS. We implement
`sync_remote_windows_repo.py` with `--pull-reports` flag that performs
text-only artifact reverse-pull (`.json/.md/.csv/.svg/.png/.jpg/.log/.txt`)
via PowerShell-packed zip archives, sidestepping flaky scp behavior on the
cpolar SSH tunnel. This separates large binary artifacts (ONNX, engines,
weights) from text-only reproducible artifacts in the round-trip.

### Unified figure regeneration entry point

Four figure subsystems (speedup, cosine, tradeoff, layer ablation) each have
their own builder function. `build_all_figures.py` (round 23) provides a
single CLI entry that invokes all four with consistent `--allow-missing`
semantics and emits a top-level `figures_index.json` cross-subsystem index.
This makes figure regeneration idempotent and diff-friendly across runs.

## Limitations and Future Work

1. **ImageNet val unavailability** — `ILSVRC/imagenet-1k` is gated on
   Hugging Face; without VPN access on the Windows GPU node, we use
   Imagenette2-320 val (10 classes, 13K images) as a proxy for both
   calibration and evaluation. While Imagenette is a strict subset of
   ImageNet, the smaller class diversity may underestimate the worst-case
   quantization error of full-distribution evaluation.

2. **QAT not implemented** — V1.3 QAT (ADR-011) requires four conditions
   simultaneously: ImageNet unblock, ≥ 5 GPU-day on A100/H100, 1-2 month
   dedicated time including paper writing, and a downstream task baseline
   (depth/segmentation FP32 + DPT head training pipeline). None of the
   four are currently met; we provide the design ADR but defer
   implementation.

3. **TRT version dependence** — Several findings (BF16 + Myelin Fill
   incompatibility, `--layerPrecisions` no-op behavior under explicit
   Q/DQ) are specific to TRT 10.13.2.6 and may be resolved by upgrading
   to a future TRT version. We have not exhaustively tested TRT 10.16.1
   (the V1.0.1 ADR-008 ideal target version), which may close some of
   these gaps.

4. **Hardware specificity** — All measurements are on RTX 5080 (Blackwell
   sm_120). DINOv3 + TRT performance characteristics on Ada Lovelace
   (sm_89) or Hopper (sm_90) may differ; in particular, sm_89's lack of
   FP8 hardware support would invert the FP8 vs INT8 trade-off observed
   here.

## Project Artifacts

- **Decision documents** (Architecture Decision Records):
  - `Wiki/0-项目计划/项目计划报告_V1.0.1.md` ADR-001 through ADR-009
  - `Wiki/0-项目计划/ADR-010-V1.2-ONNX-Q-DQ-stripping_2026-05-01.md` (Implemented · Negative result)
  - `Wiki/0-项目计划/ADR-011-V1.3-QAT-future-work_2026-05-01.md` (Proposed)
- **Result indices**:
  - `Wiki/2-实验结果/M1-M6-当前验收矩阵_2026-04-30.md` (V1.0.0 main path frozen)
  - `Wiki/2-实验结果/V1.1-stretch-summary_2026-05-01.md` (V1.1 + V1.2 synthesis)
- **Reports**:
  - `Wiki/2-技术报告/技术报告_V1.0.0.md` (full engineering report)
  - `Wiki/2-技术报告/汇报材料_V1.0.0.md` (executive summary)
  - `Wiki/2-技术报告/答辩问答预案_V1.0.0.md` (10 Q&A defense playbook)
  - `Wiki/2-技术报告/复现与许可说明_V1.0.0.md` (reproducibility + license)
- **Machine-readable artifacts**:
  - `Code/Artifacts/reports/formal_benchmark_matrix.csv` (56 rows)
  - `Code/Artifacts/reports/figures/*.svg` (8 figures)
  - `Code/Artifacts/reports/figures/figures_index.json` (4-subsystem index)
  - `Code/Artifacts/reports/artifact_manifest_formal_with_sha256.json` (419+ files)
- **Progress log**:
  - `Wiki/0-项目计划/milestones/M1-progress.md` (32 heartbeat rounds)

## Citation

This work uses Meta DINOv3 ViT-L/16 LVD-1689M
(`facebook/dinov3-vitl16-pretrain-lvd1689m`) under the DINOv3 License
(repository copy: `LICENSES/DINOv3_LICENSE.md`). Built with DINOv3.
