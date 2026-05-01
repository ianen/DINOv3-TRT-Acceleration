# DINOv3 TensorRT Acceleration Code

This directory contains the executable research code for the project plan in
`../Wiki/0-项目计划`.

Built with DINOv3. DINOv3 Materials are governed by the DINOv3 License; the
repository-level copy is `../LICENSES/DINOv3_LICENSE.md`.

Current focus: **V1.0.0 + V1.1 stretch + V1.2 mixed-precision + V1.3 future
work all closed** (rounds 14-30 of `../Wiki/0-项目计划/milestones/M1-progress
.md`). Heavy artifacts such as weights, ONNX files, TensorRT engines,
calibration caches, and datasets stay outside git.

For the high-level status, decision tree, and Q&A index, start with the
project root README and `../Wiki/2-技术报告/答辩问答预案_V1.0.0.md`. This file
focuses on developer-facing commands.

### V1.1 + V1.2 + V1.3 command index (rounds 14-30 additions)

| Module / Script | Round | Purpose |
|---|---|---|
| `scripts/run_layer_ablation_pytorch.py` | 15 | 4-layer ablation PyTorch hooks (project / dpt / late) |
| `scripts/build_layer_ablation_figure.py` | 20 | layer ablation diversity-vs-balance SVG |
| `scripts/build_layer_precisions_arg.py` | 17 | trtexec `--layerPrecisions` arg generator from ONNX |
| `scripts/build_mixed_precision_engine_windows.py` | 18 | trtexec build via Python subprocess (bypasses cmd.exe limit) |
| `scripts/build_all_figures.py` | 23 | Unified entry for all 4 figure subsystems |
| `scripts/inspect_qdq_pairs_for_blocks.py` | 24 | V1.2 step 1 — identify Q/DQ pairs per block (ADR-010) |
| `scripts/strip_qdq_for_blocks.py` | 25 | V1.2 step 2 — strip internal Q/DQ pairs in ONNX |
| `src/dinov3_trt/quantization/layer_precision.py` | 17 | Pure-Python helper (no onnx dep) |
| `src/dinov3_trt/quantization/onnx_qdq_stripper.py` | 24 | Pure-Python pair classifier |
| `src/dinov3_trt/quantization/onnx_qdq_strip_planner.py` | 25 | Pure-Python strip planner with conflict detection |
| `src/dinov3_trt/reports/benchmark_figures.py` | various | LayerAblation + Tradeoff + Cosine + Speedup builders |

Total tests: **271 passing, 3 skipped** (pure-Python modules unit-testable on
macOS). All ruff/mypy gates pass on **111 source files**.

## Contract

- Model: `facebook/dinov3-vitl16-pretrain-lvd1689m`
- Output layers: 4, 12, 16, 20; zero-based indices are `3, 11, 15, 19`
- Main token contract at 224x224: `[B, 197, 1024]`
- Register tokens: 4 exist in the full model sequence, but the project main path
  uses the official API behavior that drops register tokens from intermediate
  outputs.
- ONNX: opset 18 or newer, dynamic batch only
- TensorRT: 10.13 or newer, ideally 10.16.1
- INT8: ModelOpt explicit Q/DQ is the main path; legacy calibrator is only a
  baseline comparison.

The canonical deployment contract remains 224x224. Export and trtexec build
helpers also accept one static `--image-size` per ONNX/engine so 336x336 and
518x518 benchmark补点 can be produced as separate artifacts with static spatial
profiles.

## Local quick start

```bash
cd Code
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
```

The repository host may not have the CUDA stack. The lightweight tests here do
not require Torch, ONNX, TensorRT, model weights, or ImageNet.

## Model assets

By default, local artifacts are expected under `Artifacts/` relative to the
current working directory:

- `Artifacts/source/dinov3/`
- `Artifacts/weights/dinov3-vitl16-pretrain-lvd1689m/`
- `Artifacts/onnx/dinov3_vitl16_4out.onnx`
- `Artifacts/engines/dinov3_vitl16_4out.{fp32,fp16}.engine`
- `Artifacts/engines/dinov3_vitl16_4out.bf16.prefer.engine`

Random-weight PoC artifacts use `.random` in the filename and are tracked by
`check_assets.py` separately:

- `Artifacts/onnx/dinov3_vitl16_4out.random.onnx`
- `Artifacts/engines/dinov3_vitl16_4out.random.fp32.engine`
- `Artifacts/engines/dinov3_vitl16_4out.random.fp16.engine`
- `Artifacts/engines/dinov3_vitl16_4out.random.timing.cache`
- `Artifacts/engines/dinov3_vitl16_4out.random.fp32.timing.cache`

Create the artifact directories and inspect what is already present:

```bash
python scripts/check_assets.py --create-dirs
```

The same command emits per-file `file_info` with sizes. Add `--with-sha256`
when a reproducibility manifest needs content digests for ONNX, engine, timing
cache, weight, or report files:

```bash
python scripts/check_assets.py --with-sha256
```

For formal result archival, require the formal assets and write a reproducible
manifest:

```bash
python scripts/check_assets.py \
  --require weights \
  --require onnx \
  --require fp32-engine \
  --require fp16-engine \
  --require bf16-engine \
  --require reports \
  --with-sha256 > Artifacts/reports/artifact_manifest_formal_with_sha256.json
```

The manifest includes both the canonical assets and directory-level
`onnx-artifacts` / `engine-artifacts` entries so INT8 sensitivity ONNX files,
engines, and timing caches are covered without making them deployment
candidates.

The reproducibility and license handoff note is maintained in
`../Wiki/2-技术报告/复现与许可说明_V1.0.0.md`. Keep HF tokens, weights, datasets,
ONNX files, TensorRT engines, and timing caches outside git; only the license
copy and reproducibility instructions are tracked.

Clone or validate the official DINOv3 source tree:

```bash
python scripts/prepare_dinov3_source.py
```

Download the gated Hugging Face weights after accepting the license and logging
in on the target machine:

```bash
python scripts/download_hf_snapshot.py
```

If HF access is not available yet, download the official source-compatible
ViT-L/16 LVD-1689M checkpoint:

```bash
python scripts/download_official_weights.py
```

The official source-tree export path requires a source-compatible `.pth`
checkpoint. HF `safetensors` snapshots are valid model artifacts, but they need
a Transformers-based export path rather than the local official source loader.
After installing the export extras and authenticating to the gated HF repo, use:

```bash
python -m pip install -e ".[export]"
python scripts/export_hf_dinov3_onnx.py \
  --model-path Artifacts/weights/dinov3-vitl16-pretrain-lvd1689m \
  --local-files-only \
  --output Artifacts/onnx/dinov3_vitl16_4out.onnx \
  --validate-no-if
```

On the Windows RTX 5080 host, after the formal HF snapshot is present and
verified, run the formal pipeline wrapper from `Code/`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_formal_hf_pipeline_windows.ps1
```

The wrapper uses system `python` for the Torch/Transformers ONNX export and
`.venv\Scripts\python.exe` for TensorRT tooling. It checks weights, exports and
inspects ONNX, builds FP16/FP32 engines plus a BF16-prefer candidate, writes
short `trtexec` benchmark reports, and compares FP32 vs FP16/BF16 on
deterministic batch 1 input. Use `-SkipBf16` only when reproducing the original
FP16/FP32-only run.

If the Windows GPU host cannot reach Hugging Face directly and is not on the
same LAN, prepare small fixed-size parts plus a sequential PowerShell downloader
for an SSH reverse HTTP tunnel:

```bash
python scripts/prepare_reverse_http_parts.py \
  --input Artifacts/weights/dinov3-vitl16-pretrain-lvd1689m/model.safetensors \
  --parts-dir /tmp/dinov3-weight-parts-8m \
  --script-output /tmp/download_dinov3_parts_8m.ps1 \
  --merge-script-output /tmp/merge_dinov3_parts_8m.ps1 \
  --remote-dir 'C:\Users\USER\dinov3_weight_parts_8m' \
  --remote-output 'D:\WorkPlace\ZMP\DINOv3-TRT-Acceleration\Code\Artifacts\weights\dinov3-vitl16-pretrain-lvd1689m\model.safetensors' \
  --expected-sha256 dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179 \
  --chunk-size-mib 8
```

Serve the generated parts from the local machine, then expose the server to the
Windows host through the existing SSH route:

```bash
cd /tmp/dinov3-weight-parts-8m
python3 -m http.server 8765 --bind 127.0.0.1
ssh -N -o ExitOnForwardFailure=yes -R 18765:127.0.0.1:8765 windows-pc
```

Run the generated PowerShell script on the Windows side, then concatenate the
parts with the generated merge script. The merge script validates every part
size, final byte count, and final SHA256 before the weight file is used for
export.

## ImageNet validation data

When Hugging Face access to `ILSVRC/imagenet-1k` is fully available, download
the validation parquet shards under `Artifacts/datasets/imagenet-1k-hf/data/`.
Convert them into the ImageNet-style image tree expected by the existing
manifest/eval/calib scripts:

```bash
python -m pip install -e ".[data]"
python scripts/export_hf_imagenet_parquet_images.py \
  --parquet-dir Artifacts/datasets/imagenet-1k-hf/data \
  --glob "validation-*.parquet" \
  --output-root Artifacts/datasets/imagenet-val \
  --manifest-output Artifacts/manifests/imagenet_val_all.json \
  --split validation
```

Then create disjoint eval/calib manifests from the exported image tree:

```bash
python scripts/prepare_image_subset_manifests.py \
  --image-root Artifacts/datasets/imagenet-val \
  --eval-output Artifacts/manifests/imagenet_eval_1000.json \
  --calib-output Artifacts/manifests/imagenet_calib_500.json \
  --eval-count 1000 \
  --calib-count 500
```

Current status on 2026-04-30: the logged-in account can list the 14 validation
parquet shards, but direct download of `data/validation-00000-of-00014.parquet`
still returns `403 GatedRepoError`.

Export any importable model factory that returns an official DINOv3-compatible
module with `get_intermediate_layers()`:

```bash
python scripts/export_onnx.py \
  --factory my_model_module:create_model \
  --validate-no-if
```

Before pretrained weights are available, validate the official source contract
with a randomly initialized ViT-L/16 on the GPU machine:

```bash
python scripts/check_official_dinov3_contract.py --device cuda --dtype float32
```

Export the official random-weight ONNX with the RoPE export patch enabled:

```bash
python scripts/export_official_dinov3_onnx.py \
  --device cuda \
  --dtype float32 \
  --output Artifacts/onnx/dinov3_vitl16_4out.random.onnx \
  --validate-no-if
```

Inspect the exported ONNX:

```bash
python scripts/inspect_onnx.py Artifacts/onnx/dinov3_vitl16_4out.random.onnx
```

## Remote RTX 5080 probe

The project's current GPU workstation is reachable as `windows-pc` and uses:

```text
D:\WorkPlace\ZMP\DINOv3-TRT-Acceleration
```

Run the probe from the repository root:

```bash
python Code/scripts/probe_remote_windows.py --host windows-pc
```

The probe prints JSON containing GPU, Python, Torch/CUDA, TensorRT Python package
availability, `trtexec` location, and whether the expected project directory
exists on Windows.

## TensorRT engine dry run

Once a 4-output ONNX file exists, use the existing TensorRT 10.13.2.6 install on
the Windows workstation through `trtexec` from the `Code/` directory:

```bash
python scripts/build_engine_trtexec.py \
  --onnx Artifacts/onnx/dinov3_vitl16_4out.onnx \
  --engine Artifacts/engines/dinov3_vitl16_4out.fp16.engine \
  --precision fp16 \
  --dry-run
```

The generated command uses the project profile:

- input name: `pixel_values`
- min/opt/max batch: `1/8/32`
- static resolution: `224x224`
- `--noTF32`
- workspace: `4G`

Formal HF ViT-L/16 weights currently expose a TensorRT FP16 correctness issue:
the default FP16 engine can build and benchmark, but Python TensorRT runtime
checks show all four outputs become `NaN`. The comparison path now fails fast on
non-finite outputs instead of writing `NaN` metrics. To reproduce block-level
mixed-precision diagnostics, keep complete transformer blocks in FP32:

```bash
python scripts/build_engine_trtexec.py \
  --onnx Artifacts/onnx/dinov3_vitl16_4out.onnx \
  --engine Artifacts/engines/dinov3_vitl16_4out.fp16.blocksfp32.engine \
  --precision fp16 \
  --fp32-transformer-blocks 0-19 \
  --timing-cache Artifacts/engines/dinov3_vitl16_4out.fp16.blocksfp32.timing.cache
```

`--fp32-transformer-blocks` expands to TensorRT layer specs such as
`/model/layer.0/*:fp32` for both `--layerPrecisions` and
`--layerOutputTypes`. The all-block FP32 diagnostic engine is finite and close
to the FP32 baseline, but its speed is also FP32-like; it is a correctness
diagnostic, not the final acceleration target.

On RTX 5080 / TensorRT 10.13, the current best finite low-precision candidate is
BF16 with a global layer precision preference:

```bash
python scripts/build_engine_trtexec.py \
  --onnx Artifacts/onnx/dinov3_vitl16_4out.onnx \
  --engine Artifacts/engines/dinov3_vitl16_4out.bf16.prefer.engine \
  --precision bf16 \
  --precision-constraints prefer \
  --layer-precision "*:bf16" \
  --timing-cache Artifacts/engines/dinov3_vitl16_4out.bf16.prefer.timing.cache
```

This engine uses real BF16 tactics, keeps output bindings in FP32, avoids the
formal FP16 `NaN` failure, and should be compared against FP32 before reporting
speedups:

```bash
python scripts/compare_trt_engines.py \
  --reference-engine Artifacts/engines/dinov3_vitl16_4out.fp32.engine \
  --candidate-engine Artifacts/engines/dinov3_vitl16_4out.bf16.prefer.engine \
  --output Artifacts/reports/compare_fp32_vs_bf16_prefer_b1.json \
  --batch-size 1
```

Use `--input-mode random-normal|uniform-0-1|zeros|ones` to repeat the same
comparison on basic deterministic input distributions.

For V1.0.1's real-image eval/calibration split, prepare disjoint manifests from
an ImageNet-style class-folder directory:

```bash
python scripts/prepare_image_subset_manifests.py \
  --image-root Artifacts/datasets/imagenet-val \
  --eval-output Artifacts/manifests/imagenet_eval_1000.json \
  --calib-output Artifacts/manifests/imagenet_calib_500.json \
  --eval-count 1000 \
  --calib-count 500 \
  --seed 20260430
```

When full ImageNet val access is not ready, Imagenette2-320 can be used as a
public ImageNet-style real-image subset to unblock the eval/calibration
pipeline:

```bash
mkdir -p Artifacts/datasets
curl -L -C - --fail --retry 5 --retry-delay 5 \
  -o Artifacts/datasets/imagenette2-320.tgz \
  https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar -xzf Artifacts/datasets/imagenette2-320.tgz -C Artifacts/datasets
python scripts/prepare_image_subset_manifests.py \
  --image-root Artifacts/datasets/imagenette2-320/val \
  --eval-output Artifacts/manifests/imagenette_eval_1000.json \
  --calib-output Artifacts/manifests/imagenette_calib_500.json \
  --eval-count 1000 \
  --calib-count 500 \
  --seed 20260430
```

Then evaluate an engine pair on the manifest and aggregate per-output metrics:

```bash
python scripts/evaluate_engine_pair_on_images.py \
  --reference-engine Artifacts/engines/dinov3_vitl16_4out.fp32.engine \
  --candidate-engine Artifacts/engines/dinov3_vitl16_4out.bf16.prefer.engine \
  --manifest Artifacts/manifests/imagenet_eval_1000.json \
  --output Artifacts/reports/eval_imagenet_fp32_vs_bf16_prefer.json \
  --batch-size 8
```

Current RTX 5080 real-image BF16 check uses
`Artifacts/manifests/imagenette_selected_eval_1000.json` and writes
`Artifacts/reports/eval_imagenette1000_fp32_vs_bf16_prefer.json`. Its
per-layer cosine mean is `0.9999535 / 0.9997878 / 0.9993771 / 0.9991266`
for `feat_layer_4/12/16/20`.

Before starting INT8, verify the ModelOpt/Polygraphy/ONNX Runtime/TensorRT
Python stack, CUDA visibility, and eval/calib manifests:

```bash
python scripts/check_quant_prereqs.py \
  --eval-manifest Artifacts/manifests/imagenet_eval_1000.json \
  --calib-manifest Artifacts/manifests/imagenet_calib_500.json
```

Use `--allow-missing-data` only for environment-only checks before the real
ImageNet subset has been staged. INT8 is still expected to use the ModelOpt
explicit Q/DQ ONNX path; the legacy TensorRT calibrator remains a baseline only.

Run ModelOpt ONNX PTQ from the calibration manifest:

```bash
python scripts/quantize_onnx_modelopt.py \
  --onnx Artifacts/onnx/dinov3_vitl16_4out.onnx \
  --calib-manifest Artifacts/manifests/imagenet_calib_500.json \
  --output Artifacts/onnx/dinov3_vitl16_4out.int8.modelopt.onnx \
  --calibration-method max \
  --high-precision-dtype fp32
```

Use `--dry-run --max-calibration-images 2` with smoke manifests to validate
calibration loading without writing a quantized ONNX.

For ablations, the quantization entry point exposes ModelOpt controls such as
`--op-types-to-quantize`, `--op-types-to-exclude`, `--nodes-to-exclude`,
`--disable-mha-qdq`, and `--mha-accumulation-dtype`. ModelOpt calibration EPs
use aliases such as `cpu`, `cuda:0`, and `trt`, not ONNX Runtime provider names.
For example:

```bash
python scripts/quantize_onnx_modelopt.py \
  --onnx Artifacts/onnx/dinov3_vitl16_4out.onnx \
  --calib-manifest Artifacts/manifests/imagenet_calib_500.json \
  --output Artifacts/onnx/dinov3_vitl16_4out.int8.modelopt.matmul_only.onnx \
  --calibration-eps cpu \
  --op-types-to-quantize MatMul \
  --high-precision-dtype fp32 \
  --mha-accumulation-dtype fp32
```

Build a TensorRT engine from the explicit Q/DQ ONNX with the normal engine
builder:

```bash
python scripts/build_engine_trtexec.py \
  --onnx Artifacts/onnx/dinov3_vitl16_4out.int8.modelopt.onnx \
  --engine Artifacts/engines/dinov3_vitl16_4out.int8.modelopt.engine \
  --precision int8 \
  --timing-cache Artifacts/engines/dinov3_vitl16_4out.int8.modelopt.timing.cache
```

The 2-image smoke calibration path is expected to validate plumbing only. Treat
its numerical metrics as invalid unless the eval/calib manifests use the planned
representative image subsets.

Before attributing an INT8 failure to TensorRT, compare the FP32 ONNX and the
explicit Q/DQ ONNX directly with ONNX Runtime:

```bash
python scripts/compare_onnx_outputs.py \
  --reference-onnx Artifacts/onnx/dinov3_vitl16_4out.onnx \
  --candidate-onnx Artifacts/onnx/dinov3_vitl16_4out.int8.modelopt.onnx \
  --output Artifacts/reports/compare_onnx_fp32_vs_int8_modelopt_b1.json \
  --batch-size 1 \
  --providers CPUExecutionProvider
```

For image-manifest diagnostics:

```bash
python scripts/evaluate_onnx_pair_on_images.py \
  --reference-onnx Artifacts/onnx/dinov3_vitl16_4out.onnx \
  --candidate-onnx Artifacts/onnx/dinov3_vitl16_4out.int8.modelopt.onnx \
  --manifest Artifacts/manifests/imagenet_eval_1000.json \
  --output Artifacts/reports/eval_onnx_imagenet_fp32_vs_int8_modelopt.json \
  --batch-size 8 \
  --providers CPUExecutionProvider
```

If Q/DQ ONNX already has zero output norms or poor cosine in ONNX Runtime, the
issue is in the quantized ONNX/calibration setup rather than TensorRT engine
building.

Current real-calib findings on RTX 5080:

- Default ModelOpt with `imagenette_selected_calib_500.json` still produces
  zero candidate L2 norm for `feat_layer_4/12/16` in ONNX Runtime.
- Real64 ablations show `MatMul-only` avoids zero norms but remains inaccurate,
  while adding `LayerNormalization` or `Add` reproduces the zero-norm collapse.
  Adding `Mul` avoids zero norms but degrades cosine heavily. Continue INT8 work
  with MatMul node-level whitelists or block-level sensitivity experiments
  before building more TensorRT INT8 engines.
- A node whitelist that quantizes only MatMul nodes in `/model/layer.16` through
  `/model/layer.19` is controllable: `feat_layer_4/12/16` remain unchanged, and
  TensorRT 1000-image Imagenette eval reports `feat_layer_20` cosine mean
  `0.989177`. It is useful for sensitivity analysis, but its benchmark is only
  about `1.44x-1.51x` faster than FP32 and slower than BF16 prefer, so it is not
  the current deployment candidate.
- A narrower whitelist that quantizes only `/model/layer.19` MatMul nodes raises
  TensorRT 1000-image Imagenette `feat_layer_20` cosine mean/min to
  `0.995659 / 0.995549`, while the first three outputs remain effectively
  unchanged. Its locked `trtexec` GPU median speedup is only
  `1.05x / 1.07x / 1.06x` over FP32 and `0.43x / 0.38x / 0.33x` relative to
  BF16 prefer, so it improves INT8 correctness but gives up the acceleration
  needed for a deployable candidate.

Run the follow-up MatMul-only block sweep before building more TensorRT INT8
engines:

```bash
python scripts/run_modelopt_matmul_block_sweep.py \
  --variant 19 \
  --variant 18-19 \
  --variant 17-19 \
  --variant 16-19 \
  --summary-output Artifacts/reports/modelopt_matmul_block_sweep_imagenette64_summary.json \
  --skip-existing
```

Use inline groups or `--node-group` for finer layer-internal sensitivity:

```bash
python scripts/run_modelopt_matmul_block_sweep.py \
  --variant 19 \
  --node-group attention \
  --node-group mlp \
  --prefix imagenette64_matmul_fine \
  --summary-output Artifacts/reports/modelopt_matmul_fine_sweep_imagenette64_summary.json \
  --skip-existing
```

Supported node groups are `all`, `attention`, `qkv`, `attention-core`,
`attention-out`, `mlp`, `mlp-up`, and `mlp-down`. Inline specs such as
`--variant 19:mlp` override the global `--node-group` list.

The sweep creates exact node whitelists such as
`/model/layer.18/attention/q_proj/MatMul`, runs ModelOpt with
`--op-types-to-quantize MatMul`, then checks each generated Q/DQ ONNX with both
deterministic random input and a 32-image Imagenette manifest.

Current sweep result on the 32-image ONNX Runtime gate: `layer19` reaches
`feat_layer_20` cosine `0.9956007`, `layers18_19` reaches `0.9930622`, and
`layers17_19` reaches `0.9908769`; all leave `feat_layer_4/12/16` unchanged.
The older `layers16_19` point remains below the 0.99 threshold on the same gate
at about `0.988997`, which confirms the expected quality/speed tradeoff as more
late blocks are quantized.

The C++ TensorRT runtime has also been run on the formal engines. BF16 prefer
passes finite-output smoke and reaches end-to-end latency speedups of about
`2.27x / 2.47x / 2.83x` over FP32 for batch `1 / 8 / 32`. The partial INT8
engine is only about `1.17x-1.23x` faster than FP32 in the same C++ runtime and
is slower than BF16 prefer.

Formal `trtexec` reports should use the locked-clock + spin-wait files for
publishable GPU compute comparisons. On the RTX 5080 host, BF16 prefer reaches
`2.45x / 2.81x / 3.25x` GPU median latency speedup over FP32 for batch
`1 / 8 / 32` in
`Artifacts/reports/trtexec_formal_fp32_vs_bf16_prefer_locked2752_spinwait_speedup.json`.
Additional locked batch `4 / 16`补点 reports show BF16 prefer speedup
`2.55x / 3.08x` in
`Artifacts/reports/trtexec_formal_fp32_vs_bf16_prefer_b4_b16_locked2752_spinwait_speedup.json`.
The partial INT8 MatMul layers16-19 engine reaches only
`1.18x / 1.22x / 1.22x` over FP32 and is slower than BF16 prefer
(`0.48x / 0.43x / 0.38x`), so it remains sensitivity evidence rather than the
deployable candidate.
The layer19-only variant improves 1000-image cosine to `0.995659` at
`feat_layer_20`, but locked `trtexec` speedup falls to
`1.05x / 1.07x / 1.06x` over FP32 and remains much slower than BF16 prefer.
The same layer19-only engine is bit-identical between Python and C++ runtimes,
but C++ runtime speedup is still only `1.07x / 1.08x / 1.06x` over FP32 and
`0.47x / 0.43x / 0.37x` relative to BF16 prefer.
Finer `layer19_attention` gives even higher real-image cosine at
`feat_layer_20` (`0.998994` mean, `0.998941` min), but locked `trtexec` speedup
is only `1.04x / 1.05x / 1.04x` over FP32 and
`0.42x / 0.37x / 0.32x` relative to BF16 prefer. This confirms that
fine-grained late-layer MatMul INT8 fixes quality by reducing quantized scope,
but does not recover enough speed to challenge BF16 prefer.

For cross-language parity, dump C++ runtime outputs and compare them against the
Python TensorRT runtime on the same deterministic input:

```bash
python scripts/compare_cpp_python_parity.py \
  --engine Artifacts/engines/dinov3_vitl16_4out.bf16.prefer.engine \
  --cpp-runner build/cpp-trt-inspect-msvc/dinov3_trt_dump_outputs.exe \
  --dump-dir Artifacts/reports/cpp_python_parity_bf16_prefer_b1_dump \
  --output Artifacts/reports/cpp_python_parity_bf16_prefer_b1.json \
  --batch-size 1
```

Current RTX 5080 parity reports for FP32, BF16 prefer, and partial INT8 are
bit-identical between Python and C++ runtimes (`max_abs_error=0`, `cosine=1` for
all four outputs).

After the real-image eval and speedup reports exist, build the consolidated
formal summary:

```bash
python scripts/build_formal_report_summary.py \
  --reports-dir Artifacts/reports \
  --output-json Artifacts/reports/formal_summary.json \
  --output-md Artifacts/reports/formal_summary.md
```

The current summary decision is BF16 prefer as the deployable candidate; partial
INT8 remains sensitivity evidence.

Build the formal benchmark matrix CSV/JSON/Markdown from the same speedup
reports:

```bash
python scripts/build_benchmark_matrix.py \
  --reports-dir Artifacts/reports \
  --output-json Artifacts/reports/formal_benchmark_matrix.json \
  --output-csv Artifacts/reports/formal_benchmark_matrix.csv \
  --output-md Artifacts/reports/formal_benchmark_matrix.md
```

The current matrix covers 224x224 locked `trtexec` GPU compute and C++ runtime
rows for BF16 prefer, late-block partial INT8, `layer19`, and
`layer19_attention`, plus 336x336 locked `trtexec` BF16/FP32 rows for batch
`1/4/8`. It is the current P5 handoff artifact for the report table; future
ImageNet or 518/batch补点 reports should be added through this same entry point.

Build lightweight SVG figures for the technical report from the matrix:

```bash
python scripts/build_benchmark_figures.py \
  --matrix-csv Artifacts/reports/formal_benchmark_matrix.csv \
  --output-dir Artifacts/reports/figures
```

This writes `benchmark_trtexec_bf16_speedup.svg`,
`benchmark_trtexec_int8_speedup.svg`, `benchmark_cpp_runtime_speedup.svg`, and
`benchmark_figures_manifest.json`.

For the random-weight PoC, the same command is:

```bash
python scripts/build_engine_trtexec.py \
  --onnx Artifacts/onnx/dinov3_vitl16_4out.random.onnx \
  --engine Artifacts/engines/dinov3_vitl16_4out.random.fp16.engine \
  --precision fp16 \
  --timing-cache Artifacts/engines/dinov3_vitl16_4out.random.timing.cache
```

For the random-weight FP32 baseline:

```bash
python scripts/build_engine_trtexec.py \
  --onnx Artifacts/onnx/dinov3_vitl16_4out.random.onnx \
  --engine Artifacts/engines/dinov3_vitl16_4out.random.fp32.engine \
  --precision fp32 \
  --timing-cache Artifacts/engines/dinov3_vitl16_4out.random.fp32.timing.cache
```

Smoke-test a saved engine:

```bash
trtexec \
  --loadEngine=Artifacts/engines/dinov3_vitl16_4out.random.fp16.engine \
  --shapes=pixel_values:1x3x224x224 \
  --duration=3 \
  --warmUp=200
```

Write a repeatable JSON benchmark report for one or more batch sizes:

```bash
python scripts/benchmark_trtexec.py \
  --engine Artifacts/engines/dinov3_vitl16_4out.random.fp16.engine \
  --output Artifacts/reports/trtexec_random_fp16_smoke.json \
  --batches 1,8,32 \
  --duration 10
```

Add `--use-spin-wait` for a lower-jitter benchmark pass. GPU clocks still need
to be locked for publishable numbers. On the Windows RTX 5080 host, lock and
reset graphics clocks around the benchmark run:

```bash
nvidia-smi -lgc 2752,2752
python scripts/benchmark_trtexec.py \
  --engine Artifacts/engines/dinov3_vitl16_4out.random.fp16.engine \
  --output Artifacts/reports/trtexec_random_fp16_locked2752.json \
  --batches 1,8,32 \
  --duration 10 \
  --use-spin-wait
nvidia-smi -rgc
```

Summarize paired FP32/FP16 benchmark reports into JSON and Markdown speedup
tables:

```bash
python scripts/summarize_trtexec_benchmarks.py \
  --reference-report Artifacts/reports/trtexec_random_fp32_locked2752.json \
  --candidate-report Artifacts/reports/trtexec_random_fp16_locked2752.json \
  --reference-label FP32 \
  --candidate-label FP16 \
  --output-json Artifacts/reports/trtexec_random_locked2752_speedup.json \
  --output-md Artifacts/reports/trtexec_random_locked2752_speedup.md
```

Compare two saved TensorRT engines with the same deterministic input and write
per-output numerical metrics:

```bash
python scripts/compare_trt_engines.py \
  --reference-engine Artifacts/engines/dinov3_vitl16_4out.random.fp32.engine \
  --candidate-engine Artifacts/engines/dinov3_vitl16_4out.random.fp16.engine \
  --output Artifacts/reports/trt_random_fp32_vs_fp16_b1.json \
  --batch-size 1 \
  --seed 20260429 \
  --input-mode random-normal
```

This Python runtime path needs TensorRT Python plus `cuda-python` in the target
environment:

```bash
python -m pip install -e ".[trt]"
```

Use the FP32 engine path and report filename for the FP32 baseline:

```bash
python scripts/benchmark_trtexec.py \
  --engine Artifacts/engines/dinov3_vitl16_4out.random.fp32.engine \
  --output Artifacts/reports/trtexec_random_fp32_smoke.json \
  --batches 1,8,32 \
  --duration 10
```

## Remote source sync

When the Windows workspace exists but is missing the current repository content,
sync only the project sources and docs:

```bash
python Code/scripts/sync_remote_windows_repo.py --host windows-pc
```

The sync command uploads a filtered zip bundle of `CLAUDE.md`, `.gitignore`,
`.claude/`, `Wiki/`, and `Code/`, while skipping local caches, virtualenvs,
datasets, weights, ONNX files, TensorRT engines, and other ignored artifacts. It
extracts into `D:\WorkPlace\ZMP\DINOv3-TRT-Acceleration` and initializes a git
repository there if `git` exists and `.git/` is absent.

## C++ TensorRT runtime

The C++ surface contains the low-level contract types plus a TensorRT runtime
smoke path:

- `cpp/include/dinov3_trt/status.h`
- `cpp/include/dinov3_trt/tensor.h`
- `cpp/include/dinov3_trt/preprocess.h`
- `cpp/include/dinov3_trt/trt_inferer.h`
- `cpp/src/trt_inferer.cpp`
- `cpp/tools/inspect_engine.cpp`
- `cpp/tools/runtime_smoke.cpp`

Build the current contract test without requiring TensorRT headers or libraries:

```bash
cmake -S cpp -B build/cpp -G Ninja
cmake --build build/cpp
ctest --test-dir build/cpp --output-on-failure
```

On Windows, TensorRT C++ utilities must be built with the MSVC toolchain that
matches NVIDIA's import libraries. MinGW can compile the headers but is not a
valid runtime ABI for `nvinfer_10.lib`.

```cmd
"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64
cd /d D:\WorkPlace\ZMP\DINOv3-TRT-Acceleration\Code
set "TENSORRT_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT-10.13.2.6"
cmake -S cpp -B build\cpp-trt-inspect-msvc -G Ninja ^
  -DCMAKE_CXX_COMPILER=cl ^
  -DDINOV3_TRT_CPP_ENABLE_TENSORRT=ON
cmake --build build\cpp-trt-inspect-msvc
```

The same build is wrapped for the remote Windows workstation:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_cpp_trt_inspector_windows.ps1
```

Inspect a saved engine's TensorRT I/O metadata:

```cmd
build\cpp-trt-inspect-msvc\dinov3_trt_inspect_engine.exe ^
  Artifacts\engines\dinov3_vitl16_4out.random.fp16.engine
```

Or generate FP16/FP32 random-engine metadata reports:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\inspect_cpp_engines_windows.ps1
```

Run a real C++ TensorRT enqueue smoke with deterministic host input, CUDA device
buffers, `enqueueV3`, and host output statistics:

```cmd
build\cpp-trt-inspect-msvc\dinov3_trt_runtime_smoke.exe ^
  Artifacts\engines\dinov3_vitl16_4out.random.fp16.engine ^
  1
```

Or generate FP16/FP32 random-engine runtime smoke reports. The same script can
be parameterized with `-BatchSize 8` or `-BatchSize 32` to validate the dynamic
batch profile:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_cpp_runtime_smoke_windows.ps1
```

The generated reports live under `Artifacts\reports\` and contain the input
shape plus per-output shape, element count, finite count, min/max, mean, and
RMS. For the random-weight engines, batches `1/8/32` have been validated on the
RTX 5080 for both FP16 and FP32 engines.

Run the C++ runtime benchmark. This measures the `TRTInferer::infer` endpoint:
host-to-device copy, TensorRT enqueue, device-to-host copy, and stream
synchronization. It is intentionally different from `trtexec` GPU compute time.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\benchmark_cpp_runtime_windows.ps1 `
  -WarmupIterations 10 `
  -Iterations 50
```

Summarize FP32 vs FP16 C++ runtime reports:

```cmd
.venv\Scripts\python.exe scripts\summarize_cpp_runtime_benchmarks.py ^
  --reference-report Artifacts\reports\cpp_runtime_benchmark_random_fp32.json ^
  --candidate-report Artifacts\reports\cpp_runtime_benchmark_random_fp16.json ^
  --reference-label FP32-CppRuntime ^
  --candidate-label FP16-CppRuntime ^
  --output-json Artifacts\reports\cpp_runtime_random_fp32_vs_fp16_speedup.json ^
  --output-md Artifacts\reports\cpp_runtime_random_fp32_vs_fp16_speedup.md
```

The current random-weight RTX 5080 C++ runtime median latency speedups are:
batch 1 `3.27x`, batch 8 `3.46x`, and batch 32 `4.09x`.
