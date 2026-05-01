# DINOv3 TensorRT Acceleration

本项目研究 Meta DINOv3 ViT-L/16 在 NVIDIA TensorRT 上的推理加速，覆盖 PyTorch/HF 权重、ONNX 导出、TensorRT engine 构建、真实图片精度评估、ModelOpt INT8 定位、`trtexec` benchmark、C++ runtime 与 Python/C++ parity。

Built with DINOv3. DINOv3 Materials 受 DINOv3 License 约束；仓库级副本见 `LICENSES/DINOv3_LICENSE.md`。

## 当前状态

截至 2026-05-01，**V1.0.0 主线 + V1.1 stretch + V1.2 mixed-precision + V1.3 future work 决策树全部就位**（V1.0.1 计划 + 第 14-29 轮心跳）：

- **当前有效候选：BF16 prefer TensorRT engine**（唯一在 G2 ideal region 的候选）。
  - 224/336/518 三档全覆盖；trtexec GPU median 顶点 `3.86×`（r518 b8），cpp end-to-end 顶点 `3.40×`（r518 b8）。
  - 1000 张真实图片 eval 三档分辨率最低 cosine ≥ 0.998。
- **FP16 engine** 在正式权重下输出全 NaN — 已记录为 Blackwell sm_120 + TRT 10.13 + DINOv3 RoPE 的工程负例。
- **INT8 路径全部 negative 闭合**：默认 ModelOpt 塌缩 / 节点级 partial 速度收益消失 / SmoothQuant α=0.8 best cos_mean 0.982 (cos_min 0.968) speed 3.48×（speed 达标但 cos 差 0.022）/ V1.0+V1.1+V1.2 三层 mixed-precision 工具链（ModelOpt skip 16-19 / trtexec --layerPrecisions / ONNX graph-level Q/DQ stripping）等价 negative，cos_min 都在 0.97 附近。**Root cause**：前段 blocks 0-15 累积 INT8 量化噪声 — 详见 ADR-010 § 5.3。
- **V1.3 方向**：QAT 量化感知 fine-tuning（ADR-011，**Proposed** 状态，4 条启动门槛全状态可查）。
- **跨语言一致性**：Python ↔ C++ parity 在 224/336/518 三档 batch 1 全部 bit-identical（max_abs=0, cosine=1.0）。
- **Benchmark 矩阵**：**87 行**机器可读 CSV（含 V1.1 mixed l16-19:fp32 + V1.2 ONNX-stripped + r336/r518 b16/b32 高 batch 数据），8 张 SVG（含 12 点 tradeoff 散点 + 4 层 ablation diversity-vs-balance）。
- **r336/r518 高 batch memory-bound 实证**（第 44 轮）：r336/r518 b≥16 进入 memory-bound 区，BF16 vs FP32 speedup ≈ 1×（vs r518 b8 顶点 3.86×），与 R5 VRAM 风险登记同时是 batch saturation 边界证据。
- **完整 ImageNet val** 仍被 Hugging Face gated `403 GatedRepoError` 阻塞；当前真实图片口径为公开 Imagenette2-320 val（1000 eval / 500 calib 互斥）。

## 目录

| 路径 | 说明 |
|---|---|
| `Wiki/INDEX.md` | **顶层 Wiki 导航**（20+ 份 .md 文档按 9 种 tone 分类 + 不同 stakeholder 入口路径 + 完整决策树 + 心跳索引） |
| `Code/` | 可执行代码、脚本、C++ runtime、轻量测试（323 pytest 用例 + 112 Python 源文件 ruff/mypy 全绿 + line coverage **81%**（≥ V1.0.1 §12.1 80% 阈值）） |
| `Code/README.md` | 详细命令入口（远端 RTX 5080、导出、量化、benchmark） |
| `Wiki/0-项目计划/项目计划报告_V1.0.1.md` | 主计划 + ADR-001 ~ ADR-009（frozen） |
| `Wiki/0-项目计划/ADR-010-V1.2-ONNX-Q-DQ-stripping_2026-05-01.md` | V1.2 ONNX 层 Q/DQ stripping 设计 + 实施 + 实测（**Implemented · Negative result**） |
| `Wiki/0-项目计划/ADR-011-V1.3-QAT-future-work_2026-05-01.md` | V1.3 QAT 量化感知 fine-tuning 设计文档（**Proposed**，4 条启动门槛） |
| `Wiki/0-项目计划/milestones/M1-progress.md` | 持续推进记录（**45+ 轮心跳**） |
| `Wiki/2-实验结果/M1-正式结果摘要_2026-04-30.md` | V1.0.0 正式结果摘要 |
| `Wiki/2-实验结果/M1-M6-当前验收矩阵_2026-04-30.md` | V1.0.0 主线验收矩阵（frozen） |
| `Wiki/2-实验结果/V1.1-stretch-summary_2026-05-01.md` | V1.1 stretch + V1.2 实施 + ADR-011 引用综合表 |
| `Wiki/2-技术报告/技术报告_V1.0.0.md` | 技术报告（详细版） |
| `Wiki/2-技术报告/汇报材料_V1.0.0.md` | 答辩/汇报版执行摘要（含 V1.0+V1.1+V1.2 三层闭合证据链 + 12 点 tradeoff 解读） |
| `Wiki/2-技术报告/答辩问答预案_V1.0.0.md` | **答辩 10 大 Q&A + 通用速答模板**（5 秒答 + 30 秒展开 + 引用产物） |
| `Wiki/2-技术报告/research_contributions_V1.0.0.md` | Academic-tone 论文 abstract + 6 contributions（论文 intro 素材种子） |
| `Wiki/2-技术报告/PPT_outline_V1.0.0.md` | 答辩 PPT 18 页大纲 + speaker notes + 答辩 Q 映射 |
| `Wiki/2-技术报告/ppt_slides/output/DINOv3-TRT-Acceleration_V1.0.0.pptx` | **答辩 PPTX 终稿**（18 slides, 583 KB, 5 SVG embedded） |
| `Wiki/2-技术报告/paper_*_draft_V1.0.0.md` × 6 | 学术论文 IMRaD 8 段完整 draft（Abstract+Intro / LitReview / Method / Results+5tables / Discussion / Limit+Concl，~7700 词 EN） |
| `Wiki/2-技术报告/R2_emergency_acceptance_analysis_V1.0.0.md` | **V1.0.1 R2 应急方案 verdict**（cos_mean 4/4 ≥ 0.97 ✅；cos_min 2/4，feat_layer_16/20 缺口 0.0001/0.0017） |
| `Wiki/2-技术报告/复现与许可说明_V1.0.0.md` | 复现与许可说明 |
| `LICENSES/DINOv3_LICENSE.md` | DINOv3 License 副本 |

## 关键结果

### BF16 prefer

Imagenette1000 真实图片 eval 中，BF16 prefer 相对 FP32 的逐输出 cosine 摘要（三档分辨率均覆盖）：

| resolution | tokens | feat_layer_4 mean | feat_layer_12 mean | feat_layer_16 mean | feat_layer_20 mean | 最低 cosine |
|---:|---:|---:|---:|---:|---:|---:|
| 224 | 197 | 0.999953 | 0.999788 | 0.999377 | 0.999127 | 0.998749 |
| 336 | 442 | 0.999947 | 0.999766 | 0.999432 | 0.999360 | 0.998394 |
| 518 | 1025 | 0.999945 | 0.999800 | 0.999655 | 0.999721 | 0.998604 |

`feat_layer_20`（最深层）在 r518 上 cosine_mean `0.999721` 反而高于 r224 的 `0.999127`，因为 1024 个 patch token 形成更强的 cosine 平均化效应。最低 cosine 三档均 ≥ 0.998，无 NaN/Inf/零范数。

Locked `trtexec` GPU median latency speedup：

| resolution | batch | speedup |
|---:|---:|---:|
| 224 | 1 | 2.45x |
| 224 | 4 | 2.55x |
| 224 | 8 | 2.81x |
| 224 | 16 | 3.08x |
| 224 | 32 | 3.25x |
| 336 | 1 | 2.80x |
| 336 | 4 | 2.96x |
| 336 | 8 | 3.25x |
| 518 | 1 | 3.12x |
| 518 | 2 | 3.50x |
| 518 | 4 | 3.76x |
| 518 | 8 | 3.86x |

C++ runtime end-to-end latency speedup（BF16 prefer vs FP32，warmup 10 + iter 50，锁频 2752 MHz）：

| resolution | batch | speedup |
|---:|---:|---:|
| 224 | 1 | 2.27x |
| 224 | 8 | 2.47x |
| 224 | 32 | 2.83x |
| 336 | 1 | 2.53x |
| 336 | 4 | 2.60x |
| 336 | 8 | 2.84x |
| 518 | 1 | 2.82x |
| 518 | 4 | 3.30x |
| 518 | 8 | 3.40x |

518 batch `8` 的 BF16-prefer 与 FP32 engine 用独立 profile `min=1, opt=4, max=8` 单独构建（`*.r518.fp32.b8.engine` / `*.r518.bf16.prefer.b8.engine`），16 GB VRAM 上 build/inference 均正常。

### INT8 / FP8（V1.1 stretch）

默认 ModelOpt INT8 仍不可用。后段 MatMul partial INT8 的结论是：

- `layers16-19`：`feat_layer_20` cosine mean `0.989177`，但 locked `trtexec` 仅 `1.18x / 1.22x / 1.22x` vs FP32。
- `layer19`：cosine mean `0.995659`，速度仅 `1.05x / 1.07x / 1.06x` vs FP32。
- `layer19_attention`：cosine mean `0.998994`，速度仅 `1.04x / 1.05x / 1.04x` vs FP32。

INT8 缩小量化范围后可恢复正确性，但速度收益几乎消失，且明显慢于 BF16 prefer。

V1.1 stretch goal **FP8 PTQ via TensorRT Model Optimizer**（Blackwell sm_120 5th-gen Tensor Core）已开启实验：默认 FP8 ModelOpt PTQ + imagenette 500 校准取得**当前所有候选最高速度**（vs FP32 batch 1/8/32 = `2.93× / 4.70× / 5.05×`，vs BF16 prefer = `1.20× / 1.67× / 1.55×`），但 1000 张真实图片 cosine_mean 0.20（feat_layer_20），与默认 INT8 同模式塌缩。FP8 partial sensitivity sweep 是后续工作；当前 BF16 prefer 仍是唯一可部署候选。

## 主力环境

远端 Windows RTX 5080 工作站：

- SSH：`ssh windows-pc`
- 目录：`D:\WorkPlace\ZMP\DINOv3-TRT-Acceleration`
- GPU：NVIDIA GeForce RTX 5080
- TensorRT：10.13.2.6
- CUDA：12.8
- Python：3.10.10
- PyTorch：2.12.0.dev20260408+cu128

## 快速检查

本地轻量验证：

```bash
cd Code
.venv/bin/python -m pytest tests/test_benchmark_matrix.py tests/test_formal_summary.py tests/test_remote_sync.py
.venv/bin/python -m ruff check src scripts tests
.venv/bin/python -m mypy src scripts tests
```

同步代码和文档到 Windows（local → remote）：

```bash
cd Code
.venv/bin/python scripts/sync_remote_windows_repo.py --host windows-pc --no-git-init
```

把远端生成的 reports（SVG / matrix / manifest 等 text-only 产物）回拉本地（remote → local）：

```bash
cd Code
.venv/bin/python scripts/sync_remote_windows_repo.py --pull-reports --host windows-pc
```

远端生成正式 benchmark matrix：

```bash
ssh windows-pc 'cmd /c "cd /d D:\WorkPlace\ZMP\DINOv3-TRT-Acceleration\Code && .venv\Scripts\python.exe scripts\build_benchmark_matrix.py --reports-dir Artifacts\reports --output-json Artifacts\reports\formal_benchmark_matrix.json --output-csv Artifacts\reports\formal_benchmark_matrix.csv --output-md Artifacts\reports\formal_benchmark_matrix.md"'
```

远端生成正式 summary：

```bash
ssh windows-pc 'cmd /c "cd /d D:\WorkPlace\ZMP\DINOv3-TRT-Acceleration\Code && .venv\Scripts\python.exe scripts\build_formal_report_summary.py --reports-dir Artifacts\reports --output-json Artifacts\reports\formal_summary.json --output-md Artifacts\reports\formal_summary.md"'
```

远端生成 benchmark 图表：

```bash
ssh windows-pc 'cmd /c "cd /d D:\WorkPlace\ZMP\DINOv3-TRT-Acceleration\Code && .venv\Scripts\python.exe scripts\build_benchmark_figures.py --matrix-csv Artifacts\reports\formal_benchmark_matrix.csv --output-dir Artifacts\reports\figures"'
```

## 正式报告产物

主要产物位于 Windows 侧 `Code\Artifacts\reports\`：

- `formal_summary.{json,md}`
- `formal_benchmark_matrix.{csv,json,md}`（**87 行**机器可读，含 V1.1 mixed-precision + V1.2 ONNX-stripped + r336/r518 高 batch memory-bound 数据）
- `figures\benchmark_trtexec_bf16_speedup.svg`、`figures\benchmark_trtexec_int8_speedup.svg`、`figures\benchmark_cpp_runtime_speedup.svg`：3 张速度对比图
- `figures\benchmark_bf16_cosine_min.svg`、`figures\benchmark_bf16_cosine_mean.svg`：2 张多分辨率 BF16 4 输出 cosine 图
- `figures\benchmark_bf16_vs_int8_tradeoff.svg`：cosine vs speedup 散点图（**12 个点**含 V1.0+V1.1+V1.2 全部候选 + G2 ideal region 阴影）
- `figures\layer_ablation_diversity_vs_balance.svg`：4 层 ablation diversity vs magnitude balance（project 蓝 / dpt 绿 / late 红 三色编码）
- `figures\figures_index.json`：4 子系统统一入口顶层索引（speedup / cosine / tradeoff / layer_ablation 各自 manifest 摘要）
- `figures\benchmark_figures_manifest.json` + `figures\cosine_figures_manifest.json` + `figures\tradeoff_figures_manifest.json` + `figures\layer_ablation_figures_manifest.json`：4 子系统 manifest
- `artifact_manifest_formal_with_sha256.json`：所有产物的 SHA256 索引（**438+ 文件**）

一键重生命令：`python scripts\build_all_figures.py --allow-missing`（4 子系统统一入口，第 23 轮 DRY 重构）。

本地也保留了当前生成的 `Code/Artifacts/reports/formal_benchmark_matrix.*` 与 `figures/*.svg` 作为 P5 表格入口（macOS 通过 `scripts/sync_remote_windows_repo.py --pull-reports` 反向拉回）。

## 复现与许可

可复现命令、ImageNet 替换流程、正式 artifact manifest 与许可说明见 `Wiki/2-技术报告/复现与许可说明_V1.0.0.md`。当前正式 artifact manifest 为 `Code/Artifacts/reports/artifact_manifest_formal_with_sha256.json`；大体积权重、ONNX、engine、数据集和访问凭证不提交到 git。

正式 manifest 现在通过 `check_assets.py --output PATH` 写入：脚本对目标路径做原子写入（temp 文件 + `os.replace`）并把目标文件加进 reports 扫描的 exclude 集合，避免 manifest 自身被记录成 0-byte SHA。命令样板：

```bash
ssh windows-pc 'cmd /c "cd /d D:\WorkPlace\ZMP\DINOv3-TRT-Acceleration\Code && .venv\Scripts\python.exe scripts\check_assets.py --artifact-root Artifacts --require all --require bf16-engine --with-sha256 --output Artifacts\reports\artifact_manifest_formal_with_sha256.json"'
```

## 下一步

V1.0.0 + V1.1 + V1.2 + V1.3（设计）全部决策树就位后，剩余 2 项均为非工程性：

1. **完整 ImageNet val 解锁**（外部 blocker，HF 403）：授权放行后用 `scripts/export_hf_imagenet_parquet_images.py` 一键替换 manifest，重跑 `formal_summary` + 触发 V1.3 QAT（ADR-011 § 8 启动门槛）。
2. **PPT/海报排版**（**已生成 583 KB PPTX 终稿**：`Wiki/2-技术报告/ppt_slides/output/DINOv3-TRT-Acceleration_V1.0.0.pptx` 18 slides + 5 SVG embedded；如需进一步排版调整，参考答辩问答预案 + 87 行 matrix CSV 即可）。

V1.3 QAT 启动条件（ADR-011 § 8）：

- 数据集 unblock（ImageNet val 50K 或 ≥10K 等价）。
- 训练资源到位（A100/H100 ≥ 5 GPU-day 或 RTX 5080 ≥ 1 week）。
- 专项时间预算（1-2 个月，含论文写作）。
- 下游任务 baseline（depth/segmentation FP32 + DPT 头训练 pipeline）。

详细 V1.0+V1.1+V1.2 三层闭合证据链与 V1.3 路径设计见 ADR-010 § 5.3 与 ADR-011 全文。
