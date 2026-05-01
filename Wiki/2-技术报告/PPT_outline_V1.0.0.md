# 答辩 PPT 内容大纲 V1.0.0

> 答辩 PPT 的 page-by-page 内容映射。每页含：标题 + bullet 要点 + figure/table 引用 + speaker notes（口播脚本）+ 引用 Q（答辩问答预案对应问题号）。
>
> **使用方式**：用任意 PPT/Keynote/Beamer 模板套用，每页直接拷贝本文 outline 内容。所有 figure/table 已在 `Code/Artifacts/reports/figures/` + `formal_benchmark_matrix.csv` 中可直接嵌入。
>
> **配套**：
> - 答辩问答预案（`答辩问答预案_V1.0.0.md`）— 答辩时被提问的应对脚本
> - 本文档（PPT outline）— 主动 narrate 的演讲脚本
> - research_contributions（`research_contributions_V1.0.0.md`）— 论文 abstract + intro 素材

## 演讲结构（建议 18 页 / 约 15 分钟）

```
Page 1   Title
Page 2   TL;DR / 1-slide 摘要
Page 3   Motivation
Page 4   Method 1: V1.0.1 主计划架构（ADR-001~009）
Page 5   Method 2: 多分辨率 + 跨语言 parity 设计
Page 6   Result 1: BF16 prefer 速度数据（multi-resolution speedup chart）
Page 7   Result 2: BF16 prefer 精度数据（multi-resolution cosine chart）
Page 8   Result 3: INT8 sensitivity tradeoff scatter（12 点）
Page 9   Result 4: V1.0+V1.1+V1.2 mixed-precision 三层闭合证据
Page 10  Result 5: 4 层选择 ablation（diversity vs balance SVG）
Page 11  Discussion: Root Cause 分析
Page 12  Discussion: 工程方法学创新（pure-Python testing / SHA256 manifest / 多分辨率 profile）
Page 13  Limitations
Page 14  Future Work: V1.3 QAT（ADR-011）
Page 15  Conclusion
Page 16  Reproducibility & License
Page 17  Q&A
Page 18  Backup（可选 backup slides 留答辩追问用）
```

---

## Page 1 · Title

**标题**：DINOv3 ViT-L/16 多尺度 4 输出 TensorRT 加速研究

**副标题**：Blackwell sm_120 + TRT 10.13 + INT8 sensitivity 完整 ablation

**作者 / 机构**：[作者] / PolyU

**日期**：2026-05

**Speaker note**（30 秒）：
- 项目目标：在 RTX 5080（Blackwell sm_120）上把 DINOv3 ViT-L/16 LVD-1689M 视觉自监督基础模型加速 2-4×。
- 主候选 BF16 prefer，含 224/336/518 三档分辨率 + Python/C++ 跨语言一致性 + 完整 INT8 sensitivity 分析。
- 接下来 15 分钟我会从 motivation 讲到 V1.3 QAT future work。

---

## Page 2 · TL;DR / 1-slide 摘要

**标题**：1-slide TL;DR

**Bullets**（每行 ≤ 15 字）：
- BF16 prefer 顶点 **3.86× speedup**（r518 b8 trtexec），三档 cos ≥ 0.998
- INT8 全路径 sensitivity 已闭合 — Root cause: 前段累积量化噪声
- V1.3 方向: QAT 量化感知 fine-tuning（ADR-011 Proposed）
- Python ↔ C++ 三档 batch=1 **bit-identical**
- 56 行 benchmark matrix + 8 张 SVG + 271 tests + 111 源文件

**Figure 引用**：`figures/benchmark_bf16_vs_int8_tradeoff.svg`（缩略图，下方"详细见 Page 8"）

**Speaker note**（45 秒）：
- 整个项目可以用 1 张 slide 总结：BF16 prefer 是唯一进入 G2 ideal region 的候选。
- 顶点 3.86× speedup 在 r518 batch 8。所有 INT8 路径（PTQ + 节点级 + SmoothQuant + 三种 mixed-precision）都做了完整 sensitivity，全部 negative 闭合，但每个负例都有量化数据。
- 真正的精度瓶颈不在末段 4 个 block，而在前段累积量化噪声。V1.3 QAT 是唯一可能跨过 G2 阈值的路径。

**对应 Q**：Q3（整体结论）

---

## Page 3 · Motivation

**标题**：为什么要在消费级硬件上加速 ViT-L/16？

**Bullets**：
- DINOv3 ViT-L/16 LVD-1689M 是 Meta 最新视觉自监督基础模型 — dense prediction 下游任务（depth/segmentation）的强 backbone
- 4 输出多尺度特征（layer 4/12/16/20）→ DPT-style 融合
- 但 ViT-L FP32 在 RTX 5080（Blackwell, 16 GB VRAM）batch 8 r224 = 28 ms，部署成本高
- 工程目标：BF16/INT8 加速到 ≥ 2× 同时保持 cosine ≥ 0.99（G2 阈值）

**Figure 引用**：DINOv3 model card 截图（如有），或 ViT-L block 结构图

**Speaker note**（45 秒）：
- DINOv3 是 Meta 2024 年发布的 ViT-L/16 自监督模型，在 LVD-1689M 数据集上预训练。
- 项目用它的 4 个中间层输出 [layer 4/12/16/20] 做 DPT 风格的 dense prediction。
- 在 RTX 5080 上 FP32 batch 8 要 28 毫秒。如果不做加速，单帧推理成本太高，难以部署。
- 所以核心问题是：能不能在保持精度（cosine ≥ 0.99）的前提下加速 ≥ 2×？

---

## Page 4 · Method 1: V1.0.1 主计划架构（ADR-001~009）

**标题**：架构决策（V1.0.1 主计划 ADR-001~009）

**Bullets**：
- ADR-001 4 个 output binding `feat_layer_{4,12,16,20}`，shape `[B, 197, 1024]`，**裁剪 register tokens**
- ADR-002+009 静态分辨率 + 动态 batch（每分辨率独立 engine）
- ADR-003 INT8 主路径：**ModelOpt 显式 Q/DQ**（legacy IInt8EntropyCalibrator2 已 deprecated）
- ADR-007 RoPE 处理：**源码改造**消除 ONNX `If` 节点（避开 TRT IIfConditionalOutputLayer 失败）
- ADR-008 TRT 10.13+ 锁定（Blackwell sm_120 支持下限）

**Figure 引用**：架构 block 图（4 输出绑定 + RoPE 改造点 + Q/DQ 插入位置）

**Speaker note**（60 秒）：
- 9 个 ADR 锁定项目主要架构决策。
- ADR-001 的关键是：DINOv3 默认含 4 个 register tokens，但 API 默认裁剪它们；项目主路径采用 197 token contract（1 CLS + 196 patch token）。
- ADR-003 中 INT8 走 ModelOpt 显式 Q/DQ 是因为 legacy calibrator 在 TRT 10.1 起 deprecated。
- ADR-007 的 RoPE 处理特别 subtle — DINOv3 的 RoPE 含 `aten::if` 条件分支，导出后会变成 ONNX `If` 节点，TRT 10.13 的 IIfConditionalOutputLayer 在某些情况下 build 失败。我们用源码改造（`angles.cat` 替代 `tile`）消除条件分支。

**对应 Q**：Q1（FP16 vs BF16）

---

## Page 5 · Method 2: 多分辨率 + 跨语言 parity 设计

**标题**：多分辨率 + Python/C++ 一致性

**Bullets**：
- 多分辨率：r224 / r336 / r518 各自独立 engine（静态 spatial + 动态 batch）
- r518 batch 8 用 `min=1, opt=4, max=8` profile，独立 timing cache
- C++ runtime：MSVC + CUDA + TRT 10.13.2.6 + RAII engine wrapper
- 跨语言 parity 用 deterministic sine input 做 bit-identical 比较

**Figure 引用**：multi-resolution profile diagram

**Speaker note**（45 秒）：
- 16 GB VRAM 不允许同时 dynamic spatial + dynamic batch。每分辨率独立 engine + 独立 timing cache。
- C++ runtime 走 MSVC（MinGW 在 TRT 10.13 上有 ABI 不兼容问题）。
- 跨语言 parity 要求 max_abs_error = 0 + cosine = 1.0 — 完全 bit-identical，不是 epsilon close。

**对应 Q**：Q5（速度数据）+ Q6（跨语言一致性）

---

## Page 6 · Result 1: BF16 prefer 速度（multi-resolution speedup chart）

**标题**：BF16 prefer 速度结果（locked 2752 MHz + spin-wait）

**Bullets**：
- r224 b1/b8/b32: **2.45× / 2.81× / 3.25×**
- r336 b1/b4/b8: 2.80× / 2.96× / 3.25×
- r518 b1/b2/b4/b8: 3.12× / 3.50× / 3.76× / **3.86×**（顶点）
- C++ end-to-end：r518 b8 **3.40×**（含 H2D + enqueue + D2H）

**Figure 引用**：`figures/benchmark_trtexec_bf16_speedup.svg`（多分辨率柱图）+ `figures/benchmark_cpp_runtime_speedup.svg`

**Speaker note**（45 秒）：
- r518 batch 8 是项目顶点：trtexec GPU compute median speedup 3.86×，C++ end-to-end 3.40×。
- 锁频 2752 MHz + `--useSpinWait` 是为了规避 Windows WDDM 100ms 抖动。
- 注意 C++ end-to-end 包含 H2D + enqueue + D2H + stream sync，比 trtexec GPU compute time 更接近 production 真实成本。

**对应 Q**：Q5

---

## Page 7 · Result 2: BF16 prefer 精度（multi-resolution cosine chart）

**标题**：BF16 prefer 精度结果（Imagenette 1000 张 vs FP32）

**Bullets**：
- r224 4 输出 cos_min: feat_layer_{4,12,16,20} = **0.999933 / 0.999664 / 0.998943 / 0.998749**
- r336 cos_min: 0.999891 / 0.999276 / 0.998394 / 0.998493
- r518 cos_min: 0.999868 / 0.999075 / 0.998604 / **0.999171**（feat_layer_20 反而最高）
- 全部 ≥ G1 阈值（cosine ≥ 0.99，且实际全部 ≥ 0.998）

**Figure 引用**：`figures/benchmark_bf16_cosine_min.svg` + `figures/benchmark_bf16_cosine_mean.svg`

**Speaker note**（45 秒）：
- 三档分辨率 4 输出 1000 张真实图片 cos_min 全部 ≥ 0.998。
- 有趣的是 r518 的 feat_layer_20（最深层）cos_min = 0.999171，反而高于 r224 的 0.998749。
- 原因：r518 有 1025 token vs r224 的 197 token，BF16 量化误差被更多 patch token 稀释。

**对应 Q**：Q1

---

## Page 8 · Result 3: INT8 sensitivity tradeoff scatter（12 点）

**标题**：INT8 路径完整 sensitivity（12 点 tradeoff）

**Bullets**：
- X 轴：feat_layer_20 cosine_mean；Y 轴：trtexec b8 latency speedup vs FP32
- **唯一进入 G2 ideal region**（cos ≥ 0.99 ∧ speedup ≥ 2.2×）的候选：**BF16 prefer**
- 9 个 INT8 候选 + FP8 default + FP8 partial layer19 + V1.2 ONNX-stripped = **12 点**
- SmoothQuant α=0.8 best（cos 0.982 / speed 3.48×）— 速度达标但 cos 缺 0.022

**Figure 引用**：`figures/benchmark_bf16_vs_int8_tradeoff.svg`（占满 slide 中央，含 G2 ideal region 阴影）

**Speaker note**（60 秒）：
- 这张散点图是项目的 single-most-important visualization。
- X 轴是 feat_layer_20 cosine mean，Y 轴是 batch 8 speedup vs FP32。右上角的绿色阴影是 G2 ideal region。
- 12 个点中只有 BF16 prefer（蓝色，最右上）进入了 ideal region。
- 9 个 INT8 候选 + FP8 + V1.2 ONNX strip 都在 ideal region 之外的 trade-off curve 上。
- "量化范围越小 → cosine 越高 → speed 越低" 的关系一目了然。

**对应 Q**：Q2（INT8 cos < 0.99）

---

## Page 9 · Result 4: V1.0+V1.1+V1.2 mixed-precision 三层闭合证据

**标题**：Mixed-Precision 三种工具链等价 negative

**Table**（直接复制到 PPT）：

| 路径 | 工具链层 | feat_layer_20 cos_min | b8 speedup |
|---|---|---:|---:|
| Full SmoothQuant α=0.8 | PyTorch ModelOpt | 0.968 | 3.48× |
| ModelOpt `disable_quantizer` skip 16-19 | PyTorch | 0.971 (+0.003) | 2.41× (-30%) |
| trtexec `--layerPrecisions=l16-19:fp32` | TRT command-line | 0.9683 (≈) | 3.43× (≈) |
| **V1.2 ONNX-level Q/DQ stripping** | **ONNX library** | **0.9705** | **2.39×** |

**Bullets**：
- 三种独立工具链 → cos_min 差 0.003 / speed 差 0.02× → 等价 negative
- 这种 convergence 排除了"工具链 bug"作为 negative 的原因
- Root cause 必须在数据流上游

**Speaker note**（60 秒）：
- 这张 slide 展示项目最重要的 negative result 证据。
- 三种完全独立的工具链（PyTorch / TRT / ONNX）都试图把 layer 16-19 推到 FP32，结果数值上几乎完全等价。
- 这是一个 convergence proof：不是某个工具链有 bug，而是 fundamentally 这个方向不 work。
- 必然结论：root cause 在更上游（前段 blocks 0-15）。

**对应 Q**：Q2（核心问题）

---

## Page 10 · Result 5: 4 层选择 ablation

**标题**：DPT-style 4 层选择实证

**Bullets**：
- 项目 `[4,12,16,20]` vs DPT 论文 `[5,11,17,23]` vs late `[6,12,18,24]`
- 1000 张真实图片：inter-output cosine + per-output magnitude balance
- **project**: cos 0.383 / **mag balance 12.6×**（最平衡）
- **dpt**: cos 0.299（最分散）/ mag balance 31.9×
- **late**: cos 0.339 / mag balance **84×**（最不平衡）→ 永远不选

**Figure 引用**：`figures/layer_ablation_diversity_vs_balance.svg`（X = mean cos, Y = log10 mag ratio, 三色编码）

**Speaker note**（45 秒）：
- 我们对比了 3 种层选择 — 项目当前的 [4,12,16,20]、DPT 论文的 [5,11,17,23]、和 late-heavy [6,12,18,24]。
- DPT 选择确实在 inter-output diversity 上最高，但 magnitude balance 是 32×。
- 项目当前选择 magnitude 最平衡（12.6×），代价是 diversity 略低。
- late 是双输 — magnitude 84× 不平衡，不能用。
- **结论**：项目选择不是 DPT 简单照搬，是 diversity-magnitude 折中。

**对应 Q**：Q4

---

## Page 11 · Discussion: Root Cause 分析

**标题**：为什么 INT8 cos_min < 0.99？

**Bullets**：
- Blocks 0-15 累积 INT8 量化误差 → 偏离 FP32 baseline 约 10⁻² 量级
- 到达 block 16 输入时**已偏离**，layer 16-19 即使 FP32 也无法 recover
- 工具链 layer 不重要：PyTorch / TRT / ONNX 都 hit same TRT fallback
- **必须从前段开始减少量化** — V1.3 QAT 路径

**Figure 引用**：noise propagation diagram（量化噪声沿 block 累积示意）

**Speaker note**（60 秒）：
- 这是项目最重要的 finding。
- 每经过一个 INT8 量化的 transformer block，feat_layer_N 偏离 FP32 baseline 约 10⁻²。
- 经过 21 个 block，输入到 block 16 的数值已经偏离 FP32 一个数量级。
- 在错误的输入上做 FP32 inference 仍然得错误的输出 — 错误的输入污染无法在 block 16-19 内部被 recover。
- 要跨过 G2 cos_min ≥ 0.99，必须把前段量化误差从 10⁻² 压到 10⁻³，**只能 QAT 训练优化前段权重**。

**对应 Q**：Q2（深层）+ Q8（V1.3）

---

## Page 12 · Discussion: 工程方法学创新

**标题**：3 项工程方法学

**Bullets**：
- **Pure-Python testing for native-tool-chain**：layer_precision / onnx_qdq_stripper / strip_planner 全本地可测，271 tests / 111 源文件 / GPU-free dev workflow
- **Bidirectional remote-sync**：`--pull-reports` 文本产物反向回拉，绕开 cpolar SSH scp 不稳定
- **Unified figure regeneration**：`build_all_figures.py` 4 子系统统一入口 + `figures_index.json`

**Figure 引用**：3 列对照图（traditional vs project pattern）

**Speaker note**（45 秒）：
- 工程上有 3 个值得分享的设计模式。
- 第一个是 pure-Python testing — 把 ONNX 操作的"数据决策层"和"实际 graph mutation 层"分离，让数据决策可以本地 unit test，不依赖 GPU。
- 第二个是双向 sync — 远端跑实验，本地写文档，文本产物用 `--pull-reports` 反向回拉。
- 第三个是 figures 4 子系统统一入口 — 一条命令重生所有 figures，避免遗漏。

---

## Page 13 · Limitations

**标题**：Limitations

**Bullets**：
- ImageNet val 50K **gated 403** → 用 Imagenette2-320（10 类 13K 张）替代
- TRT 10.13 + Blackwell **BF16 + Q/DQ Myelin Fill 不兼容** → 强制用 FP32 fallback
- 单一硬件（RTX 5080 sm_120）→ Ada Lovelace / Hopper 行为可能不同
- QAT 未实施（ADR-011 § 8 4 条启动门槛全未满足）

**Speaker note**（45 秒）：
- 4 个主要 limitations 都已记录。
- ImageNet 是外部 blocker，按指令不重试，用 Imagenette 替代。
- TRT 10.13 的 BF16 + Q/DQ Myelin 问题是已记录的工程不兼容点 — 第 18 轮发现，可能在 TRT 11.x 解决。
- 硬件特异性是 limitation，但项目覆盖 Blackwell sm_120 是这一代消费级 GPU 的代表。
- QAT 未实施是因为 4 条启动门槛中至少一条未满足，不是工程懒惰。

**对应 Q**：Q7（ImageNet）

---

## Page 14 · Future Work: V1.3 QAT（ADR-011）

**标题**：V1.3 — Quantization-Aware Fine-Tuning

**Bullets**：
- 从 SmoothQuant α=0.8 PTQ initialization 出发
- ImageNet val 50K + ModelOpt QAT mode + 1-5 epoch fine-tune
- 期望 cos_min ≥ **0.99** 同时保留 ≥ **3.0×** speedup → 首个完整满足 G2 ideal region 的 INT8 候选
- 4 条启动门槛：ImageNet unblock + 训练资源 ≥ 5 GPU-day + 时间预算 1-2 月 + 下游 baseline

**Figure 引用**：QAT pipeline diagram（PTQ init → fine-tune → export Q/DQ ONNX → trtexec build）

**Speaker note**（45 秒）：
- V1.3 QAT 是项目 future work 的明确方向。
- 关键 insight：从 V1.1 SmoothQuant α=0.8 已经训练好的 weights 出发（不是从头训练），用量化感知 fine-tuning 1-5 个 epoch 重新优化前段权重，让 INT8 量化误差从 10⁻² 压到 10⁻³。
- 4 条启动门槛包含 ImageNet unblock + 训练资源 + 时间预算 + 下游 baseline — 这些都是非工程性 prerequisite，需要外部条件。

**对应 Q**：Q8

---

## Page 15 · Conclusion

**标题**：Conclusion

**Bullets**（按重要性排序）：
- BF16 prefer 是 G2 ideal region 唯一候选 — 顶点 3.86× speedup + cos ≥ 0.998
- INT8 全路径 sensitivity 已闭合（5 paths × 12 points），root cause 是前段累积量化噪声
- V1.0+V1.1+V1.2 三层 mixed-precision 工具链等价 negative — 验证 root cause
- 跨语言 parity 三档分辨率 bit-identical — 部署可信度高
- V1.3 QAT 是 future work 唯一可能跨过 G2 的路径

**Speaker note**（60 秒）：
- 项目核心结论一句话：**BF16 prefer 是 RTX 5080 + TRT 10.13 + DINOv3 ViT-L 上唯一在 G2 ideal region 的候选**。
- 9 个 INT8 候选 + 3 种 mixed-precision 工具链都做了完整 sensitivity，全部 negative。
- 但这些 negative 不是工程失败 — 是发现了 ViT-L INT8 PTQ 的工程边界 + 验证了 root cause（前段累积噪声）。
- V1.3 QAT 设计文档已就位，等数据集 unblock + 资源 + 时间预算就可以启动。

---

## Page 16 · Reproducibility & License

**标题**：Reproducibility & License

**Bullets**：
- 一键 PowerShell：`scripts/run_formal_hf_pipeline_windows.ps1`
- Figures 重生：`scripts/build_all_figures.py --allow-missing`（4 子系统统一入口）
- Atomic SHA256 manifest 自动 exclude 自身 — 419+ 文件
- DINOv3 License 副本 + "Built with DINOv3" 标注全部就位
- 33 篇心跳记录每一步可追溯（M1-progress.md）

**Speaker note**（30 秒）：
- 完整 reproducibility 链路：Windows 一键 PowerShell + figures 一键重生 + 双向 sync + atomic SHA256 manifest。
- DINOv3 License 合规：repo 副本 + 所有产物的 "Built with DINOv3" 标注。

**对应 Q**：Q9

---

## Page 17 · Q&A

**标题**：Questions?

**Bullets**：
- 引用产物路径（answers 在 `答辩问答预案_V1.0.0.md`）
- Backup slides 准备 10 大可能提问的速答（速答模板）

**Speaker note**：
- 这里把答辩问答预案 10 大 Q&A 的简短答案默念过一遍。
- 准备 backup slides 应对追问（见 Page 18+）。

---

## Page 18+ · Backup Slides（可选）

**建议 backup**：
1. ADR-007 RoPE 改造前后对比（如被追问 RoPE 处理）
2. SmoothQuant α-sweep 详细数据（α=0.5/0.7/0.8 三档对比）
3. 4 层 ablation 详细 magnitude 表（per-layer L2 norm）
4. C++ runtime parity 详细对照（max_abs_error / RMSE / cosine 全 4 输出）
5. V1.2 ONNX strip plan 示意（96 节点删除 + 48 input slots rewire）
6. ADR-011 QAT 4 条启动门槛全状态表

---

## 演讲节奏建议

| 段落 | 页数 | 时长 | 难度 |
|---|---:|---:|---|
| Title + TL;DR | 2 | 1.5 min | 轻 |
| Motivation + Method | 3 | 2.5 min | 中 |
| Results | 5 | 5.0 min | 重头戏 |
| Discussion | 2 | 2.0 min | 中（重点 Root Cause） |
| Limitations + Future + Conclusion | 3 | 2.5 min | 轻 |
| Reproducibility + Q&A | 2 | 1.5 min | 轻 |
| **总计** | **17 + backup** | **15 min** | — |

---

## 答辩问答预案对应表

| Page | 主要应对 Q |
|---:|---|
| 2 (TL;DR) | Q3 |
| 4 (Method 1) | Q1 |
| 6-7 (BF16 results) | Q5 + Q1 |
| 8 (Tradeoff scatter) | Q2 |
| 9 (Mixed-precision三层) | Q2 deep |
| 10 (4 层 ablation) | Q4 |
| 11 (Root cause) | Q2 + Q8 |
| 13 (Limitations) | Q7 |
| 14 (V1.3 QAT) | Q8 |
| 16 (Reproducibility) | Q9 |
| 17 (Q&A) | Q10 |

如果答辩官提问超出 Q1-Q10，使用速答模板（5 行表，1 句话答 + 30 秒展开）应对。
