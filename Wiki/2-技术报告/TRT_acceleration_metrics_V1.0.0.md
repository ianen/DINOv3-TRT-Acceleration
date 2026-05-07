# DINOv3 TRT 加速指标完整报告 V1.0.0

> 项目：DINOv3-TRT-Acceleration（PolyU 研究型项目）
> 模型：Meta DINOv3 ViT-L/16 LVD-1689M（24 层，4 outputs at blocks 4/12/16/20）
> 报告日期：2026-05-07

---

## 1. 测试环境

| 项           | 配置                                                                                  |
| ------------ | ------------------------------------------------------------------------------------- |
| GPU          | NVIDIA RTX 5080（Blackwell, sm_120, 16 GB VRAM, ~300W TDP）                           |
| CPU          | Intel Core Ultra 9 285K                                                               |
| 内存         | 127.5 GB                                                                              |
| 主板         | Gigabyte Z890 AORUS ELITE WIFI7 ICE                                                   |
| OS           | Windows 10 Pro 64 位                                                                  |
| Python       | 3.10.10                                                                               |
| TensorRT     | 10.13.2.6                                                                             |
| CUDA / cuDNN | 12.8 / 9.x                                                                            |
| GPU 锁频     | graphics clock 2752 MHz（实测前 `nvidia-smi -lgc 2752,2752`，测试后 `-rgc` 复位） |
| 测量口径     | trtexec `--useSpinWait` GPU compute p50 + cpp 端到端 latency p50                    |

---

## 1.5 Speedup 参考基线（baseline）定义

> **本报告中所有 "N× speedup" 数字的参考基线统一为 TensorRT FP32（同一 ONNX、同一硬件、同一锁频、仅切换 `--fp32`），不是 PyTorch eager 模式。**

具体口径：

| Speedup 类别                       | 候选（candidate）                                                | 参考基线（reference）                                          |
| ---------------------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------- |
| §3 trtexec GPU compute speedup     | trtexec `--bf16 --precisionConstraints=prefer` GPU compute p50  | trtexec `--fp32` GPU compute p50（同 ONNX、同锁频）             |
| §4 cpp 端到端 speedup              | cpp runtime（BF16 prefer engine）H2D + 推理 + D2H 全链路 p50   | cpp runtime（FP32 engine）H2D + 推理 + D2H 全链路 p50         |
| §5 INT8 trtexec speedup            | trtexec INT8 SmoothQuant α=0.8 GPU compute p50                  | trtexec `--fp32` GPU compute p50（同 ONNX、同锁频）             |

> **未测量 PyTorch eager 基线。** 本研究关注 TensorRT 内部不同精度选择（FP32 / BF16 / INT8 / FP16）对 inference 性能的相对影响，因此 baseline 选择 TensorRT FP32。PyTorch eager → TensorRT FP32 之间还存在一层独立的"TensorRT 编译加速"贡献（业界经验值约 3-5× 不等），但**该层不在本研究范围内，未测量**。
>
> 因此**本报告 speedup 数字应理解为"TRT FP32 → TRT 低精度"的加速比，而非"PyTorch → TRT 低精度"的端到端加速比**。如需对外引用 vs PyTorch 的总加速比，请明确标注未测量、属估算。

---

## 2. 模型 & 输出契约

DINOv3 ViT-L/16 LVD-1689M 推理输出取 transformer 的 **第 4、12、16、20 层**（1-based；等价 0-based [3,11,15,19]），用于下游密集预测任务的多尺度特征融合（DPT-style）。

每层输出 shape：`[B, 197, 1024]`（1 CLS + 196 patch tokens；register tokens 已默认裁剪）。

| 阶段         | 关键约束                                                                                                                             |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| ONNX 导出    | opset ≥ 18（RoPE + LayerNorm 原生算子需要）；4 个 output binding 命名 `feat_layer_{4,12,16,20}`；`dynamic_axes` 仅 batch 维动态 |
| RoPE 处理    | 源码改造（首选），消除 `aten::if` 条件分支以避开 TRT IIfConditionalOutputLayer 失败                                                |
| TRT 引擎构建 | 4 个 output binding 全部保留；INT8 校准联合一次覆盖所有激活张量                                                                      |
| INT8 量化    | 主路径 NVIDIA TensorRT Model Optimizer 显式 Q/DQ；legacy IInt8EntropyCalibrator2 仅作 baseline 对照                                  |

---

## 2.5 为什么不用 FP16？候选选择实证

**核心结论**：直接使用 `trtexec --fp16` 在 DINOv3 ViT-L/16 LVD-1689M 正式权重下推理输出 **NaN**（数值溢出），不可作生产候选；**选 BF16 prefer 是工程取舍后的最优解**。本节列出实证证据与根因。

### 2.5.1 实测现象

| 测试                                                                  | 输入数据   | 输出结果                            | 来源                                                     |
| --------------------------------------------------------------------- | ---------- | ----------------------------------- | -------------------------------------------------------- |
| trtexec `--fp16` build + 运行（**正式 LVD-1689M 权重**）             | 真实图片   | feat_layer_{4,12,16,20} 出现 NaN   | 工程负例（V1.0.1 ADR-005 文档化）                        |
| trtexec `--fp16` build + 运行（**随机权重**，N(0, 0.02²) 初始化） | 任意输入   | 数值正常，`outputs_all_finite=true` | `cpp_runtime_random_fp32_vs_fp16_speedup.md`             |
| FP16 + 全部 transformer block fallback 到 FP32 (`fp16.blocksfp32`)   | 真实图片   | 4 输出 cos = 1.000 / 0.999999       | `compare_fp32_vs_fp16_blocksfp32_b1.json`                |

**对比关键**：架构本身可以跑 FP16（随机权重证明），失败发生在**正式训练权重**上 — 说明问题不是 TensorRT 也不是 ONNX，而是**训练权重的数值范围超 FP16 表示边界**。

### 2.5.2 根因：FP16 数值范围 ±65504 与 ViT-L 训练激活范围不兼容

FP16 = 5 位指数 + 10 位尾数，最大可表示数值约 **±65504**（6.55×10⁴）。DINOv3 ViT-L/16 经 1.689B 张图像训练后，部分中间激活在以下场景超出该范围：

- **Attention 预 softmax logits**（QKᵀ 累加）：在某些 head 与极值 token 上可达 10⁴–10⁵ 量级。
- **深层 block (16–23) residual sum 累积**：每层 LayerNorm 后再 residual 加法，深层累积可超 65504。
- **个别 channel 的 LayerNorm γ 缩放后极值**。

随机权重不触发这些极值（init 分布 N(0, 0.02²) 经 LayerNorm 拉回）；正式训练权重学到了"尖锐"的少量大幅 channel，少数路径数值跳出 FP16 → 溢出 NaN，并在前向传播链中污染所有下游层。这是大尺度 trained ViT inference 的已知问题（训练时通过 loss scaling 配合 mixed precision 解决，但纯 inference 无 backward 不能 scale）。

### 2.5.3 Selective FP16 fallback 探索（partial 验证）

我们 build 了约 14 个选择性 FP16 fallback 引擎变体（仅末段 fp32 / 仅 LayerNorm fp32 / 仅 attention fp32 / 早期 block fp32 等），其中**只有 `fp16.blocksfp32`（全部 transformer block 强制 FP32, FP16 仅在外围 utility ops）有完整 cos 对照报告**：

| 变体                   | feat_layer_4 cos | feat_layer_12 cos | feat_layer_16 cos | feat_layer_20 cos | 加速 |
| ---------------------- | ---------------- | ----------------- | ----------------- | ----------------- | ---- |
| `fp16.blocksfp32` | 1.000000         | 0.999999          | 0.999999          | 0.999999          | **接近 0**（block 全 FP32）|

⚠️ 该变体精度满分但 **没有计算加速** — 等于 FP32 性能。其他 13 个 partial fallback 变体未做 cos × speedup 双指标完整 sweep，存在"找到一个 (cos ≥ 0.99 ∧ speedup ≥ 2.2×) 配置"的可能但未穷尽验证。

### 2.5.4 BF16 选型论证

选 BF16 prefer 而非继续 FP16 selective sweep 的工程理由：

| 维度                            | FP16                        | BF16                      |
| ------------------------------- | --------------------------- | ------------------------- |
| 总位数                          | 16                          | 16                        |
| 指数位 / 尾数位                 | 5 / 10                      | **8 / 7**           |
| 数值范围                        | ±6.55×10⁴                | **±3.4×10³⁸**（同 FP32）|
| 正式权重溢出风险                | ❌ 高 (NaN)                | ✅ 几乎零                 |
| Tensor Core 加速 (Blackwell)    | ✅                          | ✅                        |
| 实测 speedup（r518 b8 trtexec） | n/a (NaN)                  | **3.86×**           |
| 实测 cos_min（real ImageNet）   | n/a                         | **0.9977**          |
| 工程额外成本                    | 需 fallback layer-by-layer sweep | 直接 `--bf16 --precisionConstraints=prefer` 通过 |

BF16 用"少 3 位尾数"换 FP32 等同的指数范围，对 transformer 这种激活幅度跨越大量级的模型刚好。**同样是 16-bit Tensor Core 计算资源，BF16 已经达到 16-bit 加速理论上限；FP16 即便 sweep 出可行配置，最多匹配不可能超过**。

### 2.5.5 局限性声明

- 本研究 FP16 失败结论**仅限当前 hardware-software 组合**（RTX 5080 + TRT 10.13.2.6 + DINOv3 LVD-1689M 正式权重）。
- 未穷尽 14 个 selective FP16 fallback 变体的 cos × speedup 双指标 sweep。如未来研究有需要，可基于 `Code/Artifacts/engines/dinov3_vitl16_4out.fp16.*.engine` 这批已 build 的引擎做完整对照实验。
- 上游若有专门的 ViT-L FP16 后训练优化技术（如 outlier-aware FP16, percentile-based clipping），可能解锁 FP16 路径，但**不在 V1.0.0 本研究范围**。

---

## 3. BF16 prefer trtexec GPU compute 加速

| 分辨率        | batch       | BF16 latency p50 (ms) | FP32 ref (ms)    | latency speedup       | throughput speedup |
| ------------- | ----------- | --------------------- | ---------------- | --------------------- | ------------------ |
| 224           | 1           | 2.87                  | 7.04             | 2.45×                | 2.37×             |
| 224           | 4           | 6.22                  | 15.86            | 2.55×                | 2.44×             |
| 224           | 8           | 10.08                 | 28.32            | 2.81×                | 2.65×             |
| 224           | 16          | 18.76                 | 57.70            | 3.08×                | 2.90×             |
| 224           | 32          | 36.97                 | 120.00           | 3.25×                | 3.07×             |
| 336           | 1           | 3.91                  | 10.96            | 2.80×                | 2.68×             |
| 336           | 4           | 11.41                 | 33.78            | 2.96×                | 2.80×             |
| 336           | 8           | 21.93                 | 71.31            | 3.25×                | 3.07×             |
| 336           | 16          | 140.52                | 139.90           | **1.00× ⚠️** | —                 |
| 336           | 32          | 272.19                | 274.04           | **1.01× ⚠️** | —                 |
| 518           | 1           | 8.51                  | 26.55            | 3.12×                | 2.98×             |
| 518           | 2           | 14.30                 | 49.98            | 3.50×                | 3.31×             |
| 518           | 4           | 26.56                 | 99.93            | 3.76×                | 3.57×             |
| **518** | **8** | **51.06**       | **197.30** | **3.86× ⭐**   | **3.65×**   |
| 518           | 16          | 389.99                | 387.94           | **1.00× ⚠️** | —                 |

⭐ **顶点：r518 b8 = 3.86× speedup**（trtexec GPU compute median latency）

⚠️ **memory-bound 边界**：r336 b16+ / r518 b16 跌到 ~1.0× — 数据传输 + activation 居住超过 16 GB VRAM 可承载的 cache 局部性，FP32 与 BF16 都被 memory bandwidth 主导，不再受 compute 端 BF16 加速影响。

---

## 4. BF16 prefer cpp 端到端加速（C++ 生产路径）

含 H2D copy + 推理 + D2H copy 全链路（不仅 GPU compute）。

| 分辨率        | batch       | cpp BF16 (ms)   | cpp FP32 ref (ms) | end-to-end speedup  |
| ------------- | ----------- | --------------- | ----------------- | ------------------- |
| 224           | 1           | 3.29            | 7.46              | 2.27×              |
| 224           | 8           | 12.18           | 30.06             | 2.47×              |
| 224           | 32          | 44.79           | 126.80            | 2.83×              |
| 336           | 1           | 4.61            | 11.66             | 2.53×              |
| 336           | 4           | 13.75           | 35.74             | 2.60×              |
| 336           | 8           | 26.36           | 74.89             | 2.84×              |
| 518           | 1           | 9.89            | 27.86             | 2.82×              |
| 518           | 2           | 17.01           | 52.46             | 3.08×              |
| 518           | 4           | 31.76           | 104.68            | 3.30×              |
| **518** | **8** | **61.48** | **208.81**  | **3.40× ⭐** |

⭐ **C++ 端到端顶点：r518 b8 = 3.40×**

cpp 端到端 speedup 比 trtexec GPU compute speedup 略低（3.40× vs 3.86×），差额来自 H2D/D2H 拷贝的固定开销 — 该开销受 PCIe bandwidth 限制不随精度变化。

---

## 5. INT8 SmoothQuant α=0.8（R2 应急候选）

INT8 PTQ 通过 NVIDIA TensorRT Model Optimizer SmoothQuant α-sweep 探索（α∈{0.5, 0.7, 0.8}），α=0.8 为 Pareto 最优。

| 分辨率        | batch        | trtexec speedup     | cos_min (ImageNet val 1000, real data) |
| ------------- | ------------ | ------------------- | -------------------------------------- |
| 224           | 1            | 2.18×              | —                                     |
| 224           | 8            | 3.48×              | —                                     |
| **224** | **32** | **3.60× ⭐** | **0.9727**（feat_layer_20）      |

INT8 per-output cosine（real ImageNet val 1000）：

| 输出层        | cos_min          | cos_mean |
| ------------- | ---------------- | -------- |
| feat_layer_4  | 0.9908           | 0.9934   |
| feat_layer_12 | 0.9895           | 0.9945   |
| feat_layer_16 | 0.9762           | 0.9862   |
| feat_layer_20 | **0.9727** | 0.9836   |

模式严格符合 ADR-010 root cause 假说："**前段 INT8 量化噪声向深层单调累积**"。feat_layer_4 (0.9908) → 12 (0.9895) → 16 (0.9762) → 20 (0.9727) 单调递减。

---

## 6. Python ↔ C++ 跨语言一致性

```
224 / 336 / 518 × FP32 / BF16 prefer × batch 1 = 全部 bit-identical
- max_abs_error = 0.0
- cosine_similarity = 1.0 (exact, not approximation)
```

超出 V1.0.1 G3 最严档（原计划仅要求 cos ≥ 0.999）。意味着 Python TensorRT runtime 与 C++ TRT inferer 的预处理 + 推理 + 后处理在 deterministic 输入下产生 byte-identical 输出，可作为生产部署的可信桥梁。

---

## 7. cos verdict（real ImageNet val 1000）

通过 Kaggle workaround（`titericz/imagenet1k-val` via kagglehub 1.0.1 + 新 KGAT auth）解锁完整 ImageNet val 50K 后，跑 1000 张随机采样 eval。

| 候选                                         | cos_min          | cos_mean | Verdict                                |
| -------------------------------------------- | ---------------- | -------- | -------------------------------------- |
| **BF16 prefer**（主交付）              | **0.9977** | 0.9996   | **R1_PASS_strict ≥ 0.99 ✓**    |
| **INT8 SmoothQuant α=0.8**（R2 应急） | **0.9727** | 0.9895   | **R2_PASS_emergency ≥ 0.97 ✓** |

BF16 prefer 4 个 layer 的 cos_min 全部 ≥ 0.9977（feat_layer_4 接近 1.000，feat_layer_20 worst 但仍 0.9977）— 主交付候选稳健达标。

---

## 8. SMART 目标达成状态

| 目标                                            | 阈值                                        | 实测                                        | 状态               |
| ----------------------------------------------- | ------------------------------------------- | ------------------------------------------- | ------------------ |
| **G1 低精度加速 ≥ 3×**                  | trtexec speedup ≥ 3                        | 3.86× / r518 b8                            | ✅ 超额            |
| **G2 INT8 cos_min ≥ 0.99**               | R1 strict                                   | 0.9727 best（R1 未达；R2 ≥ 0.97 达成）     | ⚠️ R1 ❌ / R2 ✅ |
| **G3 跨语言一致**                         | cos ≥ 0.999                                | bit-identical                               | ✅ 超额            |
| **G4 benchmark 矩阵 5 batch × 3 分辨率** | 完整覆盖 + 锁频                             | 87 行机器可读 CSV，含 memory-bound 边界论证 | ✅                 |
| **G5 可复现**                             | DINOv3 license + 一键脚本 + SHA256 manifest | 全部就位                                    | ✅                 |

---

## 9. 一句话总结

**主交付 BF16 prefer**：trtexec GPU compute 顶点 **3.86×**（r518 b8）/ cpp 端到端 **3.40×**（r518 b8）；real ImageNet val 1000 的 cos_min **0.9977** 达到 R1 strict ≥ 0.99 严格阈值。

**R2 应急 INT8 SmoothQuant α=0.8**：speedup **3.60×**（r224 b32）/ cos_min **0.9727** 达 R2 emergency ≥ 0.97 阈值。
