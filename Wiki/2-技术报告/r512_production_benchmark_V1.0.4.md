# DINOv3 ViT-L/16 r=512 印花布数据集生产环境推理 Benchmark 报告 V1.0.4

> 项目：DINOv3-TRT-Acceleration（PolyU 研究型项目）
> 模型：Meta DINOv3 ViT-L/16 LVD-1689M（24 层，4 outputs at blocks 4/12/16/20）
> 报告日期：2026-05-07（**草稿 — 待 benchmark 实测数据填充**）
> 状态：⏸ Awaiting Windows SSH unblock to populate §3-§7 with empirical data

---

## ⚠ 草稿状态说明

本报告骨架已就位（结构、CSV schema、Pareto 表模板），实测数据（§3 cos verify、§4 Python 端时序、§5 C++ 端时序、§6 对照、§7 可选扩展）需 RTX 5080 工作站上电后跑 V1.0.4 launcher 自动产出。

**已完成**：
- §1 测试环境（沿用 V1.0.x 主力机配置）
- §1.5 Speedup baseline 定义（继承 V1.0.0 §1.5）
- §2 数据集介绍（144 张离线 1024 → 512 完成）
- §8 / §9 框架性内容

**待数据填充**：§3 / §4 / §5 / §6 / §7

---

## 1. 测试环境

| 项 | 配置 |
|---|---|
| GPU | NVIDIA RTX 5080（Blackwell, sm_120, 16 GB VRAM, 360 W TBP） |
| CPU | Intel Core Ultra 9 285K |
| 内存 | 127.5 GB |
| OS | Windows 10 Pro 64 位 |
| Python | 3.10.10 |
| TensorRT | 10.13.2.6 |
| CUDA / cuDNN | 12.8 / 9.x |
| GPU 锁频 | graphics clock 2752 MHz |
| 测量口径 | per-stage `time.perf_counter`（Python） / `std::chrono::steady_clock`（C++）+ `cudaStreamSynchronize` per stage |

数据集存放：`Artifacts/datasets/good_r512/` （144 张 r=512×512 印花布 JPG，9.9 MB 总）

---

## 1.5 Speedup 参考基线（baseline）定义

> **本报告所有 "N× speedup" 数字的参考基线统一为 TensorRT FP32**（同一 ONNX、同一硬件、同一锁频，仅切换 `--fp32`），不是 PyTorch eager。继承 V1.0.0 `TRT_acceleration_metrics_V1.0.0.md` §1.5 协议。

V1.0.4 报告侧重**6 段独立 latency 时序分解**，speedup 数字非主指标；如出现，参考基线为本研究 r=512 FP32 engine 的同段 latency p50。

---

## 2. 数据集与离线 resize 协议

### 2.1 来源

`Artifacts/datasets/good/` — 144 张 1024×1024 JPG 印花布图像，命名带 grid 坐标 `r{0-9}c{5-9}`（疑为布料检测多相机/工位 tile 输出）。无 metadata、无 split、无 label。

### 2.2 离线 resize（ADR-024）

**操作**：PIL `Image.LANCZOS` 高质量降采样 1024 → 512，JPEG quality=95 + optimize 落盘到 `Artifacts/datasets/good_r512/`。

**SHA256 manifest**：每张图含 SHA256 + 原 src_path 双向追溯，存 `Artifacts/datasets/good_r512/manifest.json`。

**验收实测**（2026-05-07）：
- 144 张 / 144 张 OK，0 errors
- 总输出 9.9 MB
- 所有图 PIL.size = (512, 512) RGB

**resize 不算 inference 时序** — 后续 §4 / §5 6 段时序分解的 `disk_read` 起点已是离线 resize 后的 r=512 JPG 文件。

---

## 3. r=512 Engine Build & cos verify

> **⏸ 待实测**

### 3.1 Engines

| Precision | Engine 文件 | 构建命令 | 大小（待实测）|
|---|---|---|---|
| FP32 | `dinov3_vitl16_4out.r512.fp32.engine` | `build_engine_trtexec.py --image-size 512 --precision fp32 ...` | 待 build |
| BF16 prefer | `dinov3_vitl16_4out.r512.bf16.prefer.engine` | `--precision bf16 --precision-constraints prefer --layer-precisions *:bf16` | 待 build |

`min/opt/max-batch = 1/4/16`，`--useSpinWait`。

### 3.2 cos verify（任 1 张图，BF16 vs FP32）

> **⏸ 待 build 完成后用 `evaluate_engine_pair_on_images.py` 跑**

预期（沿用 V1.0.0/1 r=224/336/518 经验）：

| 输出层 | cos vs FP32 reference | 阈值 |
|---|---|---|
| feat_layer_4 | 待实测 | > 0.999 |
| feat_layer_12 | 待实测 | > 0.999 |
| feat_layer_16 | 待实测 | > 0.999 |
| feat_layer_20 | 待实测 | > 0.999 |

**G2 PASS 条件**：4 输出 cos 全 > 0.999。

---

## 4. Python 端 6 段时序分解 Pareto 表

> **⏸ 待实测**（4 batch × 2 必选精度 = 8 必选数据点）

### 4.1 必选 Pareto 表

每行一组 (precision, batch) 实测数据；6 段独立 latency p50 ms + 总 + 吞吐。

| Precision | batch | disk_read | jpg_decode | preprocess | h2d | enqueueV3 | d2h | **total** | imgs/sec |
|---|---|---|---|---|---|---|---|---|---|
| FP32 | 1 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| FP32 | 4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| FP32 | 8 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| FP32 | 16 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| BF16 prefer | 1 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| BF16 prefer | 4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| BF16 prefer | 8 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| BF16 prefer | 16 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### 4.2 时序分解可视化（可选 matplotlib stack bar）

> 📊 待数据齐后视情况补图

---

## 5. C++ 端 6 段时序分解 Pareto 表

> **⏸ 待实测**（schema 严格 mirror §4，仅 language 列差异）

### 5.1 必选 Pareto 表

| Precision | batch | disk_read | jpg_decode | preprocess | h2d | enqueueV3 | d2h | **total** | imgs/sec |
|---|---|---|---|---|---|---|---|---|---|
| FP32 | 1 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

---

## 6. Python ↔ C++ 对照（G5 acceptance）

> **⏸ 待实测**

每个 (precision, batch) 在两语言下的 6 段 p50 diff：

| (P, B) | language | total p50 ms | enqueueV3 p50 | preprocess p50 | total diff vs Py |
|---|---|---|---|---|---|
| FP32, 8 | python | TBD | TBD | TBD | — |
| FP32, 8 | cpp | TBD | TBD | TBD | TBD% |
| BF16, 8 | python | TBD | TBD | TBD | — |
| BF16, 8 | cpp | TBD | TBD | TBD | TBD% |

**G5 PASS 条件**：所有 (precision, batch) 对的 6 段 diff ≤ 10%。

**预期**：disk_read / preprocess 在 C++ 显著更快（Python overhead）；enqueueV3 / d2h 应一致（GPU 主导）。

---

## 7. （可选扩展）FP16 / INT8 数据点

> 仅在 user 启用可选扩展（`run_v104_benchmark_windows.ps1 -Precisions @("fp32","bf16","fp16","int8")`）时填充。

### 7.1 FP16 r=512 实证（V1.0.0 r=224/336/518 NaN 复现/反证？）

**待验证假说**：r=512 因 token 数 1025 远大于 r=224（197），attention pre-softmax 累加值更大，**预期同样 NaN**。如实证 NaN，记录工程负例；如奇迹通过，作 V1.0.4 stretch 数据点。

### 7.2 INT8 SmoothQuant α=0.8 r=512（V1.0.2 R2 候选 r=512 复测）

**待验证假说**：r=512 INT8 cos_min 是否 ≥ 0.97（R2 阈值）。

---

## 8. 生产配置推荐

> **⏸ 待数据齐**

基于 §4 / §5 实测，给出生产部署的最优 (precision, batch) 推荐：

- 候选 1：低延迟优先 → 推荐 ___ + batch=___（total p50 ___ ms / imgs/sec ___）
- 候选 2：高吞吐优先 → 推荐 ___ + batch=___（total p50 ___ ms / imgs/sec ___）
- 候选 3：平衡 → 推荐 ___ + batch=___

依据 §4-§6 实测后填充。

---

## 9. 局限性 + 后续工作

### 9.1 局限性

- **样本量**：144 张印花布图像（V1.0.0 是 1000 张 ImageNet），p50 latency 噪声相对较高 — 通过 warmup ≥ 10 + iters ≥ 100 → ≥ 14400 inferences/run 缓解
- **单分辨率**：仅 r=512，未与 r=224/336/518 在同数据集上对比（其他分辨率没有对应数据集）
- **单 GPU**：仅 RTX 5080；其他 sm_89/sm_90 未验证
- **单线程**：production_benchmark 是顺序循环；多 worker 流水线（DataLoader 风格）作 V1.0.5 候选

### 9.2 后续工作

- **V1.0.5 候选**：DataLoader 风格 多 worker 异步预加载 + GPU 推理流水线重叠
- **V1.3 QAT**（独立 milestone）：r=512 INT8 / FP8 / 2:4-sparse via QAT 解锁 Tensor Core 剩余 30% 容量
- 跨数据集对比（V1.0.4 仅印花布）

---

## 10. 一句话总结

> **⏸ 待数据齐后 1 句话最优配置 + 关键 finding 收尾**

预占位（V1.0.0 风格）：

**主交付 BF16 prefer @ batch=___**：r=512 印花布数据集 production 端到端 p50 ___ ms（含 disk + decode + preprocess + GPU），imgs/sec ___；6 段时序分解显示 ___ 占主要时间，___ 是优化重点；Python ↔ C++ 6 段 diff 全 ≤ 10% 实证两语言部署可互换。

---

**附**：
- ADR 文档详见 `Wiki/0-项目计划/ADR-024..028-V1.0.4-*.md`
- 完整复现协议详见 `Wiki/0-项目计划/项目计划报告_V1.0.4.md` §12
- V1.0.4 mid-status snapshot: `Wiki/2-实验结果/V1.0.4-implementation-status_2026-05-07.md`
