# DINOv3-TRT-Acceleration

## 项目概述

**目标**：使用 NVIDIA TensorRT 对 Meta DINOv3 视觉自监督基础模型进行推理加速研究。

**性质**：研究型项目（PolyU），探索 PyTorch → ONNX → TensorRT 全链路在 DINOv3 上的延迟、吞吐与精度权衡。

**部署目标**：
- Python：研究/实验/benchmark 主语言
- C++：生产推理封装（最终阶段）

**预期产出**：
- 实验代码（导出/量化/引擎构建/推理 wrapper）
- Benchmark 报告（FP32/FP16/INT8 × 不同 batch size × 不同输入分辨率的延迟与精度矩阵）

## 技术栈

| 维度 | 选型/约束 |
|------|----------|
| 模型 | DINOv3 ViT-L/16 LVD-1689M（`facebook/dinov3-vitl16-pretrain-lvd1689m`） |
| 训练框架 | PyTorch 稳定版 cu128 优先（2.7+）；nightly `cu128` 仅在 exporter bug 触发时启用 |
| 中间表示 | ONNX opset **≥ 18** |
| 推理引擎 | NVIDIA TensorRT **≥ 10.13**（理想 10.16.1；10.8 仅作 baseline 对照） |
| CUDA | **≥ 12.8**（Blackwell sm_120 支持下限），cuDNN 9.x |
| 量化 | NVIDIA TensorRT Model Optimizer 显式 Q/DQ INT8（主路径）+ legacy calibrator baseline + FP16 |
| 部署语言 | Python（研究）+ C++（生产） |
| 测试集（精度对齐） | ImageNet val 子集（建议 ≥ 1000 张） |

## 代码目录

本项目实现代码放在 `Code/` 下；根目录保留项目文档、Wiki、agent 配置与仓库级说明。`Code/` 内的轻量单测不依赖模型权重、ONNX、TensorRT 或 ImageNet，用于先固定项目契约与可复用工具。

## 模型输出规约（关键约束）

> **来源依据**：本节基于 `Wiki/0-项目计划/项目计划报告_V1.0.1.md`（V1.0.1 修订版）。完整决策上下文见该报告 §2.1 与 ADR-001/007。

**DINOv3 推理输出**：取 transformer blocks 的 **第 4、12、16、20 层**（1-based；等价 0-based `[3,11,15,19]`）中间特征，用于下游密集预测任务的多尺度特征融合（DPT-style）。

**模型规模约束**：第 20 层要求 ≥ **ViT-L/16（24 层）**；ViT-S/B 仅 12 层不适用。本项目使用 `facebook/dinov3-vitl16-pretrain-lvd1689m`（safetensors 格式，自定义 DINOv3 License）。

### Token 序列结构（V1.0.1 关键修正）

DINOv3 ViT-L/16 LVD-1689M 默认含 **4 个 register tokens**（`Dinov3Config.num_register_tokens=4` 硬编码）。完整 token 序列在 224×224 输入下：

```
[CLS] + [reg₁ reg₂ reg₃ reg₄] + [196 patch tokens] = 201 tokens
```

**官方 `get_intermediate_layers()` API 默认裁剪 register tokens**，仅返回 197 tokens — 这是本项目主路径，与既定形状契约 `[B, 197, 1024]` 一致。

| 模式 | 序列长度 | 形状 |
|------|---------|------|
| **API 默认（裁剪 register）** ✅ 本项目主路径 | 197 | `[B, 197, 1024]` |
| 保留 register tokens（仅 V1.1+） | 201 | `[B, 201, 1024]` |

### 对各阶段的影响

| 阶段 | 必须遵守的约束 |
|------|---------------|
| **PyTorch 推理** | 通过 `get_intermediate_layers([3,11,15,19])`（0-based）抓取 4 个 block 输出；register 默认裁剪 |
| **ONNX 导出** | opset **≥ 18**（RoPE 与 LayerNorm 原生算子需要）；模型必须暴露 4 个 output binding，命名 `feat_layer_{4,12,16,20}`；`dynamic_axes` 仅 batch 维动态、分辨率固定（197 token 形状契约） |
| **RoPE 处理** | DINOv3 RoPE 含 `aten::if` 条件分支，导出后保留为 `If` 节点会触发 TRT IIfConditionalOutputLayer 失败（Issue #4603/#4558）。**必须源码改造**（首选）或 onnx-graphsurgeon surgery（fallback）—— 详见 ADR-007 |
| **TRT 引擎构建** | TRT **≥ 10.13（理想 10.16.1）**；network 含 4 个 output binding；INT8 校准覆盖所有激活张量（联合校准一次） |
| **INT8 量化** | 主路径：**NVIDIA TensorRT Model Optimizer 显式 Q/DQ**（IInt8EntropyCalibrator2 自 TRT 10.1 deprecated，仅作 baseline 对照） |
| **Benchmark** | 衡量"多输出"端到端延迟；含 p50 std ≤ 5% 与中位数去极值（Windows WDDM 100ms 尖峰风险） |
| **精度对齐** | 4 个 feature map **逐输出独立**汇报（cosine sim + MaxAbsErr），不允许仅给均值 |

### 典型形状（以 ViT-L/16, input 224×224 为例）

每层输出形状：`[B, 197, 1024]`（1 CLS + 14×14 patch tokens；register 已裁剪）。4 层叠加后输出张量数 = 4，每个均为该形状。

## 工作流入口

本项目的开发流程由 `.claude/agents/` 中的 12 个 agent 团队支撑，详见 `.claude/agents/README.md`。

**典型路径**：
1. `engineering-codebase-onboarding-engineer` 阅读 DINOv3 源码
2. `engineering-software-architect` 设计加速管线
3. `engineering-ai-engineer` 实施 ONNX 导出 + TRT 引擎构建
4. `testing-performance-benchmarker` 跑性能矩阵
5. `specialized-model-qa` 验证量化精度
6. `performance-optimizer` 攻克瓶颈
7. `engineering-technical-writer` 写报告

## 项目硬件 · 主力设备

### RTX 5080 Windows 工作站（当前主力）

**连接信息**：
- 简化命令：`ssh windows-pc`
- IP 地址：192.168.1.233
- 隧道服务：cpolar (`8.tcp.vip.cpolar.cn:13495`)
- 工作目录：`D:\WorkPlace\ZMP\DINOv3-TRT-Acceleration\`

**硬件配置**：
- 主板：Gigabyte Z890 AORUS ELITE WIFI7 ICE
- CPU：Intel Core Ultra 9 285K
- 内存：127.5 GB
- **GPU：NVIDIA GeForce RTX 5080（Blackwell, sm_120, ~300W TDP, 16 GB VRAM）**
- 存储：约 1.9 TB
- 操作系统：Windows 10 Pro 64 位

**软件栈**：
- Python 3.10.10
- PyTorch 稳定版 cu128 优先；当前远端已验证 `2.12.0.dev20260408+cu128`
- CUDA 12.8 / cuDNN 9.x
- TensorRT 10.13.2.6（已在远端 `C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT-10.13.2.6`；`trtexec`、C++ lib/include、本地 Python wheel 均可用）

**关键约束**：
- **16 GB VRAM 上限**：DINOv3 ViT-L/16 单 batch FP32 推理约 ~3 GB，FP16 ~1.5 GB；INT8 校准的 workspace 需谨慎设置（建议 4 GB）；高分辨率（518×518）+ 大 batch 需逐步压测，避免 OOM
- **Blackwell 新架构**：TRT 10.8 是当前 sm_120 支持下限，部分 ViT 算子（如 GELU、LayerNorm fusion）在 Blackwell 上的 kernel 调优可能仍在演进，性能数据需以本机实测为准
- **Windows 路径**：所有脚本路径使用 Windows 风格（`D:\WorkPlace\...`）；远程执行命令模板见下文

### 远程执行命令模板（macOS 主控 → Windows 工作站）

```bash
# 检查 GPU 状态
ssh windows-pc 'nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv'

# 切换工作目录并执行
ssh windows-pc 'cd /d D:\WorkPlace\ZMP\DINOv3-TRT-Acceleration && python Code\<script>.py'

# 后台 benchmark（脱离 SSH 会话）
ssh windows-pc 'powershell -ExecutionPolicy Bypass -File D:\WorkPlace\ZMP\DINOv3-TRT-Acceleration\Script\<launcher>.ps1'

# 查看日志
ssh windows-pc 'cmd /c "type D:\WorkPlace\ZMP\DINOv3-TRT-Acceleration\<log>.log"'
```

### 备用与对照设备

如有备用 GPU 设备（如 RTX 3090/4090 用作架构对照、或同事/实验室双卡机），请在 `CLAUDE_PRIVATE.md`（已 .gitignore）中记录，避免泄漏内网 IP/账号等信息。

## 仓库

- **远程**：`git@github.com:ianen/DINOv3-TRT-Acceleration.git`
- **当前阶段**：**V1.0.0 + V1.1 stretch + V1.2 mixed-precision + V1.3 future work 全部决策树就位**（项目计划 V1.0.1 + 第 14-28 轮心跳）。
  - **G1 低精度加速**：候选改为 **BF16 prefer**（FP16 在正式权重下输出 NaN，是负例）。224/336/518 三档分辨率全覆盖；trtexec GPU median latency speedup 顶点 `3.86×`（r518 b8），cpp end-to-end latency speedup 顶点 `3.40×`（r518 b8）。
  - **G2 INT8**：所有路径 negative 闭合（详见 ADR-010 § 5.3 三层证据链）。SmoothQuant α=0.8 best：cos_mean 0.982 / cos_min 0.968 / speed 3.48× — speed ✅ / cos ❌。三种 mixed-precision 工具链（V1.1 ModelOpt skip / V1.1 trtexec layerPrecisions / V1.2 ONNX strip）等价 negative，cos_min 都在 0.97 附近。**Root cause**：前段 blocks 0-15 累积 INT8 量化噪声，末段 mixed-precision 不能 recover。V1.3 方向 = QAT（ADR-011，Proposed）。
  - **G3 Python/C++ 跨语言一致**：224/336/518 三档分辨率 × FP32/BF16 prefer batch 1 全部 **bit-identical**（`max_abs_error=0`、`cosine=1.0`），超出 V1.0.1 G3 最严档。
  - **G4 benchmark 矩阵**：`Code/Artifacts/reports/formal_benchmark_matrix.csv` 共 **87 行**机器可读（含 V1.1 mixed l16-19:fp32 + V1.2 ONNX-stripped + r336/r518 b16/b32 高 batch memory-bound 数据），trtexec/cpp × 224/336/518 全覆盖。锁频 2752 MHz + `trtexec --useSpinWait` 口径。
  - **G5 可复现**：DINOv3 license + "Built with DINOv3" 标注 + 一键 PowerShell + 原子写入 SHA256 manifest（自动 exclude 自身）+ `scripts/build_all_figures.py` 4 子系统统一入口（speedup / cosine / tradeoff / layer_ablation）。
  - **真实图片 eval**：使用 Imagenette2-320 val（1000 eval / 500 calib 互斥），完整 HF ImageNet `ILSVRC/imagenet-1k` 仍受 `403 GatedRepoError` 阻塞（外部 blocker）；BF16 prefer 在三档分辨率最低 cosine ≥ 0.998。
- **可视化产物**（`Code/Artifacts/reports/figures/`，**8 张 SVG + 5 manifest**）：
  - 3 速度对比：`benchmark_trtexec_bf16_speedup.svg` / `benchmark_trtexec_int8_speedup.svg` / `benchmark_cpp_runtime_speedup.svg`
  - 2 cosine：`benchmark_bf16_cosine_min.svg` / `benchmark_bf16_cosine_mean.svg`
  - 1 tradeoff（**12 点**含 V1.0+V1.1+V1.2 全部候选 + G2 ideal region 阴影）：`benchmark_bf16_vs_int8_tradeoff.svg`
  - 1 layer ablation：`layer_ablation_diversity_vs_balance.svg`（project 蓝 / dpt 绿 / late 红 三色编码）
  - 5 manifest：benchmark / cosine / tradeoff / layer_ablation / **figures_index.json**（`scripts/build_all_figures.py` 一键统一入口产出）
- **报告交付**：
  - `Wiki/2-技术报告/技术报告_V1.0.0.md`（详细版）
  - `Wiki/2-技术报告/汇报材料_V1.0.0.md`（执行摘要 + 12 点 tradeoff 解读）
  - `Wiki/2-技术报告/答辩问答预案_V1.0.0.md`（**10 大 Q&A + 通用速答模板**，第 28 轮新增）
  - `Wiki/2-技术报告/复现与许可说明_V1.0.0.md`（含双向 sync `--pull-reports` 命令）
- **决策文档（ADR）**：
  - `Wiki/0-项目计划/项目计划报告_V1.0.1.md` ADR-001 ~ ADR-009（V1.0.1 主计划架构决策，frozen）
  - `Wiki/0-项目计划/ADR-010-V1.2-ONNX-Q-DQ-stripping_2026-05-01.md`（**Implemented · Negative result**，第 25 轮）
  - `Wiki/0-项目计划/ADR-011-V1.3-QAT-future-work_2026-05-01.md`（**Proposed**，未实施；4 条启动门槛）
- **顶层导航**：`Wiki/INDEX.md`（19 份 .md 按用途分类 + 不同 stakeholder 入口路径 + 完整决策树 + 心跳轮次索引）
- **结果索引**：
  - `Wiki/2-实验结果/M1-M6-当前验收矩阵_2026-04-30.md`（V1.0.0 主线 frozen）
  - `Wiki/2-实验结果/V1.1-stretch-summary_2026-05-01.md`（V1.1 stretch 7 轮 + V1.2 实施 + ADR-011 引用）
  - `Wiki/0-项目计划/milestones/M1-progress.md`（**33 轮心跳**详细记录）
- **同步工具**：`scripts/sync_remote_windows_repo.py` 双向（默认 push；`--pull-reports` 反向回拉文本产物）。
- **测试与质量门**：本地 `pytest 323 passed, 3 skipped` + ruff/mypy 全绿（**112 Python 源文件**）+ **line coverage 81%**（pytest-cov 已配置，跨过 V1.0.1 §12.1 ≥ 80% 阈值）；远端 Windows pytest 同步绿。`tests/test_layer_precision.py` / `test_onnx_qdq_stripper.py` / `test_onnx_qdq_strip_planner.py` / `test_trt_runtime.py` 等 pure-Python 与 mock-based 模块本地 macOS 可单元测试。

剩余未闭合（仅 2 项，全非工程性）：
1. 完整 ImageNet val 授权放行后用 `scripts/export_hf_imagenet_parquet_images.py` 一键替换重跑 `formal_summary` + 触发 V1.3 QAT（ADR-011 § 8 启动门槛）。
2. PPT/海报排版稿：**已生成 583 KB PPTX 终稿**（`Wiki/2-技术报告/ppt_slides/output/DINOv3-TRT-Acceleration_V1.0.0.pptx`，18 slides + 5 SVG embedded）。
