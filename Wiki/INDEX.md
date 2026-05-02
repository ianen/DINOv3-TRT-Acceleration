# Wiki Index — DINOv3-TRT-Acceleration

> 顶层 Wiki 导航。19 份 .md 文档按用途分类，让不同 stakeholder（论文 reviewer / 答辩官 / 工程接手 / GitHub 浏览者 / 新 Claude session）都能快速找到对应入口。
>
> **入口起点选择**：
> - 不熟悉项目 → 项目根 `README.md`（GitHub 第一眼）+ 本文档（Wiki 导航）。
> - 接手 Claude session → `CLAUDE.md`（自动加载，Claude 入口）。
> - 准备答辩 → `Wiki/2-技术报告/答辩问答预案_V1.0.0.md`（10 Q&A + 速答模板）。
> - 写论文 → `Wiki/2-技术报告/research_contributions_V1.0.0.md`（abstract + 6 contributions）。
> - 上手开发 → `Code/README.md`（命令索引 + V1.1+V1.2+V1.3 新 scripts 表）。
> - 复现实验 → `Wiki/2-技术报告/复现与许可说明_V1.0.0.md`。

## 项目计划与决策（`Wiki/0-项目计划/`）

| 文档 | 状态 | 用途 |
|---|---|---|
| `项目计划报告_V1.0.0.md` | **Superseded** | V1.0.0 初版项目计划（已被 V1.0.1 修订替代） |
| `项目计划报告_V1.0.1.md` | **Frozen** | V1.0.1 修订版主计划 + ADR-001~009（含 Token 序列结构 / RoPE 处理 / TRT 版本锁定等关键架构决策） |
| `项目计划报告_V1.0.2.md` | **Proposed** | V1.0.2 主计划：runtime mechanics + sparsity + custom kernels；旗舰目标 5.0× cpp r518 b8（实测 envelope 3.45× — V1.0.2 系统性论证 PTQ 路径已穷尽） |
| `项目计划报告_V1.0.3.md` | **In-Progress** | V1.0.3 主计划：throughput-oriented serving (Triton + C++ pool) + 全 GPU 利用率 (G7 SMART 目标) — G7 utilization 4 regime ✅; G1/G2/G3 throughput 双 user-blocker (Docker for ADR-019 + TRT 10.16 for ADR-020) |
| `项目计划报告_对外.md` | Frozen | 对外简版项目计划 |
| `ADR-010-V1.2-ONNX-Q-DQ-stripping_2026-05-01.md` | **Implemented · Negative result** | V1.2 ONNX 层 Q/DQ stripping 设计 + 实施 + 实测结论（第 25 轮闭合） |
| `ADR-011-V1.3-QAT-future-work_2026-05-01.md` | **Proposed** | V1.3 QAT 量化感知 fine-tuning 设计文档 + 4 条启动门槛（第 27 轮新增，未实施） |
| `ADR-012-V1.0.2-CUDA-Graphs-and-Pinned-Memory_2026-05-02.md` | **Implemented · Partial** | CUDA Graphs in C++ runtime；实测 r224 b1 1.135× speedup bit-exact，r518 b8 1.005× (compute-dominated)。Pinned staging 推迟到第二切。 |
| `ADR-013-V1.0.2-Persistent-Timing-Cache-and-Multi-Profile_2026-05-02.md` | **Implemented · Mixed** | trtexec.py 多 profile + builder optimization level + persistent cache + sparsity flag；实测 opt level 5 在 BF16 r518 b8 上 negative（V1.0.1 已最优）；persistent cache 需 TRT 10.16+。 |
| `ADR-014-V1.0.2-TRT-10.16.1-Upgrade-Validation_2026-05-02.md` | **Proposed**（user-blocked）| TRT 10.13.2.6 → 10.16.1 升级前置 milestone（解锁 sparsity 成熟 kernel + persistent cache）；需 user NVIDIA Developer login 下载 ~2GB 才能触发。 |
| `ADR-015-V1.0.2-Multi-Stream-Inference_2026-05-02.md` | **Implemented · Partial** | Multi-stream concurrent inference Python wrapper；r224 b1 N=2 sweet spot 1.513× aggregate；r518 b8 GPU 饱和 multi-stream 完全无效（1.023×）。 |
| `ADR-016-V1.0.2-2to4-Structured-Sparsity_2026-05-02.md` | **Implemented · Negative** | 2:4 structured sparsity 实施 + 完整 ablation k∈{1,2,4,8,12,16,20,24}+ block 19 reverse 假说全部 FAIL；仅 block 0 single-block PASS（1/24 网络贡献 ~0% latency）。 |
| `ADR-017-V1.0.2-FP8-Refined-Scaling_2026-05-02.md` | **Confirmed · Negative** | FP8 ModelOpt PTQ 在 TRT 10.13.2.6 + Blackwell sm_120 上 cos_min 0.1299 catastrophic FAIL；同 ADR-010 INT8 / ADR-016 sparsity 同 root-cause precision wall。 |
| `ADR-018-V1.0.2-Custom-Fused-Attention-Kernel_2026-05-02.md` | **Gated-PASS · Limited-ROI** | Custom CUDA fused attention kernel；trtexec --exportProfile 实证 attention 占 26.4% (gating ≥ 20% PASS) 但 ROI 上限 ~7%（TRT Myelin 已 fuse 部分），4 周研究级工作 ROI 不足。 |
| `ADR-019-V1.0.3-Triton-Inference-Server_2026-05-02.md` | **Parked**（user-blocker）| Triton Inference Server 集成 (V1.0.3 §4.1 主推) — Windows host 缺 Docker/WSL2, 提供 3 unblock paths (A install Docker / B cloud Linux GPU / C skip), 默认 C 直到 user 选 A/B |
| `ADR-020-V1.0.3-CPP-Multi-Context-Pool_2026-05-02.md` | **Implemented Phase 1 · Blocked** | C++ TRTInfererPool 完整设计 + Phase 1 实现 (per-slot TRTInferer + counting_semaphore + per-slot mutex), Windows MSVC 2022 build clean, N=1 single-thread ✅, concurrent N=2 ❌ TRT 10.13 Myelin runner.cpp:778 thread-safety blocker (V1.0.3 §8 Risk #2 实证 materialize) |
| `ADR-022-V1.0.3-CUDA-MPS-Evaluation_2026-05-03.md` | **Parked-Confirmed-Negative** | CUDA MPS 仅 Linux, Windows host 不支持; V1.0.3 G7 saturation 数据 (SM 96-99%) 直接证实 plan §3 "ViT-L compute-bound 收益 < 5%" 预测 — ADR-020 in-process pool 已实现 MPS 等价价值 |
| `ADR-023-V1.0.3-TRT-LLM-vLLM-Inapplicability_2026-05-02.md` | **Confirmed-Negative** | TRT-LLM/vLLM 与 ViT pure encoder 不适用论证 (paper §7 future-work 引用素材) — TRT-LLM 90% LLM-decode-specific, vLLM 唯一适用项 (CUDA Graph for ViT) 已被 V1.0.2 ADR-012 覆盖 |
| `V1.3_QAT_launch_threshold_evaluation_2026-05-01.md` | **Actionable evaluation** | V1.3 QAT 4 条启动门槛逐条评估 — 难度排序 / 推荐先满足路径 / 启动决策树（不启动 / 仅 V1.3 / +workshop paper / +full conference paper 4 选项 + 成本估算） |
| `imagenet_403_workaround_manual_2026-05-01.md` | **Actionable manual V1.0.1** | ImageNet 403 GatedRepoError unblock 完整手册（第 50 轮 V1.0.1 修订）— 兼容 Kaggle 新 KGAT_/access_token + legacy kaggle.json 双格式 + 修正 dataset slug `titericz/imagenet1k-val` + Kaggle CLI 已 install + kagglehub 1.0.1 已 install + user 配置步骤（5 min）+ pkg 升级路径 + ImageNet val 50K 替换 cosine eval 一键命令 |
| `milestones/M1-progress.md` | **Live** | 持续推进记录（**67+ 轮心跳 V1.0.1 + V1.0.2 持续推进**）；含每轮诊断、改动、远端实验、文档同步、剩余未做 |

## 技术调研（`Wiki/1-技术调研/`）

| 文档 | 状态 | 用途 |
|---|---|---|
| `调研Prompt_DeepResearch_V1.md` | Frozen | 深度调研 prompt 模板 |
| `DINOv3-TRT-Acceleration 深度技术调研报告（Claude）.md` | Frozen | Claude 视角调研报告（V1.0 之前） |
| `DINOv3-TRT-Acceleration 深度技术调研报告（GPT）.md` | Frozen | GPT 视角调研报告（V1.0 之前） |
| `DINOv3-TRT-Acceleration 深度技术调研与工程部署基准报告（Gemini）.md` | Frozen | Gemini 视角调研报告（V1.0 之前） |

调研报告作为 V1.0 之前的输入证据保留，不在本期工程范围更新。

## 技术报告（`Wiki/2-技术报告/`，按 tone 分类）

| 文档 | Tone | 用途 |
|---|---|---|
| `技术报告_V1.0.0.md` | **Engineering** | 详细工程报告（实施 + 数据 + 结果对照）；论文 method/results 段素材 |
| `汇报材料_V1.0.0.md` | **Executive** | 答辩用执行摘要（决策 + 关键数字 + 12 点 tradeoff 解读） |
| `答辩问答预案_V1.0.0.md` | **Defense** | 10 大 Q&A + 通用速答模板（5 秒答 + 30 秒展开 + 引用产物） |
| `research_contributions_V1.0.0.md` | **Academic** | 论文 abstract + intro 种子（6 大 Key Contributions + Methodological Innovations + Limitations） |
| `PPT_outline_V1.0.0.md` | **Presentation** | 答辩 PPT page-by-page 内容大纲（18 页 + speaker notes + 对应 Q 编号） |
| `paper_outline_IMRaD_V1.0.0.md` | **Academic publication** | 学术论文 IMRaD outline（10 大节 + abstract + evidence map + submission strategy；用 academic-paper skill outline-only mode 生成） |
| `paper_abstract_intro_draft_V1.0.0.md` | **Academic submission** | 投稿就绪：bilingual abstract（EN 312 words + 中文 720 字）+ Introduction §1.1-1.5 完整草稿 ~1500 词英文（用 academic-paper skill abstract-only mode + Introduction draft 生成） |
| `paper_methodology_draft_V1.0.0.md` | **Academic submission** | Methodology §3.1-3.9 完整草稿 ~1500 词英文（hardware / model contract / RoPE patch / multi-resolution / 12 candidates / cosine eval / cross-language parity / pure-Python testing / reproducibility infrastructure） |
| `paper_results_draft_V1.0.0.md` | **Academic submission** | Results §4.1-4.6 完整草稿 ~1500 词英文 + 5 tables + 6 figure references（BF16 speedup / BF16 cosine / 12-point sensitivity scatter / three-tool-chain convergence / 4-layer ablation / cross-language parity sanity check） |
| `paper_discussion_draft_V1.0.0.md` | **Academic submission** | Discussion §5.1-5.5 完整草稿 ~1200 词英文（root cause / tool-chain convergence implications / V1.3 QAT path / methodological innovations / 3-line related work comparison） |
| `paper_limitations_conclusion_draft_V1.0.0.md` | **Academic submission** | Limitations §6.1-6.4 + Conclusion §7 完整草稿 ~700 词英文（dataset proxy / single-hardware / TRT version / QAT deferred + 4 段 paper 收尾） |
| `paper_literature_review_draft_V1.0.0.md` | **Academic submission** | Literature Review §2.1-2.5 完整草稿 ~1000 词英文（PTQ vs QAT theoretical / ViT INT8 PTQ / TensorRT mixed-precision / DPT fusion / synthesis & gap）— **paper draft 至此 100% completion** |
| `paper_full_draft_V1.0.0.md` | **Academic submission (assembled)** | Single-file paper draft V1.0.0 — 6 份分散 draft 合并成完整 IMRaD（Title block + Abstract bilingual + §1-7 + §8 References + Acknowledgments + Word Count Summary）；**总 9,235 词 EN + 5 tables + 6 figure refs + 12 preliminary citations**，~67 KB markdown |
| `paper_full_draft_V1.0.0.tex` | **Academic submission (LaTeX)** | Pandoc 自动转换的 LaTeX article（~85 KB），可直接 `pdflatex` 编译为 PDF（待 venue submission 模板替换 `\documentclass`） |
| `复现与许可说明_V1.0.0.md` | Reproducibility | 一键 PowerShell + 数据替换流程 + License 副本说明 |
| `R2_emergency_acceptance_analysis_V1.0.0.md` | **Acceptance / Verdict** | V1.0.1 R2 应急方案适用性官方分析 — SmoothQuant α=0.8 4 输出 cos_mean / cos_min × R2 阈值 ≥ 0.97 双视角对照（cos_mean 4/4 达成；cos_min 2/4 达成，feat_layer_16/20 缺口 0.0001/0.0017）+ 工程语义解读 + 3 种交付建议 |
| `TRT_acceleration_metrics_V1.0.0.md` + `.pdf` | **Reportable / Stakeholder** | TRT 加速指标完整报告 V1.0.0（779 KB PDF，11 节）— 含 testing env / 模型契约 / G1-G5 实测 / R1/R2 verdict / SMART 5 目标对照 / 11 ADR 简表 |
| `TRT_acceleration_metrics_V1.0.2-delta.md` + `.pdf` | **V1.0.2 Delta Report** | V1.0.2 增量章节（813 KB PDF，11 节）— 4 ADRs Implemented + PTQ precision wall 5-vector 综合分析（INT8/sparsity/FP8 全部 FAIL）+ stacked envelope 3.45× / 5.0× target unreachable 数学证明 + V1.3 QAT 启动 implication |

## 实验结果（`Wiki/2-实验结果/`）

| 文档 | 状态 | 用途 |
|---|---|---|
| `M1-正式结果摘要_2026-04-30.md` | Frozen（V1.0.0 主线快照） | V1.0.0 主线正式结果数据快照 |
| `M1-M6-当前验收矩阵_2026-04-30.md` | **Frozen（V1.0.0 主线 G1-G5 闭合状态）** | V1.0.0 主线验收矩阵（G1-G5 + M1-M7 状态对照表） |
| `V1.1-stretch-summary_2026-05-01.md` | **Live（V1.1 + V1.2 综合）** | V1.1 stretch 6 轮 + V1.2 实施综合表 + ADR-011 引用 + V1.1 期总产物增量 |
| `V1.0.3-first-G7-datapoint_2026-05-02.md` | **Live（V1.0.3 G7 4-regime 闭合）** | r518 b8 99.08% / r336 b8 96.39% / r224 b1 N=1 88.24% / r224 b1 N=2 95.77% — BF16 dense path 三档 regime 全部接近 saturation；V1.3 QAT motivation 量化数据基础 |
| `V1.0.3-implementation-status_2026-05-02.md` | **Live（V1.0.3 mid-impl status snapshot）** | TL;DR + Goal Progress + ADR Status + 今日 builds + measurements + 5 大决策记录 + Open user-side decisions（Path A/B/C for ADR-019 + TRT 10.16 upgrade for ADR-020）|

## 代码与实验产物（`Code/Artifacts/`）

不在 Wiki 但密切相关的可复现产物：

| 路径 | 用途 |
|---|---|
| `Code/Artifacts/reports/formal_benchmark_matrix.{csv,json,md}` | **56 行**机器可读 benchmark matrix |
| `Code/Artifacts/reports/figures/*.svg` | **8 张 SVG**（3 速度 + 2 cosine + 1 tradeoff 12 点 + 1 layer ablation + 1 figures_index.json） |
| `Code/Artifacts/reports/figures/figures_index.json` | 4 子系统统一入口顶层索引（`scripts/build_all_figures.py` 一键产出） |
| `Code/Artifacts/reports/artifact_manifest_formal_with_sha256.json` | **419+ 文件**的 SHA256 完整索引（atomic write + self-exclude） |
| `Wiki/2-技术报告/ppt_slides/output/DINOv3-TRT-Acceleration_V1.0.0.pptx` | **答辩 PPTX 终稿**（18 slides，583 KB，5 SVG embedded） |
| `Wiki/2-技术报告/ppt_slides/slide-NN.js` × 18 | 各 slide 源码（PptxGenJS + standalone preview + compile.js 统一入口） |

## 对应不同 stakeholder 的入口路径

| Stakeholder | 入口 | 后续阅读顺序 |
|---|---|---|
| **新 Claude session** | `CLAUDE.md`（自动加载） | 项目根 README → 本 Wiki INDEX → 按需 deep-dive |
| **GitHub 浏览者** | 项目根 `README.md` | § 当前状态 → § 目录 → § 正式报告产物 → § 下一步 |
| **答辩官** | `答辩问答预案_V1.0.0.md` | 10 Q&A → 通用速答模板 → 必要时 § 引用产物 |
| **论文 reviewer** | `research_contributions_V1.0.0.md` | Abstract → § Key Contributions → § Detailed Findings → § Limitations |
| **工程接手** | `Code/README.md` § V1.1+V1.2+V1.3 command index | 11 行表 → ADR-010/011 → 验收矩阵 |
| **复现实验者** | `复现与许可说明_V1.0.0.md` | 一键 PowerShell → 数据替换流程 → license |
| **架构理解** | `项目计划报告_V1.0.1.md` ADR-001~009 → ADR-010 → ADR-011 | 主计划决策 → V1.2 negative → V1.3 future |
| **进度审计** | `milestones/M1-progress.md`（**33 篇心跳**） | 按时间顺序看每轮诊断 + 改动 + 结论 |

## 完整决策树概览

```
V1.0.1 主计划（项目计划报告_V1.0.1.md）
  ├── ADR-001 ONNX 多输出导出方式（裁剪 register tokens）
  ├── ADR-002 动态 shape 策略
  ├── ADR-003 INT8 量化方案（ModelOpt 显式 Q/DQ）
  ├── ADR-004 Python 推理 Runtime（cuda-python）
  ├── ADR-005 C++/Python 接口边界
  ├── ADR-006 预/后处理位置
  ├── ADR-007 DINOv3 RoPE 节点处理
  ├── ADR-008 TRT 版本锁定（10.13+，理想 10.16.1）
  └── ADR-009 静态分辨率 + 动态 batch
   │
   ├── V1.0.0 主线（G1-G5 闭合，BF16 prefer 主候选）
   │
   ├── V1.1 stretch（FP8 PTQ / FP8 partial / SmoothQuant α-sweep / mixed-precision via ModelOpt+TRT / 4 层 ablation）
   │     全部 negative 闭合（rounds 14-21）
   │
   ├── V1.2 ADR-010（ONNX 层 Q/DQ stripping）
   │     Implemented · Negative result（rounds 22-26）
   │     与 V1.1 ModelOpt skip 16-19 等价（cos 差 0.0005，speed 差 0.02×）
   │     验证 root cause = 前段 blocks 0-15 累积 INT8 量化噪声
   │
   └── V1.3 ADR-011（QAT 量化感知 fine-tuning）
         Proposed · 未实施（round 27）
         4 条启动门槛全状态：ImageNet ❌ + 训练资源 ❌ + 时间预算 ❌ + 下游 baseline ❌
```

## 心跳轮次索引（M1-progress 第 14-32 轮 V1.1+V1.2+V1.3+文档同步）

| 轮次 | 主题 | 性质 |
|---|---|---|
| 14 | SmoothQuant + skip 16-19 mixed-precision | negative-ish |
| 15 | 4 层组合 ablation | 研究证据 |
| 16 | layer ablation 可视化 SVG | 工程交付 |
| 17 | trtexec --layerPrecisions 工程基础设施 | 工程交付 |
| 18 | trtexec mixed-precision 实际 build | negative |
| 19 | mixed-precision 数据补 matrix + tradeoff 11 点 | 一致性 |
| 20 | layer ablation figure 入 manifest 系统 | 一致性 |
| 21 | V1.1 stretch summary single source of truth | 文档 |
| 22 | ADR-010 V1.2 设计文档 | 文档 |
| 23 | figures 系统统一入口 build_all_figures.py | DRY |
| 24 | V1.2 step 1（识别+分类，pure-Python） | 工程 |
| 25 | V1.2 step 2（完整实施 + DEFINITIVE NEGATIVE） | 工程闭合 |
| 26 | V1.2 数据补 matrix 第 56 行 + tradeoff 第 12 点 | 一致性 |
| 27 | ADR-011 V1.3 QAT 设计文档 | future work |
| 28 | 答辩问答预案 10 Q&A | 文档 |
| 29 | CLAUDE.md 入口同步 | 同步 |
| 30 | 项目根 README.md 同步 | 同步 |
| 31 | Code/README.md 命令索引同步 | 同步 |
| 32 | research_contributions academic tone 文档 | 文档 |
| 33 | 本文档（Wiki INDEX 顶层导航） | 文档 |

V1.0+V1.1+V1.2+V1.3 决策树 + 4 种 tone 文档生态 + 完整文档同步链路 + 顶层 Wiki 导航 全部就位。

## 剩余未闭合（仅 2 项，全非工程性）

1. **完整 ImageNet val** — Hugging Face gated repo 403 阻塞（外部 blocker，按指令不重试）。当前真实图片口径用 Imagenette2-320 val 替代；unblock 后用 `scripts/export_hf_imagenet_parquet_images.py` 一键替换 manifest 重跑 `formal_summary` + 触发 V1.3 QAT（ADR-011 § 8 启动门槛）。
2. **PPT/海报排版** — 纯排版工作，无新工程内容。可基于答辩问答预案（10 Q&A）+ 8 张 SVG + 56 行 matrix CSV 直接套模板。

## 测试与质量门

- 本地 `pytest 271 passed, 3 skipped`
- ruff + mypy 全绿（**111 Python 源文件**）
- 远端 Windows pytest 同步绿
- C++ tests `dinov3_trt_cpp_contract_tests` + `dinov3_trt_inspect_engine` 在 MinGW + MSVC 双工具链验证
- `tests/test_layer_precision.py` / `test_onnx_qdq_stripper.py` / `test_onnx_qdq_strip_planner.py` 等 pure-Python 模块本地 macOS 可单元测试

## License

DINOv3 Materials 使用自定义 DINOv3 License；仓库副本 `LICENSES/DINOv3_LICENSE.md`；所有发布产物含 "Built with DINOv3" 标注。
