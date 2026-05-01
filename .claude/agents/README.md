# DINOv3-TRT-Acceleration · Agent 团队

本目录是项目级 Claude Code agent 工作区，由 `project-agent-team` skill 自动从 `~/.claude/agents/` 精选并复制而来。

**项目定位**：使用 NVIDIA TensorRT 对 Meta DINOv3 视觉自监督基础模型进行推理加速研究。

**部署目标**：Python（研究/实验）+ C++（生产推理封装）双栈。

**主要产出**：实验代码 + benchmark 报告。

---

## 团队组成（12 个 agent）

### Tier 1 — 核心技术栈（5 个）

| 文件 | 职责 | 调用时机 |
|------|------|---------|
| **engineering-ai-engineer.md** | ML/AI 工程化主负责人 | DINOv3 加载、ONNX 导出、TRT 引擎构建、推理 API |
| **testing-performance-benchmarker.md** | 性能基准测试专家 | 延迟/吞吐/GPU 显存测量；不同 batch/分辨率/精度对比 |
| **performance-optimizer.md** | 性能瓶颈优化专家 | GPU 算子热点分析、kernel fusion、显存优化 |
| **engineering-software-architect.md** | 系统架构师 | ONNX→TRT 管线架构、量化策略 ADR、Python/C++ 接口划分 |
| **specialized-model-qa.md** | 模型 QA 审计 | FP16/INT8 引擎 vs PyTorch FP32 baseline 精度对齐验证 |

### Tier 2 — 代码质量与理解（4 个）

| 文件 | 职责 | 调用时机 |
|------|------|---------|
| **python-reviewer.md** | Python 代码审查 | 导出脚本、量化标定、推理服务的 PEP 8/类型审查 |
| **cpp-reviewer.md** | C++ 代码审查 | C++ TRT 推理 wrapper 的内存安全、并发安全、现代 C++ 习惯 |
| **engineering-codebase-onboarding-engineer.md** | 代码库导览 | 阅读上游 DINOv3 源码，绘制模块/前向路径地图 |
| **pytorch-build-resolver.md** | PyTorch/CUDA 错误修复 | ONNX 导出 shape 不匹配、AMP、device 错误调试 |

### Tier 3 — 流程与文档（3 个）

| 文件 | 职责 | 调用时机 |
|------|------|---------|
| **planner.md** | 任务分解与计划 | 多阶段路线图（导出→量化→引擎→benchmark）拆解 |
| **architect.md** | 通用架构决策 | 跨模块技术决策、依赖管理、目录结构 |
| **engineering-technical-writer.md** | 技术文档撰写 | README、benchmark 报告、复现指南 |

---

## 典型工作流

### 工作流 A · DINOv3 ONNX 导出验证

```
codebase-onboarding-engineer  →  ai-engineer  →  pytorch-build-resolver  →  python-reviewer
   (理解上游模型结构)              (写导出脚本)         (修 shape/device 问题)         (代码审查)
```

### 工作流 B · 量化与基准对比（核心研究流）

```
software-architect  →  ai-engineer  →  performance-benchmarker  →  model-qa  →  performance-optimizer
   (定量化策略)         (FP16/INT8 引擎)     (latency/throughput 矩阵)    (精度回归)      (深度优化)
```

### 工作流 C · C++ 部署封装

```
software-architect  →  ai-engineer  →  cpp-reviewer
   (Python/C++ 接口)    (C++ wrapper)    (内存/线程安全审查)
```

### 工作流 D · 实验报告产出

```
performance-benchmarker  →  technical-writer
    (汇总数据)                 (写 benchmark 报告/README)
```

---

## 调用示例

通过 Claude Code 的 Agent 工具按 `subagent_type` 调用（文件名去除 `.md` 即为类型名）：

```python
# 示例 1：让 AI engineer 写 ONNX 导出脚本
Agent(
    description="Write DINOv3 ONNX export script",
    subagent_type="engineering-ai-engineer",
    prompt="编写 DINOv3-ViT-L/16 的 4 输出 ONNX 导出脚本,opset=18,dynamic_axes 仅支持 batch 维度,
            输出命名 feat_layer_{4,12,16,20},register tokens 默认裁剪,
            导出后用 onnx.checker 验证,并逐输出对比 PyTorch 与 ONNX Runtime 的余弦相似度 >= 0.9999。"
)

# 示例 2：让 benchmarker 跑性能矩阵
Agent(
    description="Benchmark TRT engine across precisions",
    subagent_type="testing-performance-benchmarker",
    prompt="对 FP32、FP16、INT8-modelopt、INT8-legacy 四档 engine 进行 benchmark:
            batch_size ∈ {1,4,8,16,32},input_size ∈ {224,336,518},warmup=50,iters=1000,
            输出 P50/P95/P99 latency、p50 std、trimmed median、throughput(img/s) 与逐输出精度,保存为 reports/bench.csv。"
)

# 示例 3：让 model-qa 验证量化精度
Agent(
    description="Validate INT8 engine accuracy",
    subagent_type="specialized-model-qa",
    prompt="在 ImageNet val 1000 张子集上对比 dinov3_int8.engine 与 PyTorch FP32 baseline,
            分别报告 feat_layer_{4,12,16,20} 的 cosine sim 与 MaxAbsErr,
            不允许仅给 4 输出平均值。"
)
```

---

## 维护说明

### 同步全局更新

```bash
# 从全局 agents 目录刷新本地副本(覆盖)
for f in *.md; do
  [ "$f" != "README.md" ] && cp ~/.claude/agents/"$f" .
done
```

### 添加新 agent

```bash
# 例:加入 cpp-build-resolver 处理 C++ 编译错误
cp ~/.claude/agents/cpp-build-resolver.md .claude/agents/
# 然后手动编辑本 README 的"团队组成"表格
```

### 移除 agent

直接删除对应 `.md` 文件并同步本 README。

---

## 项目下一步建议

1. **创建 `CLAUDE.md`**（项目根级）：写明目标、技术栈版本约束（PyTorch、CUDA、TensorRT 期望版本）、当前阶段。
2. **创建 `.gitignore`**：忽略 `*.engine`、`*.onnx`、`*.pth`、`__pycache__/`、`build/`、`CLAUDE_PRIVATE.md`、`.venv/`。
3. **DINOv3 源码接入**：通过 git submodule 或直接下载 facebookresearch/dinov3 的预训练权重与参考实现。
4. **建立基线**：先用 PyTorch 跑通 DINOv3 推理，记录 baseline latency/精度，再开始 ONNX→TRT 优化。
5. **目录规划**（建议）：
   ```
   ├── src/                # 核心代码（Python）
   │   ├── export/         # ONNX 导出
   │   ├── engine/         # TRT 引擎构建
   │   ├── infer/          # 推理 wrapper
   │   └── quantize/       # 量化标定
   ├── cpp/                # C++ 部署封装
   ├── benchmarks/         # 基准测试脚本
   ├── reports/            # benchmark 报告输出
   ├── notebooks/          # 探索性 Jupyter
   └── tests/
   ```

---

## 风险与扩展

- **未包含底层开发 agent**（CUDA plugin/kernel）：若 DINOv3 中存在 TRT 不原生支持的算子（如某些 attention 变体），需追加 `cpp-build-resolver.md` 与考虑自定义 plugin。
- **未包含 build resolver**（`cpp-build-resolver.md`、`build-error-resolver.md`）：当 C++ 部署进入 CMake/编译阶段、或遇到大量构建错误时，可按需补充。
- **未包含通用 code-reviewer/security-reviewer**：研究阶段已被 python/cpp 专项 reviewer 覆盖，公开发布前再补充。
