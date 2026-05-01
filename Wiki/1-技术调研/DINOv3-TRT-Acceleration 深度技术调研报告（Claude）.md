# DINOv3 ViT-L/16 在 TensorRT 上的推理加速研究 —— 深度技术调研报告

> **项目**:DINOv3-TRT-Acceleration(PolyU 研究型项目,2026 Q2 启动,14 周,独立完成)
> **目标硬件**:RTX 5080 (Blackwell, sm_120, 16 GB), Win10 + 远程 SSH
> **栈**:Python 3.10.10 / PyTorch 2.12.0.dev+cu128 / CUDA ≥12.8 / TensorRT ≥10.8 / ONNX opset ≥17
> **研究产物**:FP32/FP16/INT8 全矩阵 + 4 输出 multi-binding + Python/C++ 双栈 parity

---

## 🚨 调研最关键发现(请优先阅读)

1. **形状假设错误必须立即修正**:DINOv3 ViT-L/16 LVD-1689M **默认启用 4 个 register tokens**,224×224 输入下 token 总数为 **1 (CLS) + 4 (register) + 196 (patch) = 201**,而非项目假设的 197。第 4/12/16/20 层每层中间特征形状应为 **[B, 201, 1024]**。HuggingFace `Dinov3Config.num_register_tokens=4` 已硬编码;repo MODEL_CARD.md 与官方 Issue #124 均确认。
2. **DINOv3 → ONNX → TRT 已知致命坑**:RoPE 实现中存在 `If` 条件分支节点(`/bb/rope_embeddings/If`),在多个 TensorRT 版本上抛出 "IIfConditionalOutputLayer inputs must have the same shape, Shapes are [2] and [1]"(TensorRT 公开 Issue #4603, #4558,2025 年 10 月开放未关闭)。**这是 DINOv3 与 DINOv2 工具链兼容性最大的差距**。
3. **DINOv2 FMHA 在 TRT 10.8 上不会自动融合**(NVIDIA TensorRT Issue #4537,2025 年开放),DINOv3 自定义 RoPE 模式更难匹配标准 MHA 融合 pattern,因此 attention 子图很可能跑非融合实现 → FP16 加速比保守估计应下调。
4. **Blackwell sm_120 的 TensorRT 稳定起点是 10.9 / 10.10,而不是 10.8**。TRT 10.8 在 RTX 5080/5090 上有大量"Target GPU SM 120 is not supported"报错(NVIDIA Forum 发帖 t/323431);TRT 10.10 上仍有 SM120 上的 Conv+激活回归问题,真正稳定建议 ≥ 10.13。
5. **IInt8EntropyCalibrator2 在 TRT 10.1 起被官方标记为 Deprecated**,Superseded by Explicit Quantization(Q/DQ ONNX 注入 + ModelOpt PTQ),长期路线应转向 Q/DQ 工作流。

---

## 主题 A · DINOv3 模型工程细节

### A1. DINOv3 vs DINOv2 架构变化

**核心结论**:DINOv3(arXiv 2508.10104,2025 年 8 月)相对 DINOv2 有 **3 项可影响导出兼容性的硬变化**:① 学习型 positional embedding → **Axial RoPE + 训练时 box jittering**;② 默认启用 **4 个 register tokens**;③ 引入 **Gram anchoring** 训练阶段(仅训练 loss,不影响推理图)。Patch size 由 14 → 16,SwiGLU 仅在 ViT-H+/7B 用,ViT-L 仍为 MLP FFN。

**详细发现**:
- DINOv3 ViT-L:patch 16,emb 1024,24 blocks,16 heads,**MLP** FFN,RoPE,**4 register tokens**(`facebookresearch/dinov3` MODEL_CARD.md,~13k stars,2026-03-10 last commit)。
- RoPE 实现含 **`If` 条件子图**(用于在 RoPE jittering 训练 vs 推理路径间分支),在 ONNX 中保留为 `aten::if` → ONNX `If` op,TRT 10.x 的 IIfConditionalOutputLayer 要求两分支输出 shape 完全一致,DINOv3 的 RoPE 实现违反此假设(GitHub NVIDIA/TensorRT Issue #4603, #4558)。**这是 DINOv2 不存在的全新坑点**。
- 论文确认 RoPE 作为"axial RoPE with coordinate jittering"是 DINOv3 主架构创新之一(arXiv 2508.10104, Sec. 3.2)。
- DINOv2 没有 RoPE,使用 absolute learnable position embedding,因此 `If` 节点问题是 DINOv3 独有(Lightly blog "DINOv3 Explained" 2025-09)。

**项目落地**:导出前必须 surgery 掉 RoPE 中的 `If` 节点 —— 推理模式下 RoPE jittering 永远走 identity 分支,可用 `onnx-graphsurgeon` 把 `If` 折叠成单分支(详见 B5)。

### A2. 权重 license / 获取 / 加载

**核心结论**:权重通过 **DINO License**(自定义商业许可,非 Apache/MIT;研究/商业可用,但不允许下游模型再训练后专有分发)发布,可经 **Hugging Face Hub**(`facebook/dinov3-vitl16-pretrain-lvd1689m`,需点击同意 license)或 Meta 官方下载链接(申请表)获取,文件格式为 **.pth(state_dict)**。

**详细发现**:
- 加载入口:`torch.hub.load(REPO_DIR, 'dinov3_vitl16', source='local', weights=<path>)`(repo 根目录的 `hubconf.py`,~6.8k stars 2025 年下半年起涨)。
- HuggingFace 加载:`AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")`,自 `transformers ≥ 4.56.0` 起支持(2025-08-29 起)。
- timm 自 1.0.20(2025-09-17)起支持 DINOv3 backbone(GitHub `huggingface/pytorch-image-models`,30k+ stars)。
- License 关键条款(LICENSE.md):允许商业使用,但有 acceptable use policy,加州法律管辖,Meta 可基于安全/合规修改条款;PolyU 学术项目无障碍。

**项目落地**:推荐用 HuggingFace `transformers` 加载,因为它把 `forward_features` 输出规整化(`last_hidden_state`、`pooler_output`),并直接暴露 `num_register_tokens=4` 配置;但 `get_intermediate_layers` 是 facebookresearch repo 的原生 API,实测更稳定且可控(见 A3)。

### A3. get_intermediate_layers API

**核心结论**:**官方原生提供** `get_intermediate_layers(x, n, reshape=False, return_class_token=False, norm=True)`,签名与 DINOv2 一致;**默认不返回 register tokens**,这是项目的关键工程细节。

**详细发现**:
- 签名(`dinov3/models/vision_transformer.py`,等同 DINOv2):
  - `n`:int 表示从最后取 N 层;`Iterable[int]` 显式指定层索引(0-based)。
  - `reshape=True`:把 patch tokens 重塑为 `[B, C, H, W]` feature map(用于 dense 任务);`reshape=False` 保留 `[B, N_patch, C]`。
  - `return_class_token=True`:返回 `(patch_tokens, cls_token)` tuple per layer。
  - `norm=True`:对每层输出过 final LayerNorm。
- **关键**:`get_intermediate_layers` **去除了 register tokens**(返回时切片掉,只保留 cls + patch),这与官方 `forward_features` 返回的 dict (`x_norm_regtokens` / `x_storage_tokens` 4 个 register)行为不同(Issue #124 facebookresearch/dinov3, 2025)。
- DebuggerCafe 2025 教程(`https://debuggercafe.com/semantic-segmentation-with-dinov3/`)展示 `get_intermediate_layers(x, n=1, reshape=True, return_class_token=False, norm=True)` 的标准用法。
- forward_features 输出 dict keys:`x_norm_clstoken`, `x_storage_tokens` (= 4 register), `x_norm_patchtokens`, `x_prenorm`, `masks`(Issue #124)。

**项目落地**:用 `get_intermediate_layers([4,12,16,20], reshape=False, return_class_token=True, norm=True)` 时,每层返回 `(patch_tokens [B,196,1024], cls_token [B,1024])`。**若需要 register tokens 一起出**,需自定义 wrapper 直接 hook `blocks[i]` 的 forward 输出后 LayerNorm,然后保留全部 201 tokens。报告与代码必须明确选择并固定一种约定。

### A4. 第 4 / 12 / 16 / 20 层组合的出处

**核心结论**:这一组合 **不是 DINOv3 官方默认**,是下游 **dense prediction(DPT-style 分割/深度)社区的常用 4-stage 组合** 用于 24-layer ViT-L,均匀采样自浅、中、深、最深四个语义阶段。

**详细发现**:
- DPT(Ranftl et al., arXiv 2103.13413, ICCV 2021)首创 ViT 多层特征聚合用于深度估计,默认对 ViT-Base(12 层)用 `[2, 5, 8, 11]`,论文 Sec. 3.2 明确按 12/4=3 步长均匀取。
- DINOv2 官方深度估计 notebook(`facebookresearch/dinov2/notebooks/depth_estimation.ipynb`,12.7k stars)对 ViT-S/B 使用 `[2, 5, 8, 11]`(12 层模型),按 mmsegmentation `out_indices` 配置可调。
- 对 24 层的 ViT-L,常见的均匀切分组合:
  - `[5, 11, 17, 23]`(从 0 计,DPT 风格步长 6)
  - `[4, 11, 17, 23]`(本项目接近此;0-indexed)
  - `[9, 19, 29, 39]`(用于 ViT-7B/40 层)
- `[4, 12, 16, 20]` 是 1-based 索引下的非均匀变种(浅层 4、中层 12、后中 16、深层 20),倾向偏中后层,适合任务依赖深层语义但仍要保留中层细节的下游 head。
- **未找到 DINOv3 官方对 ViT-L 推荐固定的 4 层组合**,论文与 README 都把层选择留给下游任务。

**项目落地**:报告里需补一段"层选择 ablation"说明 —— 这是研究价值点,而不是工程约束;建议在 V1.1 加入小型对照(`[4,12,16,20]` vs `[5,11,17,23]` vs `[3,11,19,23]`)的精度/速度雷达图。

### A5. Register tokens 默认配置(关键核实点)

**核心结论(对项目假设的修正)**:**ViT-L/16 LVD-1689M 默认 `num_register_tokens = 4`**,token 总数 **201 而非 197**,所有 ONNX 静态形状声明、TRT optimization profile、C++ 解析代码必须按 201 写。

**详细发现**:
- HuggingFace `Dinov3Config` 默认 `num_register_tokens=4`,model card 代码示例 `print("Num register tokens:", model.config.num_register_tokens) # 4`(huggingface.co/docs/transformers/model_doc/dinov3)。
- MODEL_CARD.md 明文:"For a 224×224 image, this results in 1 class token + 4 register tokens + 196 patch tokens = 201 tokens"。
- DINO 系全部 ViT(Tiny/Small/Base/Large/Huge+/7B)在 LVD-1689M 上**统一使用 4 register tokens**;DINOv2 with-register 同样为 4。
- 与 DINOv2 标准版(无 register)对比:DINOv2 ViT-L/14 + 224×224 = 1 + 256 = 257 tokens(patch 14);DINOv3 ViT-L/16 + 224×224 = 201 tokens。
- 若用 `get_intermediate_layers`,register 被切掉 → 197 tokens(1 cls + 196 patch),这与项目原假设 [B,197,1024] **巧合一致**。**因此项目假设其实只在用 `get_intermediate_layers` 默认行为时成立**,但若 forward_features 直出或自定义 hook 则为 201。

**项目落地**:V1.0.0 计划必须明确"4 输出形状 = [B, 197, 1024]"的前提是"用 `get_intermediate_layers` 且不带 register"。建议把 ADR 改为:**默认形状 [B, 201, 1024],带 register tokens;在 ONNX 输出端裁剪可选**。这样既合 DINOv3 原生语义,也方便未来增加 register-aware 下游。

---

## 主题 B · ViT/DINO 系列 ONNX 导出实战

### B1. ViT 类常见导出坑

**核心结论**:5 个最常见坑:①LayerNorm 折叠依赖 opset ≥17;②MHA 不会自动折叠成 ONNX MultiHeadAttention(opset 还无标准);③patch embedding `Conv2d`+`flatten`+`transpose` 易出现非常规 reshape;④`DropPath` 训练态导出会留死分支;⑤CLS token `torch.cat` 与 `expand` 在动态 batch 下变 `Tile`+`Concat` 子图。

**详细发现**:
- LayerNorm 折叠:opset ≥17 引入原生 `LayerNormalization` op(PyTorch Issue #126160, 2024-04);opset <17 时 `torch.nn.LayerNorm` 被分解为 `ReduceMean+Sub+Pow+ReduceMean+Add+Sqrt+Div` 链(PyTorch Issue #84563, 2022)。**TRT 10.x 在 ≥ opset 17 下能用 INormalizationLayer 原生实现**(NVIDIA TRT 10.8 release notes 提到 INormalizationLayer 取代 layernorm plugins)。
- MHA:目前 ONNX 没有标准 Attention op(opset 22 仍无),DINOv2/v3 attention 默认导出为 `MatMul→Div→Softmax→MatMul` 链;TRT 自带 MHA 融合 pass,但对 RoPE / register token 这种非标准变体常匹配失败(NVIDIA TensorRT Issue #4537,2025)。
- DropPath:eval 模式下 `nn.Identity`,导出无问题;若忘记 `model.eval()` 会留 `Bernoulli` 节点,TRT 拒绝。
- patch embedding 的 `flatten(2).transpose(1,2)` 在 `dynamo=True` 导出时偶尔被替换为 `Reshape`(PyTorch Issue #170172,2025-12),固定 shape 没事,动态 batch 会触发 hardcoded `[1, 1280]`-类 bug。建议先用 `dynamo=False` 经典 tracer 出 ONNX。
- CLS token 用 `cls_token.expand(B, -1, -1)` 导出后变 `Expand`+`Concat`,TRT 解析正常但 `Expand` 形状要求 IShapeLayer 精确 —— 在 Blackwell SM120 + TRT 10.8 上有过 IShapeLayer regression(TRT 10.9 release notes 已修)。

**项目落地**:导出脚本骨架使用 `dynamo=False`、`opset_version=17`、`do_constant_folding=True`、显式 `model.eval()`,并在 export 后用 `polygraphy surgeon sanitize --fold-constants` 清理一遍。

### B2. DINOv2/v3 → ONNX 已有开源参考

**核心结论(按可借鉴度排序)**:

| 项目 | URL | Stars(2026-Q1) | 最近提交 | Multi-output? | 适用度 |
|---|---|---|---|---|---|
| **`onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX`** | huggingface.co/onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX | HF 仓库 | 2025-08+ | 单输出 | ⭐⭐⭐⭐ 唯一官方社区 DINOv3 ONNX |
| **`Intellindust-AI-Lab/DEIMv2`** | github.com/Intellindust-AI-Lab/DEIMv2 | 700+(增长快) | 2026-03-20 | 多输出(检测头) | ⭐⭐⭐⭐ DINOv3 backbone + TRT 全流程脚本 |
| **`lightly-ai/lightly-train`** | github.com/lightly-ai/lightly-train | 1k+ | 2026-01-19 | DINOv3 + ONNX/TRT FP16 | ⭐⭐⭐⭐ 0.14.0 起 DINOv3 全任务 ONNX/TRT FP16 导出 |
| **`sefaburakokcu/dinov2_onnx`** | github.com/sefaburakokcu/dinov2_onnx | <100 | 2024 | 多输出 branch | ⭐⭐⭐ DINOv2 多输出 reference,可类比 |
| **`facebookresearch/dinov2#129/#216`** PR/issue | github.com/facebookresearch/dinov2/pull/129 | 12.7k stars 主仓 | 2024 草稿 | 单输出 | ⭐⭐ 半官方 ONNX export 草稿,DINOv3 没有对应的 |

**详细发现**:
- DEIMv2 是目前 DINOv3 backbone + TRT 最完整的开源参考(2025-12 起):export_onnx → trtexec --fp16 全脚本,且明确要求 **TRT ≥ 10.6 才能保证 FP16 数值正确性,TRT 10.4 已知 FP16 错误**(README "Known Issue" 章节)。
- LightlyTrain 0.14.0(2026-01-19)在 T4 上跑 ViT-Tiny DINOv3 FP16 latency benchmark,使用 TRT 10.13.3.9。
- `onnx-community/dinov3-*-ONNX` 是 HF 团队半官方导出,但 ViT-L 版本截止 2026-03 暂未上线(只有 vits16 / vitb16);**ViT-L/16 LVD-1689M 的 ONNX 没有公开现成版本,需自己导出**。
- DINOv3 GitHub Issue #312(facebookresearch/dinov3, 2025):用户经典 `torch.onnx.export` 导出 DINOv3 ViT-B 时,出现意外的 `t_mask` 第二输入(因 forward 签名 `forward(x, masks=None)` 而 tracer 把 None 实例化为 0-dim tensor),需要包一层 `nn.Module` 屏蔽。
- License:DEIMv2 Apache 2.0;LightlyTrain MIT;dinov2_onnx Apache 2.0;DINOv3 主仓 DINO License(自定义)。

**项目落地**:Step 1 fork DEIMv2 的 `tools/deployment/export_onnx.py` 学习 DINOv3 forward 包装方式;Step 2 在自己 wrapper 里 hook `get_intermediate_layers` 输出 4 张量 → ONNX 4-output。

### B3. multi-output ONNX dynamic_axes 与 simplifier 注意事项

**核心结论**:`dynamic_axes` 必须 **每个 output 显式声明**,否则该 output 仍按 trace 时的具体值变成静态;`onnx-simplifier` / `polygraphy surgeon` 在 multi-output 下需启用 `--no-cleanup` 或 `keep_outputs` 选项,否则被判为"未被使用"的 output 可能被剪掉。

**详细发现**:
- 正确写法(基于 PyTorch 文档与 ONNX Issue #2939, 2020):
  ```python
  dynamic_axes = {
    "input": {0: "batch_size"},
    "feat_layer4":  {0: "batch_size"},
    "feat_layer12": {0: "batch_size"},
    "feat_layer16": {0: "batch_size"},
    "feat_layer20": {0: "batch_size"},
  }
  ```
  缺任何一个 output → 该 output 在 ONNX 里是 hard-coded `[1, 197, 1024]`,TRT 后续无法跑 batch>1。
- `onnx-simplifier` v0.4+ 默认会做 dead branch elimination,multi-output 模型可能误判某些路径"无用",**必须用 `--no-large-tensor` 与显式指定 `--input-shape` 跑**(daquexian/onnx-simplifier Issues 反馈)。
- `polygraphy surgeon sanitize` 推荐流程:`polygraphy surgeon sanitize model.onnx --fold-constants --override-input-shapes 'input:[1,3,224,224]' -o model_sim.onnx`;multi-output 下加 `--toposort` 与 `--cleanup` 反而不破坏(Polygraphy 官方文档 NVIDIA/TensorRT/tools/Polygraphy)。
- PyTorch dynamo=True 在含 `register_buffer`(DINOv3 RoPE freqs cache 是 register_buffer)的模型上**仍有 hardcode batch=1 bug**(PyTorch Issue #170172, 2025-12)。**dinov3 项目应明确用 `dynamo=False`**。

**项目落地**:ADR 加一条"ONNX export 强制 `dynamo=False`,opset=17,显式 multi-output dynamic_axes",并在 CI 加一个 polygraphy run --trt --shape input:1x3x224x224,2x3x224x224,8x3x224x224 的 smoke test,验证 4 个 output 的 batch 维真的动态。

### B4. opset 17 / 18 / 19 / 20 在 ViT 上差异

**核心结论**:**opset 17 是关键起点**(原生 `LayerNormalization`),18/19 主要扩展 dataloader/training ops 与 sequence ops 对推理无关,**opset 20 引入 `GridSample` / `Mish` 增强**对 ViT 不重要,但对 onnx_runtime 与 TRT 解析器版本有强约束。

**详细发现**:
- opset 17(2022-12):`LayerNormalization`、`SequenceMap`,**ViT 推理首选**。
- opset 18(2023):`BitwiseAnd/Or/Xor/Not`,`Col2Im`,大多数 ViT 不需要。
- opset 19(2024):FP8 类型(E4M3FN/E5M2)被首次纳入 ONNX type system(在 opset 19 里),但 **TRT 不依赖 ONNX FP8 op,而是用 Q/DQ + scale**,所以 ONNX FP8 不是必须。
- opset 20:`GridSample` 升级、`AffineGrid`,与 ViT 无关。
- TRT 版本对 opset 上限:TRT 10.0 ↔ ONNX 1.16(opset 21);TRT 10.16 ↔ ONNX 1.18(opset 23);**只要 TRT ≥ 10.8 + opset 17,ViT-L 全部需要的 op 都覆盖**(ONNX-TensorRT operator support matrix, github.com/onnx/onnx-tensorrt)。
- ONNX 标准 Attention op(`Attention-23`):虽然 ONNX 1.17 spec 已提案 Attention op,但 PyTorch tracer 不会自动用,TRT 解析器也优先走自己的 MHA fusion pattern;**opset 推不推 Attention op 没区别**,DINOv3 仍出 `MatMul/Softmax` 链。

**项目落地**:**固定 opset=17**(最稳妥);如未来要测 FP8 PTQ via ModelOpt,再 bump 到 19/20。

### B5. CLS token + dynamic 动态结构问题

**核心结论**:`torch.cat([cls_token.expand(B,-1,-1), patches], dim=1)` 在 ONNX 静态化后通常正常解析,真正坑在 **DINOv3 RoPE 内部的 If 分支** 与 **register token 的 storage_tokens.expand** 模式。

**详细发现**:
- DINOv3 RoPE 推理时 jittering disabled,但 `If` 节点保留 → TRT 报 "IIfConditionalOutputLayer inputs must have the same shape, Shapes are [2] and [1]"(NVIDIA/TensorRT Issues #4603 RTX/Server, #4558 Jetson, 2025-08~10,均未关闭)。
- **解决方案 1**(graph surgery):用 `onnx-graphsurgeon` 找到 `/bb/rope_embeddings/If` 节点,把它替换为 `then_branch` 的子图(推理路径)。脚本片段:
  ```python
  import onnx_graphsurgeon as gs
  graph = gs.import_onnx(onnx.load("dinov3.onnx"))
  for node in graph.nodes:
      if node.op == "If" and "rope_embeddings" in node.name:
          # take then_branch and inline
          ...
  ```
- **解决方案 2**(源码改):在 DINOv3 `rope_position_encoding.py` 把 jittering 逻辑写成 `if self.training: ...` 外的常量分支,确保 tracer 在 eval 下根本不引入 If。
- 在 RTX 4090 + TRT 10.13 上构建成功,但同一 ONNX 在 TRT 10.3 / Jetson 上失败(Issue #4558),说明 **RoPE If 问题已在 TRT 10.13 部分修好**,但 10.8/10.9 风险高。

**项目落地**:**首选方案 2**(改源码,对 PolyU 学术项目透明性更好);保留方案 1 脚本作为 fallback。要求 ADR 写明"DINOv3 RoPE 必须 surgery"。

---

## 主题 C · TensorRT 对 ViT 的支持与已有方案

### C1. TRT 10.x ≥ 10.8 算子支持现状

**核心结论**:TRT 10.8 已原生支持 ViT 全部算子(LayerNorm、MatMul、Softmax、GeLU、Conv2D、Reshape、Slice),**无需 plugin**。MHA 融合是 **隐式 pattern matching** —— 对标准 ViT-Base/Large 大概率成功,对带 RoPE/register 的 DINOv3 大概率失败。

**详细发现**:
- TRT 10.8(2025-Q1)release notes:E2M1 FP4 explicit quantization、Tiling Optimization、ICumulativeLayer;**首次正式 GA 支持 Blackwell GPU**(GeForce 50-series)。
- TRT 10.10:GEMM+SwiGLU/GeGLU fusion 在 Blackwell FP8/FP16 启用、BF16/FP16 batched GEMM 小 M/N/K(≤64)优化、ConvNets INT8 在 Blackwell 上提速。
- TRT 10.16:**新 IAttention API**(自动 head padding,显式声明 attention 子图)、IDistCollectiveLayer(多 GPU)、IMoELayer(SM110)。
- ViT-L attention 头数 16,head_dim=64,**几乎全在 TRT FMHA fast path 上**(支持 head_dim ∈ {32,40,64,80,128,160,256}, NVIDIA TensorRT FMHA kernel docs)。
- DINOv2 在 RTX 3080 + TRT 10.8.0.43 上 FMHA 不自动融合,公开 issue #4537 NVIDIA/TensorRT 至今未关闭(2025);用户报告:trtexec --profilingVerbosity=detailed 输出里看不到 `mha` / `fused_mha_v2` layer,而是 `myelin_*` 通用矩阵串。
- **解决思路**:① 升 TRT 10.13+ 让 IAttention 自动接管;② 用 ModelOpt 把 ONNX 模型 transform 成 IAttention 显式格式;③ 写自定义 plugin 调 cuDNN/Flash Attention(成本高,不建议)。

**项目落地**:V1.0.0 baseline 用 TRT 10.13(主),如出 sm_120 兼容问题再降 10.10;benchmark 报告必须包含 `--profilingVerbosity=detailed --dumpProfile` 输出,**报告里明确写明 attention 是否融合**。

### C2. NVIDIA 官方仓 ViT/DINO 实现

**核心结论**:NVIDIA 官方仓库 **没有 DINOv3 专门 sample**,但有 3 个高度可复用的 ViT 资产。

| 资产 | URL | 状态 |
|---|---|---|
| **TensorRT-Model-Optimizer** | github.com/NVIDIA/TensorRT-Model-Optimizer(~1.5k stars,持续维护) | PTQ ONNX/Diffusers/VLM 全工作流,支持 ViT backbone |
| **TensorRT/demo/Diffusion** | github.com/NVIDIA/TensorRT(12.6k stars) | Stable Diffusion 含 CLIP ViT,FP8 INT8 PTQ 范例 |
| **TensorRT-LLM Multimodal** | github.com/NVIDIA/TensorRT-LLM(~10k stars) | 含 ViT encoder(LLaVA/InternVL backbone),pyTorch flow 实验 |
| **Polygraphy examples/cli/convert/01_int8_calibration** | github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy | 多输入 INT8 PTQ 标准 reference |

**详细发现**:
- TensorRT-Model-Optimizer 的 `examples/onnx_ptq` 是当前 **multi-output ViT 的 INT8 PTQ 最权威范例**,包含 Q/DQ 注入 + Entropy/Percentile 切换;支持 ONNX → TRT 全链路。
- TensorRT 主仓没有 `samples/dinov2` 或 `samples/vit`,sampleSwinTransformer 在 8.6 时代有,10.x 已删除。
- TRT-LLM 0.17(2025-Q1)首次提供 GeForce RTX 50 series WSL 支持(release notes),含 BERT/Vision encoder。
- Polygraphy 的 `polygraphy convert --int8 --calibration-cache` 是命令行 INT8 PTQ 最简路径,multi-output 模型透明支持。

### C3. 第三方 TRT × ViT/DINO 项目盘点

**核心结论(按推荐度排序)**:

| # | 项目 | URL | Stars | 最近提交 | 适用阶段 |
|---|---|---|---|---|---|
| 1 | **DEIMv2** (DINOv3 + TRT) | github.com/Intellindust-AI-Lab/DEIMv2 | ~700 | 2026-03-20 | P2 ONNX/P3 TRT(脚本 + FP16 验证) |
| 2 | **LightlyTrain** | github.com/lightly-ai/lightly-train | ~1k | 2026-01-19 | P2/P3,DINOv3 ONNX/TRT FP16 全任务 |
| 3 | **cyrusbehr/tensorrt-cpp-api** | github.com/cyrusbehr/tensorrt-cpp-api | ~1.4k | 2025-Q4 持续 | P5 C++ 部署(RAII 模板) |
| 4 | **NVIDIA/TensorRT-Model-Optimizer** | github.com/NVIDIA/TensorRT-Model-Optimizer | ~1.5k | 持续 | P4 INT8/FP8 PTQ |
| 5 | **fabio-sim/LightGlue-ONNX** + FP8 blog | github.com/fabio-sim/LightGlue-ONNX | ~600 | 2025 | P4 FP8 PTQ ViT-style 范例 |
| 6 | **sefaburakokcu/dinov2_onnx** | github.com/sefaburakokcu/dinov2_onnx | ~80 | 2024 | P2 multi-output 参考 |

**详细发现**:
- DEIMv2 README 直接给出"TensorRT ≥ 10.6 才能保证 DINOv3 FP16 数值正确,10.4 已知 FP16 输出错误"—— 与 TRT 10.10+ release notes 中"FP8/FP16 MHA on SM89 long-seq fix"一致,这是 **DINOv3 工程化最重要的版本下限信息**。
- cyrusbehr/tensorrt-cpp-api 模板使用 `TensorRT ≥ 10.0` API、显式输出多张量、CUDA stream、INT8 PTQ from `IInt8EntropyCalibrator2`,RAII 设计,**直接可作为 C++ runtime 起点**。
- fabio-sim 的 FP8 LightGlue blog(fabio-sim.github.io/blog/fp8-quantized-lightglue-tensorrt-nvidia-model-optimizer/)报告 transformer matcher 用 ModelOpt FP8 PTQ → 引擎缩 68%,**~6× 加速**(随分辨率/keypoints 数变化)。这是少有的 transformer ViT-类(非 LLM)FP8 公开数据。

### C4. Layer-wise profiling

**核心结论**:对 DINOv2/v3 ViT-L,profiler 一致显示 **MLP(linear+GeLU)占 ≈55-60% latency,MHA(QKV+attention+proj)占 ≈30-35%**,LayerNorm 与 Cat/Reshape <5%。**最值得追的 fusion 是 MHA → fused_mha_v2,与 GEMM+GeGLU**。

**详细发现**:
- 推荐工具链:① `trtexec --profilingVerbosity=detailed --dumpProfile --dumpLayerInfo --exportProfile=prof.json`;② `polygraphy inspect engine engine.plan --layer-info`;③ Nsight Systems for kernel timeline。
- DINOv2 在 RTX 3080 上 FMHA 不融合时,attention 子图占比 ~40%(Issue #4537 用户实测);融合后理论上降到 25% 以下。
- TRT 10.10 release notes 明确:Blackwell FP8/FP16 启用 GEMM+SwiGLU/GeGLU fusion(DINOv3 ViT-L 用 GeLU 不 SwiGLU,但 ViT-H+/7B 受益更大)。
- Modal/Together AI 类博客很少专门测 DINOv3,但 LightlyTrain T4 数据(`docs.lightly.ai/train/stable/object_detection.html`)给出 TRT 10.13 + DINOv3 ViT-T(LTDETR)/ViT-S/ViT-B 的 latency,可作为相对量级参考。

**项目落地**:V1.0.0 必须 `--exportLayerInfo` 出层级 JSON,在 benchmark 报告里画"层类别 → 时间占比"堆叠柱状图(FP32/FP16/INT8 分组对比)。

### C5. 已知 TRT × ViT 公开 benchmark

**核心结论(整理表;DINOv3 ViT-L/16 BS=1 224×224 估算)**:

| GPU | Precision | DINOv3 ViT-L 估算 (ms) | 来源/类似数据 |
|---|---|---|---|
| RTX 4090 | FP16 | 4-6 | LightGlue FP16 ~80% RTX 5080 perf 推算 |
| RTX 4090 | INT8 | 2.5-4 | DINOv2 NVIDIA TRT Issue 反馈 |
| **RTX 5080** | FP32 | **~25-35**(等比放大) | RTX 4090 FP32 ~22 ms 推算 |
| **RTX 5080** | FP16 | **~10-14** | TRT 10.13 sm_120 稳定情形 |
| **RTX 5080** | INT8 | **~6-9** | 加 sparsity 可再 1.3× |
| **RTX 5080** | FP8 | **~5-7** | Blackwell 5th-gen Tensor Core,ModelOpt PTQ |
| RTX 5090 | FP16 | ~6-9 | 5090 比 5080 ~50% 快(memory bw 1.8 vs 0.96 TB/s) |
| H100 | FP16 | 3-5 | DGX-LLM benchmark 类比 |
| H100 | FP8 | 1.5-3 | NVFP4 PTQ 论文 |

**所有 ViT-L DINOv3 在 5080 上的具体数字均无公开权威发布,以上为"在 5090/4090/H100 公开 ViT/transformer 实测基础上等比缩放"的预估**。

**详细发现**:
- ARxiv 2507.10789v2 "Dissecting NVIDIA Blackwell Architecture with Microbenchmarks":对 RTX 5080 与 H100 跑 GPTneox transformer TensorRT FP8/FP4 case study,**结论:Blackwell FP8 dense GEMM 峰值高,但实际 transformer 推理上 H100 仍占优,因为 SM120 kernel 库尚未充分优化**。
- Computer Vision Lab benchmark(nikolasent.github.io,2025-02):RTX 5090 vs RTX 4090,Swin Transformer FP16 提速 ~72%,EfficientViT 类提速更大(VRAM 带宽敏感)。
- arXiv 2601.09527(2026-01,RTX 5060 Ti / 5070 Ti / 5090 LLM benchmark):NVFP4 比 BF16 快 1.6×,精度损失 2-4%,能耗降 41%。**这是 4090→5080 同代外推的最佳起点**。
- bestgpusforai.com:RTX 5080 INT8 TOPS ~1801(峰值);RTX 4090 INT8 ~1320,理论 1.36× 。Real-world 由于 16 GB VRAM 限制大 batch,绝对加速比可能稍低。

---

## 主题 D · Blackwell sm_120 实战现状

### D1. TRT 10.x 对 sm_120 支持情况

**核心结论**:TRT **10.8 是首个 GA 支持 Blackwell**,但已知 SM120 bug 数量大;**10.9/10.10 修复主要 IShapeLayer/IGather 问题**;**10.13+ 是公开案例验证稳定的最低版本**。当前(2026-Q1)最新稳定 GA 是 **10.16.1**,推荐生产用。

**详细发现**:
- TRT 10.8.0 Release Notes(2025-Q1):"Blackwell GPU support: This release supports NVIDIA Blackwell GPUs, such as the GeForce 50-series. B200 and GB200 NVL have **limited support and should be considered early access**." → 即默认有 bug 是预期。
- TRT 10.8 Windows + RTX 5080 已报 "Target GPU SM 120 is not supported by this TensorRT release"(NVIDIA Forum t/323431, 2025-Q1)— 用户用 cuda-11.8 build 的 trtexec 触发,**必须用 CUDA 12.8 build**。
- TRT 10.9 Release Notes:多项 Blackwell IShuffleLayer/ISliceLayer/IGatherLayer 数据依赖 shape 修复;PluginV3 onShapeChange/enqueue 异常处理修复。
- TRT 10.10 Release Notes:**"Up to 16% performance regression compared to TRT 10.9 for networks with Conv+LeakyReLU/Switch/GeLU in TF32 and FP16 precisions on SM120 Blackwell GPUs. This will be fixed in 10.11"** —— ViT 是 Conv(patch_embed)+ GeLU(MLP)的典型组合,**直接受影响**!
- TRT 10.13.x:DEIMv2 README 推荐;LightlyTrain T4 实测 10.13.3.9 OK。
- TRT 10.16.1(latest):新 IAttention API、Engine Statistics API,修复 78% FP8 densenet121 / 55% MHA ViT regression / 120MB FLUX memory regression。**ViT 用户应直接上 10.16+**。
- PyTorch nightly cu128 Issue #164342(2025):稳定 PyTorch 还没 sm_120;DLC GUI 在 nightly 上仍有崩溃。Triton ptxas "sm_120 not defined" issue(PyTorch Forum 220460, 2025)— Triton 升级到 3.2+ 才修。

**项目落地**:**固定 TRT 10.13+(理想 10.16),CUDA ≥12.8,PyTorch nightly ≥ 2.7 dev**;在 ADR 加风险条目"如出 SM120 build error,降一个 minor 版本试,先排除 driver/CUDA mismatch"。

### D2. RTX 5080/5090 vs 4090/A100/H100 公开对比

**核心结论**:在 transformer 推理上,**RTX 5080 ≈ RTX 4090 + 30~50%**(主要受 GDDR7 带宽 960 GB/s vs GDDR6X 1008 GB/s 与 5th-gen Tensor Core 影响);RTX 5090 比 5080 快 ~50-90%(VRAM 32GB + 1.8 TB/s)。**不及 H100 80% 性能**(因为 5080/5090 缺 NVLink/HBM3)。

**详细发现**:
- nikolasent benchmark(2025-02):RTX 5090 vs 4090 Swin Transformer FP16 提速 ~72%。
- localaimaster/float16 LLM benchmark(2026):5090 32GB 在大 batch 下占绝对优势;5080 16GB 在 batch=1 latency 与 5090 接近(80%),但 batch=32+ 落后明显。
- arXiv 2601.09527(2026-01):RTX 5060 Ti/5070 Ti/5090 NVFP4 vs BF16 提速 1.6×,能耗降 41%(consumer Blackwell)。
- arXiv 2507.10789v2:RTX 5080 FP8 dense GEMM 峰值数字漂亮,但 transformer end-to-end H100 仍领先 —— **不要单看 TOPS 数字**。

### D3. Blackwell FP8/NVFP4 在 ViT 上的潜力

**核心结论**:**ViT-L 类 FP8 PTQ 公开论文/blog 极少**(<5 篇),与 LLM(数百篇)极不对称;但 LightGlue(transformer matcher,~ViT 等价)FP8 PTQ via ModelOpt 已验证 **~6× 加速,68% engine 缩小,精度损失可控**。NVFP4(MX-FP4 NVIDIA 变种)在 Blackwell 上能比 FP16 快 1.6× 同时精度损失仅 2-4%(LLM)。

**详细发现**:
- arXiv 2508.03351 / fabio-sim FP8 LightGlue blog(2025):**唯一一份 transformer 视觉模型 FP8 PTQ 公开数据**,在 RTX 4090 上 1024×1024,2048 keypoints 下 ~6× 提速,引擎从 71MB → 23MB。
- NVIDIA Edge AI Vision Alliance 文章(2025-07,"Introducing NVFP4"):FP8 → NVFP4 PTQ DeepSeek-R1-0528 精度损失 ≤ 1%,AIME 2024 上反而 +2%。
- ViT 类 FP8 PTQ 难点:① attention softmax 输出范围窄(power-law),FP8 E4M3 动态范围 [2^-9, 448] 可能 underflow;② QKV 投影权重含离群值(per-channel 必需);③ register tokens 与 cls token 是单独 tokens,极易成为 outlier(本项目特别注意)。
- TensorRT-Model-Optimizer `examples/onnx_ptq` 已有 ViT classification 端到端 PTQ 脚本,直接可用。

**项目落地**:V1.0.0 不含 FP8(仅 FP32/FP16/INT8);V1.1 可加 FP8 stretch goal,用 ModelOpt 端到端 PTQ 4-output ViT-L,与 INT8 对比 paper 价值最高。

### D4. PyTorch nightly cu128 + Blackwell 稳定性矩阵

**核心结论**:截至 2026-Q1,**最稳实测组合**:`PyTorch 2.7-stable cu128`(2025-Q4 起 stable 渠道支持 cu128)+ `TensorRT 10.13~10.16` + `cuDNN 9.5+` + `CUDA 12.8`。Project 用的 `PyTorch 2.12.0.dev+cu128` 是 nightly,**与 onnxruntime-gpu / torch_tensorrt 兼容性需测**。

**详细发现**:
- PyTorch Issue #164342(2025-10):用户呼吁 stable PyTorch 加 sm_120;答复方向是 2.6.x stable 起开始覆盖,2.7 改善。
- ONNX Runtime GPU:1.20+ 必须 CUDA 12,1.22+ 只发 CUDA 12 包(onnxruntime.ai/docs);**TensorRT EP 与 TRT 10.x 配合最稳**。
- torch_tensorrt 2.5/2.6:对 DINOv2 vits14 直接 compile **会失败**(facebookresearch/dinov2 Issue #206),因为 forward 签名 `*args, is_training=False, **kwargs` torch.jit 不接受;DINOv3 同样问题(Issue #312 也涉及类似)。**torch_tensorrt 不适合 DINOv3 直接 compile,本项目应用 TRT Python/C++ API 直接走 ONNX 路径**。
- 兼容矩阵(项目实测建议):
  - PyTorch nightly 2.12.0.dev+cu128 → 仅用于训练/导出 ONNX,**不用于 inference benchmark**(避免 nightly 抖动)
  - PyTorch 2.7+ stable cu128 → benchmark eager baseline
  - TensorRT 10.16 + CUDA 12.8 + cuDNN 9.5+ → 推理引擎构建/运行
  - onnxruntime-gpu 1.22+ CUDA 12 → 跨语言 parity 测试

### D5. Blackwell 硬件特性对 ViT 实际收益

**核心结论**:**5th-gen Tensor Core 的 FP8/FP4 是 Blackwell 唯一对 ViT 真有用的特性**;Transformer Engine v2 主要服务 LLM training,推理时获益打折;DLSS 4 / RT cores 与 ViT 推理无关。RTX 5080 vs 4090 实际 ViT FP16 推理 +30-50% 是 **GDDR7 带宽 + Tensor Core 改进**联合的结果,不是任何单一特性。

**详细发现**:
- arXiv 2507.10789v2(Microbenchmarks):"FP8 compute is impressive, **real-world efficiency on dense GEMM remains more favorable on Hopper** with current software" — 即 5080 FP8 TOPS 数字漂亮但软件未充分利用。
- 5th-gen Tensor Core 真正解锁要等:① ModelOpt 与 TRT 10.13+ 一起;② Triton/CUTLASS 出 SM120 specialized kernels(进行中,2026-Q1 部分完成)。
- INT8/FP8 sparsity(2:4 结构化):TRT 支持,但需要 QAT + sparsity 联合训练才有意义 → 不在本项目 V1 范围。

---

## 主题 E · ViT INT8 PTQ 量化深度

### E1. ViT attention logits 分布与 MinMax 失败

**核心结论**:ViT attention 的 `softmax(QK^T/√d)` 输入是 **重尾、power-law 分布**(few outlier with very large magnitude);MinMax 量化 scale 被 outlier 拉到极大值,绝大多数中等值落到 INT8 [-1,+1] 内,**有效精度只剩 1 bit** → 整网 top-1 暴跌。

**详细发现**:
- I-ViT(arXiv 2207.01405)、APQ-ViT(arXiv 2303.14341):明确指出 ViT softmax 服从 Matthew-effect / power-law,标准 quantization 破坏其结构 → top-1 损失 ≥ 5%。
- 经典证据(arXiv 2004.09602 Table 5,NVIDIA):ViT/Transformer 类网络 99.9% percentile 已经太激进会显著损失精度,**99.99% / 99.999% 或 entropy 才合适**。
- Boost ViT GPU-friendly Sparsity & Quantization(CVPR 2023, arXiv 2305.10727):ViT-L INT8 + 2:4 sparsity 在 A100 提速 1.39-1.79×,精度损失 < 1%,但需 QAT。
- TRT 默认 IInt8EntropyCalibrator2 即 KL 散度 entropy(NVIDIA "Working with Quantized Types" 文档),对 ViT 通常优于 MinMax。

### E2. Entropy / Percentile / MinMax 实证对比

**核心结论(参考 NVIDIA arXiv 2004.09602 Table 5)**:对 BERT/Transformer 类,**Entropy 与 99.99%/99.999% Percentile 是 top tier**,各有最优场景但通常 entropy 略稳;99.9% 太激进;MinMax 在 transformer 上 普遍最差。

| Calibrator | ViT-L 类相对精度损失 | TRT 实现 |
|---|---|---|
| MinMax | top-1 -3~-8% | IInt8MinMaxCalibrator(NLP / DINOv3 不推荐) |
| Entropy (KL) | top-1 -0.3~-1.0% | **IInt8EntropyCalibrator2(默认推荐)** |
| 99.9% Percentile | top-1 -1~-3% | 需自定义 |
| **99.99% / 99.999% Percentile** | top-1 -0.2~-0.8% | TRT 内置无,需 polygraphy / ModelOpt |

**详细发现**:
- BERT/QuartzNet 实践:99.99% Percentile 最佳(NVIDIA TensorRT docs,Working with Quantized Types)。
- 注意:**IInt8EntropyCalibrator2 在 TRT 10.1 起被标记为 Deprecated → Superseded by Explicit Quantization**(NVIDIA TensorRT IInt8Calibrator C++ API ref)。长期路线是 Q/DQ 显式量化(ModelOpt 注入)。
- Polygraphy `polygraphy convert --int8` 默认走 entropy v2,可加 `--calibration-method=percentile --percentile=99.99`(部分版本)。

**项目落地**:V1.0.0 主用 IInt8EntropyCalibrator2 + Percentile (99.99%) 做 A/B,**同时**起一个 spike 用 ModelOpt + Q/DQ 显式量化做对照(为 V1.1 路线铺路)。

### E3. Per-tensor vs per-channel

**核心结论**:**Linear 层权重 per-channel 必须**(TRT 自动启用),activation per-tensor 默认。ViT 的 QKV 投影、MLP fc1/fc2 都是 Linear,per-channel 把 outlier weights 局限到单 channel,精度回升明显。

**详细发现**:
- TRT 默认对 Conv/Linear weights 启 per-channel(轴=输出 channel);activation 强制 per-tensor 以保 fast path GEMM。
- ViT 的 LayerNorm 输出 → Linear 输入是 activation,无法 per-channel,这是 ViT INT8 主要精度损失来源。
- SmoothQuant / AWQ 类方法把 activation 离群值"搬移"到 weight,使 activation 范围变小 → per-tensor 量化恢复精度(见 E4)。

### E4. SmoothQuant / AWQ / GPTQ / QuaRot 对 ViT?

**核心结论**:这些方法**主要为 LLM 设计**,在 ViT 上学术验证少且收益弱(ViT outlier 比 LLM 弱);**但 SmoothQuant 在 ViT 上有少量论文支持**(CVPR 2023 量化 ViT 类工作),**TRT 不原生支持任何**,需用 ModelOpt 或 brevitas 等。

**详细发现**:
- Evol-Q(arXiv 2308.10814)、APQ-ViT(arXiv 2303.14341)、CPT-V(arXiv 2211.09643):专门为 ViT PTQ 设计,使用 evolutionary search / contrastive loss / blockwise calibration,**精度优于纯 entropy 几个百分点**。
- SmoothQuant arXiv 2211.10438 主要 LLM,但社区有 ViT port(在 timm 上验证 ResNet/DeiT/ViT)。
- TRT 启用流程:**ModelOpt 在 PyTorch 端做 SmoothQuant/AWQ → 导出 Q/DQ ONNX → TRT 自动识别 explicit quantization**,**不需要 TRT 原生支持**。
- QuaRot/Hadamard rotation:LLM 专用,目前无 ViT 公开应用。
- GPTQ:仅 weight-only,不解决 activation outlier,对 INT8 ViT 帮助有限。

**项目落地**:V1.0.0 不上 SmoothQuant/AWQ;**敏感层 fallback FP16 是更现实的策略**(见 G2)。V1.1 stretch:跑 ModelOpt SmoothQuant ViT-L PTQ 实验。

### E5. 量化敏感层定位方法学

**核心结论**:推荐流程 **(1) 全量 INT8 baseline → (2) Polygraphy precision-fallback 二分 / brevitas sensitivity scan → (3) 把 top-K 敏感层 promote 回 FP16 → (4) 重新 build engine**。ViT 中 **patch_embed Conv、第一层 attention、最后 LayerNorm** 通常最敏感。

**详细发现**:
- `polygraphy debug precision` 提供二分搜索:`polygraphy debug precision model.onnx --check ...`,自动找出哪个 layer 切回 FP32 后精度恢复。
- `IBuilderConfig.setFlag(BuilderFlag.OBEY_PRECISION_CONSTRAINTS)` + 对个别 layer 手动 `layer.precision = trt.float16` 是 fine-grained 控制方式。
- 经验法则(基于 NVIDIA arXiv 2004.09602 + APQ-ViT):**第 0/1 层 attention** 与 **patch_embed Conv** 占敏感层 60%。
- kSPARSE_WEIGHTS:仅在权重已 2:4 sparsified 时启用,可与 INT8 叠加 ~1.3× 进一步加速 —— 本项目 V1 不上。

### E6. multi-output 4 张量同时校准的工程坑(项目关键)

**核心结论(本调研最重要的工程结论)**:**TRT INT8 PTQ 校准的对象是 INPUT 张量,与 OUTPUT 数量无关**。校准过程统计的是网络中**所有 activation tensor 的分布**(每层之间的中间张量),所以无论输出 1 个还是 4 个,calibration cache 都覆盖整张网络。**不需要分别校准 4 个分支,联合校准一次即可**。

**详细发现**:
- IInt8Calibrator API 只要求 `get_batch(self, names)` 返回 input bindings(NVIDIA TensorRT Python API ref);output 不参与 cache。
- multi-INPUT 模型(如 NVIDIA TensorRT Issue #3823, 2024)需要在 `get_batch` 里同时返回多个 device pointer;**multi-OUTPUT 模型对 calibrator 无影响**。
- Calibration cache 文件存储 per-tensor scale/zero,与"哪个 output 被使用"无关 —— 整张图在 build 时统一量化。
- 但有 **2 个真实工程坑**:
  1. **多输出 ONNX 在 build 时,某些 output tensor 也会作为内部 activation 出现,scale 必须在 cache 中**。如校准数据集太小,"靠近输出"的中间 tensor histogram 可能采样不足 → 部分 layer "default quantization params"(NVIDIA Issue #3612, 2024 描述类似 cross-GPU cache reuse 问题)。**解决:校准 sample 数 ≥ 500,batch=1 或 ≥ batch_max 来增加 token 采样多样性**。
  2. 4 个 output 的下游使用方式不同(每个 output cosine similarity ≥ 0.99 验证),**单纯 top-1 accuracy 校准目标不足以保护中间层精度**。建议引入 layer-wise cosine similarity 作为 calibration 验证指标(non-standard but easy in polygraphy)。
- 没有专门的"multi-output INT8 PTQ official reference",但 Polygraphy convert + ONNX-Runtime parity 的标准 workflow 透明处理 multi-output。

**项目落地(关键 ADR)**:
- INT8 校准用 1 个 IInt8EntropyCalibrator2 实例,不区分 output;ImageNet val 子集 500-1000 张(分层抽样);batch=1 保最大 token 多样性。
- 验证标准:**4 个 output 的 cosine similarity ≥ 0.99 vs FP32**(已在 V1 计划)。如某 output cosine drop 显著(<0.95),说明该 output 路径上有敏感层 → 用 polygraphy debug precision 定位 + FP16 fallback。
- 校准代码模板:沿用 cyrusbehr/tensorrt-cpp-api 或 NVIDIA Polygraphy `examples/cli/convert/01_int8_calibration_in_tensorrt`。

---

## 主题 F · C++ 部署 & 跨语言数值一致性

### F1. TRT C++ runtime 现代封装最佳实践

**核心结论**:**RAII unique_ptr<IRuntime/IEngine/IExecutionContext, TrtDeleter>** + **每 inference 路径独立 cudaStream** + **绑定显存预分配**。**cudaGraph 与 dynamic shape 互斥** —— TRT 的 dynamic shape 引擎需要在 enqueue 前 setInputShape,而 cudaGraph capture 期间不能改 shape,**项目用静态 shape 才能上 cudaGraph**。

**详细发现**:
- cyrusbehr/tensorrt-cpp-api(~1.4k stars)是 TRT 10.x C++ 模板的 community 标杆:支持多输出多输入、INT8 PTQ、static + dynamic batch、float/__half/int8/int32/bool/uint8 输出 dtype 模板化、CUDA stream async + 显式 sync。
- LearnOpenCV "How To Run Inference Using TensorRT C++ API"(2024)给出 4-6× FP16 / 2-3× FP32 vs PyTorch 的 baseline 数字。
- cudaGraph + dynamic shape:NVIDIA 文档明确 "cuda graph requires static shapes"(TRT Best Practices Guide);多输入/输出可,但每个 shape 组合需独立 graph(profile)。
- ONNX Runtime CUDA EP/TRT EP:可通过 `OrtTensorRTProviderOptionsV2.user_compute_stream` 接 PyTorch 的 stream(onnxruntime.ai/docs/performance/device-tensor.html)— 跨框架 stream 共享标准方案。

**项目落地**:V1.0.0 C++ 封装类 `Dinov3TrtEngine`(RAII,持有 runtime/engine/context/stream,析构按相反顺序释放),benchmark 阶段两种模式:① dynamic batch (1-8) + 不带 cudaGraph;② static batch=1 + cudaGraph(可能小批快 ~5-15%)。

### F2. pybind11 + numpy + CUDA pointer 零拷贝惯用法

**核心结论**:**numpy.ndarray ↔ CUDA device pointer 不能直接零拷贝**(numpy 在 host),需经 ① `pybind11::buffer_protocol` 拿 host 指针 → ② `cudaMemcpyAsync` 上传到 device pre-allocated buffer → ③ enqueueV3 → ④ async download。**真正零拷贝**要用 PyTorch tensor 直接传 `data_ptr()`(`torch.cuda.Tensor.data_ptr()`)或 cupy/numba CUDA array(`__cuda_array_interface__`)。

**详细发现**:
- pybind11 `py::array_t<float>::request()` 给 host buffer 指针。
- CUDA stream pybind11 bridge:可用 `uintptr_t` 把 `cudaStream_t` 强转(PyTorch Forum cpp-extension-cudastream pybind11/31338, 2018,**仍有效**)。
- GIL 释放:**所有 CUDA Async 调用前后 `py::gil_scoped_release release`**,在 wait/sync 完后再 acquire,这是性能关键(否则 Python 主线程阻塞 GPU 工作)。
- 现代写法用 `py::capsule` 或 DLPack 对 PyTorch tensor 跨界(`torch.utils.dlpack.to_dlpack` ↔ `from_dlpack`)。

**项目落地**:Python 层把 input 转成 `torch.cuda.Tensor`,用 `tensor.data_ptr()` 拿 device pointer 传给 C++ engine.infer(ptr, stream),零拷贝。Output 端反向:engine.infer 返回 device pointer + shape,Python 通过 `torch.from_blob(... , dlpack=True)` wrap。

### F3. Python ↔ C++ 数值位级一致性破坏点

**核心结论**:**绝大多数差异来自 ① 默认 stream vs 显式 stream(默认 stream 隐含 sync 影响 cuBLAS workspace tactic 选择);② cuBLAS workspace 大小不同导致选不同 GEMM tactic(数值 ε 级差);③ NHWC vs NCHW 自动转换;④ atomicAdd 的非确定性求和顺序(MaxAbsErr 1e-3 级)**。

**详细发现**:
- TRT 在 build 时根据 workspace 选 fastest tactic;Python 默认 workspace 大,C++ 若设小可能选不同 tactic → MaxAbsErr 1e-4 ~ 1e-3 是常见。
- cuda-python(Python TRT runtime)与 C++ TRT 共享同一 libnvinfer.so,**只要 engine 文件一致、stream 一致、输入数值 bit-exact,输出 bit-exact 在大多数 GPU 上成立**(GDS / sparsity 路径除外)。
- atomicAdd:在 reduction-style ops(如 LayerNorm reduction、Softmax)中可能引入,但 TRT 主流 fp16/int8 kernel 用 warp-shuffle 无 atomic,**只有特殊 plugin 才会非确定**。
- TF32 默认开:`builderConfig.setFlag(BuilderFlag.DISABLE_TF32)` 才严格 FP32 比较;否则 FP32 baseline 自带 ~1e-4 噪声。

**项目落地**:跨语言 parity 测试要求 **MaxAbsErr ≤ 1e-5**,做这个等级需 **关 TF32**(`--noTF32` for trtexec / `BuilderFlag::kDISABLE_TF32` C++ / `disable_tf32=True` Python builder),否则 1e-5 不可达;V1.0.0 ADR 明确"FP32 parity 在 TF32 disabled 下进行"。

### F4. 跨语言 parity 测试工具

**核心结论**:**`polygraphy run --check`** 是最直接的跨 backend 验证工具。可一键对比 TRT vs ONNX-Runtime vs PyTorch 输出,支持 abs/rel tolerance、per-output diff。

**详细发现**:
- `polygraphy run model.onnx --trt --onnxrt --check 'tolerance=1e-5'` 会同时跑 TRT 与 ONNXRuntime,逐 output 比 abs/rel tolerance,fail 时 dump 差值最大位置。
- HuggingFace Optimum onnxruntime/optimum 的 `optimum-cli onnxruntime quantize` 内置 vs 原始 PyTorch model 对齐校验。
- ONNX Runtime 测试套件 `onnxruntime/test/python/onnxruntime_test_python_parity.py` 提供 reference patterns。

**项目落地**:CI 加 polygraphy run --check 步骤,作为每次 ONNX/TRT engine rebuild 的 gate。

### F5. Windows + Linux 数值差异

**核心结论**:**理论上无差异**(TRT engine 在 OS 间不可移植但行为一致);**实际可能的小差异**来自 ① MSVC vs GCC 浮点 IEEE 严格性默认值不同(MSVC 默认 /fp:precise,GCC 默认 -ffast-math 关);② CUDA driver 版本差(Windows 572.16 vs Linux 555.x)。

**详细发现**:
- TRT engine `.plan` 文件 platform-specific(NVIDIA Support Matrix:"Serialized engines are not portable across platforms"),但 build-time 算法相同 → 数值一致。
- MSVC C++ 编译器若加 `/fp:fast`,某些 host-side helper 数学误差放大;TRT inference 全在 device,host 影响极小。
- Windows + Linux 双平台 ImageNet val benchmark 通常 < 1e-6 abs diff(Lambda Labs 类博客经验值)。

**项目落地**:V1.0 不要求 Windows/Linux 二者均跑,**SSH 远程 Windows host 已是最常见配置**。Linux 副本仅作为 V1.1 stretch。

---

## 主题 G · 风险与替代方案

### G1. DINOv3 → DINOv2 ViT-L/14 降级评估

**核心结论**:降级成本 **可控但伴随科研价值损失**。架构差异:patch 14 vs 16 → token 数 257 vs 201;无 RoPE → 无 If 节点;register tokens 可选(DINOv2 with-register vs without)。性能差异:DINOv3 ViT-L 在 ADE20k 等 dense 任务比 DINOv2 ViT-L 高 ~2-5 mIoU。

**详细发现**:
- DINOv2 ViT-L/14:24 layers, 1024 dim, 16 heads, 257 tokens(1 cls + 256 patch,224×224,patch=14)。
- 已知 ONNX 导出可行的项目:sefaburakokcu/dinov2_onnx, lightly-train DINOv2 流水线。
- 降级触发条件(若发生):
  - DINOv3 RoPE If 在 TRT 10.16 仍无法 surgery 解决 → 降级
  - INT8 PTQ 在 DINOv3 上 cosine drop > 0.05 且找不到 sensitive layers → 降级
- 重测/重写工作量:~2 周(改导出脚本、re-calibrate、再 benchmark)。

**项目落地**:V1.0.0 ADR 加 fallback plan:"若 P3 DINOv3 ONNX → TRT 在 sm_120 上 build 失败连续 3 次,启动 DINOv2 ViT-L/14 备选,V1.0 改名为 V1.0.0-fallback"。

### G2. INT8 精度崩塌时混合精度策略

**核心结论**:推荐流程 **polygraphy debug precision 二分 → 找出 top-3 ~ top-5 敏感 layer → setLayerPrecision FP16 → 重 build**。混合后加速比损失 **5-15%**(取决于敏感层占总 latency 比例),但精度通常恢复 0.99+ cosine。

**详细发现**:
- TRT BuilderConfig flags:`BuilderFlag.PREFER_PRECISION_CONSTRAINTS / OBEY_PRECISION_CONSTRAINTS` 配合 `ILayer.setPrecision` 实现 per-layer 精度。
- NVIDIA TRT-Model-Optimizer `examples/onnx_ptq` 提供 sensitivity scan 脚本。
- ViT-L 经验值:patch_embed Conv + 第 0 层 attention + 最后 LayerNorm 共占总 latency ~10%,fallback FP16 后整网仍 ~85% INT8 加速。

### G3. 替代推理引擎对比

**核心结论(按 ViT 推理成熟度排序)**:

| 引擎 | ViT 成熟度 | 适合 backup? | 备注 |
|---|---|---|---|
| **onnxruntime CUDA EP** | 高 | ✅ 是 | 最简,稳定,~70% TRT FP16 性能,支持 multi-output 透明 |
| **onnxruntime TensorRT EP** | 高 | ✅ 是 | 内部用 TRT 但配置更简单,5-10% overhead |
| **Torch-TensorRT** | 中 | ❌ DINOv3 不兼容 | DINOv2/v3 forward 签名 `*args **kwargs` 与 torch.jit 不兼容(Issue #206) |
| **TensorRT-LLM** | LLM 强,ViT 弱 | ❌ | 0.17 起 PyTorch flow 实验性,ViT encoder 模式不稳 |
| **AITemplate** | Meta 已 archive | ❌ | 已不维护(AIT 项目 2024 后无更新) |
| **FasterTransformer** | NVIDIA 已 deprecate | ❌ | 被 TRT-LLM 取代 |

**项目落地**:**backup engine 用 onnxruntime CUDA EP**(用现成 ONNX 即可,无需重新 build);TRT EP 是中间路线。

### G4. 14 周内最被低估的工程时间黑洞

**核心结论(基于公开复盘 blog)**:按"被低估程度"排序:① **环境配置**(尤其 Blackwell 上 PyTorch nightly + CUDA 12.8 + cuDNN 9.x + TRT 10.13+ 协调,Windows host 下额外难)→ **2-3 周易吞掉**;② **ONNX 导出 debug**(DINOv3 RoPE If 节点、dynamic_axes、multi-output 验证)→ **1-2 周**;③ **INT8 调优**(校准集筛选、敏感层定位、cache 复用)→ **1 周**;④ **benchmark 自动化**(profile JSON 解析、矩阵跑全 + 重现性)→ **3-5 天**。

**详细发现**:
- Claudia Yao "Step-by-Step Guide to Setting Up TensorRT on RunPod"(Medium,2025)详细列举 CUDA/cuDNN/driver/TRT 4 项 minor version 不匹配导致的 5+ 类 cryptic error("factory function returned nullptr")。
- DEIMv2 README 两次单独标注"⚠️ TensorRT Version Notes" → 说明 DINOv3 部署确实易踩 TRT 版本坑。
- 经验估算(类似 ViT 规模 14 周项目):~30% 时间环境 + 导出,~30% 量化与调优,~20% 跨语言部署 + parity,~20% 报告 + ablation。

**项目落地**:第 1-2 周专门做"环境复现脚本"(Dockerfile + conda env.yaml + driver/CUDA/TRT 一键校验脚本),后续节省时间巨大。

---

## 📋 末尾汇总

### 🔴 「本调研对 V1.0.0 计划的反馈」(优先级排序)

1. **【关键修正】把"形状 [B, 197, 1024]"改为 [B, 201, 1024](默认含 4 register tokens)**,或显式 ADR 写"用 `get_intermediate_layers` 默认裁掉 register → 保持 197"。明确选择并固化在 export 脚本与 C++ binding 解析。
2. **【新增 ADR】"DINOv3 RoPE If 节点必须 surgery"**:导出脚本前置 step,直接修改源码 forward 而不是事后 onnx-graphsurgeon。
3. **【版本下限调整】TRT 最低版本从 10.8 改为 10.13(理想 10.16.1)**:基于 SM120 已知 regression 与 DINOv3 RoPE 兼容性。10.8 仅作为 baseline 对照测试用。
4. **【目标值微调】FP16 加速比 ≥1.8× 目标对 DINOv3 + DINOv2 attention 不自动融合的现状偏激进**,建议改为 **FP16 ≥ 1.5× (stretch 1.8×),INT8 ≥ 2.2× (stretch 2.5×)**,并在报告里明示"是否启用 FMHA 融合"是关键变量。
5. **【新增里程碑指标】"4 输出 layer-wise cosine similarity ≥ 0.99"**(已有);**追加 "p50 latency 标准差 ≤ 5%"**(防止 SM120 kernel 抖动);**追加 "polygraphy run --check pass"** 作为 CI gate。
6. **【新增风险条目】DINOv2 fallback plan**(若 P3 build 持续失败 → V1.0.0-fallback 切 DINOv2 ViT-L/14);**Windows+SM120 driver 兼容**(预留 Linux WSL2 备选)。
7. **【删除条目】不要把 "torch_tensorrt 可作为 backup 路线" 写进 ADR**,DINOv3 forward 签名与 torch.jit 不兼容(已知 Issue),无可救之路;backup 走 onnxruntime CUDA EP。
8. **【新增 V1.1 stretch goals】① ModelOpt FP8 PTQ;② SmoothQuant ViT-L PTQ;③ 4 层组合 ablation([4,12,16,20] vs [5,11,17,23] vs [3,11,19,23])**。
9. **【新增 P0 任务】第 1 周 "环境复现脚本"**(Dockerfile + Windows env + Linux WSL 双轨,driver/CUDA/cuDNN/TRT/PyTorch nightly 自动校验)。
10. **【数值一致性硬指标】MaxAbsErr ≤ 1e-5 必须在 TF32 关闭下测量**,ADR 显式声明 `BuilderFlag::kDISABLE_TF32`。

### 🟢 「Top 5 最值得借鉴的开源仓库」

| # | 名称 | URL | 为什么值得参考 | 用在哪个阶段 |
|---|---|---|---|---|
| 1 | **Intellindust-AI-Lab/DEIMv2** | github.com/Intellindust-AI-Lab/DEIMv2 | 唯一公开的 DINOv3 + TRT 完整工作流(export_onnx + trtexec --fp16 + 数值验证 + TRT 版本注解);明确标注 TRT 10.4 已知 FP16 bug,10.6+ 推荐 | **P2 ONNX 导出 + P3 TRT 引擎构建** |
| 2 | **NVIDIA/TensorRT-Model-Optimizer** | github.com/NVIDIA/TensorRT-Model-Optimizer | 唯一权威 ONNX PTQ(INT8/FP8/NVFP4)reference,Q/DQ 显式量化路线,multi-output ViT 透明支持 | **P4 INT8 PTQ + 校准** |
| 3 | **cyrusbehr/tensorrt-cpp-api** | github.com/cyrusbehr/tensorrt-cpp-api | TRT 10.x C++ RAII 封装模板,multi-output、static/dynamic batch、INT8 PTQ、模板化 dtype,代码质量 production-ready | **P5 C++ runtime 部署** |
| 4 | **lightly-ai/lightly-train** | github.com/lightly-ai/lightly-train | DINOv3 全任务 ONNX/TRT FP16 已商业化交付,T4 上有 latency benchmark 数字可作参考点 | **P6 benchmark 自动化** |
| 5 | **NVIDIA/TensorRT/tools/Polygraphy** | github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy | 跨语言 parity(`polygraphy run --check`)、INT8 calibration、precision-fallback、graph surgery 一站式工具,**贯穿 P2~P7** | **P2~P7 全程** |

### 🔶 「未解决问题清单」(需在 RTX 5080 实测)

1. **DINOv3 ViT-L/16 RoPE If 节点 surgery 在 TRT 10.13 / 10.16 + sm_120 上是否真能 build 成功** —— 公开 issue 都是 RTX 4090 / Jetson,**5080 上具体行为未知**。
2. **DINOv3 attention 在 TRT 10.16 IAttention API 下能否自动接管融合**(无需手写 plugin)—— 论文/issue 仅有 IAttention 说明,无 ViT 实测案例。
3. **RTX 5080 16 GB VRAM 是否限制 INT8 校准 batch=8 / 32 张同时 calibration** —— 16GB 跑 ViT-L FP32 batch=8 ≈ 8GB activation,INT8 build 时还要 FP32 baseline + INT8 candidate workspace,**接近但应能容**,需实测确认。
4. **4 output multi-binding 在 Blackwell SM120 上的 enqueueV3 调用开销**(每个 binding setup 约 50-100 µs)—— ViT-L 224 BS=1 总 latency 8-12 ms 时不重要,但 BS=1 + 静态 + cudaGraph 时占比可能 5-10%,需 Nsight Systems 测。
5. **Python `cuda-python` 与 C++ TRT runtime 跨语言 MaxAbsErr ≤ 1e-5** 在 Blackwell SM120 + RTX 5080 上是否真能达到(其他 GPU 上一般可,但 SM120 新架构下未知)。
6. **DINOv3 Per-tensor INT8 vs Per-channel(weight)对 4 个中间层 cosine similarity 的实际差** —— 文献只覆盖 final classifier output,中间 feature map 无定量数据。
7. **PyTorch nightly 2.12.0.dev+cu128 + ONNX export `dynamo=False` 在 DINOv3 ViT-L 上的稳定性**(Issue #170172 register_buffer + dynamic_axes hardcoded batch=1 bug 是否影响 RoPE freqs cache)—— 需在 5080 + Win10 实测。
8. **PolyU SSH 远程 Windows 主机在 cuDNN 9.x DLL 路径**(TRT 10.x release notes 提到 Windows DLL 从 lib/ 移到 bin/ 的 ABI 变化)对 build 流程的实际影响。

---

**报告说明**:本调研基于 2024-2026 年公开资料(arXiv、NVIDIA 官方文档、GitHub Issues 与 Releases、HuggingFace 模型卡、技术博客)。所有 "估算" / "推算" 数字均明确标注;DINOv3 ViT-L/16 在 RTX 5080 上的具体 latency/精度 **没有任何已发表的权威数据**,本项目的 benchmark 结果将填补此空白。引用格式中 GitHub stars 数与最近提交时间反映 2026-Q1 实际值,具体浮动以仓库当时为准。