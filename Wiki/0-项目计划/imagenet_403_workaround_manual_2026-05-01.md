# ImageNet 403 GatedRepoError Workaround Manual V1.0.0

> **目的**：项目唯一硬性外部 blocker — `ILSVRC/imagenet-1k` HF gated repo `403 GatedRepoError`。本手册给出 actionable 的 unblock 路径 + 第 48 轮远端网络可达性 audit 实证 + Kaggle CLI 已就位的 ready-to-use download 脚本。
>
> **状态**：脚本就位（13 单元测试 PASSED）+ 远端 Kaggle CLI 已 install + 待 user 配置 Kaggle API token 即可执行下载。
>
> **触发**：V1.0.1 §12.1 第 5 条"完整 ImageNet val 解锁"是项目 30+ 轮心跳唯一未闭合的硬性条款；R5（VRAM）+ R2（cos 阈值）已通过应急方案缓解，**只剩这一条 external blocker**。

---

## 1. 第 48 轮远端网络可达性 audit

`Test-NetConnection -Port 443` 实测（Windows + RTX 5080 + cpolar SSH 隧道）：

| 域名 | 可达性 | 说明 |
|---|:---:|---|
| `huggingface.co` | ❌ False | 项目原 ImageNet 403 根因；TCP connect to 199.59.149.210:443 直接失败 |
| `hf-mirror.com` | ⚠️ Reachable but gated | TCP 443 连通，metadata API 200 OK；但实际 LFS 下载链路返回 403（gated 沿袭上游） |
| **`kaggle.com`** | ✅ **True** | TCP 443 连通；可作为 ImageNet val 替代 distribution 源 |

**结论**：HF 整链路（含 mirror）对 ImageNet val 50K 不可用；Kaggle 是当前唯一 viable workaround。

## 2. 远端 Kaggle 环境就位

第 48 轮执行：

```bash
ssh windows-pc
cd /d D:\WorkPlace\ZMP\DINOv3-TRT-Acceleration\Code
.venv\Scripts\python.exe -m pip install kaggle
# ✅ Successfully installed kaggle-1.7.4.5 + 9 deps
```

实际执行 `from kaggle.api.kaggle_api_extended import KaggleApi` import 报：

```
OSError: Could not find kaggle.json. Make sure it's located in C:\Users\USER\.kaggle.
```

— 这是预期行为，CLI 已 install，**只缺 user-side API token**（一次性配置）。

## 3. User 配置步骤（一次性，~5 min）

### 3.1 获取 Kaggle API token

1. 注册 / 登录 https://www.kaggle.com（免费）
2. 浏览器右上 → "Account" → 滚动到 "API" → 点击 **"Create New Token"**
3. 浏览器自动下载 `kaggle.json`（约 100 字节，内含 username + API key）

### 3.2 上传到 Windows 远端

任选一种方式：

**方式 A · scp（macOS → Windows，本项目主路径）：**

```bash
# 在 macOS 本机
scp ~/Downloads/kaggle.json windows-pc:C:/Users/USER/.kaggle/kaggle.json
```

如目标目录不存在，先在远端创建：

```bash
ssh windows-pc 'powershell -Command "New-Item -ItemType Directory -Path C:\Users\USER\.kaggle -Force"'
```

**方式 B · 远程粘贴（cpolar 隧道下更快）：**

```bash
ssh windows-pc
mkdir C:\Users\USER\.kaggle
notepad C:\Users\USER\.kaggle\kaggle.json
# 粘贴 kaggle.json 内容并保存
```

### 3.3 接受目标 Kaggle dataset 的 license（一次性）

在浏览器登录后，访问目标数据集页面（默认 `titericz/imagenet1k-validation`）并点击 **"Late Submission" / "I Understand and Accept"** 同意 Kaggle 数据集 terms。

不同 mirror 选择：

| Kaggle 数据集 | 内容 | 大小 | 推荐度 |
|---|---|---|---|
| `titericz/imagenet1k-validation` | 50K val images | ~6.4 GB | ⭐⭐⭐⭐ |
| `lijiyu/imagenet` | full train + val | ~150 GB | 仅当需要 train |
| `karthik4321/imagenet-val-2012` | 50K val images（替代 mirror） | ~6 GB | 备选 |

## 4. 执行下载（user 配置完后）

### 4.1 验证 token 有效（dry-run，不实际下载）

```bash
ssh windows-pc
cd /d D:\WorkPlace\ZMP\DINOv3-TRT-Acceleration\Code
.venv\Scripts\python.exe scripts\download_imagenet_val_via_kaggle.py --output-dir Artifacts\datasets\imagenet_val_kaggle --dry-run
```

期望输出：
```
[auth] kaggle.json located at: C:\Users\USER\.kaggle\kaggle.json
[dry-run] would download dataset='titericz/imagenet1k-validation' to ...
```

### 4.2 实际下载（约 1-2 hour）

```bash
.venv\Scripts\python.exe scripts\download_imagenet_val_via_kaggle.py
```

脚本流程：
1. 验证 `kaggle.json` 可读 + Kaggle API authenticate
2. `kaggle datasets download -d titericz/imagenet1k-validation -p Artifacts/datasets/imagenet_val_kaggle`（~6.4 GB / Kaggle CDN ~50-100 MB/s = 1-2 min；端到端含 unzip 约 1 hour）
3. 自动 unzip 所有 .zip 文件
4. 写出 `manifest.json` 列出全部 50K JPEG 路径

### 4.3 切换到完整 ImageNet val 重跑 cosine eval

下载完成后，复用项目现有 manifest-based eval pipeline：

```bash
.venv\Scripts\python.exe scripts\select_imagenet_subset.py \
    --image-root Artifacts\datasets\imagenet_val_kaggle\imagenet1k-validation \
    --output Artifacts\manifests\imagenet_val_50k_eval_1000.json \
    --eval-count 1000 --calib-count 500 --seed 42

# 然后用现有 evaluate_engine_pair_on_images.py 重跑
.venv\Scripts\python.exe scripts\evaluate_engine_pair_on_images.py \
    --reference-engine Artifacts\engines\dinov3_vitl16_4out.fp32.engine \
    --candidate-engine Artifacts\engines\dinov3_vitl16_4out.bf16.prefer.engine \
    --manifest Artifacts\manifests\imagenet_val_50k_eval_1000.json \
    --batch-size 8 --image-size 224 \
    --output Artifacts\reports\eval_imagenet50k_fp32_vs_bf16_prefer.json
```

## 5. 等价性验证（推荐）

下载后 cross-verify Kaggle mirror 与 ILSVRC2012 原版的等价性：

```bash
# 检查 50K count
.venv\Scripts\python.exe -c "import json; d = json.load(open('Artifacts/manifests/imagenet_val_50k_eval_1000.json')); print(d['image_count'])"
# 期望: 1000 (eval subset) / 50000 (full)

# 抽样 SHA256 对比官方 manifest（如有）
.venv\Scripts\python.exe -c "
import hashlib
from pathlib import Path
p = Path('Artifacts/datasets/imagenet_val_kaggle/imagenet1k-validation/ILSVRC2012_val_00000001.JPEG')
print('SHA256:', hashlib.sha256(p.read_bytes()).hexdigest())
# 比对 https://image-net.org 官方 manifest 第一张
"
```

## 6. 影响范围（unblock 后预期变化）

完成 ImageNet val 50K 替换后，以下条款预期解锁：

| 项 | 之前 | 替换后 |
|---|---|---|
| §12.1 第 5 条"ImageNet val 解锁" | ⏳ 待外部 (HF 403) | ✅ 完整达成 |
| 论文 §6.1 Limitations "Imagenette proxy" | acknowledged limitation | 升级为"已用 ImageNet val 50K 验证主候选" |
| BF16 cos_min 数据 | Imagenette 1000 张（10 类） | ImageNet 1000 张（1000 类完整 distribution） |
| SmoothQuant α=0.8 cos_min 0.968 | Imagenette 数据点 | 可能轻微下移（更难数据集 worst-case 更差） |
| V1.3 QAT 启动门槛 1（数据集 unblock） | ❌ 未满足 | ✅ 满足 |

**重要 caveat**：ImageNet 完整 1000 类比 Imagenette 10 类更难，**SmoothQuant α=0.8 cos_min 在 ImageNet 上可能比 0.968 更低**（worst-case 样本更多）。这恰好暴露 R2 应急方案 cos_min ≥ 0.97 视角的 robustness — 真正交付时如选 R2 cos_mean 视角，则不受影响。

## 7. 质量门

- 脚本 `Code/scripts/download_imagenet_val_via_kaggle.py` (~190 行) 已 + 13 单元测试 PASSED。
- 本地 `pytest 336 passed, 3 skipped`、coverage 81% 保持、ruff/mypy 全绿。
- 远端 `kaggle` Python pkg 已 install（venv 内 1.7.4.5）。

## 8. 下一步行动

**项目方决策点**：

| 选项 | 工作量 | 收益 |
|---|---|---|
| **A. 立即配置 Kaggle token + 下载** | ~5 min user setup + 1-2 hour 下载 + 30-60 min 重跑 cosine eval | 解锁 §12.1 唯一硬性 blocker，达成 30+ 轮心跳后的最终 closure |
| **B. 推迟，保持 Imagenette 主路径** | 0 | 当前 V1.0.1 §12.1 已有 8 ✅ + 1 ⚠️ R2 应急 + 1 ⏳ 外部，按 R2 视角已可交付 |
| **C. 等论文 venue submission 时再触发** | 0 (本期) | 若发 paper，§6.1 limitations 可同步用 ImageNet val 数据强化 |

**推荐**：选项 A —— 一次 ~2 hour 投入（含 user setup + 下载 + 重跑）即可达成 V1.0.1 §12.1 9/9 完整闭合（除最终 review 外）。

## 9. 相关文档

- `Code/scripts/download_imagenet_val_via_kaggle.py`（download 脚本，13 单元测试 PASSED）
- `Code/tests/test_download_imagenet_val_via_kaggle_script.py`（mock-based 测试）
- `Wiki/0-项目计划/V1.3_QAT_launch_threshold_evaluation_2026-05-01.md` § 1（4 路径 + Kaggle 推荐）
- `Wiki/2-技术报告/复现与许可说明_V1.0.0.md`（项目复现总入口）
