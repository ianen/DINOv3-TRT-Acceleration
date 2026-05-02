# ImageNet 403 GatedRepoError Workaround Manual V1.0.1

> **目的**：项目唯一硬性外部 blocker — `ILSVRC/imagenet-1k` HF gated repo `403 GatedRepoError`。本手册给出 actionable 的 unblock 路径 + 第 48 轮远端网络可达性 audit 实证 + Kaggle CLI 已就位的 ready-to-use download 脚本。
>
> **状态**：脚本就位（**15 单元测试 PASSED**，已兼容新 KGAT 与 legacy kaggle.json 双格式）+ 远端 Kaggle CLI 已 install + 待 user 配置 Kaggle API token 即可执行下载。
>
> **触发**：V1.0.1 §12.1 第 5 条"完整 ImageNet val 解锁"是项目 30+ 轮心跳唯一未闭合的硬性条款；R5（VRAM）+ R2（cos 阈值）已通过应急方案缓解，**只剩这一条 external blocker**。
>
> **V1.0.1 修订记录（2026-05-01 第 50 轮）**：
> 1. Kaggle 在 2025/2026 升级 API token 体系，UI 不再下载 `kaggle.json`，改为单字符串 `KGAT_*` token + `~/.kaggle/access_token` 文件。脚本与本手册同步兼容两种格式。
> 2. 数据集 slug 修正：`titericz/imagenet1k-val` → **`titericz/imagenet1k-val`**（404 fix）。

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

注意：旧版 `kaggle 1.7.4.5` 可能仅识别 `kaggle.json`；若使用新 KGAT 格式而 auth 失败，按 §3.4 升级 pkg。

## 3. User 配置步骤（一次性，~5 min）

### 3.1 获取 Kaggle API token（**新版 UI**）

1. 注册 / 登录 https://www.kaggle.com（免费）
2. 头像 → **Settings** → 滚动到 **"API"** → 点击 **"Create New Token"**
3. 弹窗显示 `KGAT_*` 单字符串 + 配置命令：

   ```bash
   mkdir -p ~/.kaggle && echo KGAT_xxxxxxxx > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
   ```

   旧版 UI 仍可能直接下载 `kaggle.json`；脚本已兼容两种格式。

### 3.2 上传到 Windows 远端

任选一种方式：

**方式 A · scp（macOS → Windows，本项目主路径）：**

```bash
# 在 macOS 本机
ssh windows-pc 'powershell -Command "New-Item -ItemType Directory -Path C:\Users\USER\.kaggle -Force"'

# 新格式
scp ~/.kaggle/access_token windows-pc:C:/Users/USER/.kaggle/access_token

# 或旧格式
scp ~/Downloads/kaggle.json windows-pc:C:/Users/USER/.kaggle/kaggle.json
```

**方式 B · 远程直接配置（cpolar 隧道下更快）：**

```bash
ssh windows-pc
mkdir C:\Users\USER\.kaggle
notepad C:\Users\USER\.kaggle\access_token
# 粘贴 KGAT_* 字符串并保存（不要带换行）
```

### 3.3 接受目标 Kaggle dataset 的 license（一次性）

在浏览器登录后，访问目标数据集页面并点击 **"I Understand and Accept"** 同意 Kaggle 数据集 terms。

| Kaggle 数据集 | 内容 | 大小 | 推荐度 |
|---|---|---|---|
| **`titericz/imagenet1k-val`** | **50K val images（默认）** | ~6.4 GB | **⭐⭐⭐⭐⭐** |
| `tusonggao/imagenet-validation-dataset` | 50K val images（备选 mirror） | ~6 GB | ⭐⭐⭐ |
| `lijiyu/imagenet` | full train + val | ~150 GB | 仅当需要 train |

> ⚠️ **过期 slug**：`titericz/imagenet1k-val`（带 `-validation` 后缀）现 404。正确 slug 已更正为 `titericz/imagenet1k-val`。

### 3.4 验证远端 kaggle pkg 支持新 KGAT 格式

```bash
ssh windows-pc 'cd /d D:\WorkPlace\ZMP\DINOv3-TRT-Acceleration\Code && .venv\Scripts\python.exe -c "import kaggle; print(kaggle.__version__)"'
```

新 KGAT/access_token 至少需要 `kaggle >= 1.7.4` 系列；若过旧报错则升级：

```bash
ssh windows-pc 'cd /d D:\WorkPlace\ZMP\DINOv3-TRT-Acceleration\Code && .venv\Scripts\python.exe -m pip install -U kaggle'
```

## 4. 执行下载（user 配置完后）

### 4.1 验证 token 有效（dry-run，不实际下载）

```bash
ssh windows-pc
cd /d D:\WorkPlace\ZMP\DINOv3-TRT-Acceleration\Code
.venv\Scripts\python.exe scripts\download_imagenet_val_via_kaggle.py --output-dir Artifacts\datasets\imagenet_val_kaggle --dry-run
```

期望输出（新格式）：
```
[auth] credential file located at: C:\Users\USER\.kaggle\access_token
[dry-run] would download dataset='titericz/imagenet1k-val' to ...
```

旧格式同样 OK（脚本自动兼容）：
```
[auth] credential file located at: C:\Users\USER\.kaggle\kaggle.json
```

### 4.2 实际下载（约 1-2 hour）

```bash
.venv\Scripts\python.exe scripts\download_imagenet_val_via_kaggle.py
```

脚本流程：
1. 验证 `kaggle.json` 可读 + Kaggle API authenticate
2. `kaggle datasets download -d titericz/imagenet1k-val -p Artifacts/datasets/imagenet_val_kaggle`（~6.4 GB / Kaggle CDN ~50-100 MB/s = 1-2 min；端到端含 unzip 约 1 hour）
3. 自动 unzip 所有 .zip 文件
4. 写出 `manifest.json` 列出全部 50K JPEG 路径

### 4.3 切换到完整 ImageNet val 重跑 cosine eval

**推荐路径（一键 orchestrator）**：

```bash
ssh windows-pc 'cd /d D:\WorkPlace\ZMP\DINOv3-TRT-Acceleration\Code && .venv\Scripts\python.exe scripts\run_imagenet_val_post_download.py'
```

`scripts/run_imagenet_val_post_download.py`（V1.0.1 第 52 轮新增，18 单元测试 PASSED）自动完成：

1. 读取 `Artifacts/datasets/imagenet_val_kagglehub/download.success` marker 解析 dataset 路径
2. 用现有 `prepare_image_subset_manifests.py` 生成 1000 eval / 500 calib 互斥 manifest
3. 跑两个候选 cosine eval：
   - **BF16 prefer**（V1.0.1 主交付候选，预期 cos_min ≥ 0.998）
   - **INT8 SmoothQuant α=0.8**（R2 应急候选，验证 cos_min ≥ 0.97 在完整 ImageNet 是否仍成立）
4. 写出 unified summary `Artifacts/reports/imagenet50k_post_download_summary.json` 含每候选 R1/R2 verdict

CLI 选项：`--eval-count`（默认 1000）/ `--calib-count`（500）/ `--seed`（42）/ `--batch-size`（8）/ `--image-size`（224）/ `--skip-pair {bf16_prefer,int8_smoothquant_a080}` / `--dry-run`（验证管线不触发实际 eval）。

**手动路径（原有逐步命令保留）**：

```bash
.venv\Scripts\python.exe scripts\prepare_image_subset_manifests.py \
    --image-root <kagglehub_extracted_path> \
    --eval-output Artifacts\manifests\imagenet_val_50k_eval_1000.json \
    --calib-output Artifacts\manifests\imagenet_val_50k_calib_1000.json \
    --eval-count 1000 --calib-count 500 --seed 42

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

- 脚本 `Code/scripts/download_imagenet_val_via_kaggle.py`（~210 行）+ **15 单元测试 PASSED**（新增 2 测：`test_find_kaggle_credentials_finds_new_access_token_format`、`test_find_kaggle_credentials_prefers_access_token_over_legacy`）。
- 脚本 `Code/scripts/run_imagenet_val_post_download.py`（~330 行 orchestrator）+ **18 单元测试 PASSED**（覆盖 image_root 解析、manifest 生成、cosine summary 解析、R1/R2 verdict 分级、dry-run 模式）。
- 本地 `pytest 356 passed, 3 skipped`、coverage 81% 保持、ruff/mypy 全绿（含 `mypy --strict`）。
- 远端 `kaggle` Python pkg 已 install（venv 内 1.7.4.5）+ `kagglehub 1.0.1` 已 install（用户建议路径，第 50 轮已验证 KGAT auth 工作）；如旧 kaggle pkg 新 KGAT 格式 auth 失败则按 §3.4 升级。
- 远端 `_check_kagglehub_progress.ps1` 一键查询 STATUS / size / 吞吐 / ETA / log_tail / success/failed marker，便于 unattended 心跳监控。
- 远端下载进程通过 **WMI Win32_Process.Create** 完全脱离 ssh（survives ssh disconnect / mac shutdown / session 结束）。

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
