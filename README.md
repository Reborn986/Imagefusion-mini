# ImageFusion-R1 minimal handoff

这是从实验工作区抽出的可交付版本。代码仓库只保留四条主链，不包含模型、数据、历史输出或密钥：

1. Qwen3-VL 生成并校验三阶段 CoT；
2. mGPT2-Omni CE-SFT 基础训练；
3. 一个统一的 full-COT 三图 inference；
4. MSRS text + image reward 的 GRPO/RL。

## 目录

```text
imagefusion_r1/preprocess/   CoT 清洗与 SFT 预编码
imagefusion_r1/trainers/    唯一保留的 CE-SFT trainer
imagefusion_r1/inference/   唯一保留的 mGPT2-Omni inference solver
imagefusion_r1/rl/          GRPO、两路 reward、Qwen judge
scripts/                    四条链路的少量入口
configs/                    正式 6000 条 SFT 配置
third_party/lumina_mgpt_2/  必需的上游源码；不含模型权重
tests/                      RL/reward/logits processor 单元测试
```

## 私有交付边界

- GitHub 私有仓库：只推本目录源码。
- 私有模型仓库或对象存储：Lumina 基座、SFT checkpoint、Qwen reward/COT 模型、MoVQGAN 权重。
- 私有对象存储或 SSH：数据 bundle。
- 不分享自己的 SSH 私钥或个人 token。接收者使用她自己的 GitHub SSH 公钥和自己的细粒度只读 token。

建议让接收者从原始发布方下载 Lumina、Qwen 和 MoVQGAN，只私下分发本项目 SFT/RL 权重。上传前先核对各模型和 MSRS 数据的再分发许可。GitHub 大文件说明、Hugging Face 私有仓库与 token 文档见：

- https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github
- https://huggingface.co/docs/hub/repositories-settings
- https://huggingface.co/docs/hub/security-tokens

## 目标机准备

需要 Linux、NVIDIA 驱动以及与 PyTorch 匹配的 CUDA。建议建两个环境，避免 mGPT2 与 vLLM/新 transformers 冲突：

```bash
python -m venv .venv-main
source .venv-main/bin/activate
# 先按 pytorch.org 为目标 CUDA 安装 torch/torchvision
pip install -r requirements-main.txt

python -m venv .venv-qwen
source .venv-qwen/bin/activate
# 先安装与 vLLM 匹配的 torch
pip install -r requirements-qwen.txt
```

复制 `.env.example` 为 `.env.local`，只在本机填写模型路径：

```bash
cp .env.example .env.local
set -a
source .env.local
set +a
```

不要在 `.env.local` 中保存 HF/GitHub 长期 token；优先使用 `hf auth login` 的本机凭据或短期只读凭据。

## 模型布局

```text
models/
  Lumina-mGPT-2.0-Omni/       # 约 15 GB
  imagefusion-sft-epoch4/     # fresh RL 只需模型权重，约 15 GB
  Qwen3-VL-8B-Instruct/       # RL reward，约 17 GB
  Qwen3-VL-32B-Thinking/      # 仅生成 CoT 时需要
third_party/lumina_mgpt_2/lumina_mgpt/movqgan/270M/
  movqgan_270M.ckpt            # 约 1 GB，不进 Git
```

从原工作区导出 fresh-RL 所需 SFT 权重，不复制 60 GB optimizer：

```bash
bash scripts/export_model_checkpoint.sh \
  /path/to/sft/epoch4 /path/to/private-upload/imagefusion-sft-epoch4 init
```

只有续跑同一 RL checkpoint 时才使用第三个参数 `resume`；这时 optimizer 和 trainer state 必须一起带上，并保持相同 world size。

## 1. CoT 生成

输入 manifest 每条需包含 degraded/clean IR、degraded/clean VIS 和 fused GT 路径。使用 Qwen 环境：

```bash
MODEL_PATH="$QWEN_COT_MODEL" \
MANIFEST=data/manifests/cot_input.json \
OUTPUT_JSON=outputs/cot/generated.json \
PYTHON_BIN=.venv-qwen/bin/python \
CUDA_VISIBLE_DEVICES=0,1 TP_SIZE=2 \
bash scripts/run_msrs_qwen3vl_cot_worker.sh
```

随后构建最终训练项并预编码：

```bash
bash scripts/build_msrs_final_6000_training_items.sh
bash scripts/pretokenize_msrs_final_6000_480x640.sh
```

## 2. CE-SFT

使用 main 环境：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 \
PRETRAINED="$BASE_MODEL" \
OUTPUT_DIR=outputs/sft \
bash scripts/train_mgpt2_ce_cot_final_6000_5epoch.sh
```

## 3. 唯一 inference

```bash
python scripts/batch_infer_msrs_mgpt2_ce.py \
  --model_path "$INIT_CKPT" \
  --manifest data/manifests/eval.json \
  --save_root outputs/inference \
  --max_samples 1 \
  --output_protocol auto
```

## 4. RL/GRPO

先执行 9 条 preflight，不要直接开长任务：

```bash
RL_TIER=preflight \
INIT_CKPT="$INIT_CKPT" \
QWEN_REWARD_MODEL="$QWEN_REWARD_MODEL" \
TRAIN_MANIFEST=data/manifests/msrs_level2_rl_preflight_9.json \
bash scripts/train_msrs_level2_rl_small_epoch4.sh
```

preflight 通过后再将 `RL_TIER` 改为 `pilot100`。训练日志包含 `[ENTRY]`、`[DIST]`、`[PHASE]`、`[HEARTBEAT]`、`[GEN]`、`[REWARD]`、`[GROUP]` 和 optimizer 状态，长时间停在哪个阶段可以直接看出来。

## 数据打包

不要复制全部 18 GB 数据来跑 9/100 条实验。按 manifest 收集所需文件并重写为仓库相对路径：

```bash
python scripts/package_manifest_data.py \
  --input-manifest /path/to/original_manifest.json \
  --output-dir data/bundles/pilot100
```

将整个 `data/bundles/pilot100/` 私下传输并解压到接收者仓库同一路径。打包后先在新机器运行 `--max_samples 1` inference 或 RL preflight，确认路径与环境。

## 推 GitHub 前

```bash
git init
git status --short
git add .
git status --short
git commit -m "Initial minimal ImageFusion-R1 handoff"
git branch -M main
git remote add origin git@github.com:YOUR_ORG/YOUR_PRIVATE_REPO.git
git push -u origin main
```

创建远端仓库时必须选 Private。`git add` 后若看到 `models/`、`data/`、`outputs/`、`.env.local`、权重或 token，立即停止提交。
