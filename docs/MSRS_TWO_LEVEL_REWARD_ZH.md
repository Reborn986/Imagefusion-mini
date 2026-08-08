# MSRS-R1 Two-Level Reward 设计

本文固定第一版 RL reward 的命名、公式和实现边界。目标是保持 reward 简洁：只有 text-level
reward 和 image-level reward 两类。Format 只作为 gate，不作为第三类 reward。

## 1. Reward 总式

```text
R = R_gate * (lambda_t * R_text + lambda_i * R_image) * (0.5 + 0.5 * R_text)
```

默认权重：

```text
lambda_t = 0.4
lambda_i = 0.6
```

其中：

- `R_gate`：格式门控。输出图像数量不对、没有 fused image、关键标签缺失时为 0。
- `R_text`：文本级 reward，检查退化识别和简单融合规划。
- `R_image`：图像级 reward，用 Qwen3-VL-8B-Instruct 作为冻结视觉专家评价最终 fused image。
- `(0.5 + 0.5 * R_text)`：柔和调制项。文本完全错时降低 image reward 影响，但不硬截断。

## 2. Text-Level Reward

```text
R_text = 0.7 * R_label + 0.3 * R_plan
```

`R_label` 检查模型是否覆盖 expected IR/VIS degradation components。标签来自 manifest 或 degraded
path，只用于 RL reward，不用于推理。

`R_plan` 是简单融合规划完整性检查，不单独命名为 understanding reward。它只检查必要项：

```text
infrared preservation: infrared/thermal/target/silhouette/contour
visible preservation: visible/color/texture/detail/edge/structure/layout
artifact suppression: suppress/remove/reduce/artifact/noise/stripe/haze/rain/blur
fusion intent: fusion/fuse/combine/integrate/balance/complement
```

实现：

```text
imagefusion_r1/rl/msrs_two_level_reward.py
```

## 3. Image-Level Reward

Qwen3-VL-8B-Instruct 输入三张图：

```text
1. degraded infrared image
2. degraded visible image
3. generated fused image
```

同时提供 expected IR/VIS degradation labels，帮助 judge 明确应该抑制哪些退化。Qwen judge 不读取
policy 生成的 CoT，避免文字影响视觉判断。

Qwen 输出 JSON 分项：

```json
{
  "artifact_suppression": 0,
  "visible_preservation": 0,
  "infrared_preservation": 0,
  "fusion_naturalness": 0,
  "semantic_consistency": 0,
  "overall": 0,
  "reason": "one short sentence"
}
```

训练分数不用 `overall`，而用分项加权：

```text
R_image =
0.25 * R_artifact
+ 0.20 * R_visible
+ 0.20 * R_infrared
+ 0.20 * R_natural
+ 0.15 * R_semantic
```

所有分项从 0-10 归一化到 0-1。

实现：

```text
imagefusion_r1/rl/qwen3vl_reward_judge.py
```

## 4. 离线验证

先对已有 batch inference 输出跑 offline reward，确认完整模型与 fused-only 消融的 reward 分布合理，再接
online RL。

只跑 text/gate：

```bash
cd /home/zhuyiming/data/lsy/ImageFusion-R1-mgpt2-omni
python scripts/evaluate_msrs_two_level_reward.py \
  --eval_root outputs/your_eval_root \
  --skip_image_reward
```

启用 Qwen3-VL-8B image reward：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_msrs_two_level_reward.py \
  --eval_root outputs/your_eval_root \
  --qwen_model_path /home/zhuyiming/data/lsy/models/Qwen3-VL-8B-Instruct
```

输出：

```text
two_level_reward_details.jsonl
two_level_reward_summary.json
```

Qwen judge cache 默认写到：

```text
outputs/reward_cache/qwen3vl8b_msrs_reward_cache.jsonl
```

## 5. 模型下载位置

Qwen3-VL-8B-Instruct 固定下载到：

```text
/home/zhuyiming/data/lsy/models/Qwen3-VL-8B-Instruct
```

推荐下载方式：

```bash
source /home/zhuyiming/data/anaconda3/etc/profile.d/conda.sh
conda activate lumina-mgpt2
cd /home/zhuyiming/data/lsy/ImageFusion-R1-mgpt2-omni

python -m pip install -U huggingface_hub hf_transfer
python scripts/download_qwen3vl8b_reward_model.py
```

如果确认 `hf_transfer` 在当前环境可用，可加：

```bash
python scripts/download_qwen3vl8b_reward_model.py --hf_transfer
```
