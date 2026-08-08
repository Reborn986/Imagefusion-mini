# RealSR-R1 RL 代码解读与 MSRS 恢复融合迁移方案

本文记录本地 `RealSR-R1/` 的 RL 代码真实做了什么，以及迁移到当前
`ImageFusion-R1-mgpt2-omni` 任务时应如何改。当前任务不是单图超分，而是：

```text
输入：degraded infrared image + degraded visible image
输出：infrared CoT + clean infrared image
     visible CoT  + clean visible image
     fused CoT    + clean fused image
```

因此 RealSR-R1 只能作为 GRPO 训练骨架和 reward 设计参考，不能直接复制。

## 1. RealSR-R1 代码地图

### 1.1 README 的方法描述

`RealSR-R1/README.md` 把方法称为 VLCoT-GRPO，用于 real-world image
super-resolution。它在文字上定义四类 reward：

1. Format reward：规范 CoT 生成格式。
2. Degradation reward：奖励正确估计图像退化。
3. Understanding reward：奖励对图像内容理解准确。
4. Generation reward：用视觉专家/IQA 模型评价生成图质量。

注意：这是论文/README 层面的抽象。当前本地代码并没有把这四个 reward 都作为独立函数完整实现。

### 1.2 数据构建：`src/realsr-r1/create_json.py`

该脚本把单张低质图和 tag 文本做成 HuggingFace Dataset。

输入目录：

```text
RealSR-R1/data/demo/images_lq/
RealSR-R1/data/demo/tags/
```

每条样本包含：

```python
{
    "image_path": ".../images_lq/00001.png",
    "problem": "Perceive the degradation ... <|image|> ...",
    "tag": "<degradation>...</degradation> ... <final_image>...</final_image>"
}
```

`problem` 要求模型按以下格式生成：

```xml
<degradation>...</degradation>
<rough_understand>...</rough_understand>
<rough_image>...</rough_image>
<middle_understand>...</middle_understand>
<middle_image>...</middle_image>
<final_understand>...</final_understand>
<final_image>...</final_image>
```

这对应 RealSR 的“单图、粗到细、最终超分图”协议。它不包含多模态输入，也不包含三张目标图。

### 1.3 RL 入口：`src/realsr-r1/grpo.py`

这个文件负责：

- 解析 GRPO 参数；
- 定义 reward 函数；
- 把 dataset 样本映射成 conversation；
- 选择 `Qwen2VLGRPOTrainer` 或 `Qwen2VLGRPOVLLMTrainer`；
- 启动训练。

关键事实：

1. `accuracy_reward_iou` 和 `accuracy_reward_confidence` 是 bbox/数学验证风格的历史函数，和当前 RealSR demo 的恢复任务关系不大。
2. 真正注册进训练的 reward 是：

```python
script_args.reward_funcs = ["tag", "format"]
```

也就是 `tag_reward` 和 `format_reward`。

3. 代码里硬编码加载：

```python
dataset = DatasetDict.load_from_disk("test500_v2")
```

这会覆盖命令行传入的 dataset 路径，是 demo/实验代码写法，迁移时必须去掉。

4. `tag_reward` 当前提取 `<rough_understand>` 里的文本，然后检查样本 tag 中的关键词是否出现。它把
README 中的 Degradation/Understanding reward 合并成了一个很粗的文本关键词 reward。

5. 本地 demo 的 `create_json.py` 把 `tag` 存成字符串，但 `tag_reward` 里使用 `tag[i][0]`，
这更像旧版 `f.readlines()` 的 list 结构。也就是说本地 RealSR 代码存在数据形状历史遗留，不能按原样照抄。

### 1.4 自定义 GRPO Trainer：`src/realsr-r1/trainer/grpo_trainer.py`

这是最有参考价值的部分。它继承 HuggingFace `Trainer`，在 `compute_loss` 里手写 GRPO 流程。

主要流程：

1. 初始化 IQA 模型：

```python
TOPIQ, MUSIQ, MANIQA, CLIPIQA
```

2. 如果模型名包含 `GRPO`，用 `ChameleonForConditionalGeneration` 加载模型。

3. 创建 `FlexARInferenceSolver`：

```python
self.inference_solver = FlexARInferenceSolver(
    model_path=model,
    precision="bf16",
    target_size=512,
)
```

4. 在 `compute_loss` 里构造 prompt token。

5. 用当前 policy model 采样 `num_generations` 个 completion。

6. 用 inference solver 把 token decode 成：

```python
(generated_text, generated_images)
```

7. 保存每步样本的生成图和文本，方便肉眼检查。

8. 计算文本 reward：

```python
tag_reward
format_reward
```

9. 额外硬加图像 IQA reward：

```python
iqa_reward(completions, TOPIQ, MUSIQ, MANIQA, CLIPIQA)
```

其中 `iqa_reward` 只取 `generated_images[-1]`，即最后一张图，作为最终生成图质量。

10. 所有 reward 直接求和：

```python
rewards = rewards_per_func.sum(dim=1)
```

11. 按同一个 prompt 的 `num_generations` 个样本做 group mean/std，得到 advantage：

```python
advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
```

12. 用 policy-gradient 形式计算 loss：

```python
per_token_loss = -exp(logp - logp.detach()) * advantage
```

当前代码没有真正使用 reference model KL，`self.beta` 被设置但没有参与 loss。因此这份实现更像
“无 KL 的简化 GRPO”。迁移到我们项目时，建议加入 KL 或至少加入 SFT mixing/format gate，避免 RL 漂移太快。

### 1.5 RealSR inference solver：`PURE/pure/inference_solver.py`

这个 solver 负责：

- 把 conversation 和输入图编码成 prompt；
- 调用 `model.generate`；
- 解析图像 token；
- 用 VQ decoder 解码成 PIL image；
- 创建 multimodal logits processor。

本地代码里有多处硬编码：

```text
target_size = 512
eos_token_id = 8710
cfg 默认 3.0，但 trainer 调用时 cfg=0.8
text_top_k 在 trainer 调用时为 1
```

这些都不适合当前 MSRS mGPT2-Omni 项目。我们已在当前项目的
`imagefusion_r1/inference/inference_solver_mgpt2_ce.py` 中修过 stop token：

```text
Conversation.sep_token = <reserved08706>
```

迁移 RL 时必须继续使用这个 stop token，而不能回到 `8710`。

## 2. RealSR 四类 reward 在代码里的真实对应

| README 名称 | 代码中对应 | 当前实现强度 | 对我们是否可复用 |
|---|---|---|---|
| Format reward | `format_reward` regex | 较明确 | 可复用思想，格式需完全改成三阶段三图 |
| Degradation reward | 部分体现在 `tag_reward` | 很弱，不是独立函数 | 需重写为 IR/VIS 退化标签匹配 |
| Understanding reward | 部分体现在 `tag_reward` | 很弱，关键词匹配 | 需重写为跨模态理解/可靠性 reward |
| Generation reward | `iqa_reward` | 只评价最后一张图的无参考质量 | 可借鉴，但我们应加入 fused GT 全参考指标 |

一句话总结：RealSR 的代码提供了“GRPO 训练如何采样、decode、打 reward、算 advantage”的骨架；
reward 本身对我们而言几乎都要重写。

## 3. 为什么不能直接搬 RealSR-R1

当前项目至少有以下差异：

1. 输入数量不同：

```text
RealSR：1 张 LQ image
MSRS：1 张 degraded IR + 1 张 degraded VIS
```

2. 输出数量不同：

```text
RealSR：多段理解 + 最终 1 张 SR image
MSRS：3 段 CoT + clean IR + clean VIS + clean fused，共 3 张图
```

3. 图像尺寸不同：

```text
RealSR：512
MSRS：480 x 640
```

4. stop token 不同：

```text
RealSR demo：8710
MSRS mGPT2-Omni：Conversation.sep_token
```

当前本地 tokenizer 检查得到 `Conversation.sep_token` 为 `<reserved08706>`，id 为 `171384`。
后续代码不应硬编码该数值，而应运行时从 tokenizer 查询。

5. reward 目标不同：

```text
RealSR：最终图自然、清晰、像真实高分辨率图
MSRS：最终融合图既保留 VIS 结构/颜色，又保留 IR 热目标，同时 IR/VIS 恢复不能崩
```

6. RealSR 只有 no-reference IQA；我们有 clean IR、clean VIS、fused GT，因此应优先使用
full-reference reward，再把 no-reference IQA 作为辅助。

## 4. 当前项目的 RL 迁移目标

用户偏好已经明确：

- fused image 是主目标；
- clean IR / clean VIS 是辅助过程，但不能太差；
- fusion 中红外热目标容易被可见光淹没，因此 cross-modal reward 中 IR:VIS 建议为 `1.5:1`；
- 需要惩罚 fused image 偷懒复制 clean VIS 或 clean IR；
- 希望一次性实现完整、可调、可诊断的 reward，而不是只做低配版本。

因此 RL 目标应定义为：

```text
在保持三阶段输出格式正确的前提下，
提升最终 fused image 的融合质量，
同时约束 clean IR / clean VIS 过程输出不崩，
并显式奖励红外热目标保留、可见光结构保留，惩罚单模态复制。
```

## 5. 建议的 MSRS reward 设计

### 5.1 总体公式

建议把 format reward 做成 gate，而不是简单加和：

```text
R_total = G_format * (
    0.08 * R_degradation_text
  + 0.25 * R_aux_restore
  + 0.22 * R_cross_modal_reliability
  + 0.35 * R_fused_generation
  + 0.10 * R_visual_quality
  - 0.15 * P_lazy_copy
)
```

如果格式严重错误，例如三图不全、顺序错、tag 错位，则：

```text
G_format = 0
```

如果格式基本正确但有轻微文本问题，可以给 `0.3-0.8` 的 partial gate。所有子项都要单独记录日志，
不能只记录总 reward。

### 5.2 Format gate

检查生成文本和图像是否满足：

```xml
<infrared_cot>...</infrared_cot>
<clean_infrared_image><|image|></clean_infrared_image>
<visible_cot>...</visible_cot>
<clean_visible_image><|image|></clean_visible_image>
<fused_cot>...</fused_cot>
<clean_fused_image><|image|></clean_fused_image>
```

必须检查：

- 三段 CoT tag 是否完整；
- 三张 image block 是否完整；
- 顺序是否为 infrared -> visible -> fused；
- 是否多图、少图、提前 EOS；
- `generated_images` 是否能成功 decode 成三张 PIL image。

### 5.3 Degradation text reward

已在 `imagefusion_r1/rl/rewards_msrs.py` 中实现为
`score_msrs_degradation_text(...)` / `msrs_degradation_text_reward(...)`。

真实标签优先从样本 metadata 中读取；如果正式 6000 条 JSON 里没有显式
`infrared_label` / `visible_label` 字段，则从 degraded image path 解析：

```text
infrared_label
visible_label
infrared_level
visible_level
```

检查：

- `<infrared_degradation>` 是否覆盖 IR 退化成分；
- `<visible_degradation>` 是否覆盖 VIS 退化成分；
- compound label 如 `haze_rain` 是否同时提到 haze 和 rain；
- IR `noise_stripe_noise` 是否同时提到 noise 和 stripe/stripe noise。

默认只奖励正确退化成分覆盖，不因额外描述词扣分，因为 GT CoT 中经常会用
`blurred edges`、`streaks` 等词描述某种退化的视觉后果；额外命中的退化词会保留在
diagnostic 字段里，后续可在 cross-modal/text consistency reward 中再处理。

这个 reward 只负责“说对退化”。图像是否真的修好，由图像 reward 负责；外层 XML
错位也由 format reward 负责。

### 5.4 Auxiliary restoration reward

clean IR / clean VIS 是辅助，但不能太差。按用户偏好设置 IR:VIS = `1.5:1`：

```text
R_aux_restore = 0.6 * R_clean_ir + 0.4 * R_clean_vis
```

每个子项可由以下指标组合：

```text
R_clean_ir  = SSIM(pred_ir, gt_ir) + LPIPS inverse + PSNR normalized
R_clean_vis = SSIM(pred_vis, gt_vis) + LPIPS inverse + PSNR normalized
```

如果暂时没有 LPIPS，可先用 PSNR/SSIM/edge SSIM，但 reward 接口要预留 LPIPS/DISTS。

### 5.5 Cross-modal reliability reward

这是 MSRS 融合任务和 RealSR 最大的不同。它奖励 fused image 从可靠模态取可靠信息。

建议：

```text
R_cross_modal_reliability =
    0.6 * R_ir_saliency_preservation
  + 0.4 * R_vis_structure_preservation
```

对应 IR:VIS = `1.5:1`。

`R_ir_saliency_preservation`：

- 从 clean IR 或 degraded IR 中取热目标显著区域，例如 top percentile mask、局部对比度或梯度显著区域；
- 检查 fused image 在这些区域是否保留足够对比和亮度差；
- 防止 fused image 过度像 VIS，导致人/车热目标消失。

`R_vis_structure_preservation`：

- 从 clean VIS 或 visible GT 中提取边缘/梯度；
- 检查 fused image 是否保留道路、建筑、行人轮廓等结构；
- 防止 fused image 变成灰暗热图，丢失可见光空间结构。

### 5.6 Fusion generation reward

这是主 reward。因为 MSRS 有 `fused_gt_path`，应优先使用 full-reference 指标：

```text
R_fused_generation =
    a * SSIM(pred_fused, fused_gt)
  + b * LPIPS_inverse(pred_fused, fused_gt)
  + c * PSNR_norm(pred_fused, fused_gt)
  + d * fusion_metric_norm
```

可加入的融合指标：

```text
EN, SD, MI, VIF, Qabf, SCD
```

注意不要让 no-reference IQA 主导 reward，否则模型可能学成“好看的夜景图”，但红外信息不足。

### 5.7 Visual quality reward

对应 RealSR 的 Generation reward/IQA 部分。可使用：

```text
TOPIQ, MUSIQ, MANIQA, CLIPIQA
```

但它只应作为辅助项，主要解决：

- 输出发脏；
- VQ token 伪影；
- 颜色/纹理不自然；
- 指标高但肉眼差。

建议权重不超过 `0.10`，并且所有 IQA 值要单独记录，防止 reward 被某个无参考指标带偏。

### 5.8 Lazy-copy penalty

需要惩罚 fused image 偷懒复制单一模态，但不能误伤正常融合。建议使用条件 penalty：

```text
如果 fused 与 clean_vis_pred 过像，
且 IR saliency preservation 低，则惩罚。

如果 fused 与 clean_ir_pred 过像，
且 VIS structure/color preservation 低，则惩罚。
```

不要简单地“fused 像 VIS 就罚”，因为正常融合图本来会大量继承 VIS 的结构和颜色。

可实现为：

```text
P_lazy_copy =
    max(0, sim(fused, pred_vis) - tau_vis) * max(0, tau_ir_saliency - R_ir_saliency)
  + max(0, sim(fused, pred_ir)  - tau_ir)  * max(0, tau_vis_structure - R_vis_structure)
```

`sim` 可先用 SSIM/LPIPS inverse，后续替换为特征相似度。

## 6. 需要新建或改造的代码模块

建议不要在 RealSR-R1 目录中改，而是在当前项目内实现：

```text
imagefusion_r1/rl/
  rewards_msrs.py
  grpo_trainer_mgpt2_ce.py
  grpo_data.py
  image_metrics.py

scripts/
  build_msrs_rl_dataset.py
  train_msrs_grpo_mgpt2_ce.sh
  score_msrs_rewards_offline.py

configs/rl/
  msrs_grpo_reward.yaml
```

### 6.1 `grpo_data.py`

负责把 MSRS JSON/manifest 转成 RL dataset item：

```python
{
    "id": ...,
    "infrared_degraded_path": ...,
    "visible_degraded_path": ...,
    "infrared_clean_path": ...,
    "visible_clean_path": ...,
    "fused_gt_path": ...,
    "infrared_label": ...,
    "visible_label": ...,
    "infrared_level": ...,
    "visible_level": ...,
}
```

RL prompt 使用当前 SFT 的 human 模板，不喂 GT answer。

### 6.2 `grpo_trainer_mgpt2_ce.py`

参考 RealSR 的 `Qwen2VLGRPOTrainer`，但必须改：

- 使用当前 `ImageFusionMGPT2CESolver` 或拆出的 decode helper；
- 目标尺寸固定 `480x640`；
- stop token 使用 `Conversation.sep_token`；
- 支持三张输出图；
- `num_generations` 个 completion 都要保存文本和三张图；
- reward 函数返回完整字典，而不只是标量；
- 加入 KL 或 SFT CE mixing，防止无 KL RL 漂移。

### 6.3 `rewards_msrs.py`

实现：

```python
format_reward(...)
degradation_text_reward(...)
aux_restore_reward(...)
cross_modal_reliability_reward(...)
fused_generation_reward(...)
visual_quality_reward(...)
lazy_copy_penalty(...)
combine_rewards(...)
```

当前已实现：

- `msrs_format_reward(...)`
- `msrs_degradation_text_reward(...)`

每个函数都返回：

```python
{
    "score": float,
    "submetrics": {...},
    "ok": bool,
    "reason": str,
}
```

训练日志必须保存每个子项，便于诊断 reward hacking。

### 6.4 `score_msrs_rewards_offline.py`

先对 SFT batch inference 输出离线打 reward。这个脚本非常重要，因为它能在不训练的情况下检查：

- reward 是否和肉眼质量一致；
- lazy-copy penalty 是否误伤；
- IR:VIS = `1.5:1` 是否过强；
- format gate 是否太严格；
- fused GT 是否偏可见光导致红外被压。

这不是“低配版本”，而是完整 reward 上线前的校准步骤。

## 7. 推荐实施顺序

1. 完成 SFT `epoch0-epoch4` 的 test8 及更大 balanced batch 评估。
2. 选定 RL 初始化 checkpoint。
3. 实现 `rewards_msrs.py` 和 offline reward scorer。
4. 用已生成的 SFT outputs 校准 reward 权重，至少抽查：
   - 热目标是否消失；
   - 融合图是否复制 VIS；
   - 融合图是否复制 IR；
   - 图像是否发灰/发绿/横纹；
   - 文本标签是否正确。
5. 实现 GRPO trainer。
6. 先做 `num_generations=2`、`max_samples=4` 的训练 smoke，检查是否能生成、decode、反传、保存。
7. 再扩大到 100-500 条 balanced RL subset。
8. 最后跑正式 RL。

## 8. 当前决策记录

已确定：

- fused image 是主目标；
- clean IR / clean VIS 是辅助目标，但必须防止中间图太差；
- cross-modal reward 中 IR:VIS = `1.5:1`，对应权重 `0.6:0.4`；
- 需要 lazy-copy penalty；
- fused GT 应作为主参考之一；
- no-reference IQA 可用，但不能支配 reward；
- 所有 reward 都要可配置、可日志化、可离线校准。

仍需进一步确认：

1. fused GT 是否足够可信，是否偏向 visible；
2. fused image 是否要求自然 RGB 色彩，还是允许更强红外热目标表现；
3. lazy-copy penalty 的阈值如何通过 SFT batch 输出校准；
4. RL dataset 是否从 6000 train item 中抽取，还是先补齐完整 21660 后按 scene-level 重新划分；
5. 是否在 GRPO loss 中加入 reference KL，或加入少量 SFT CE mixing。

## 9. 和 RealSR-R1 的一一对应改造表

| RealSR-R1 模块 | 当前作用 | MSRS 迁移方案 |
|---|---|---|
| `create_json.py` | 单图 LQ + tag -> HF Dataset | 改为 IR/VIS degraded paths + 三个 GT paths + labels |
| `problem` prompt | 单图超分粗到细格式 | 使用当前 MSRS human prompt 和三阶段输出格式 |
| `format_reward` | 检查 RealSR 7 个 tag | 检查 MSRS 三段 CoT + 三张 image tag |
| `tag_reward` | 粗略检查关键词 | 拆成 degradation text reward 和 cross-modal text reward |
| `iqa_reward` | 评价最后一张图 | 评价 fused 图，同时保留 IR/VIS 辅助图指标 |
| `FlexARInferenceSolver` | 512 单图 decode | 使用当前 mGPT2-Omni 480x640 inference/decode |
| `eos=8710` | RealSR/PURE 硬编码 | 改为 `Conversation.sep_token` |
| `generated_images[-1]` | 最终 SR 图 | 解析为 IR/VIS/Fused 三张图 |
| reward sum | 直接相加 | format gate + weighted reward + lazy-copy penalty |
| GRPO loss | 无 KL 简化版 | 建议加入 reference KL 或 SFT CE mixing |
