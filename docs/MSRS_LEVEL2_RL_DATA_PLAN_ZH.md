# MSRS Level2 RL 数据选择与 GRPO 训练计划

本文用于规划 MSRS 恢复融合任务的第一版 RL 数据选择。当前目标不是重新做 SFT，而是在已有
CE-SFT checkpoint 上做 GRPO：输入退化红外和退化可见光，输出三段 CoT、clean IR、clean VIS
和 clean fused image。数据选择重点放在 level2 的中等偏难退化组合。

## 1. 结论先行

推荐第一版正式 main RL 数据量为 1500 条 level2 prompt，但不直接从 1500 起跑。
实际执行分为 smoke、pilot、main 三阶段：先用 120-200 条检查训练链路，再用 600 条判断
reward 是否有效，最后冻结 1500 条作为第一版可汇报主实验。

原因：

- RealSR-R1 的本地 README 示例使用 `num_generations=5`、`per_device_train_batch_size=1`、
  `max_prompt_length=5000`、`max_completion_length=7000`、6 张 GPU、2 epoch。
- MSRS 每个 prompt 需要两张输入图并生成三张 480x640 图，单条 completion 长很多，不能直接照搬
  RealSR 的 group size 和总吞吐。
- 当前 MSRS 全量 level2 训练子集是 10 个组合 × 300 条 = 3000 条。直接全量 RL 成本较高，且
  reward server 需要逐个 fused image 调 Qwen3-VL 打分。
- 1500 条是 pilot 通过后的 main set。它配合 `num_generations=4` 时，一轮约 6000 个 sampled completions；配合
  `num_generations=2` 时，一轮约 3000 个 completions。这个量级既接近 RealSR 的千级/数千级
  rollout 思路，也足够作为第一版正式 RL 主实验。

建议分三步跑：

| 阶段 | 数据量 | 用途 | 推荐设置 |
|---|---:|---|---|
| smoke | 120-200 | 检查生成、reward、显存、日志 | `NUM_GENERATIONS=2`, `RL_EPOCHS=1` |
| pilot | 600 | 看 reward 分布和是否有明显退化 | `NUM_GENERATIONS=2`, `GRAD_ACCUM_STEPS=2` |
| main | 1500 | pilot 通过后的第一版可汇报 RL 主实验 | 优先 `NUM_GENERATIONS=4`；不稳定则 `NUM_GENERATIONS=2` |
| expand | 3000 | 若 1500 有收益，再覆盖全部 level2 | 10 组合各 300 |

## 2. 现有证据

### 2.1 RealSR-R1 怎么做

本地 `RealSR-R1/README.md` 的 GRPO 启动例子使用：

| 项 | RealSR-R1 示例 |
|---|---:|
| GPU 数 | 6 |
| per-device batch | 1 |
| grad accumulation | 1 |
| num generations | 5 |
| max prompt length | 5000 |
| max completion length | 7000 |
| epochs | 2 |
| learning rate | 2e-6 |
| max pixels | 401408 |

但是 RealSR 是单图超分，输出最终 1 张 SR 图；MSRS 是双输入、三阶段、三张目标图，所以这里只能参考
GRPO 形态：同一个 prompt 采样多个 completion，组内 reward 标准化为 advantage，再更新 policy。

### 2.2 当前 MSRS GRPO 默认

`scripts/train_msrs_two_level_rl.sh` 和 `imagefusion_r1/rl/grpo_trainer_mgpt2.py` 当前默认：

| 项 | 当前 MSRS 默认 |
|---|---:|
| policy GPU | 3 张 |
| reward GPU | 1 张 |
| per-device prompt batch | 1 |
| global prompt batch | `NPROC * 1`，默认 3 |
| num generations | 2 |
| reward batch size | 1 |
| max position embeddings | 28672 |
| max new tokens | 24576，代码会按剩余上下文自动截到可用预算 |
| target size | 480x640 |
| learning rate | 5e-7 |
| KL | 默认 `NO_REFERENCE_KL=1`, `KL_BETA=0` |

这里 batch 要拆成两个概念：

- prompt batch：每次并行多少个不同样本。当前 3 张 policy GPU、每卡 1，就是每个 micro-step 3 个
  prompt。
- GRPO group size：每个 prompt 采样多少个 completion。RealSR 用 5；当前 MSRS 默认 2。

实际每个 micro-step 的 completion 数：

```text
global_completions = NPROC * PER_DEVICE_BATCH_SIZE * NUM_GENERATIONS
```

默认是 `3 * 1 * 2 = 6`。如果 `NUM_GENERATIONS=4`，就是 12 个 completion。

### 2.3 MSRS prompt/completion size

从已通过检查的 480x640 PKL 看：

| 协议 | 总序列长度 | supervised labels | 推断 prompt 长度 |
|---|---:|---:|---:|
| full CoT + 三图 | 25326-25909 | 15539-16122 | 约 9786-9787 |
| fused-only CoT + 三图 | 24711-24973 | 14924-15186 | 约 9787 |
| no-CoT GT 三图 | 24413 | 14627 | 9786 |

所以 MSRS 的 prompt 约 9.8k tokens，completion 约 14.6k-16.1k tokens。RealSR 的
`max_prompt_length=5000`、`max_completion_length=7000` 对 MSRS 不够用。MSRS 应继续使用
`max_position_embeddings=28672`，`MAX_NEW_TOKENS` 建议保留默认或显式设为 `18000` 以上。

## 3. 退化组合选择

### 3.1 已有 eval 暴露的弱项

当前已有 level1 full eval 样本数较少，每个组合只有 2 个测试场景，因此只能作为弱项线索，不能直接
作为最终结论。它给出的信号是：

| level1 组合 | 弱项迹象 |
|---|---|
| `stripe_noise + haze` | 可见光恢复明显最差，2 个样本 mean VIS PSNR 约 19.29，mean VIS SSIM 约 0.66 |
| `noise + rain` | IR/VIS 都偏低，雨线和噪声叠加后恢复更不稳 |
| `noise + noise` | 可见光 PSNR 偏低，噪声类退化较难彻底清掉 |
| `noise + haze` | 个别场景 IR 指标掉得明显，说明天气退化会放大跨模态不稳 |

因此 level2 里应优先覆盖 haze、rain、noise 这些组合，同时保留 blur4 作为强模糊项。

### 3.2 level2 候选分层

level2 的 IR 固定为 `noise+stripe_noise`。VIS 从五种基础退化中取两个。

| 优先级 | VIS level2 组合 | 选择理由 |
|---|---|---|
| A | `haze+noise` | haze 在已有 eval 中有明显弱项，叠加 noise 后更难恢复纹理和对比度 |
| A | `haze+rain` | 两种天气/散射类退化叠加，容易残留雾感、雨线和低对比 |
| A | `noise+rain` | 对应 level1 的弱项，噪声和雨线都容易残留 |
| B | `blur4+haze` | 强模糊叠加雾，边缘和远处结构恢复难 |
| B | `blur4+noise` | 强模糊叠加噪声，细节和纹理容易被过平滑 |
| B | `blur4+rain` | 强模糊叠加雨线，结构边缘和雨痕分离难 |
| C | `blur2+noise` | 中等难度 anchor，包含噪声但不至于过难 |
| C | `blur2+rain` | 中等难度 anchor，雨线可测试 artifact suppression |
| C | `blur2+haze` | 中等难度 anchor，测试去雾和结构保持 |
| D | `blur2+blur4` | 两种 blur 叠加，缺少噪声/天气，不进入 main train，保留作 held-out diagnostic eval |

第一版 1500 条建议按 A/B/C 混合，不只选最难的 A 组。RL 数据如果全是最难样本，reward 方差会大，
而且容易把模型推向过度去噪或过锐化。D 组不进入 main RL train，但应作为 level2 held-out
diagnostic eval：如果 RL 后在 `blur2+blur4` 上出现过锐化、伪纹理或边缘发硬，说明 reward 可能推偏。

## 4. 推荐抽样配比

### 4.1 1500 条主实验

| 优先级 | 组合 | 每组条数 | 小计 |
|---|---|---:|---:|
| A | `haze+noise`, `haze+rain`, `noise+rain` | 250 | 750 |
| B | `blur4+haze`, `blur4+noise`, `blur4+rain` | 180 | 540 |
| C | `blur2+noise`, `blur2+rain`, `blur2+haze` | 70 | 210 |
| D | `blur2+blur4` | 0 | 0 |
| 合计 | 9 个组合 |  | 1500 |

这个配比是“中等偏难”：A 组占 50%，B 组占 36%，C 组占 14%。它强调已有弱项和难恢复噪声，
但仍保留一些中等难度样本帮助稳定训练。D 组计数为 0 只表示不参与 main train；评测时仍建议作为
held-out diagnostic eval 单独报告。

### 4.2 600 条 pilot

| 优先级 | 组合 | 每组条数 | 小计 |
|---|---|---:|---:|
| A | `haze+noise`, `haze+rain`, `noise+rain` | 100 | 300 |
| B | `blur4+haze`, `blur4+noise`, `blur4+rain` | 80 | 240 |
| C | `blur2+noise`, `blur2+rain`, `blur2+haze` | 20 | 60 |
| 合计 | 9 个组合 |  | 600 |

### 4.3 3000 条扩展

如果 1500 条带来稳定收益，再使用全部 level2：10 组合 × 300 条 = 3000 条。扩展实验用于判断
难样本重加权是否比 level2 均衡覆盖更好。

## 5. Batch 与超参建议

### 5.1 smoke

```bash
NUM_GENERATIONS=2
PER_DEVICE_BATCH_SIZE=1
GRAD_ACCUM_STEPS=1
RL_EPOCHS=1
LR=5e-7
MAX_SAMPLES=120
REWARD_BATCH_SIZE=1
```

目的只是保证训练链路通：三图能生成、Qwen reward 能打分、GRPO loss 能回传。

### 5.2 pilot

```bash
NUM_GENERATIONS=2
PER_DEVICE_BATCH_SIZE=1
GRAD_ACCUM_STEPS=2
RL_EPOCHS=1
LR=5e-7
MAX_SAMPLES=600
REWARD_BATCH_SIZE=1
```

默认 3 张 policy GPU 时：

```text
每个 micro-step: 3 prompt × 2 generations = 6 completions
每个 optimizer update: 6 prompt × 2 generations = 12 completions
```

这里的 group size 只有 2，advantage 方差会偏大，但适合省时间判断方向。

### 5.3 main

main 阶段准备两套参数。若 reward server 和显存稳定，使用 `main-preferred`；若 Qwen reward 成本过高、
生成不稳定或排队时间过长，使用 `main-safe`。

`main-safe`：

```bash
NUM_GENERATIONS=2
PER_DEVICE_BATCH_SIZE=1
GRAD_ACCUM_STEPS=4
RL_EPOCHS=1
LR=5e-7
MAX_SAMPLES=1500
REWARD_BATCH_SIZE=1
```

默认 3 张 policy GPU 时：

```text
每个 micro-step: 3 prompt × 2 generations = 6 completions
每个 optimizer update: 12 prompt × 2 generations = 24 completions
一轮 1500 prompt: 3000 sampled completions
```

`main-preferred`：

```bash
NUM_GENERATIONS=4
PER_DEVICE_BATCH_SIZE=1
GRAD_ACCUM_STEPS=2
RL_EPOCHS=1
LR=3e-7
MAX_SAMPLES=1500
REWARD_BATCH_SIZE=1
```

默认 3 张 policy GPU 时：

```text
每个 micro-step: 3 prompt × 4 generations = 12 completions
每个 optimizer update: 6 prompt × 4 generations = 24 completions
一轮 1500 prompt: 6000 sampled completions
```

`main-preferred` 的组内比较更稳定，但 rollout 和 Qwen reward 成本约为 `main-safe` 的 2 倍。
如果 `NUM_GENERATIONS=4` 太慢或显存压力大，退回 `main-safe`。此时每个 update 仍有 24 个
completion，但每个 prompt 的组内比较只有 2 个样本，reward 估计会更粗。

### 5.4 不建议一开始做的事

- 不建议 `PER_DEVICE_BATCH_SIZE > 1`。MSRS 的序列接近 26k，而且要做三图 decode 和 logprob replay，
  先保守使用每卡 1。
- 不建议第一版就加载 reference KL。当前脚本默认无 reference KL 是为了节省 3 张 policy A100 上的显存。
  如果发现 RL 漂移、格式退化或图像风格跑偏，再考虑 `KL_BETA=0.01-0.02` 或混入 SFT loss。
- 不建议只选 A 组三个最难组合。这样 reward 太尖，可能学成过度去雾/去噪，损害中等退化。

## 6. 选择流程

1. 先补 level2 eval 或至少做 level2 smoke eval。当前文档里的弱项主要来自 level1 评测线索，
   level2 的正式逐组合 eval 还需要补齐。
2. 固定候选来源为已审计的最终 6000 条 clean JSON，先筛出 level2，再按组合抽样。这样能继承
   scene-level train/test 隔离和 CoT 清洗门禁。
3. 抽样时保持 item 整体移动：degraded IR、degraded VIS、clean IR、clean VIS、fused GT 和标签不能拆开。
4. 每个组合内部按 seed 无放回抽样，并输出 report：seed、组合计数、base image 曝光直方图、路径存在性。
5. 先跑 120-200 条 smoke，检查 reward JSONL 中：
   - format gate 通过率；
   - 三图生成成功率；
   - Qwen image reward 是否有明显区分度；
   - A/B/C 组合 reward 是否符合难度预期。
6. 再跑 600 pilot，和 SFT 初始化 checkpoint 做同一测试集对比。不要只看 Qwen reward，也要看同一
   eval set 上相对 SFT 的图像质量和人工偏好。
7. pilot 通过进入 main 的门槛后，冻结 1500 条主实验 manifest，开始 main RL。

## 7. Pilot 进入 Main 的门槛

600 条 pilot 不是为了刷结果，而是决定是否值得进入 1500 条 main。建议同时满足：

- 三图生成成功率 `>= 95%`；
- format gate 通过率 `>= 90%`；
- reward std 不能长期接近 0，同一 prompt 的多个 completion 必须能被 reward 区分；
- Qwen reward error/timeout 比例 `< 10%`；
- 与 SFT checkpoint 相比，level2 fused 指标或人工偏好不下降；
- 随机抽查样本中没有明显热目标消失、单模态复制、过度平滑或过锐化。

如果 reward std 很低，说明 GRPO advantage 缺少学习信号；此时继续放大到 1500 条通常只是增加成本，
应先检查 reward 分项、采样温度或 generation diversity。

## 8. 验收标准

训练前：

- manifest 只含 level2；
- 组合计数符合计划；
- 所有路径存在；
- base scene 不进入 official test；
- 每条样本都有 `infrared_label=noise+stripe_noise` 和对应 `visible_label`；
- prompt 顺序仍是 degraded IR、degraded VIS；
- 输出协议仍是 infrared CoT/image、visible CoT/image、fused CoT/image。

训练中：

- `num_generated_images=3` 的比例接近 100%；
- format gate 不持续掉到 0；
- reward 标准差不能长期接近 0，否则 GRPO advantage 没有学习信号；
- Qwen reward error group 不能大量跳过；
- 生成图人工抽查不能出现明显 token 崩坏、单模态复制、过度去噪或热目标消失。

训练后：

- 在同一 eval 设置下比较 SFT init、RL smoke、RL pilot、RL main；
- 至少报告 level1 average、level2 average、20-combination macro-average；
- 对 A/B/C 组合分别看 fused reward 和 IR/VIS 恢复指标；
- D 组 `blur2+blur4` 作为 held-out diagnostic eval，重点检查过锐化、伪纹理和边缘发硬；
- 如果 level2 提升但 level1 明显下降，优先控制 policy drift，而不是马上改 reward 权重。

level1 明显下降时的备用方案：

- 加 `KL_BETA=0.01`，必要时再试 `0.02`；
- 或混入 `5%-10%` level1 anchor；
- 或加入 SFT loss mixing；
- 优先控制 policy drift，再考虑调整 reward 权重。

## 9. 代码入口

构建 smoke/pilot/main/diagnostic manifest：

```bash
cd /home/zhuyiming/data/lsy/ImageFusion-R1-mgpt2-omni
SEED=20260709 bash scripts/build_msrs_level2_rl_manifests.sh
```

默认输出：

```text
dataset_final/MSRS/rl_level2_seed20260709/msrs_level2_rl_smoke_180_seed20260709.json
dataset_final/MSRS/rl_level2_seed20260709/msrs_level2_rl_pilot_600_seed20260709.json
dataset_final/MSRS/rl_level2_seed20260709/msrs_level2_rl_main_1500_seed20260709.json
dataset_final/MSRS/rl_level2_seed20260709/msrs_level2_rl_diagnostic_blur2_blur4_300_seed20260709.json
dataset_final/MSRS/rl_level2_seed20260709/msrs_level2_rl_all_3000_seed20260709.json
dataset_final/MSRS/rl_level2_seed20260709/msrs_level2_rl_manifest_report_seed20260709.json
```

脚本会从 degraded image path 解析并规范化标签，例如 `haze_noise` 变成 `haze+noise`，
`noise_stripe_noise` / `stripe_noise_noise` 都变成 `noise+stripe_noise`。输出 manifest 是最小
RL schema，不包含 GT CoT 文本，避免训练入口误读监督答案。

训练入口：

```bash
bash scripts/train_msrs_level2_rl_smoke_epoch4.sh
bash scripts/train_msrs_level2_rl_pilot_epoch4.sh
bash scripts/train_msrs_level2_rl_main_safe_epoch4.sh
bash scripts/train_msrs_level2_rl_main_preferred_epoch4.sh
```

这些 wrapper 默认使用 epoch4 SFT checkpoint、3 张 policy GPU 和 1 张 Qwen reward GPU；如需换 GPU、
checkpoint 或输出目录，覆盖对应环境变量即可，例如 `POLICY_GPUS`、`REWARD_GPU`、`INIT_CKPT`、
`OUTPUT_DIR`、`TRAIN_MANIFEST`。

## 10. 推荐执行口径

第一版可以这样和学长对齐：

```text
数据：level2-only，1500 条中等偏难重加权 main set，不直接从 1500 起跑。
流程：120-200 smoke → 600 pilot → 1500 main；pilot 通过门槛后才进入 main。
退化：优先 haze+noise、haze+rain、noise+rain，其次 blur4+haze/noise/rain，
     再放少量 blur2+noise/rain/haze 稳定训练；blur2+blur4 保留作 held-out diagnostic eval。
batch：per-device prompt batch 固定 1，3 张 policy GPU。
group：smoke/pilot 用 num_generations=2；main-safe 用 2，main-preferred 用 4。
prompt：MSRS prompt 约 9.8k tokens，completion 约 15-16k tokens，继续用 28672 上下文。
reward：沿用 two-level reward，text 0.4，image 0.6，Qwen reward batch size 1。
显存：RL logprob replay 已使用 LOGPROB_CHUNK_SIZE，默认 64；显存紧张时可降到 32 或 16。
```

一句话汇报：

```text
1500 条 level2 中等偏难重加权 main set，
先用 600 条 pilot 验证 reward 有效性，
num_generations 从 2 逐步放大到 4。
```

最小可行版本是 600 条 pilot；更像正式实验的版本是 1500 条 main；如果 1500 条确实有效，再扩展到
3000 条全 level2。
