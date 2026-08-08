# MSRS ImageFusion GRPO 代码审查与 Level1/Level2 小规模 RL 方案

更新时间：2026-08-08

本文以当前仓库实际代码和 wrapper 参数为准，说明 ImageFusion-R1 的 GRPO 训练链路、text/image
两路 reward，以及在单个 completion 推理约 15 分钟时，使用 Level1 30 条、Level2 70 条是否合理。

## 1. 结论

当前代码已经具备完整的在线 GRPO 主链路：三张 policy GPU 并行 rollout、同一 prompt 生成两个
completion、text/image reward、组内 advantage、teacher-forced log-prob replay、反向更新、断点保存和
preflight 门禁。Python 与 shell 语法检查通过，使用 GT 图像构造的本地 reward smoke test 也通过。

但当前状态还不能直接认定为“正式 RL 已验证可跑”：

- 仓库中没有找到通过的 `preflight_validation.json`，也没有现存的正式 RL checkpoint；
- 已有 RL manifest 全是 Level2-only，尚没有 Level1/Level2 混合的 30/70 manifest；
- 旧文档 `MSRS_TWO_LEVEL_REWARD_ZH.md` 记录的 `text:image=0.4:0.6` 已经过时，当前 wrapper 实际使用
  `0.1:0.9`；
- 严格格式门一旦失败，总 reward 直接为 0；如果同一 prompt 的两个 completion reward 相同，该组会被
  丢弃。因此必须先通过 GPU preflight，不能直接提交 100 条长任务。

关于样本数：**Level1 30、Level2 70 作为一次“小规模方向性 pilot”是合理的，但不足以作为最终论文
RL 主实验。** 它强调更难的 Level2，同时用 30 条 Level1 作为防遗忘 anchor。考虑当前使用 3 张 policy
GPU，建议实际做成 **Level1 30 + Level2 69 = 99 条**。当前 distributed sampler 每轮只使用能被全局
prompt batch 3 整除的样本；100 条最终也只训练 99 条，并随机丢掉 1 条。显式使用 99 条能让覆盖和审计
更清楚。

## 2. 当前 GRPO 代码链路

主要入口如下：

| 功能 | 当前文件 |
|---|---|
| 总 launcher、Qwen reward server 和 torchrun | `scripts/train_msrs_two_level_rl.sh` |
| 9/100/200 条实验 wrapper | `scripts/train_msrs_level2_rl_small_epoch4.sh` |
| GRPO trainer | `imagefusion_r1/rl/grpo_trainer_mgpt2.py` |
| RL manifest 读取和分布式分片 | `imagefusion_r1/rl/grpo_data.py` |
| text/image 两路 reward 汇总 | `imagefusion_r1/rl/msrs_two_level_reward.py` |
| 三张 GT 的 PSNR/SSIM reward | `imagefusion_r1/rl/reference_image_reward.py` |
| Qwen3-VL fused-image 裁判 | `imagefusion_r1/rl/qwen3vl_reward_judge.py` |
| Qwen reward HTTP server | `scripts/serve_msrs_qwen3vl_reward.py` |
| GPU preflight 验证 | `scripts/validate_msrs_rl_preflight.py` |

一次训练 micro-step 的数据流是：

```text
每张 policy GPU 读取 1 个 prompt
    -> 每个 prompt 顺序生成 G=2 个 completion
    -> 解码并保存 clean IR / clean VIS / clean fused 三张图
    -> 计算严格格式 gate、text reward、三 GT reward、Qwen fused reward
    -> 对同一 prompt 的两个总 reward 做组内标准化得到 advantage
    -> 丢弃 reward 零方差组
    -> teacher-forced replay rollout token 的 log-prob
    -> 检查 rollout/replay log-prob 最大误差 <= 0.10
    -> GRPO/PPO clipped loss backward
    -> 每 2 个 micro-step 做一次 optimizer update
```

当前默认关键参数是：

| 参数 | 当前值 |
|---|---:|
| policy GPU | 3 张 |
| reward GPU | 1 张 |
| per-device prompt batch | 1 |
| `num_generations` | 2 |
| gradient accumulation | 2（100/200 条 wrapper） |
| learning rate | `3e-7` |
| image CFG | 3.0 |
| text top-k | 1，贪心保 XML 格式 |
| image top-k | 2000，提供组内图像差异 |
| reference KL | 关闭，`KL_BETA=0` |
| FSDP | `SHARD_GRAD_OP` |
| replay micro-batch | 1 completion |

## 3. 两路 Reward 的实际公式

Format 只是 gate，不是第三路 reward。当前运行时的总式为：

```text
R_total = R_gate * (0.1 * R_text + 0.9 * R_image)
```

代码中还保留了 text modulation 参数，但默认 `text_modulation_min=1.0`，因此当前实际 modulation 恒为
1，不会再次缩放总 reward。

### 3.1 Text reward

```text
R_text = 0.7 * R_label + 0.3 * R_plan
```

`R_label` 分别检查 IR 和 VIS 的期望退化成分是否被生成文本覆盖：

```text
R_label = 0.5 * coverage(IR degradation) + 0.5 * coverage(VIS degradation)
```

标签来自 manifest 中的 `infrared_label`、`visible_label`，缺失时才从 degraded path 解析。当前支持的成分
为 `noise`、`stripe_noise`、`blur2`、`blur4`、`haze`、`rain`。

`R_plan` 检查 fused planning 文本中是否覆盖四类意图，每类占 0.25：

- 保留红外热目标、轮廓或显著信息；
- 保留可见光颜色、纹理、边缘或结构；
- 抑制噪声、条纹、雾、雨、模糊等伪影；
- 明确 fusion/combine/balance/complement 等融合意图。

优点是便宜、确定、容易审计。局限也很明确：

- `blur2` 与 `blur4` 共用模糊关键词，不能识别模糊强度是否判断正确；
- 当前只奖励 expected component coverage，不惩罚模板化地枚举全部退化，存在 reward hacking 空间；
- `TEXT_TOP_K=1` 时同一 prompt 的两个 completion 文本往往很接近，GRPO 的主要组内方差仍来自图像；
- 它判断关键词覆盖，不判断长 CoT 的事实一致性和推理质量。

因此当前将 text 权重降到 0.1 是合理的：它主要负责方向约束和可解释日志，不应压过图像质量信号。

### 3.2 Image reward

当前 image reward 不是只有 Qwen，而是确定性 GT reward 为主、Qwen 视觉判断为辅：

```text
R_image = 0.9 * R_three_GT + 0.1 * R_Qwen
```

若 Qwen 超时、输出 JSON 错误或被跳过，代码会把有效权重重新归一化到 `R_three_GT`，不会因为外部
judge 故障浪费已经完成的昂贵 rollout。

#### 三张 GT reward

每张生成图分别与对应 GT 计算：

```text
R_pair = 0.6 * SSIM + 0.4 * normalized_PSNR
normalized_PSNR = clip((PSNR - 10) / (40 - 10), 0, 1)
```

然后聚合三种输出：

```text
weighted_mean = 0.25 * R_IR + 0.25 * R_VIS + 0.50 * R_fused
weakest       = min(R_IR, R_VIS, R_fused)
R_three_GT    = 0.70 * weighted_mean + 0.30 * weakest
```

这样 fused image 权重最高，同时通过 `weakest` 防止只优化 fused、牺牲 clean IR 或 clean VIS。代码禁止
隐式 resize，生成图与 GT 尺寸不一致会显式报错。

#### Qwen fused-image reward

Qwen3-VL-8B-Instruct 接收：degraded IR、degraded VIS、generated fused image 和期望退化标签，不读取
policy 生成的 CoT。它输出五个 0--10 分项，归一化后按下式汇总：

```text
R_Qwen = 0.25 * artifact_suppression
       + 0.20 * visible_preservation
       + 0.20 * infrared_preservation
       + 0.20 * fusion_naturalness
       + 0.15 * semantic_consistency
```

Qwen server 对同步 vLLM 调用加锁，因此三个 policy rank 同时请求时会在 reward GPU 上串行执行。它通常
不是 15 分钟 rollout 的主要成本，但必须在 preflight 中记录每个 reward batch 的实际耗时和超时率。

### 3.3 Format gate

Full protocol 必须同时满足：

- IR CoT/image -> VIS CoT/image -> fused CoT/image 的标签顺序正确；
- 三个 stage 内 required tag 全部存在、非空、无重复；
- 文本中恰好 3 个 `<|image|>`；
- 实际成功解码 3 张图。

满足时 `R_gate=1`，否则为 0。最终 fused image 生成后立即停止时，代码允许
`<clean_fused_image><|image|>` 没有闭合标签。

严格 gate 可以阻止格式崩坏，但也会产生“两个 completion 都是 0 -> 组内 reward 方差为 0 -> 不更新”的
情况。trainer 连续 3 个 distributed batch 无信号会主动停止。这也是 preflight 必须先通过的原因。

## 4. GRPO 更新是否接通

当前实现不是只做推理打分，policy update 已接通：

1. 同一 prompt 生成 `G=2` 个 completion；
2. 对组内 reward 计算均值和 population standard deviation；
3. 计算 `A_i = (R_i - mean(R)) / (std(R) + 1e-4)`；
4. replay rollout 分布，并检查新旧 log-prob 一致性；
5. 使用 PPO clip range 0.2 的 token-level policy-gradient loss；
6. completion 内先按有效 token 平均，再按 completion 平均；
7. replay 每次只保留 1 个 completion 的反向图，降低约 25k token 长序列的显存峰值；
8. Adam moments 可在 step 之外 offload 到 CPU；
9. 保存 policy、optimizer shard 和 `trainer_state.json`，支持同 world size resume。

当前关闭 reference KL，所以小学习率、少量 update、Level1 anchor 和独立 dev 评估非常重要。若 Level2 提升
而 Level1 明显下降，再考虑 `KL_BETA=0.01`，不建议第一次 pilot 就额外加载 reference model。

## 5. Level1 30 / Level2 70 的合理性

### 5.1 为什么可以作为 pilot

- SFT 已使用 Level1/Level2 各 3000 条，RL 不需要重新覆盖 6000 条；
- Level2 是更难且当前更希望改善的区域，70% 训练预算倾斜是有目的的 curriculum；
- 30% Level1 可作为稳定 anchor，检查和缓解只训 Level2 导致的简单退化性能回退；
- 100 个 prompt、`G=2` 已产生约 200 个昂贵 rollout，足够判断 reward 是否有方向性信号。

### 5.2 为什么不能当最终结论

- 30 条 Level1 若均匀覆盖 10 个组合，每组合只有 3 条；
- 70 条 Level2 即使覆盖 9 个主训练组合，每组合也只有约 4--12 条；
- `G=2` 的 advantage 本质上是成对比较，方差较大；
- 99 个有效 prompt、`grad_accum=2` 只有 33 个 micro-step、17 次 optimizer update，扣掉零方差组后可能更少；
- 结果容易受场景、seed 和少数高 reward completion 影响。

因此它适合回答“这套 reward 能不能让模型往正确方向走”，不适合单独回答“RL 是否稳定提升所有 20 个
退化组合”。

### 5.3 推荐的 99 条配比

Level1 用 30 条覆盖全部 10 个组合，每组合 3 条。

Level2 延续现有方案，把 `blur2+blur4` 留作 held-out diagnostic，不进入 RL train；其余 9 个组合建议：

| Level2 难度组 | VIS 组合 | 每组合 | 小计 |
|---|---|---:|---:|
| A | `haze+noise`, `haze+rain`, `noise+rain` | 11 | 33 |
| B | `blur4+haze`, `blur4+noise`, `blur4+rain` | 8 | 24 |
| C | `blur2+haze`, `blur2+noise`, `blur2+rain` | 4 | 12 |
| D | `blur2+blur4`，held-out diagnostic | 0 | 0 |
| Level2 合计 | 9 个训练组合 |  | 69 |

总计 `30 + 69 = 99`，恰好被 3 张 policy GPU 整除，比例仍约为 30%/70%。抽样还应满足：

- 尽量使用 99 个不同 base scene；
- 与 RL dev、official test 的 base scene 零交集；
- manifest 中显式保存 level、canonical degradation 和 base image；
- 固定 seed，无放回抽样，并输出逐组合计数和路径检查 report；
- 99 条顺序随机打散，不能先连续跑完 Level1 再跑 Level2。

## 6. 按 15 分钟推理估算耗时

当前每个 prompt 需要两个 completion，并且同一 rank 上是顺序生成。3 张 policy GPU 理想并行时：

```text
纯 rollout 墙钟时间
= floor(N_prompt / 3) * G * 15 分钟
```

| 阶段 | prompt | completion | micro-step | 纯 rollout | 建议预留总时间 |
|---|---:|---:|---:|---:|---:|
| preflight | 9 | 18 | 3 | 1.5 小时 | 2--4 小时 |
| 小 pilot | 30 | 60 | 10 | 5 小时 | 7--12 小时 |
| 推荐混合 pilot | 99 | 198 | 33 | 16.5 小时 | 24--36 小时 |
| 原 100 条设想 | 实际使用 99 | 198 | 33 | 16.5 小时 | 24--36 小时 |

“建议预留总时间”不是实测值，而是在 rollout 之外为以下工作留出的保守工程预算：

- 三张图 decode、保存和本地 PSNR/SSIM；
- 单 reward GPU 上串行的 Qwen judge；
- 每个 completion 的 25k 左右 token log-prob replay 和 backward；
- optimizer CPU/GPU 搬运与 checkpoint；
- 首次模型加载、同步和可能的失败重启。

真正的总耗时必须由 9 条 preflight 的 `sec_per_micro_step` 外推。现有 validator 会生成
`projected_pilot100_hours`，应以该值替代上表估算。

## 7. 推荐执行顺序

不建议一次性盲跑 99 条。推荐按下面的止损门禁执行：

1. 修订并冻结 mixed-level manifest 构建脚本，生成嵌套的 9/30/99 条集合；
2. 先跑 9 条 GPU preflight，必须生成且通过 `preflight_validation.json`；
3. 再跑 30 条小 pilot，约 7--12 小时总预算；
4. 用固定 mixed-level dev 比较 SFT 与 RL-30：三图 PSNR/SSIM、严格格式率、reward 分项和人工图像检查；
5. 只有 reward 有方差、update 正常、Level1 未明显回退时，再扩到完整 99 条；
6. 99 条完成后仍将其称为 pilot；若要形成论文主结论，应至少增加独立 seed 或更大样本，并报告 20 个组合
   的 macro-average。

30 条小 pilot 的继续门槛建议为：

- 三图生成率和严格 gate 通过率均不低于 95%；
- 非零 reward-std group 比例不低于 50%，且不能连续出现无信号 batch；
- `replay_logp_max_abs <= 0.10`；
- 至少发生 1 次真实 optimizer update，`grad_norm > 0`；
- Qwen error/timeout 率低于 10%；
- Level1 dev 没有明显格式、热目标、结构或颜色退化；
- Level2 至少在 fused 指标或人工偏好之一出现一致改善，不只看训练 reward 上升。

## 8. 当前审查发现和开跑前清单

已经确认：

- SFT `epoch4` checkpoint 存在；
- Qwen3-VL-8B reward 模型目录存在；
- RL Python 文件 `py_compile` 通过；
- 两个主要 shell launcher 通过 `bash -n`；
- 现有 Level2 manifest report 通过路径、ID、scene 和 split 检查；
- 用三张 GT 作为生成图时，本地 reference reward 得分为 1.0，完整 format gate 可通过；
- 当前代码会丢弃外部 GT 数据错误组、零方差组，并检查 replay parity 和非零梯度。

开跑前仍必须完成：

- 生成 mixed Level1/Level2 的 30/69 manifest 和审计 report；
- 明确 9 条 preflight 是否也包含 Level1，不能只沿用旧的 Level2-only preflight 后就宣称 mixed pilot 已验证；
- 取得一次新的、通过的 `preflight_validation.json`；
- 用实际 preflight 更新 15 分钟假设和 99 条总耗时；
- 离线检查 text reward 是否能被“枚举所有退化”轻易刷满；
- 在 reward JSONL 中分别记录 Level1/Level2 和各组合统计，避免总均值掩盖某一级回退；
- 冻结 held-out `blur2+blur4` diagnostic 和 mixed-level dev，训练中不得再抽入。

最终建议口径：

```text
先做 9 条 preflight，再做 30 条小 pilot；通过门禁后扩到 99 条混合 RL。
99 条使用 Level1=30、Level2=69，近似原计划的 30/70，且适配 3-GPU 全局 batch。
该规模用于验证 reward 和训练方向，不作为最终论文主实验的充分证据。
```
