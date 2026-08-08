# MSRS 多退化图像恢复与融合：最终 CE-SFT 数据及训练流程

> **项目强制门禁（后续所有实验必须遵守）**：Qwen 生成的 JSON 是保留
> `raw_qwen_outputs` 的中间产物，不允许直接生成 PKL。必须先从原始输出重新执行
> CoT 标签清洗，修复缺失/错位的 `<think>`、`<answer>` 及阶段字段，完成结构、
> 语义、路径和清洗幂等性验证，再从最终 clean JSON 生成 PKL。任何未通过清洗
> 和验证的 JSON/PKL 均不得用于训练。

当前整理后的 MSRS 数据资产统一维护在 `dataset_final/MSRS/`。其中
`dataset_final/MSRS/README.md` 是文件用途、C01-C20 编号、zeus9 回传、6000 条训练子集和
21660 条论文发布集的当前状态入口；本文件保留完整的方法与复现流程。Conversation、五图 token
顺序、label mask、CE-only PKL schema 和 MSE 禁用边界单独固定在
`docs/MSRS_CE_PKL_CONVERSATION_ZH.md`。

## 1. 目标与当前结论

本阶段的目标是得到一个用于后续 RL 初始化的高质量 SFT checkpoint。模型接收两张退化输入图像，并按固定自回归顺序输出三个阶段的语义分析和三张干净目标图像。

最终正式训练集规划为 6000 条：

- level1：10 种红外/可见光退化组合，每种 300 条，共 3000 条；
- level2：10 种红外/可见光退化组合，每种 300 条，共 3000 条；
- 总计：20 种组合 × 300 条 = 6000 条训练 item；
- 每条 item 始终绑定 2 张退化输入、3 张干净 GT 和 3 段 CoT；
- 每条 item 有 5 次图像编码，因此 6000 条对应 30000 个图像 token block，但底层场景来自 MSRS 的 1083 张训练底图。

6000 条适用于基于 Lumina-mGPT-2.0-Omni 的领域 SFT，不代表从零训练多模态基础模型所需的数据规模。继续增加同一批 1083 个场景的组合数量会逐渐出现收益递减，因此第一版正式实验优先保证质量、均衡性和可复现性。

以下旧产物不能作为最终训练数据：

- `data/msrs_raw_items_cot_level1_level2_balanced_112_seed20260620.json`：仅为 2240 条阶段性版本；
- `data/processed_mgpt2/msrs_ce_cot_clean_balanced_112_480x640/`：曾中断于约 311 个 PKL，且生成前使用的是旧清洗结果；
- `data/msrs_raw_items_fixed.json`：只有三个 `*_understand` 字段，不是效果好的 demo checkpoint 实际使用的完整 stage-CoT 协议。

效果好的 demo PKL 位于：

```text
data/processed_mgpt2/msrs_ce_cot_clean_480x640/
```

通过直接解码其中的监督 token，可以确认它来自完整的 stage-CoT conversation，而不是 understand-only conversation。

## 2. 数据组合定义

### 2.1 level1

红外退化：

- `noise`
- `stripe_noise`

可见光退化：

- `blur2`
- `blur4`
- `haze`
- `noise`
- `rain`

二者笛卡尔组合得到 2 × 5 = 10 种组合。

### 2.2 level2

红外同时包含 `noise` 和 `stripe_noise`。目录中可能存在 `noise_stripe_noise` 与 `stripe_noise_noise` 两种顺序，构建训练集时将其视为同一个 canonical degradation。

可见光从五种基础退化中取两种，共有 10 种无序组合：

```text
blur2+blur4, blur2+haze, blur2+noise, blur2+rain,
blur4+haze, blur4+noise, blur4+rain,
haze+noise, haze+rain, noise+rain
```

因此 level2 也有 10 种组合。

## 3. 为什么需要补充 1500 条 CoT

正式数据不再使用旧 demo 中 112 条 understand-only 或旧风格 CoT 来填补 `IR=noise + VIS=blur2`，而是生成 300 条风格统一的新 golden CoT。

另外四个 level2 组合虽然已经达到约 300 条，但它们几乎使用了同一批 300 张底图：三组之间底图重合率为 100%。为降低场景内容与退化类型的偶然相关性，需要为它们补充不重合的新候选。

补充脚本生成五组、每组 300 条，共 1500 条：

| 组合 | 旧候选 | 新增候选 | 合并候选 |
|---|---:|---:|---:|
| level1: noise + blur2 | 正式集不使用旧 112 条 | 300 | 300 |
| level2: blur4 + rain | 334 | 300 | 634 |
| level2: haze + noise | 300 | 300 | 600 |
| level2: haze + rain | 300 | 300 | 600 |
| level2: noise + rain | 300 | 300 | 600 |

`blur4 + rain` 理论上补 266 条即可达到 600；统一生成 300 条会多 34 条候选，但最终仍只选 300 条，不会改变训练分布。

补充清单已经经过以下静态验证：

- 只有上述 5 种组合；
- 每组恰好 300 条；
- level1 IR 只与 level1 VIS 配对，level2 IR 只与 level2 VIS 配对；
- 四组 level2 的新增图对与各自旧图对重合数为 0；
- 新增 level2 组合之间的底图交集约为 108–121，而不是再次使用完全相同的 300 张底图。

## 4. 补充 CoT 生成

入口：

```text
scripts/run_msrs_supplement_5combos_qwen3vl_cot_gpu23.sh
```

该脚本只负责：构建不重合 manifest、每组随机抽取 300 条、一次加载 Qwen3-VL-32B-Thinking、生成五组 CoT、断点续跑和严格格式验证。

### 4.1 GPU 2、3 启动测试

GPU 2、3 各需要约 40 GB 可用显存。脚本默认使用 TP=2、`gpu_memory_utilization=0.45`、每卡 12 GB CPU offload，并显式设置每个 TP worker 使用 1 GiB KV cache。显式 KV cache 用于跳过共享 GPU 上不稳定的空闲显存 profiling；在 batch size 1、最大上下文 4096 的本任务中足够使用。先生成 2 条进行完整启动测试：

```bash
cd /home/zhuyiming/data/lsy/ImageFusion-R1-mgpt2-omni

CUDA_VISIBLE_DEVICES=2,3 MAX_SAMPLES=2 \
bash scripts/run_msrs_supplement_5combos_qwen3vl_cot_gpu23.sh
```

若测试成功，再运行完整任务。`--resume` 会跳过已完成的 2 条：

```bash
nohup env CUDA_VISIBLE_DEVICES=2,3 MAX_SAMPLES=0 \
bash scripts/run_msrs_supplement_5combos_qwen3vl_cot_gpu23.sh \
>/dev/null 2>&1 &
```

查看日志：

```bash
tail -f outputs/logs/msrs_supplement_5combos_qwen3vl_cot_gpu23.log
```

输出：

```text
data/msrs_supplement_5combos_300_each_with_fusedgt_golden_qwen3vl_cot.json
```

不要在显存预检失败时设置 `SKIP_GPU_CHECK=1` 强行启动。其他任务的显存会动态增长，初始化成功也不等于可以安全抢占正在使用的 GPU。

## 5. CoT 清洗方法

清洗实现位于：

```text
imagefusion_r1/preprocess/cot_sanitizer.py
```

### 5.1 每个 stage 的规范结构

红外：

```xml
<think>退化诊断、受损信息、保留信息和恢复依据</think>
<answer>
<infrared_degradation>退化类别和具体视觉证据</infrared_degradation>
<infrared_understand>受损/保留信息及红外对融合的贡献</infrared_understand>
<infrared_image>理想干净红外图像描述</infrared_image>
</answer>
```

可见光：

```xml
<think>退化诊断、受损信息、保留信息和恢复依据</think>
<answer>
<visible_degradation>退化类别和具体视觉证据</visible_degradation>
<visible_understand>受损/保留信息及可见光对融合的贡献</visible_understand>
<visible_image>理想干净可见光图像描述</visible_image>
</answer>
```

融合：

```xml
<think>两模态退化总结、信息保留、伪影抑制和融合目标</think>
<answer>
<fused_understand>保留两模态优势并抑制各自退化</fused_understand>
<fused_image>理想干净融合图像描述</fused_image>
</answer>
```

### 5.2 清洗步骤

1. 统一换行和错误控制标签，例如重复 `<answer>`、重复 `</think>`。
2. 从每个 opening tag 分别寻找 closing tag，避免前面未闭合的错误标签吞掉后面真正正确的 answer。
3. 对 answer 字段选择最后一个完整、合法的标签内容。
4. 去除 XML 嵌套、提示词复述和元话语，例如 “the user said”“output format”“let me draft”。
5. 合并异常空格和多余空行。
6. 最终训练 JSON 将 `<think>` 截断到最多 1800 字符，answer 单字段最多 900 字符，与效果好的 demo PKL 清洗尺度一致。
7. 若 `<think>` 少于 100 字符，或 `<think>` 内错误嵌入 answer 标签，则不保留 `"."`、`"and"` 等碎片；清洗器使用该样本已经验证过的 degradation、understand 和 image 字段重组一段完整诊断。该步骤不引入新的图像事实。
8. 再次运行同一清洗器必须得到完全相同的文本，即清洗具有幂等性。

旧 demo 补洞数据中曾发现 23/112 条 `<think>` 少于 100 字符，并存在一条 answer 标签错位。改进清洗后，2240 条阶段数据的全量审计结果为 0 个结构或语义错误；最短 `<think>` 长度分别为 IR 897、VIS 838、Fusion 1027。最终 6000 条仍需执行相同检查。

### 5.3 最终验收条件

任何一条不满足以下条件都不能进入 PKL：

- 五张图路径存在，且文件名对应同一底图；
- IR 与 VIS degradation level 相同；
- ID 唯一，IR/VIS degraded path pair 唯一；
- 三段 CoT 和全部 required tags 非空；
- `<think>` 不短于 100 字符；
- answer 字段不含嵌套标签、提示词复述或 fallback 占位文本；
- IR degradation 描述覆盖路径标签中的全部退化成分；
- VIS degradation 描述覆盖路径标签中的全部退化成分；
- fused understand 同时提到 visible、infrared，并明确伪影抑制；
- 再次清洗文本不发生变化；
- 20 个组合各 300 条；
- level1 和 level2 各 3000 条。

## 6. 最终随机抽样与无偏控制

“随机”不是将 IR、VIS、CoT 和 GT 分别打乱。一个 item 内的所有字段始终整体移动。

最终构建采用固定 seed `20260620`：

1. 根据 `(IR level, IR degradation, VIS level, VIS degradation)` 分为 20 个 strata；
2. 每个 stratum 无放回选择 300 条；
3. 候选数量少的组合先处理；
4. 在随机 tie-break 的前提下，优先选择当前累计出现次数较少的 base image；
5. 目标是让 1083 张底图在 6000 条中尽量各出现约 5–6 次；
6. 最后对 6000 个完整 item 进行全局 shuffle；
7. 保存 seed、来源文件、每组计数和底图曝光直方图到 report JSON。

分布式补充 CoT 完成、拉回并严格合并为 `merged_1500.json` 后执行：

```bash
SUPPLEMENT=data/msrs_cot_distributed_seed20260621/merged_1500.json \
bash scripts/build_msrs_final_6000_training_items.sh
```

输出：

```text
data/msrs_raw_items_cot_final_6000_clean_seed20260620.json
data/msrs_raw_items_cot_final_6000_clean_seed20260620.report.json
```

脚本在补充 CoT 不存在、验证失败、组合不足 300 或最终数量不是 6000 时会直接退出。

## 7. 论文发布版 MSRS-COT：补齐全部 21660 条

6000 条训练集与论文发布版 CoT 数据集是两个不同产物：

- AR 训练集：20 个组合各固定抽 300 条，共 6000 条，用于均衡训练；
- 论文发布版：20 个组合覆盖全部 1083 个 MSRS 场景，共 21660 条，不再抽样。

当前 1500 条分布式任务全部完成后，完整发布版仍缺 2681 条：

| 组合编号 | level | 红外退化 | 可见光退化 | 完成后已有 | 全量目标 | 仍需生成 |
|---|---|---|---|---:|---:|---:|
| C01 | level1 | noise | blur2 | 300 | 1083 | 783 |
| C17 | level2 | noise+stripe_noise | blur4+rain | 634 | 1083 | 449 |
| C18 | level2 | noise+stripe_noise | haze+noise | 600 | 1083 | 483 |
| C19 | level2 | noise+stripe_noise | haze+rain | 600 | 1083 | 483 |
| C20 | level2 | noise+stripe_noise | noise+rain | 600 | 1083 | 483 |
| 合计 |  |  |  |  |  | 2681 |

其余 C02–C16 已各有 1083 条，不再生成。过滤以
`(base_image, level, canonical IR degradation, canonical VIS degradation)` 为任务键，不依赖文件
路径根目录或 level2 退化字符串的先后顺序，因此不会与旧 CoT、当前 1500 条任务或 zeus9
任务重叠。

### 7.1 已生成的补洞清单和论文标签目录

运行入口：

```text
scripts/prepare_msrs_full_cot_dataset_completion.py
```

可复现命令：

```bash
cd /home/zhuyiming/data/lsy/ImageFusion-R1-mgpt2-omni

/home/zhuyiming/data/anaconda3/envs/qwen3vl/bin/python \
  scripts/prepare_msrs_full_cot_dataset_completion.py \
  --completed_inputs 'data/*with_fusedgt_golden_qwen3vl_cot.json' \
  --reserved_inputs data/msrs_supplement_5combos_300_each_seed20260620_pair_items.json \
  --output_dir data/msrs_cot_full_completion_seed20260622 \
  --expected_pending 2681 \
  --overwrite
```

`reserved_inputs` 是正在执行的固定 1500 条 manifest。即使其中一部分尚未生成，它们也会先被
占位排除，因此新任务不会和 A100/zeus9 当前任务重复。当前审计结果必须为：

```text
full_catalog=21660
covered_unique=18979
pending=2681
pending_overlaps_covered=false
```

产物：

```text
data/msrs_cot_full_completion_seed20260622/msrs_cot_full_catalog_21660.json
data/msrs_cot_full_completion_seed20260622/msrs_cot_completion_local_manifest.json
data/msrs_cot_full_completion_seed20260622/msrs_cot_completion_zeus9_manifest.json
data/msrs_cot_full_completion_seed20260622/completion_report.json
```

论文目录和补洞 manifest 为每条数据显式保存：

- `paper_dataset_id`：例如 `MSRS-COT-C17-S0001`；
- `paper_dataset_index`：1–21660 全局编号；
- `degradation_combo_id` / `degradation_combo_index`：C01–C20；
- `sample_index_within_combo`：1–1083；
- 原始与 canonical 红外/可见光退化标签；
- 红外/可见光退化 component 列表；
- 五张图路径、底图名和原训练 sample ID。

### 7.2 将 2681 条补洞任务发送到 zeus9

当前 1100 条接管任务完成后，在 Mac 执行：

```bash
scp -3 \
  <current-a100>:/home/zhuyiming/data/lsy/ImageFusion-R1-mgpt2-omni/data/msrs_cot_full_completion_seed20260622/msrs_cot_completion_zeus9_manifest.json \
  zeus9:/home/lushuyun/imagefusion-cot/data/
```

zeus9 使用独立输出文件启动，禁止复用当前 1100 条的 `zeus9_output.json`：

```bash
cd ~/imagefusion-cot

PYTHON_BIN="$CONDA_PREFIX/bin/python" \
MODEL_PATH="$HOME/models/qwen32" \
MANIFEST=data/msrs_cot_completion_zeus9_manifest.json \
OUTPUT_JSON=data/msrs_cot_completion_output.json \
LOG_PATH=outputs/msrs_cot_completion.log \
CUDA_VISIBLE_DEVICES=0,1 \
TP_SIZE=2 \
BSZ=4 \
KV_CACHE_MEMORY_GB=8 \
CPU_OFFLOAD_GB=0 \
GPU_MEMORY_UTILIZATION=0.80 \
MIN_FREE_MB=70000 \
MAX_SAMPLES=0 \
bash scripts/run_msrs_qwen3vl_cot_worker.sh
```

启动应显示 `input=2681 existing=0 pending=2681`。若中断，原命令重启会自动 resume。

### 7.3 拉回补洞结果并恢复 A100 本地路径

在 Mac 执行：

```bash
scp -3 \
  zeus9:/home/lushuyun/imagefusion-cot/data/msrs_cot_completion_output.json \
  <current-a100>:/home/zhuyiming/data/lsy/ImageFusion-R1-mgpt2-omni/data/msrs_cot_full_completion_seed20260622/
```

在 A100 严格检查 2681 个 ID，并从本地 manifest 恢复本地 MSRS 路径及论文标签：

```bash
cd /home/zhuyiming/data/lsy/ImageFusion-R1-mgpt2-omni

/home/zhuyiming/data/anaconda3/envs/qwen3vl/bin/python \
  scripts/merge_msrs_cot_worker_output.py \
  --manifest data/msrs_cot_full_completion_seed20260622/msrs_cot_completion_local_manifest.json \
  --worker_output data/msrs_cot_full_completion_seed20260622/msrs_cot_completion_output.json \
  --output data/msrs_cot_full_completion_seed20260622/msrs_cot_completion_2681_local.json

/home/zhuyiming/data/anaconda3/envs/qwen3vl/bin/python \
  scripts/validate_msrs_qwen3vl_cot.py \
  --input data/msrs_cot_full_completion_seed20260622/msrs_cot_completion_2681_local.json \
  --min_think_chars 100 \
  --check_paths
```

只有 `total=2681`、`unique_ids=2681`、`missing=0`、`incomplete=0` 且 validator 通过，
才能进入最终 21660 条论文数据集合并。发布版应同时保留原始
`raw_qwen_outputs`、规范化 stage CoT、显式退化标签、组合编号以及生成参数；训练使用的 6000 条
则继续由固定 seed 从发布候选中均衡抽取。

最终发布版合并命令：

```bash
/home/zhuyiming/data/anaconda3/envs/qwen3vl/bin/python \
  scripts/merge_msrs_full_cot_dataset.py \
  --catalog data/msrs_cot_full_completion_seed20260622/msrs_cot_full_catalog_21660.json \
  --inputs \
    'data/*with_fusedgt_golden_qwen3vl_cot.json' \
    data/msrs_cot_distributed_seed20260621/merged_1500.json \
    data/msrs_cot_full_completion_seed20260622/msrs_cot_completion_2681_local.json \
  --output data/msrs_cot_full_completion_seed20260622/msrs_cot_dataset_full_21660.json

/home/zhuyiming/data/anaconda3/envs/qwen3vl/bin/python \
  scripts/validate_msrs_qwen3vl_cot.py \
  --input data/msrs_cot_full_completion_seed20260622/msrs_cot_dataset_full_21660.json \
  --min_think_chars 100 \
  --check_paths
```

合并器按 canonical 任务键去重，而不是按旧文件中的 ID 字符串或路径根目录去重；若同一任务在
两个来源中出现但教师输出不同，会直接报冲突，不会静默任选一条。最终 report 必须满足：

```text
total=21660
unique_paper_dataset_ids=21660
combos=20
每个 C01-C20 均为 1083
missing=0
incomplete=0
```

## 8. 最终 conversation 与图像顺序

### 8.1 Human 输入

```text
Infrared degraded image: <|image|>
Visible degraded image: <|image|>
Please first analyze the infrared degradation and restore the clean infrared image.
Then analyze the visible degradation and restore the clean visible image.
Finally analyze how to fuse the restored infrared and visible information,
and generate the clean fused image.
```

### 8.2 GPT 监督输出

```text
<infrared_cot>
  <think>...</think>
  <answer>...</answer>
</infrared_cot>
<clean_infrared_image><|image|></clean_infrared_image>
<visible_cot>
  <think>...</think>
  <answer>...</answer>
</visible_cot>
<clean_visible_image><|image|></clean_visible_image>
<fused_cot>
  <think>...</think>
  <answer>...</answer>
</fused_cot>
<clean_fused_image><|image|></clean_fused_image>
```

内部 image list 的严格顺序为：

```text
0. infrared_degraded_path   输入
1. visible_degraded_path    输入
2. infrared_clean_path      输出 GT
3. visible_clean_path       输出 GT
4. fused_gt_path            输出 GT
```

前两张 degraded image 位于 human prompt，label 为 `-100`，不参与输出监督；GPT response 中的 CoT 文本和三张 clean target image token 均参与自回归 CE 监督。直接解码效果好的 demo PKL 已确认其监督顺序与上述结构一致。

## 9. PKL 生成

入口：

```text
scripts/pretokenize_msrs_final_6000_480x640.sh
```

运行：

```bash
CUDA_VISIBLE_DEVICES=0 \
bash scripts/pretokenize_msrs_final_6000_480x640.sh
```

该入口调用项目已有文件：

```text
imagefusion_r1/preprocess/pre_tokenize_mgpt2_ce.py
```

固定参数：

- 分辨率：480 × 640；
- tokenizer：`pretrained/Lumina-mGPT-2.0-Omni`；
- 最大序列长度：28672；
- 输出目录：`data/processed_mgpt2/msrs_ce_cot_final_6000_clean_480x640`；
- 使用 `--no_sanitize_cot`，因为最终 JSON 已经完成且通过幂等清洗，避免二次截断。

生成完成后脚本强制检查：

```text
record 行数 = 6000
record_len28672 行数 = 6000
PKL 文件数 = 6000
```

任一计数不为 6000 都会退出并提示“do not train”。特别是 `record_len28672` 少于 6000 时，说明存在过长序列，不能静默丢样本后直接训练，否则会破坏组合均衡。

## 10. 纯 CE 训练配置

训练入口：

```text
scripts/train_mgpt2_ce_cot_final_6000_5epoch.sh
```

数据配置：

```text
configs/sft/mgpt2_ce_cot_final_6000_480x640.yaml
```

默认 4 GPU 运行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 \
bash scripts/train_mgpt2_ce_cot_final_6000_5epoch.sh
```

默认训练参数：

| 参数 | 值 |
|---|---:|
| epochs | 5 |
| per-GPU batch size | 1 |
| gradient accumulation | 1 |
| 4-GPU global batch size | 4 |
| learning rate | 2e-5 |
| minimum learning rate | 0 |
| weight decay | 0 |
| precision | bf16 |
| gradient precision | fp32 |
| parallelism | FSDP |
| max sequence length | 28672 |
| z-loss weight | 0 |
| pixel MSE | 不使用 |

本实验只使用自回归 token CE。文本 token、三张目标图像的视觉 token、图像结构 token 权重均设置为 1.0，因此训练目标等价于对全部有效 GPT response token 求普通 CE：

```text
L_CE = - 1/N * Σ_t log p_theta(y_t | x, y_<t),  y_t != -100
```

其中 `x` 包括 degraded IR、degraded VIS 和指令，`y` 包括三段 CoT 以及 clean IR、clean VIS、clean fused 的视觉 token。human prompt 与输入图像 token 被 mask，不参与目标预测。

使用的 trainer 是：

```text
imagefusion_r1/trainers/finetune_solver_mgpt2_ce.py
```

不是带 pixel-MSE 的 `finetune_solver_mgpt2_ce_mse.py`。

## 11. 五个 epoch 与 checkpoint 选择

可以训练 5 个 epoch，但不要预先认定第 5 个最好。6000 条 item 仍然来自 1083 个底层场景，后期可能出现训练损失继续下降、测试恢复能力反而变差的情况。

训练脚本设置：

- `save_interval=1`：每个 epoch 保存一次；
- `ckpt_max_keep=0`：保留所有 epoch checkpoint；
- 默认 checkpoint 为 `epoch0` 到 `epoch4`，分别对应完成第 1 到第 5 个 epoch；
- 自动 resume 保持开启，任务中断后使用同一 output directory 重跑即可恢复；
- output directory：`outputs/mgpt2_ce_cot_final_6000_480x640_5epoch`。

用于后续 RL 的初始化 checkpoint 只能通过独立 validation scene 选择，而不是只看 train CE，也
不能使用最终官方 test 逐 epoch 挑选。建议在 validation 上逐 epoch 比较：

- clean infrared：PSNR、SSIM、LPIPS；
- clean visible：PSNR、SSIM、LPIPS；
- clean fused：与 fused GT 的 PSNR、SSIM、LPIPS，以及项目已有融合指标；
- conversation：tag 完整率、输出顺序正确率、三张图是否全部生成；
- 人工抽查：是否抑制输入退化、是否保留红外热目标和可见光结构/颜色、是否出现视觉 token 崩坏。

论文主结果使用 validation 预先选定的 checkpoint，在最终官方 test 上只评测一次。当前工程尚未
定义独立 validation scene，因此在 split 协议确定前不得用官方 test 选择 epoch。完整门禁和消融
协议见 `docs/MSRS_EVALUATION_PROTOCOL_ZH.md`。

## 12. 端到端执行顺序

```bash
# 1. zeus9 输出拉回后，严格合并并验证当前 1500 条补充 CoT
/home/zhuyiming/data/anaconda3/envs/qwen3vl/bin/python \
  scripts/merge_msrs_distributed_cot.py \
  --manifest data/msrs_supplement_5combos_300_each_seed20260620_pair_items.json \
  --completed_snapshot data/msrs_cot_distributed_seed20260621/completed_snapshot.json \
  --local_output data/msrs_cot_distributed_seed20260621/local_output.json \
  --zeus9_output data/msrs_cot_distributed_seed20260621/zeus9_output.json \
  --output data/msrs_cot_distributed_seed20260621/merged_1500.json

# 2. 构建并审计最终 6000 条 clean stage-CoT 训练 JSON
SUPPLEMENT=data/msrs_cot_distributed_seed20260621/merged_1500.json \
bash scripts/build_msrs_final_6000_training_items.sh

# 3. 生成并严格检查 6000 个 PKL
CUDA_VISIBLE_DEVICES=0 \
bash scripts/pretokenize_msrs_final_6000_480x640.sh

# 4. 训练 5 epoch 纯 CE checkpoint
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 \
bash scripts/train_mgpt2_ce_cot_final_6000_5epoch.sh

# 5. 论文发布版为独立支线：用第 7 节 manifest 在 zeus9 补齐 2681 条，最终汇总 21660 条
```

必须按顺序执行。上一步失败时不要绕过检查直接进入下一步。

## 13. 论文方法描述参考

可在论文中将该阶段概括为：

> 我们构建了一个面向红外–可见光联合恢复与融合的分层均衡自回归监督数据集。数据覆盖两个退化等级和 20 种跨模态退化组合，每种组合均匀采样 300 个样本，共计 6000 个训练 item。每个 item 包含退化红外图像、退化可见光图像，以及干净红外、干净可见光和干净融合图像三个目标。为降低退化类别与场景内容之间的偶然相关性，我们在组合内执行固定随机种子的无放回采样，并分别约束每个 degradation level 内的底图曝光次数。所有 split 均在退化合成前按干净 base scene 划分，保证 train、validation 和 test 之间不存在场景重叠。模型按照“红外分析与恢复—可见光分析与恢复—跨模态融合”的固定顺序，自回归生成阶段性 CoT 和三个目标图像 token 序列。训练阶段仅采用对有效响应 token 的自回归交叉熵，不额外引入像素 MSE；checkpoint 仅通过独立 validation 选择，官方 test 只用于最终报告。

清洗方法可描述为：

> 对教师模型生成的原始 CoT，我们执行标签规范化、最后有效字段提取、提示词污染去除、长度约束和幂等性检查。对于控制标签错位造成的残缺推理，仅基于该样本已验证的 degradation、understand 和 target-image 描述重构诊断文本，不额外引入图像事实。所有样本在 token 化前均通过路径一致性、退化成分覆盖、融合双模态完整性和 conversation 顺序检查。

## 14. 复现信息

正式实验至少记录：

- 数据 seed：`20260620`；
- Qwen 模型：`Qwen3-VL-32B-Thinking`；
- CoT generation temperature/top-p/top-k：0.2 / 0.8 / 20；
- CoT max tokens：1024；
- 是否提供 fused GT：是；
- SFT 输入尺寸：480 × 640；
- SFT max sequence length：28672；
- 训练 epoch：5；
- train/validation/test base-scene ID 文件及零交集审计报告；
- checkpoint 选择所用 validation 集和指标；
- final test 是否只评测一次；
- 最终 JSON report、filtered record JSONL 和训练 `args.json`。
