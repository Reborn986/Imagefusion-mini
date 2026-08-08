# MSRS 纯 CE 数据预处理、Conversation 与 PKL 协议

本文固定 MSRS 自回归恢复与融合训练从 clean JSON 到 Omni PKL 的唯一正式协议。训练目标是
CoT 文本 token 和三张干净目标图像的离散 VQ token；退化输入只作为条件，不参与监督。本文所述
PKL 不包含 pixel-MSE target，也不得交给 MSE trainer。

## 1. 正式输入与门禁

当前首选训练输入：

```text
dataset_final/MSRS/training_seed20260620_levelbalanced_v2/
  msrs_raw_items_cot_final_6000_clean_seed20260620_levelbalanced_v2.json
```

选择和 CoT 审计：

```text
msrs_raw_items_cot_final_6000_clean_seed20260620_levelbalanced_v2.report.json
msrs_raw_items_cot_final_6000_clean_seed20260620_levelbalanced_v2.cot_audit.json
seed_config.json
```

固定属性：

- seed：`20260620`；
- 20 个退化组合各 300 条；
- level1/level2 各 3000 条；
- 6000 个 ID 和实际退化图像对均唯一；
- 每个 level 内 1083 个 base scene 各出现 2–3 次；
- 全局每个 base scene 各出现 5–6 次；
- 三阶段 CoT 已通过结构、退化 component、融合语义、污染、路径和清洗幂等性审计。

PKL 生成使用 `--no_sanitize_cot`，因为 JSON 已经冻结并证明幂等。此参数不是跳过质量控制，而是
避免 tokenizer 阶段再次改变已经审计和记录哈希的文本。

## 2. 每条 clean JSON 的语义

每条训练 item 至少包含：

```text
id
infrared_degraded_path
visible_degraded_path
infrared_clean_path
visible_clean_path
fused_gt_path
infrared_cot
visible_cot
fused_cot
```

五张图属于同一个 base scene。IR/VIS degradation level 必须一致。

输入条件：

```text
infrared_degraded_path
visible_degraded_path
```

自回归 GT：

```text
infrared_cot + infrared_clean_path 的 VQ token
visible_cot  + visible_clean_path 的 VQ token
fused_cot    + fused_gt_path 的 VQ token
```

## 3. Conversation 是什么

这里的 `conversation` 不是额外的数据文件，也不是普通聊天记录。它是 Omni
`FlexARItemProcessor` 在 token 化前使用的两角色多模态监督结构：

```python
{
    "conversations": [
        {"from": "human", "value": human_value},
        {"from": "gpt", "value": gpt_value},
    ],
    "image": [path0, path1, path2, path3, path4],
}
```

Omni 的 `Conversation` 模板把 `human` 映射为 `Human`，把 `gpt` 映射为 `Assistant`，并在
每个 turn 末尾加入 `<reserved08706>`。模板同时产生 `predict=False/True` 标记：

- Human turn：`predict=False`，对应 label `-100`；
- Assistant turn：`predict=True`，对应真实监督 label。

因此 conversation 的作用是同时定义：

1. 模型看到的完整自回归序列；
2. 五个 `<|image|>` 按什么顺序替换成图像 token；
3. 哪些 token 是输入条件，哪些 token 是预测目标。

实现：

```text
third_party/lumina_mgpt_2/lumina_mgpt/data/convertsation.py
imagefusion_r1/preprocess/pre_tokenize_mgpt2_ce.py
```

## 4. Human 输入模板

精确内容：

```text
Infrared degraded image: <|image|>
Visible degraded image: <|image|>
Please first analyze the infrared degradation and restore the clean infrared image.
Then analyze the visible degradation and restore the clean visible image.
Finally analyze how to fuse the restored infrared and visible information,
and generate the clean fused image.
```

两个 image placeholder 消费 `image` list 的前两张退化图。

## 5. Assistant/GPT 监督模板

精确顺序：

```xml
<infrared_cot>
  <think>...</think>
  <answer>
    <infrared_degradation>...</infrared_degradation>
    <infrared_understand>...</infrared_understand>
    <infrared_image>...</infrared_image>
  </answer>
</infrared_cot>
<clean_infrared_image><|image|></clean_infrared_image>

<visible_cot>
  <think>...</think>
  <answer>
    <visible_degradation>...</visible_degradation>
    <visible_understand>...</visible_understand>
    <visible_image>...</visible_image>
  </answer>
</visible_cot>
<clean_visible_image><|image|></clean_visible_image>

<fused_cot>
  <think>...</think>
  <answer>
    <fused_understand>...</fused_understand>
    <fused_image>...</fused_image>
  </answer>
</fused_cot>
<clean_fused_image><|image|></clean_fused_image>
```

三个 Assistant image placeholder 消费后三张干净 GT。

## 6. 五张图的不可变顺序

`image` list 顺序固定为：

| index | 路径 | 角色 | label |
|---:|---|---|---|
| 0 | `infrared_degraded_path` | Human 输入 | 全部 `-100` |
| 1 | `visible_degraded_path` | Human 输入 | 全部 `-100` |
| 2 | `infrared_clean_path` | Assistant 输出 GT | VQ token 参与 CE |
| 3 | `visible_clean_path` | Assistant 输出 GT | VQ token 参与 CE |
| 4 | `fused_gt_path` | Assistant 输出 GT | VQ token 参与 CE |

禁止交换 IR/VIS 或三张目标图的顺序。输入图即使也被 VQ 编码进入 `token`，其 `label` 仍全为
`-100`，不会要求模型重建输入退化图。

## 7. Omni 文本与图像编码

生成入口：

```text
imagefusion_r1/preprocess/pre_tokenize_mgpt2_ce.py
```

Omni 依赖：

```text
third_party/lumina_mgpt_2/
pretrained/Lumina-mGPT-2.0-Omni
```

文本：使用本地 Lumina-mGPT-2.0-Omni tokenizer 编码 conversation 字符串。

图像：使用 Omni 自带 270M MoVQGAN/VQ encoder，将像素编码成离散图像 token。正式尺寸固定为：

```text
height = 480
width  = 640
```

480 和 640 均可被 32 整除：

```text
grid = 15 × 20
VQ latent = 60 × 80
```

每行 latent token 后加入 image newline token，整个图像块再加入：

```text
<racm3:break>       image start
height grid token
width grid token
VQ image tokens + row separators
<eoss>              image end
```

当前 checkpoint 中：

```text
image_start_token_id = 151665
image_end_token_id   = 151666
```

注意：PKL 保存的是编码后的离散 token ID，不是“decode 后的图片”，也不是图像像素 tensor。
需要可视化时才能用同一 Omni VQ decoder 将目标 image token 解码回图像。

## 8. Token 与 Label

`token` 是完整 flattened sequence：

```text
Human instruction
+ degraded IR image tokens
+ degraded VIS image tokens
+ infrared CoT tokens
+ clean IR image tokens
+ visible CoT tokens
+ clean VIS image tokens
+ fused CoT tokens
+ clean fused image tokens
```

`label` 与 `token` 等长：

```text
Human text                         -> -100
degraded IR/VIS image tokens       -> -100
Assistant CoT text tokens          -> token ID
clean IR/VIS/fused image tokens    -> token ID
Assistant image structure tokens   -> token ID
```

训练计算标准 next-token autoregressive CE，只在 `label != -100` 的位置计算损失。

## 9. PKL 实际 schema

每个 CE-only PKL 是一个 Python dict：

```python
{
    "id": "sample_id",
    "token": [int, ...],
    "label": [int, ...],
    "source_paths": {
        "infrared_degraded": "...",
        "visible_degraded": "...",
        "infrared_clean": "...",
        "visible_clean": "...",
        "fused_gt": "...",
    },
}
```

以下字段必须不存在：

```text
targets
pixel_mse
pixel tensors
```

CoT 原字符串不会作为独立字段重复保存在 PKL；它已经进入 `token/label`。原始 clean JSON 和
`source_paths` 用于审计与追溯。

## 10. 纯 CE 训练入口

正式 trainer：

```text
imagefusion_r1/trainers/finetune_solver_mgpt2_ce.py
```

该 trainer 从 PKL 只读取：

```python
tokens = record["token"]
labels = record["label"]
```

模型加载后删除 VQ encoder，因为训练阶段图像已经离散 token 化，不再需要在线编码像素。

正式训练脚本：

```text
scripts/train_mgpt2_ce_cot_final_6000_5epoch.sh
```

权重：

```text
ce_weight_text = 1.0
ce_weight_infrared_image = 1.0
ce_weight_visible_image = 1.0
ce_weight_fused_image = 1.0
ce_weight_image_structure = 1.0
z_loss_weight = 0
```

明确禁止用于本实验：

```text
imagefusion_r1/trainers/finetune_solver_mgpt2_ce_mse.py
scripts/train_mgpt2_ce_mse_cot_480x640_b200.sh
pixel_mse_loss_mgpt2.py
```

## 11. 两条 smoke 的实际验收

目录：

```text
dataset_final/MSRS/training_seed20260620_levelbalanced_v2/omni_pkl_smoke2_480x640
```

结果：

```text
record_rows = 2
filtered_rows = 2
pkl_files = 2
errors = []
passed = true
sequence_length = 25500–25525
supervised_labels = 15713–15738
```

逐 PKL 读取确认：

- keys 只有 `id/token/label/source_paths`；
- token 与 label 等长；
- 五个图像块全部进入 token；
- 只有后三个干净目标图像块进入监督 label；
- `targets` 不存在；
- `pixel_mse` 不存在。

检查入口：

```text
scripts/check_msrs_ce_pkl.py
```

## 12. 全量 PKL 生成

建议在 tmux 中执行：

```bash
cd /home/zhuyiming/data/lsy/ImageFusion-R1-mgpt2-omni

CUDA_VISIBLE_DEVICES=2 \
PYTHON_BIN=/home/zhuyiming/data/anaconda3/envs/lumina-mgpt2/bin/python \
RAW_ITEMS=dataset_final/MSRS/training_seed20260620_levelbalanced_v2/msrs_raw_items_cot_final_6000_clean_seed20260620_levelbalanced_v2.json \
OUT_DIR=dataset_final/MSRS/training_seed20260620_levelbalanced_v2/omni_pkl_final_6000_480x640 \
EXPECTED_ITEMS=6000 \
DEVICE=cuda \
bash scripts/pretokenize_msrs_final_6000_480x640.sh
```

该入口使用 `--overwrite`。中断后重新运行会删除同名输出目录并从头生成，因此必须在 tmux 中运行，
不要误按 `Ctrl-C`。

生成时同时写：

```text
files/*.pkl
0-of-1-record.jsonl
0-of-1-record_len28672.jsonl
0-of-1-progress.txt
```

只有长度不超过 `28672` 的样本进入 filtered record。正式门禁要求：

```text
record rows = 6000
filtered record rows = 6000
PKL files = 6000
```

随后执行：

```bash
python3 scripts/check_msrs_ce_pkl.py \
  --processed_dir dataset_final/MSRS/training_seed20260620_levelbalanced_v2/omni_pkl_final_6000_480x640 \
  --expected 6000 \
  --max_seq_len 28672 \
  --report dataset_final/MSRS/training_seed20260620_levelbalanced_v2/omni_pkl_final_6000_480x640/check_report.json
```

只有 `errors=[]` 且 `passed=true` 才能进入训练。

## 13. 不可混用的概念

- `conversation`：token 化前定义角色、顺序和监督 mask 的逻辑结构；
- `token`：Human+Assistant 文本和五张图的完整离散序列；
- `label`：与 token 等长的 CE 目标，输入位置为 `-100`；
- CoT GT：Assistant 中三段 CoT 的文本 token；
- image GT：Assistant 中三张干净图的 VQ token；
- `source_paths`：审计元数据，不直接进入 loss；
- decode：仅用于将 token 还原为可读文本/图像做检查，不是 PKL 生成步骤；
- pixel MSE：本实验不使用。
