from __future__ import annotations

import os
import types
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from imagefusion_r1.rl.hf_compat import relax_huggingface_hub_upper_bound

relax_huggingface_hub_upper_bound()

from model.chameleon import ChameleonForConditionalGeneration


IMAGE_TOKEN_START = 155000
IMAGE_TOKEN_END = 171383
IMAGE_START_TOKEN_ID = 151665
IMAGE_END_TOKEN_ID = 151666


def rank0_print(*args, **kwargs):
    if int(os.environ.get("RANK", "0")) == 0:
        print(*args, **kwargs, flush=True)


def _unpack_input_ids(input_ids):
    if (
        isinstance(input_ids, (list, tuple))
        and len(input_ids) > 0
        and isinstance(input_ids[0], dict)
    ):
        input_ids = [item["input_ids"] for item in input_ids]
    if isinstance(input_ids, tuple):
        input_ids = list(input_ids)
    return input_ids


def _pad_for_chameleon_forward(model_self, input_ids, labels):
    input_ids = _unpack_input_ids(input_ids)
    if isinstance(labels, tuple):
        labels = list(labels)

    max_tokens = max(len(item) for item in input_ids)
    max_tokens = min(max_tokens, model_self.config.max_position_embeddings)

    input_ids = [item[:max_tokens] for item in input_ids]
    labels = [item[:max_tokens] for item in labels]

    input_ids = [item + [0] * (max_tokens - len(item)) for item in input_ids]
    labels = [item + [-100] * (max_tokens - len(item)) for item in labels]

    device = getattr(model_self, "device", None)
    if device is None:
        device = next(model_self.parameters()).device

    input_tensor = torch.tensor(input_ids, dtype=torch.int64, device=device)
    label_tensor = torch.tensor(labels, dtype=torch.int64, device=device)
    return input_tensor, label_tensor


def find_target_image_blocks(label_tensor_b: torch.Tensor) -> List[Tuple[int, int]]:
    starts = torch.where(label_tensor_b == IMAGE_START_TOKEN_ID)[0].detach().cpu().tolist()
    ends = torch.where(label_tensor_b == IMAGE_END_TOKEN_ID)[0].detach().cpu().tolist()

    blocks = []
    end_cursor = 0
    for start in starts:
        while end_cursor < len(ends) and ends[end_cursor] <= start:
            end_cursor += 1
        if end_cursor >= len(ends):
            break
        end = ends[end_cursor]
        content = label_tensor_b[start + 3 : end]
        if bool(((content >= IMAGE_TOKEN_START) & (content <= IMAGE_TOKEN_END)).any().item()):
            blocks.append((start, end))
        end_cursor += 1
    return blocks


def image_token_positions(label_tensor_b: torch.Tensor, block: Tuple[int, int]) -> torch.Tensor:
    start, end = block
    positions = torch.arange(start + 3, end, device=label_tensor_b.device, dtype=torch.long)
    labels = label_tensor_b.index_select(dim=0, index=positions)
    is_image = (labels >= IMAGE_TOKEN_START) & (labels <= IMAGE_TOKEN_END)
    return positions[is_image]


def _build_ce_maps(
    labels: torch.Tensor,
    target_order: Tuple[str, ...],
    image_weights: Dict[str, float],
    text_weight: float,
    image_structure_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    weight_map = torch.full(
        labels.shape,
        float(text_weight),
        dtype=torch.float32,
        device=labels.device,
    )
    target_map = torch.full(labels.shape, -1, dtype=torch.long, device=labels.device)

    for batch_index in range(labels.shape[0]):
        blocks = find_target_image_blocks(labels[batch_index])
        if len(blocks) < len(target_order):
            raise ValueError(
                f"Expected at least {len(target_order)} target image blocks, got {len(blocks)}."
            )

        for target_index, target_name in enumerate(target_order):
            block_start, block_end = blocks[target_index]
            block_positions = torch.arange(
                block_start,
                block_end + 1,
                device=labels.device,
                dtype=torch.long,
            )
            weight_map[batch_index, block_positions] = float(image_structure_weight)

            positions = image_token_positions(labels[batch_index], blocks[target_index])
            weight = float(image_weights[target_name])
            weight_map[batch_index, positions] = weight
            target_map[batch_index, positions] = target_index

    return weight_map, target_map


def _chunked_weighted_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    target_order: Tuple[str, ...],
    image_weights: Dict[str, float],
    text_weight: float,
    image_structure_weight: float,
    vocab_size: int,
    chunk_size: int = 1024,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    weight_map, target_map = _build_ce_maps(
        labels=labels,
        target_order=target_order,
        image_weights=image_weights,
        text_weight=text_weight,
        image_structure_weight=image_structure_weight,
    )

    shift_logits = logits[..., :-1, :].contiguous().view(-1, vocab_size)
    shift_labels = labels[..., 1:].contiguous().view(-1).to(shift_logits.device)
    shift_weights = weight_map[..., 1:].contiguous().view(-1).to(shift_logits.device)
    shift_target_map = target_map[..., 1:].contiguous().view(-1).to(shift_logits.device)

    valid = shift_labels != -100
    if int(valid.sum().item()) == 0:
        zero = shift_logits.sum() * 0.0
        metrics = {"ce_unweighted_loss": zero.detach()}
        for target_name in target_order:
            metrics[f"ce_image_{target_name}_loss"] = zero.detach()
            metrics[f"acc_image_{target_name}"] = zero.detach()
        return zero, metrics

    chunk_size = max(1, int(chunk_size))
    total_weighted_loss = shift_logits.sum() * 0.0
    total_weight = shift_logits.new_tensor(0.0)
    total_unweighted_loss = shift_logits.new_tensor(0.0)
    total_unweighted_count = 0
    target_loss_sums = [shift_logits.new_tensor(0.0) for _ in target_order]
    target_counts = [0 for _ in target_order]
    target_correct = [0 for _ in target_order]

    for start in range(0, shift_logits.shape[0], chunk_size):
        end = min(start + chunk_size, shift_logits.shape[0])
        chunk_labels = shift_labels[start:end]
        chunk_valid = chunk_labels != -100
        if not bool(chunk_valid.any().item()):
            continue

        chunk_logits = shift_logits[start:end]
        loss_vec = F.cross_entropy(
            chunk_logits,
            chunk_labels,
            ignore_index=-100,
            reduction="none",
        )
        valid_weights = shift_weights[start:end][chunk_valid]
        valid_loss = loss_vec[chunk_valid]
        total_weighted_loss = total_weighted_loss + (valid_loss * valid_weights).sum()
        total_weight = total_weight + valid_weights.sum()
        total_unweighted_loss = total_unweighted_loss + valid_loss.detach().sum()
        total_unweighted_count += int(chunk_valid.sum().item())

        chunk_target_map = shift_target_map[start:end]
        for target_index, _target_name in enumerate(target_order):
            target_mask = chunk_valid & (chunk_target_map == target_index)
            if not bool(target_mask.any().item()):
                continue
            target_logits = chunk_logits[target_mask]
            target_labels = chunk_labels[target_mask]
            target_loss_sums[target_index] = (
                target_loss_sums[target_index] + loss_vec[target_mask].detach().sum()
            )
            target_counts[target_index] += int(target_mask.sum().item())
            target_pred = target_logits.detach().argmax(dim=-1)
            target_correct[target_index] += int((target_pred == target_labels).sum().item())

    c_loss = total_weighted_loss / total_weight.clamp_min(1.0)
    metrics: Dict[str, torch.Tensor] = {
        "ce_unweighted_loss": total_unweighted_loss / max(1, total_unweighted_count),
    }
    for target_index, target_name in enumerate(target_order):
        count = max(1, target_counts[target_index])
        metrics[f"ce_image_{target_name}_loss"] = target_loss_sums[target_index] / count
        metrics[f"acc_image_{target_name}"] = logits.new_tensor(
            float(target_correct[target_index]) / count
        )
    return c_loss, metrics


class MGPT2WeightedCEComputer:
    def __init__(self, args, target_order: Tuple[str, ...], log_fn=rank0_print):
        self.args = args
        self.target_order = target_order
        self.log_fn = log_fn
        self.printed_debug = False
        self.image_weights = {
            "infrared": float(getattr(args, "ce_weight_infrared_image", 1.0)),
            "visible": float(getattr(args, "ce_weight_visible_image", 1.0)),
            "fused": float(getattr(args, "ce_weight_fused_image", 1.0)),
        }
        self.text_weight = float(getattr(args, "ce_weight_text", 1.0))
        self.image_structure_weight = float(getattr(args, "ce_weight_image_structure", 1.0))

    def forward_with_weighted_ce(self, model_self, input_ids=None, labels=None, training=True, **kwargs):
        input_tensor, label_tensor = _pad_for_chameleon_forward(model_self, input_ids, labels)
        result = ChameleonForConditionalGeneration.forward(
            model_self,
            input_ids=input_tensor,
            labels=None,
            use_cache=False,
            **kwargs,
        )
        logits = result.logits if hasattr(result, "logits") else result[0]
        c_loss, ce_metrics = _chunked_weighted_ce_loss(
            logits=logits,
            labels=label_tensor,
            target_order=self.target_order,
            image_weights=self.image_weights,
            text_weight=self.text_weight,
            image_structure_weight=self.image_structure_weight,
            vocab_size=model_self.config.vocab_size,
            chunk_size=int(getattr(self.args, "ce_loss_chunk_size", 1024)),
        )

        if not self.printed_debug:
            self.printed_debug = True
            self.log_fn(
                "[DEBUG_MGPT2_WEIGHTED_CE]",
                f"target_order={self.target_order}",
                f"image_weights={self.image_weights}",
                f"text_weight={self.text_weight}",
                f"image_structure_weight={self.image_structure_weight}",
                f"chunk_size={int(getattr(self.args, 'ce_loss_chunk_size', 1024))}",
            )

        additional_loss_dict = {
            metric_name: (metric_value.detach(), 0.0)
            for metric_name, metric_value in ce_metrics.items()
        }
        if getattr(model_self.config, "z_loss_weight", 0.0) > 0:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = label_tensor[..., 1:].contiguous()
            valid_mask = shift_labels >= 0
            z_loss = torch.logsumexp(shift_logits, dim=-1).pow(2)[valid_mask].mean()
            additional_loss_dict["z_loss"] = (z_loss, model_self.config.z_loss_weight)

        return c_loss, additional_loss_dict


def patch_model_forward_for_mgpt2_weighted_ce(
    model,
    args,
    target_order: Tuple[str, ...],
    log_fn=rank0_print,
) -> None:
    computer = MGPT2WeightedCEComputer(args=args, target_order=target_order, log_fn=log_fn)

    def forward_with_weighted_ce(model_self, input_ids=None, labels=None, training=True, **kwargs):
        return computer.forward_with_weighted_ce(
            model_self,
            input_ids=input_ids,
            labels=labels,
            training=training,
            **kwargs,
        )

    model.forward = types.MethodType(forward_with_weighted_ce, model)
