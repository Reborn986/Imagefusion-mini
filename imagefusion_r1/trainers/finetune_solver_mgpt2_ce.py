from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
LUMINA2_ROOT = REPO_ROOT / "third_party" / "lumina_mgpt_2"
LUMINA2_IMPL = LUMINA2_ROOT / "lumina_mgpt"

for import_path in (REPO_ROOT, LUMINA2_IMPL, LUMINA2_ROOT):
    import_path = str(import_path)
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

import torch
from accelerate import init_empty_weights

from imagefusion_r1.rl.hf_compat import relax_huggingface_hub_upper_bound

relax_huggingface_hub_upper_bound()

from imagefusion_r1.trainers.ce_weight_loss_mgpt2 import (
    patch_model_forward_for_mgpt2_weighted_ce,
    rank0_print,
)

from model import ChameleonXLLMXConfig, ChameleonXLLMXForConditionalGeneration
from xllmx.data.item_processor import ItemProcessorBase
from xllmx.solvers.finetune import FinetuneSolverBase


TARGET_ORDER = ("infrared", "visible", "fused")
FUSED_ONLY_TARGET_ORDER = ("fused",)


class ItemProcessor(ItemProcessorBase):
    def process_item(self, data_item: dict, training_mode=False) -> Tuple[List[int], List[int]]:
        assert training_mode

        if "token" in data_item and "label" in data_item:
            record = data_item
        else:
            if "file" not in data_item:
                raise KeyError(f"Expected data item to contain 'file', got keys={list(data_item)}")
            with open(data_item["file"], "rb") as f:
                record = pickle.load(f)

        tokens = record["token"]
        labels = record["label"]
        if len(tokens) != len(labels):
            raise ValueError(f"tokens length {len(tokens)} != labels length {len(labels)}")

        return tokens, labels

    def predict_item_token_length(self, data_item: dict) -> int:
        if "token" in data_item:
            return len(data_item["token"])
        if "len" in data_item:
            return int(data_item["len"])
        if "file" in data_item:
            with open(data_item["file"], "rb") as f:
                record = pickle.load(f)
            return len(record["token"])
        raise ValueError(f"Cannot predict token length from keys={list(data_item)}")


class Solver(FinetuneSolverBase):
    @classmethod
    def get_args_parser(cls):
        parser = super().get_args_parser()
        parser.add_argument("--max_seq_len", default=10240, type=int, help="max token length")
        parser.add_argument("--mask_image_logits", default=False)
        parser.add_argument("--unmask_image_logits", action="store_false", dest="mask_image_logits")
        parser.add_argument("--dropout", type=float, default=0.05)
        parser.add_argument("--z_loss_weight", type=float, default=1e-5)
        parser.add_argument("--model_size", type=str, default="7B", choices=["7B"])
        parser.add_argument("--ce_loss_chunk_size", type=int, default=1024)
        parser.add_argument(
            "--target_protocol",
            choices=("full", "fused_only"),
            default="full",
            help="Select three clean-image targets or only the clean fused-image target.",
        )
        parser.add_argument(
            "--ce_weight_infrared_image",
            type=float,
            default=1.0,
            help="CE weight for clean infrared target image tokens.",
        )
        parser.add_argument(
            "--ce_weight_visible_image",
            type=float,
            default=1.0,
            help="CE weight for clean visible target image tokens.",
        )
        parser.add_argument(
            "--ce_weight_fused_image",
            type=float,
            default=1.0,
            help="CE weight for fused target image tokens.",
        )
        parser.add_argument(
            "--ce_weight_text",
            type=float,
            default=1.0,
            help="CE weight for supervised non-image answer text tokens.",
        )
        parser.add_argument(
            "--ce_weight_image_structure",
            type=float,
            default=1.0,
            help="CE weight for target image start/grid/newline/end structure tokens.",
        )
        parser.add_argument(
            "--pretrained_name_or_path",
            type=str,
            default="pretrained/Lumina-mGPT-2.0-Omni",
            help="Local Lumina-mGPT-2.0-Omni checkpoint directory.",
        )
        return parser

    def _model_func(self, init_from: str) -> Tuple[ChameleonXLLMXForConditionalGeneration, None]:
        if self.global_rank == 0:
            model = ChameleonXLLMXForConditionalGeneration.from_pretrained(
                init_from,
                max_position_embeddings=self.args.max_seq_len,
                mask_image_logits=self.args.mask_image_logits,
                dropout=self.args.dropout,
                z_loss_weight=self.args.z_loss_weight,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
            )
        else:
            with init_empty_weights():
                config = ChameleonXLLMXConfig.from_pretrained(
                    init_from,
                    max_position_embeddings=self.args.max_seq_len,
                    mask_image_logits=self.args.mask_image_logits,
                    dropout=self.args.dropout,
                    z_loss_weight=self.args.z_loss_weight,
                    torch_dtype=torch.bfloat16,
                )
                model = ChameleonXLLMXForConditionalGeneration(config)

        if hasattr(model.model, "vqmodel"):
            del model.model.vqmodel

        target_order = (
            FUSED_ONLY_TARGET_ORDER
            if self.args.target_protocol == "fused_only"
            else TARGET_ORDER
        )
        patch_model_forward_for_mgpt2_weighted_ce(
            model,
            self.args,
            target_order=target_order,
            log_fn=rank0_print,
        )
        return model, None

    def _item_processor_func(self) -> ItemProcessorBase:
        return ItemProcessor()

    def _make_and_save_starting_point(self, save_path: str) -> None:
        model = ChameleonXLLMXForConditionalGeneration.from_pretrained(
            self.args.pretrained_name_or_path,
            max_position_embeddings=self.args.max_seq_len,
            mask_image_logits=self.args.mask_image_logits,
            dropout=self.args.dropout,
            z_loss_weight=self.args.z_loss_weight,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )

        if hasattr(model.model, "vqmodel"):
            del model.model.vqmodel

        model.save_pretrained(save_path, max_shard_size="10GB")


if __name__ == "__main__":
    args = Solver.get_args_parser().parse_args()
    solver = Solver(args)
    solver.run()
