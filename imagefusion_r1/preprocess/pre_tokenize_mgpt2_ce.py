from __future__ import annotations

import json
import math
import os
import pickle
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
LUMINA2_ROOT = REPO_ROOT / "third_party" / "lumina_mgpt_2"
LUMINA2_IMPL = LUMINA2_ROOT / "lumina_mgpt"

for import_path in (REPO_ROOT, LUMINA2_IMPL, LUMINA2_ROOT):
    import_path = str(import_path)
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from imagefusion_r1.rl.hf_compat import relax_huggingface_hub_upper_bound

relax_huggingface_hub_upper_bound()

from data.convertsation import Conversation
from data.item_processor import FlexARItemProcessor
from imagefusion_r1.preprocess.cot_sanitizer import sanitize_stage_cot
from xllmx.data.data_reader import read_general


NO_COT_FORBIDDEN_KEY_FRAGMENTS = (
    "cot",
    "think",
    "answer",
    "understand",
    "raw_qwen",
)
NO_COT_FORBIDDEN_TEXT_FRAGMENTS = (
    "<cot",
    "</cot",
    "<think",
    "</think",
    "<answer",
    "</answer",
    "infrared_cot",
    "visible_cot",
    "fused_cot",
    "_understand",
)


class MSRSCEItemProcessor(FlexARItemProcessor):
    """CE-only adapter for ImageFusion-R1 raw MSRS records on Lumina-mGPT 2.0."""

    def __init__(
        self,
        tokenizer: str,
        target_size: int = 256,
        target_height: int | None = None,
        target_width: int | None = None,
        device: str = "cuda",
        sanitize_cot: bool = True,
        output_protocol: str = "auto",
    ) -> None:
        if (target_height is None) != (target_width is None):
            raise ValueError("target_height and target_width must be provided together.")
        if output_protocol not in {
            "auto",
            "no_cot",
            "fused_only",
            "fusion_cot_three_images",
        }:
            raise ValueError(f"Unsupported output_protocol={output_protocol!r}")

        self.fixed_resize_hw = None
        self.sanitize_cot = sanitize_cot
        self.output_protocol = output_protocol
        if target_height is not None and target_width is not None:
            if target_height <= 0 or target_width <= 0:
                raise ValueError(
                    f"target_height/target_width must be positive, got {target_height}x{target_width}"
                )
            if target_height % 32 != 0 or target_width % 32 != 0:
                raise ValueError(
                    "target_height and target_width must be divisible by 32 for Chameleon grid tokens, "
                    f"got {target_height}x{target_width}"
                )
            self.fixed_resize_hw = (target_height, target_width)
            # The parent still needs a scalar target_size to initialize MoVQGAN.
            target_size = max(target_height, target_width)

        cwd = Path.cwd()
        try:
            # Official Lumina-mGPT 2.0 loads MoVQGAN from ./movqgan.
            os.chdir(LUMINA2_IMPL)
            super().__init__(
                tokenizer=str((REPO_ROOT / tokenizer).resolve()) if not Path(tokenizer).is_absolute() else tokenizer,
                conv_template=Conversation,
                target_size=target_size,
                device=device,
            )
        finally:
            os.chdir(cwd)

    @torch.no_grad()
    def process_image(self, image: Any) -> Dict[str, List[int]]:
        if self.fixed_resize_hw is None:
            return super().process_image(image)

        if isinstance(image, Image.Image):
            pil = image
        else:
            pil = Image.open(read_general(image))

        target_h, target_w = self.fixed_resize_hw
        pil = self._whiten_transparency(pil).resize((target_w, target_h), resample=Image.BICUBIC)

        w_grids = pil.size[0] // self.patch_size
        h_grids = pil.size[1] // self.patch_size
        h_latent = pil.size[1] // 8
        w_latent = pil.size[0] // 8

        image_toks = self.img_tokens_from_pil(pil).view(-1)
        full_image_toks = self.get_image_token(image_toks.reshape(h_latent, w_latent))
        new_line_id = self.token2id(self.new_line_token)
        full_image_toks = torch.cat(
            (
                full_image_toks,
                torch.ones(
                    h_latent,
                    1,
                    device=full_image_toks.device,
                    dtype=full_image_toks.dtype,
                )
                * new_line_id,
            ),
            dim=1,
        ).flatten()

        result_toks = [
            self.token2id(self.image_start_token),
            self.token2id(self.get_n_grids_token(h_grids)),
            self.token2id(self.get_n_grids_token(w_grids)),
            *full_image_toks.tolist(),
            self.token2id(self.image_end_token),
        ]
        return {"input_ids": result_toks, "labels": result_toks}

    @staticmethod
    def _validate_no_cot_clean(raw_item: Dict[str, Any]) -> None:
        bad_keys = [
            key
            for key in raw_item
            if any(fragment in key.lower() for fragment in NO_COT_FORBIDDEN_KEY_FRAGMENTS)
        ]
        if bad_keys:
            raise KeyError(
                "no_cot protocol requires GT-only raw items without CoT or text-reasoning keys; "
                f"forbidden keys found: {bad_keys}"
            )

        bad_values = []
        for key, value in raw_item.items():
            if not isinstance(value, str):
                continue
            lowered = value.lower()
            if any(fragment in lowered for fragment in NO_COT_FORBIDDEN_TEXT_FRAGMENTS):
                bad_values.append(key)
        if bad_values:
            raise ValueError(
                "no_cot protocol requires GT-only raw items without CoT-like text values; "
                f"forbidden text found in keys: {bad_values}"
            )

    def _validate(self, raw_item: Dict[str, Any]) -> None:
        required_path_keys = ["visible_degraded_path", "infrared_degraded_path", "fused_gt_path"]
        if self.output_protocol != "fused_only":
            required_path_keys.extend(["visible_clean_path", "infrared_clean_path"])
        missing = [key for key in required_path_keys if key not in raw_item]
        if missing:
            raise KeyError(f"raw_item missing required image path keys: {missing}")

        if self.output_protocol == "no_cot":
            self._validate_no_cot_clean(raw_item)
            return

        if self.output_protocol == "fused_only":
            if not raw_item.get("final_cot"):
                raise KeyError("fused_only protocol requires non-empty 'final_cot'")
            return

        has_stage_cot = all(
            raw_item.get(key)
            for key in [
                "infrared_cot",
                "visible_cot",
                "fused_cot",
            ]
        )
        has_final_cot = bool(raw_item.get("final_cot"))
        has_understand = all(
            raw_item.get(key)
            for key in [
                "infrared_understand",
                "visible_understand",
                "fused_understand",
            ]
        )

        if not (has_stage_cot or has_final_cot or has_understand):
            raise KeyError(
                "raw_item must contain either stage CoT keys "
                "['infrared_cot', 'visible_cot', 'fused_cot'], "
                "'final_cot', or legacy understand keys "
                "['infrared_understand', 'visible_understand', 'fused_understand']"
            )

    @staticmethod
    def _clean_text(value: Any) -> str:
        return str(value).strip()

    def _stage_cot_text(self, raw_item: Dict[str, Any], stage: str) -> str:
        key = f"{stage}_cot"
        value = raw_item[key]
        if self.sanitize_cot:
            return sanitize_stage_cot(value, stage)
        return self._clean_text(value)

    def _build_gpt_value(self, raw_item: Dict[str, Any]) -> str:
        if self.output_protocol == "fusion_cot_three_images":
            return (
                "<clean_infrared_image><|image|></clean_infrared_image>\n"
                "<clean_visible_image><|image|></clean_visible_image>\n"
                "<fused_cot>\n"
                f"{self._clean_text(raw_item['final_cot'])}\n"
                "</fused_cot>\n"
                "<clean_fused_image><|image|></clean_fused_image>"
            )

        if self.output_protocol == "fused_only":
            return (
                "<fused_cot>\n"
                f"{self._clean_text(raw_item['final_cot'])}\n"
                "</fused_cot>\n"
                "<clean_fused_image><|image|></clean_fused_image>"
            )

        if self.output_protocol == "no_cot":
            return (
                "<clean_infrared_image><|image|></clean_infrared_image>\n"
                "<clean_visible_image><|image|></clean_visible_image>\n"
                "<clean_fused_image><|image|></clean_fused_image>"
            )

        if all(raw_item.get(key) for key in ["infrared_cot", "visible_cot", "fused_cot"]):
            return (
                "<infrared_cot>\n"
                f"{self._stage_cot_text(raw_item, 'infrared')}\n"
                "</infrared_cot>\n"
                "<clean_infrared_image><|image|></clean_infrared_image>\n"
                "<visible_cot>\n"
                f"{self._stage_cot_text(raw_item, 'visible')}\n"
                "</visible_cot>\n"
                "<clean_visible_image><|image|></clean_visible_image>\n"
                "<fused_cot>\n"
                f"{self._stage_cot_text(raw_item, 'fused')}\n"
                "</fused_cot>\n"
                "<clean_fused_image><|image|></clean_fused_image>"
            )

        if raw_item.get("final_cot"):
            return (
                "<cot>\n"
                f"{self._clean_text(raw_item['final_cot'])}\n"
                "</cot>\n"
                "<clean_infrared_image><|image|></clean_infrared_image>\n"
                "<clean_visible_image><|image|></clean_visible_image>\n"
                "<clean_fused_image><|image|></clean_fused_image>"
            )

        return (
            "<infrared_understand>\n"
            f"{self._clean_text(raw_item['infrared_understand'])}\n"
            "</infrared_understand>\n"
            "<clean_infrared_image><|image|></clean_infrared_image>\n"
            "<visible_understand>\n"
            f"{self._clean_text(raw_item['visible_understand'])}\n"
            "</visible_understand>\n"
            "<clean_visible_image><|image|></clean_visible_image>\n"
            "<fused_understand>\n"
            f"{self._clean_text(raw_item['fused_understand'])}\n"
            "</fused_understand>\n"
            "<clean_fused_image><|image|></clean_fused_image>"
        )

    def _build_human_value(self) -> str:
        if self.output_protocol == "fusion_cot_three_images":
            return (
                "Infrared degraded image: <|image|>\n"
                "Visible degraded image: <|image|>\n"
                "Restore the clean infrared image first, then restore the clean visible image. "
                "After obtaining both restored modalities, analyze how to fuse their complementary "
                "information while suppressing both degradations, and finally generate the clean fused image."
            )

        if self.output_protocol == "fused_only":
            return (
                "Infrared degraded image: <|image|>\n"
                "Visible degraded image: <|image|>\n"
                "Analyze how to fuse the complementary infrared and visible information while "
                "suppressing both degradations, then generate the clean fused image."
            )

        if self.output_protocol == "no_cot":
            return (
                "Infrared degraded image: <|image|>\n"
                "Visible degraded image: <|image|>\n"
                "Restore the clean infrared image first, then restore the clean visible image, "
                "and finally generate the clean fused image. Respond only with the three image "
                "outputs in this exact order: clean infrared, clean visible, clean fused."
            )

        return (
            "Infrared degraded image: <|image|>\n"
            "Visible degraded image: <|image|>\n"
            "Please first analyze the infrared degradation and restore the clean infrared image. "
            "Then analyze the visible degradation and restore the clean visible image. "
            "Finally analyze how to fuse the restored infrared and visible information, "
            "and generate the clean fused image."
        )

    def process_item(
        self,
        raw_item: Dict[str, Any],
        training_mode: bool = False,
        out_flatten: bool = True,
    ):
        self._validate(raw_item)

        human_value = self._build_human_value()
        gpt_value = self._build_gpt_value(raw_item)

        image_paths = [
            raw_item["infrared_degraded_path"],
            raw_item["visible_degraded_path"],
        ]
        if self.output_protocol == "fused_only":
            image_paths.append(raw_item["fused_gt_path"])
        else:
            image_paths.extend(
                [
                    raw_item["infrared_clean_path"],
                    raw_item["visible_clean_path"],
                    raw_item["fused_gt_path"],
                ]
            )

        item = {
            "conversations": [
                {"from": "human", "value": human_value},
                {"from": "gpt", "value": gpt_value},
            ],
            "image": image_paths,
        }
        return super().process_item(
            item,
            training_mode=training_mode,
            out_flatten=out_flatten,
        )


def load_raw_items(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
        return items

    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    raise ValueError(f"Unsupported input suffix: {path.suffix}")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--in_filename", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="pretrained/Lumina-mGPT-2.0-Omni",
        help="Local Lumina-mGPT-2.0-Omni tokenizer/model directory.",
    )
    parser.add_argument("--target_size", type=int, default=256)
    parser.add_argument(
        "--target_height",
        type=int,
        default=None,
        help="Optional fixed output height. Use with --target_width to keep a non-square image size.",
    )
    parser.add_argument(
        "--target_width",
        type=int,
        default=None,
        help="Optional fixed output width. Use with --target_height to keep a non-square image size.",
    )
    parser.add_argument("--max_seq_len", type=int, default=10240)
    parser.add_argument("--splits", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no_sanitize_cot",
        action="store_true",
        help="Disable CoT tag cleanup. By default stage CoTs are canonicalized before tokenization.",
    )
    parser.add_argument(
        "--output_protocol",
        choices=("auto", "no_cot", "fused_only", "fusion_cot_three_images"),
        default="auto",
        help=(
            "auto keeps the existing three-target behavior; no_cot emits three image targets "
            "without CoT; fused_only emits fused CoT followed by only the clean fused image."
        ),
    )
    args = parser.parse_args()
    effective_target_size = (
        max(args.target_height, args.target_width)
        if args.target_height is not None and args.target_width is not None
        else args.target_size
    )

    in_path = Path(args.in_filename)
    out_dir = Path(args.out_dir)
    save_dir = out_dir / "files"

    if args.overwrite and out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)

    save_dir.mkdir(parents=True, exist_ok=True)

    raw_items = load_raw_items(in_path)
    num_per_rank = math.ceil(len(raw_items) / args.splits)
    start = num_per_rank * args.rank
    end = min(num_per_rank * (args.rank + 1), len(raw_items))

    processor = MSRSCEItemProcessor(
        tokenizer=args.tokenizer,
        target_size=effective_target_size,
        target_height=args.target_height,
        target_width=args.target_width,
        device=args.device,
        sanitize_cot=not args.no_sanitize_cot,
        output_protocol=args.output_protocol,
    )

    record_path = out_dir / f"{args.rank}-of-{args.splits}-record.jsonl"
    filtered_record_path = out_dir / f"{args.rank}-of-{args.splits}-record_len{args.max_seq_len}.jsonl"
    progress_path = out_dir / f"{args.rank}-of-{args.splits}-progress.txt"

    kept = 0
    total = 0
    for idx in range(start, end):
        raw_item = raw_items[idx]
        sample_id = str(raw_item.get("id", idx))
        pkl_path = save_dir / f"{sample_id}.pkl"
        total += 1

        try:
            tokens, labels = processor.process_item(raw_item, training_mode=True)
            source_paths = {
                "infrared_degraded": raw_item["infrared_degraded_path"],
                "visible_degraded": raw_item["visible_degraded_path"],
                "fused_gt": raw_item["fused_gt_path"],
            }
            if args.output_protocol != "fused_only":
                source_paths.update(
                    {
                        "infrared_clean": raw_item["infrared_clean_path"],
                        "visible_clean": raw_item["visible_clean_path"],
                    }
                )

            record = {
                "token": tokens,
                "label": labels,
                "id": sample_id,
                "source_paths": source_paths,
                "output_protocol": args.output_protocol,
            }
            with open(pkl_path, "wb") as f:
                pickle.dump(record, f)

            line = {
                "file": str(pkl_path),
                "len": len(tokens),
                "id": sample_id,
                "target_size": effective_target_size,
                "target_height": args.target_height,
                "target_width": args.target_width,
            }
            with open(record_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
            if len(tokens) <= args.max_seq_len:
                with open(filtered_record_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
                kept += 1
        except Exception:
            import traceback

            print(f"[ERROR] item idx={idx} id={sample_id}")
            traceback.print_exc()

        if idx % 10 == 0:
            print(f"[PROGRESS] idx={idx}/{end} kept={kept}/{total}", flush=True)

        progress_path.write_text("finished" if idx == end - 1 else str(idx), encoding="utf-8")

    print(
        f"[DONE] rank={args.rank}/{args.splits} total={total} kept_len<={args.max_seq_len}: {kept}",
        flush=True,
    )


if __name__ == "__main__":
    main()
