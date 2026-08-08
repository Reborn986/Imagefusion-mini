from __future__ import annotations

import unittest

import torch

from imagefusion_r1.inference.inference_solver_mgpt2_ce import FixedGridMultiModalLogitsProcessor


class _DummyTokenizer:
    def __init__(self) -> None:
        self._encoded = {
            "<clean_infrared_image>": [10, 11],
            "<clean_visible_image>": [12, 13],
            "<clean_fused_image>": [14, 15],
        }

    def encode_wo_prefix_space(self, text: str) -> list[int]:
        return list(self._encoded[text])


class _DummyItemProcessor:
    image_start_token = "<image_start>"
    image_end_token = "<image_end>"
    new_line_token = "<new_line>"

    def __init__(self) -> None:
        self.tokenizer = _DummyTokenizer()
        self._ids = {
            self.image_start_token: 100,
            self.image_end_token: 101,
            self.new_line_token: 102,
            self.get_n_grids_token(2): 200,
            self.get_n_grids_token(3): 201,
        }

    @staticmethod
    def get_n_grids_token(n_grids: int) -> str:
        return f"<grid_{n_grids}>"

    def token2id(self, token: str) -> int:
        return self._ids[token]


class FixedGridMultiModalLogitsProcessorTest(unittest.TestCase):
    def _processor(self) -> FixedGridMultiModalLogitsProcessor:
        item_processor = _DummyItemProcessor()
        return FixedGridMultiModalLogitsProcessor(
            item_processor=item_processor,
            image_start_token_id=item_processor.token2id(item_processor.image_start_token),
            image_end_token_id=item_processor.token2id(item_processor.image_end_token),
            image_next_line_token_id=item_processor.token2id(item_processor.new_line_token),
            voc_size=171500,
            h_grids=2,
            w_grids=3,
        )

    def test_image_start_only_allowed_after_clean_image_open_tag(self) -> None:
        processor = self._processor()
        scores = torch.zeros((1, 171500), dtype=torch.float32)

        text_context = torch.tensor([[1, 2, 3]], dtype=torch.long)
        constrained = processor(text_context, scores)
        self.assertTrue(torch.isneginf(constrained[0, 100]))
        self.assertTrue(torch.isneginf(constrained[0, 155000]))
        self.assertEqual(float(constrained[0, 42].item()), 0.0)

        image_open_context = torch.tensor([[1, 10, 11]], dtype=torch.long)
        constrained = processor(image_open_context, scores)
        self.assertEqual(float(constrained[0, 100].item()), 0.0)
        self.assertTrue(torch.isneginf(constrained[0, 155000]))

    def test_inside_image_block_still_forces_grid_tokens(self) -> None:
        processor = self._processor()
        scores = torch.zeros((1, 171500), dtype=torch.float32)
        image_start_context = torch.tensor([[10, 11, 100]], dtype=torch.long)

        constrained = processor(image_start_context, scores)
        self.assertEqual(float(constrained[0, 200].item()), 0.0)
        self.assertTrue(torch.isneginf(constrained[0, 42]))


if __name__ == "__main__":
    unittest.main()
