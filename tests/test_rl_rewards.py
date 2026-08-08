from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from imagefusion_r1.rl.msrs_two_level_reward import (
    ImageLevelRewardResult,
    score_format_gate,
    score_two_level_reward,
)
from imagefusion_r1.rl.reference_image_reward import (
    ThreeImageReferenceReward,
    score_reference_pair,
    score_three_image_reference,
)


VALID_FULL_TEXT = """<infrared_cot>
<think>Suppress noise while preserving infrared thermal targets.</think>
<answer>
<infrared_degradation>noise and stripe noise</infrared_degradation>
<infrared_understand>Preserve infrared thermal contrast and structure.</infrared_understand>
<infrared_image>Restore a clean infrared image.</infrared_image>
</answer>
</infrared_cot>
<clean_infrared_image><|image|></clean_infrared_image>
<visible_cot>
<think>Remove haze while preserving visible color, texture, and edges.</think>
<answer>
<visible_degradation>haze</visible_degradation>
<visible_understand>Preserve visible scene layout and detail.</visible_understand>
<visible_image>Restore a clean visible image.</visible_image>
</answer>
</visible_cot>
<clean_visible_image><|image|></clean_visible_image>
<fused_cot>
<think>Fuse complementary infrared saliency and visible structure and suppress artifacts.</think>
<answer>
<fused_understand>Balance infrared targets with visible color and texture.</fused_understand>
<fused_image>Generate a clean natural fusion.</fused_image>
</answer>
</fused_cot>
<clean_fused_image><|image|></clean_fused_image>"""


class ReferenceRewardTest(unittest.TestCase):
    def test_identical_pair_scores_one_and_collapse_is_low(self) -> None:
        reference = Image.fromarray(np.full((32, 48, 3), 220, dtype=np.uint8))
        identical = score_reference_pair(reference, reference)
        collapsed = score_reference_pair(
            Image.fromarray(np.zeros((32, 48, 3), dtype=np.uint8)),
            reference,
        )
        self.assertAlmostEqual(identical.score, 1.0, places=6)
        self.assertGreater(identical.psnr, 40.0)
        self.assertLess(collapsed.score, 0.2)

    def test_three_outputs_map_to_their_own_gt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = [
                Image.fromarray(np.full((24, 32, 3), value, dtype=np.uint8))
                for value in (40, 120, 200)
            ]
            keys = ("infrared_clean_path", "visible_clean_path", "fused_gt_path")
            sample = {}
            for key, image in zip(keys, images):
                path = root / f"{key}.png"
                image.save(path)
                sample[key] = str(path)
            result = score_three_image_reference(images, sample)
            self.assertAlmostEqual(result.score, 1.0, places=6)
            swapped = score_three_image_reference([images[1], images[0], images[2]], sample)
            self.assertLess(swapped.score, result.score)


class TwoLevelRewardTest(unittest.TestCase):
    def test_full_protocol_uses_strict_gate(self) -> None:
        images = [object(), object(), object()]
        valid = score_format_gate(
            {"generated_text": VALID_FULL_TEXT, "generated_images": images},
            protocol="full",
        )
        malformed = score_format_gate(
            {
                "generated_text": "<infrared_cot><clean_infrared_image><|image|>"
                "<visible_cot><clean_visible_image><|image|>"
                "<fused_cot><clean_fused_image><|image|>",
                "generated_images": images,
            },
            protocol="full",
        )
        self.assertTrue(valid.ok)
        self.assertEqual(valid.score, 1.0)
        self.assertFalse(malformed.ok)
        self.assertEqual(malformed.score, 0.0)

    def test_reference_remains_primary_when_qwen_is_skipped(self) -> None:
        completion = {
            "generated_text": VALID_FULL_TEXT,
            "generated_images": [object(), object(), object()],
        }
        reward = score_two_level_reward(
            completion,
            sample={"infrared_label": "noise+stripe_noise", "visible_label": "haze"},
            image_reward=ImageLevelRewardResult(score=0.0, error="image_reward_skipped"),
            reference_reward=0.8,
            protocol="full",
        )
        self.assertTrue(reward.format_gate.ok)
        self.assertAlmostEqual(reward.combined_image_score, 0.8, places=6)
        self.assertGreater(reward.score, 0.7)

    def test_reference_fallback_on_qwen_server_error(self) -> None:
        completion = {
            "generated_text": VALID_FULL_TEXT,
            "generated_images": [object(), object(), object()],
        }
        reward = score_two_level_reward(
            completion,
            sample={"infrared_label": "noise+stripe_noise", "visible_label": "haze"},
            image_reward=ImageLevelRewardResult(score=0.0, error="URLError: connection refused"),
            reference_reward=ThreeImageReferenceReward(score=0.72),
            protocol="full",
            reference_weight=0.9,
            qwen_weight=0.1,
        )
        self.assertAlmostEqual(reward.combined_image_score, 0.72, places=6)

    def test_reference_policy_failure_is_not_replaced_by_qwen(self) -> None:
        completion = {
            "generated_text": VALID_FULL_TEXT,
            "generated_images": [object(), object(), object()],
        }
        reward = score_two_level_reward(
            completion,
            sample={"infrared_label": "noise+stripe_noise", "visible_label": "haze"},
            image_reward=ImageLevelRewardResult(score=0.9),
            reference_reward=ThreeImageReferenceReward(
                score=0.0,
                error="reference reward forbids implicit resizing",
            ),
            protocol="full",
            reference_weight=0.9,
            qwen_weight=0.1,
        )
        self.assertAlmostEqual(reward.combined_image_score, 0.09, places=6)


if __name__ == "__main__":
    unittest.main()
