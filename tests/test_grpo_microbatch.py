from __future__ import annotations

from types import MethodType, SimpleNamespace
import unittest

import torch

from imagefusion_r1.rl.grpo_trainer_mgpt2 import (
    MSRSGRPOTrainer,
    build_arg_parser,
    default_gpt_prefix,
    validate_training_args,
)


class GRPOMicrobatchTest(unittest.TestCase):
    def test_protocol_specific_default_prefix(self) -> None:
        self.assertEqual(default_gpt_prefix("full"), "<infrared_cot>\n")
        self.assertEqual(default_gpt_prefix("auto"), "<infrared_cot>\n")
        self.assertEqual(default_gpt_prefix("fused_only"), "<cot>\n")

    def test_validation_rejects_deterministic_grpo(self) -> None:
        args = build_arg_parser().parse_args(
            [
                "--base_model", "base",
                "--init_ckpt", "checkpoint",
                "--train_manifest", "manifest.json",
                "--output_dir", "output",
                "--reward_server_url", "http://127.0.0.1:18080/score",
                "--no-do_sample",
            ]
        )
        with self.assertRaisesRegex(ValueError, "deterministic rollouts"):
            validate_training_args(args)

    def test_validation_rejects_non_three_image_protocol(self) -> None:
        args = build_arg_parser().parse_args(
            [
                "--base_model", "base",
                "--init_ckpt", "checkpoint",
                "--train_manifest", "manifest.json",
                "--output_dir", "output",
                "--reward_server_url", "http://127.0.0.1:18080/score",
                "--stop_after_images", "1",
            ]
        )
        with self.assertRaisesRegex(ValueError, "hard-wired to three outputs"):
            validate_training_args(args)

    def test_microbatch_backward_matches_full_completion_mean(self) -> None:
        trainer = object.__new__(MSRSGRPOTrainer)
        trainer.rank = 0
        trainer.device = torch.device("cpu")
        trainer.args = SimpleNamespace(num_generations=2, replay_micro_batch_size=1)
        parameter = torch.nn.Parameter(torch.tensor(2.0))

        def fake_loss(self, completions, advantages):
            coefficients = torch.tensor(completions, dtype=torch.float32)
            loss = (parameter * coefficients * advantages.cpu()).mean()
            return loss, {
                "loss": float(loss.detach().item()),
                "advantage_abs": float(advantages.abs().mean().item()),
                "replay_logp_max_abs": 0.0,
            }

        trainer._compute_grpo_loss_with_advantages = MethodType(fake_loss, trainer)
        rewards = torch.tensor([0.2, 0.8], dtype=torch.float32)
        advantages, _reward_std = trainer._group_advantages(rewards)
        expected_grad = float((advantages * torch.tensor([1.0, 3.0])).mean().item() / 2.0)

        metrics = trainer.backward_grpo_loss(
            [1.0, 3.0],
            rewards,
            accumulation_steps=2,
        )

        self.assertIsNotNone(parameter.grad)
        self.assertAlmostEqual(float(parameter.grad.item()), expected_grad, places=6)
        self.assertAlmostEqual(metrics["reward"], 0.5, places=6)
        self.assertGreater(metrics["reward_std"], 0.0)


if __name__ == "__main__":
    unittest.main()
