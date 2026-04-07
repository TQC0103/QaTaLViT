import copy
import unittest

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from nets.LViT_improved import LViTImproved
from nets.improved_training import ImprovedSSLTrainer, ImprovedSemiSupervisedLoss, update_ema_model
from text_encoder import attribute_vector_size
from train_improved_ssl import split_labeled_unlabeled


class ImprovedTrainingTests(unittest.TestCase):
    def test_full_label_split_has_no_unlabeled_subset(self) -> None:
        dataset = TensorDataset(torch.arange(10))
        labeled_subset, unlabeled_subset = split_labeled_unlabeled(dataset, label_ratio=1.0, seed=666)

        self.assertEqual(len(labeled_subset), 10)
        self.assertIsNone(unlabeled_subset)

    def test_composite_loss_returns_all_expected_terms(self) -> None:
        loss_fn = ImprovedSemiSupervisedLoss(
            lambda_consistency=1.0,
            lambda_text=0.5,
            lambda_boundary=0.25,
            lambda_prototype=0.1,
        )
        labeled_logits = torch.full((2, 1, 16, 16), 1.4)
        labeled_target = torch.ones(2, 1, 16, 16)
        unlabeled_logits = torch.full((2, 1, 16, 16), 0.9)
        visual_teacher = torch.full((2, 1, 16, 16), 0.9)
        text_prior = torch.full((2, 1, 16, 16), 0.85)
        visual_summary = torch.randn(2, 32)
        prototype_summary = torch.randn(2, 32)

        losses = loss_fn(
            labeled_logits=labeled_logits,
            labeled_target=labeled_target,
            unlabeled_logits=unlabeled_logits,
            visual_teacher_prediction=visual_teacher,
            text_prior_prediction=text_prior,
            visual_summary=visual_summary,
            prototype_summary=prototype_summary,
        )

        for key in [
            "total",
            "supervised",
            "consistency",
            "text_guidance",
            "boundary",
            "prototype",
            "fusion_weight",
            "fused_pseudo",
        ]:
            self.assertIn(key, losses)

        self.assertTrue(torch.isfinite(losses["total"]).item())
        self.assertEqual(losses["fused_pseudo"].shape, labeled_logits.shape)
        self.assertGreaterEqual(float(losses["fusion_weight"].item()), 0.0)
        self.assertLessEqual(float(losses["fusion_weight"].item()), 1.0)

    def test_ema_update_moves_teacher_toward_student(self) -> None:
        teacher = nn.Linear(4, 4, bias=False)
        student = copy.deepcopy(teacher)

        with torch.no_grad():
            student.weight.add_(1.0)
        before = teacher.weight.clone()

        update_ema_model(teacher, student, decay=0.5)

        expected = before * 0.5 + student.weight * 0.5
        self.assertTrue(torch.allclose(teacher.weight, expected))

    def test_ssl_trainer_runs_one_teacher_student_step(self) -> None:
        model = LViTImproved(
            n_channels=3,
            n_classes=1,
            img_size=64,
            text_dim=32,
            base_channels=16,
            bottleneck_dim=64,
            num_bottleneck_tokens=4,
            num_heads=4,
            transformer_layers=1,
            num_prototypes=4,
            attribute_dim=attribute_vector_size(),
            text_dropout_prob=0.0,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = ImprovedSSLTrainer(student=model, optimizer=optimizer, ema_decay=0.9)

        labeled_images = torch.randn(2, 3, 64, 64)
        labeled_masks = torch.ones(2, 1, 64, 64)
        labeled_text = torch.randn(2, 5, 32)
        labeled_attrs = torch.randn(2, attribute_vector_size())
        unlabeled_images = torch.randn(2, 3, 64, 64)
        unlabeled_text = torch.randn(2, 5, 32)
        unlabeled_attrs = torch.randn(2, attribute_vector_size())

        losses = trainer.training_step(
            labeled_images=labeled_images,
            labeled_masks=labeled_masks,
            labeled_text_tokens=labeled_text,
            labeled_structured_attributes=labeled_attrs,
            unlabeled_images=unlabeled_images,
            unlabeled_text_tokens=unlabeled_text,
            unlabeled_structured_attributes=unlabeled_attrs,
            unlabeled_force_text_dropout_mask=torch.tensor([True, False]),
        )

        self.assertIn("total", losses)
        self.assertTrue(torch.isfinite(losses["total"]).item())


if __name__ == "__main__":
    unittest.main()
