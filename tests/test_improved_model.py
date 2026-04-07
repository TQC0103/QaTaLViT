import unittest

import torch

from nets.LViT_improved import LViTImproved
from text_encoder import attribute_vector_size


class LViTImprovedModelTests(unittest.TestCase):
    def _build_model(self) -> LViTImproved:
        return LViTImproved(
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
            enable_upper_scale_fusion=True,
        )

    def test_forward_with_text(self) -> None:
        model = self._build_model().eval()
        image = torch.randn(2, 3, 64, 64)
        text = torch.randn(2, 5, 32)
        attrs = torch.randn(2, attribute_vector_size())

        with torch.no_grad():
            logits, aux = model(image, text_tokens=text, structured_attributes=attrs, return_aux=True)

        self.assertEqual(logits.shape, (2, 1, 64, 64))
        self.assertEqual(aux["spatial_prior"].shape, (2, 1, 4, 4))
        self.assertEqual(aux["prototype_summary"].shape, (2, 64))
        self.assertEqual(aux["fusion_gate"].shape, (2, 1, 64))
        self.assertEqual(aux["prior_prediction"].shape, (2, 1, 64, 64))
        self.assertEqual(aux["probabilities"].shape, (2, 1, 64, 64))
        self.assertEqual(aux["used_prototype_fallback"].shape, (2,))
        self.assertTrue(torch.all(aux["used_prototype_fallback"] == 0).item())
        self.assertIsNotNone(aux["upper_scale_gate"])

    def test_forward_without_text_uses_prototype_fallback(self) -> None:
        model = self._build_model().eval()
        image = torch.randn(2, 3, 64, 64)
        attrs = torch.randn(2, attribute_vector_size())

        with torch.no_grad():
            logits, aux = model(image, text_tokens=None, structured_attributes=attrs, return_aux=True)

        self.assertEqual(logits.shape, (2, 1, 64, 64))
        self.assertEqual(aux["text_summary"].shape, (2, 64))
        self.assertTrue(torch.all(aux["used_prototype_fallback"] == 1).item())

    def test_prototype_memory_updates_in_training_mode(self) -> None:
        model = self._build_model().train()
        image = torch.randn(2, 3, 64, 64)
        text = torch.randn(2, 5, 32)
        attrs = torch.randn(2, attribute_vector_size())
        before = model.prototype_memory.memory.clone()

        model(image, text_tokens=text, structured_attributes=attrs, update_prototypes=True)

        self.assertFalse(torch.equal(before, model.prototype_memory.memory))

    def test_force_text_dropout_triggers_fallback_per_sample(self) -> None:
        model = self._build_model().train()
        image = torch.randn(2, 3, 64, 64)
        text = torch.randn(2, 5, 32)
        attrs = torch.randn(2, attribute_vector_size())
        force_drop = torch.tensor([True, False])

        _, aux = model(
            image,
            text_tokens=text,
            structured_attributes=attrs,
            return_aux=True,
            force_text_dropout_mask=force_drop,
        )

        self.assertTrue(aux["used_prototype_fallback"][0].item() == 1.0)
        self.assertTrue(aux["used_prototype_fallback"][1].item() == 0.0)


if __name__ == "__main__":
    unittest.main()
