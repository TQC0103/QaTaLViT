import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from nets.improved_ssl import (
    BottleneckCrossAttention,
    BoundaryConsistencyLoss,
    CrossModalSkipAdapter,
    PrototypeAlignmentLoss,
    PrototypeMemoryBank,
    StructuredTextConditioner,
    TextSpatialPrior,
    UncertaintyAwarePseudoLabelFusion,
)
from text_encoder import (
    StructuredReportParser,
    attribute_vector_size,
    build_cache_metadata,
    load_report_features,
    save_report_features,
    split_report_into_units,
)


class BottleneckCrossAttentionTests(unittest.TestCase):
    def test_shapes_and_gate_range(self) -> None:
        module = BottleneckCrossAttention(dim=32, num_heads=4, num_bottleneck_tokens=4)
        visual = torch.randn(2, 16, 32)
        text = torch.randn(2, 5, 32)

        fused, bottleneck, gate = module(visual, text)

        self.assertEqual(fused.shape, visual.shape)
        self.assertEqual(bottleneck.shape, (2, 4, 32))
        self.assertEqual(gate.shape, (2, 1, 32))
        self.assertTrue(torch.all(gate >= 0.0).item())
        self.assertTrue(torch.all(gate <= 1.0).item())


class TextSpatialPriorTests(unittest.TestCase):
    def test_prior_is_probabilistic_map(self) -> None:
        module = TextSpatialPrior(dim=32, height=8, width=8, attribute_dim=attribute_vector_size())
        text = torch.randn(2, 6, 32)
        attrs = torch.randn(2, attribute_vector_size())

        prior = module(text_tokens=text, structured_attributes=attrs)

        self.assertEqual(prior.shape, (2, 1, 8, 8))
        self.assertTrue(torch.all(prior >= 0.0).item())
        self.assertTrue(torch.all(prior <= 1.0).item())

    def test_fallback_summary_is_supported(self) -> None:
        module = TextSpatialPrior(dim=32, height=8, width=8, attribute_dim=attribute_vector_size())
        fallback = torch.randn(3, 32)
        attrs = torch.randn(3, attribute_vector_size())

        prior = module(text_tokens=None, fallback_summary=fallback, structured_attributes=attrs)

        self.assertEqual(prior.shape, (3, 1, 8, 8))


class StructuredTextConditionerTests(unittest.TestCase):
    def test_force_drop_uses_no_text_path(self) -> None:
        module = StructuredTextConditioner(
            text_dim=32,
            model_dim=16,
            attribute_dim=attribute_vector_size(),
            text_dropout_prob=0.0,
        )
        text = torch.randn(2, 5, 32)
        attrs = torch.randn(2, attribute_vector_size())
        prototypes = torch.randn(2, 16)
        force_drop = torch.tensor([True, False])

        tokens, summary, attribute_context, fallback_mask = module(
            text_tokens=text,
            structured_attributes=attrs,
            prototype_summary=prototypes,
            force_drop_mask=force_drop,
        )

        self.assertEqual(tokens.shape, (2, 5, 16))
        self.assertEqual(summary.shape, (2, 16))
        self.assertEqual(attribute_context.shape, (2, 16))
        self.assertTrue(fallback_mask[0].item())
        self.assertFalse(fallback_mask[1].item())


class StructuredReportParserTests(unittest.TestCase):
    def test_parser_extracts_expected_attributes(self) -> None:
        parser = StructuredReportParser()
        report = "Severe bilateral lower lung multifocal opacities with multiple lesions."

        parsed = parser.parse(report)
        vector = parser.vectorize(report)

        self.assertEqual(parsed["laterality"], "bilateral")
        self.assertEqual(parsed["vertical"], "lower")
        self.assertEqual(parsed["severity"], "severe")
        self.assertEqual(parsed["count"], "multiple")
        self.assertEqual(parsed["extent"], "multifocal")
        self.assertEqual(vector.shape[0], attribute_vector_size())
        self.assertEqual(float(vector.sum().item()), 5.0)

    def test_report_is_split_into_multiple_semantic_units(self) -> None:
        report = "Bilateral lower lobe opacity; severe disease. Multifocal lesions, patchy involvement."

        units = split_report_into_units(report)

        self.assertGreaterEqual(len(units), 3)
        self.assertIn("Bilateral lower lobe opacity", units[0])

    def test_cached_report_features_round_trip(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            text = torch.randn(4, 32)
            attributes = torch.randn(attribute_vector_size())
            metadata = build_cache_metadata("test-model", max_units=10)
            save_report_features(tmp_dir, "sample.png", text, attributes, metadata=metadata)

            loaded = load_report_features(tmp_dir, "sample.png", expected_metadata=metadata)

            self.assertIsNotNone(loaded)
            loaded_text, loaded_attributes, loaded_metadata = loaded
            self.assertTrue(torch.equal(text, loaded_text))
            self.assertTrue(torch.equal(attributes, loaded_attributes))
            self.assertEqual(loaded_metadata, metadata)

    def test_cache_metadata_mismatch_invalidates_entry(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            text = torch.randn(4, 32)
            attributes = torch.randn(attribute_vector_size())
            save_report_features(
                tmp_dir,
                "sample.png",
                text,
                attributes,
                metadata=build_cache_metadata("model-a", max_units=10),
            )

            loaded = load_report_features(
                tmp_dir,
                "sample.png",
                expected_metadata=build_cache_metadata("model-b", max_units=10),
            )

            self.assertIsNone(loaded)


class PrototypeMemoryTests(unittest.TestCase):
    def test_retrieve_and_update(self) -> None:
        module = PrototypeMemoryBank(dim=16, num_prototypes=4, momentum=0.5)
        before = module.memory.clone()
        features = torch.randn(5, 16)

        retrieved, weights = module.retrieve(features)
        module.update(features)

        self.assertEqual(retrieved.shape, (5, 16))
        self.assertEqual(weights.shape, (5, 4))
        self.assertFalse(torch.equal(before, module.memory))


class SkipAdapterTests(unittest.TestCase):
    def test_skip_adapter_preserves_shape(self) -> None:
        module = CrossModalSkipAdapter(skip_channels=16, context_dim=32)
        skip = torch.randn(2, 16, 32, 32)
        context = torch.randn(2, 32)
        prior = torch.rand(2, 1, 8, 8)

        adapted = module(skip, context, prior)

        self.assertEqual(adapted.shape, skip.shape)


class PseudoLabelFusionTests(unittest.TestCase):
    def test_prefers_visual_when_confident_and_agreeing(self) -> None:
        module = UncertaintyAwarePseudoLabelFusion(entropy_threshold=0.2, agreement_threshold=0.8)
        visual = torch.full((1, 1, 8, 8), 0.95)
        text_prior = torch.full((1, 1, 8, 8), 0.9)

        fused, weight = module(visual, text_prior)

        self.assertAlmostEqual(float(weight.item()), 1.0, places=5)
        self.assertTrue(torch.allclose(fused, visual))

    def test_falls_back_to_text_prior_when_visual_is_uncertain(self) -> None:
        module = UncertaintyAwarePseudoLabelFusion(entropy_threshold=0.2, agreement_threshold=0.1)
        visual = torch.full((1, 1, 8, 8), 0.5)
        text_prior = torch.full((1, 1, 8, 8), 0.9)

        fused, weight = module(visual, text_prior)

        self.assertAlmostEqual(float(weight.item()), 0.0, places=5)
        self.assertTrue(torch.allclose(fused, text_prior))


class AuxiliaryLossTests(unittest.TestCase):
    def test_boundary_consistency_is_near_zero_for_identical_masks(self) -> None:
        loss_fn = BoundaryConsistencyLoss()
        mask = torch.rand(2, 1, 16, 16)

        loss = loss_fn(mask, mask)

        self.assertLess(float(loss.item()), 1e-4)

    def test_prototype_alignment_loss_is_small_for_identical_vectors(self) -> None:
        loss_fn = PrototypeAlignmentLoss()
        vector = torch.randn(4, 32)

        loss = loss_fn(vector, vector)

        self.assertLess(float(loss.item()), 1e-5)


if __name__ == "__main__":
    unittest.main()
