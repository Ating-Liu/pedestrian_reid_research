from __future__ import annotations

import unittest

import torch

from reid.model import GlobalLocalReIDModel


class ModelTestCase(unittest.TestCase):
    def test_forward_shapes(self):
        model = GlobalLocalReIDModel(
            num_classes=8,
            embedding_dim=128,
            pretrained=False,
            use_local_branch=True,
            use_transformer=True,
            use_fusion_gate=True,
            num_parts=4,
            transformer_dim=64,
            transformer_heads=4,
            transformer_layers=1,
        )
        model.eval()
        images = torch.randn(2, 3, 256, 128)
        with torch.no_grad():
            outputs = model(images)
        self.assertEqual(tuple(outputs["embeddings"].shape), (2, 128))
        self.assertEqual(tuple(outputs["logits"].shape), (2, 8))
        self.assertIsNotNone(outputs["local_embeddings"])

    def test_residual_fusion_forward_shapes(self):
        model = GlobalLocalReIDModel(
            num_classes=8,
            embedding_dim=128,
            pretrained=False,
            use_local_branch=True,
            use_transformer=False,
            use_fusion_gate=False,
            fusion_mode="residual",
            local_residual_weight=0.1,
            num_parts=4,
            transformer_dim=64,
        )
        model.eval()
        images = torch.randn(2, 3, 256, 128)
        with torch.no_grad():
            outputs = model(images)
        self.assertEqual(tuple(outputs["embeddings"].shape), (2, 128))
        self.assertEqual(tuple(outputs["logits"].shape), (2, 8))
        self.assertIsNotNone(outputs["local_embeddings"])

    def test_gated_residual_fusion_forward_shapes(self):
        model = GlobalLocalReIDModel(
            num_classes=8,
            embedding_dim=128,
            pretrained=False,
            use_local_branch=True,
            use_transformer=True,
            use_fusion_gate=True,
            fusion_mode="gated_residual",
            local_residual_weight=0.1,
            num_parts=4,
            transformer_dim=64,
            transformer_heads=4,
            transformer_layers=1,
        )
        model.eval()
        images = torch.randn(2, 3, 256, 128)
        with torch.no_grad():
            outputs = model(images)
        self.assertEqual(tuple(outputs["embeddings"].shape), (2, 128))
        self.assertEqual(tuple(outputs["logits"].shape), (2, 8))
        self.assertIsNotNone(outputs["local_embeddings"])

    def test_local_auxiliary_forward_shapes(self):
        model = GlobalLocalReIDModel(
            num_classes=8,
            embedding_dim=128,
            pretrained=False,
            use_local_branch=True,
            use_transformer=False,
            use_fusion_gate=False,
            fusion_mode="residual",
            local_residual_weight=0.1,
            local_residual_learnable=False,
            use_local_auxiliary=True,
            num_parts=4,
            transformer_dim=64,
        )
        model.eval()
        images = torch.randn(2, 3, 256, 128)
        with torch.no_grad():
            outputs = model(images)
        self.assertEqual(tuple(outputs["embeddings"].shape), (2, 128))
        self.assertEqual(tuple(outputs["local_logits"].shape), (2, 8))
        self.assertFalse(model.local_residual_scale.requires_grad)


if __name__ == "__main__":
    unittest.main()
