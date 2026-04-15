from __future__ import annotations

import unittest

import torch

from reid.losses import BatchHardTripletLoss


class LossesTestCase(unittest.TestCase):
    def test_batch_hard_triplet_loss_vectorized_value(self):
        embeddings = torch.tensor([[0.0], [1.0], [1.2], [2.2]])
        targets = torch.tensor([0, 0, 1, 1])
        loss = BatchHardTripletLoss(margin=0.3)(embeddings, targets)
        self.assertAlmostEqual(float(loss), 0.6, places=5)

    def test_batch_hard_triplet_loss_without_positive_pair(self):
        embeddings = torch.tensor([[0.0], [1.0], [2.0]])
        targets = torch.tensor([0, 1, 2])
        loss = BatchHardTripletLoss(margin=0.3)(embeddings, targets)
        self.assertEqual(float(loss), 0.0)


if __name__ == "__main__":
    unittest.main()
