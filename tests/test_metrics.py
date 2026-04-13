from __future__ import annotations

import unittest

import numpy as np
import torch

from reid.metrics import compute_distance_matrix, evaluate_rankings


class MetricsTestCase(unittest.TestCase):
    def test_distance_matrix_shapes(self):
        query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        gallery = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        dist = compute_distance_matrix(query, gallery, metric="cosine")
        self.assertEqual(tuple(dist.shape), (2, 2))

    def test_ranking_metrics(self):
        distmat = np.array([[0.1, 0.9, 1.2], [0.8, 0.2, 1.0]], dtype=np.float32)
        query_ids = np.array([1, 2])
        gallery_ids = np.array([1, 2, 3])
        query_camids = np.array([0, 1])
        gallery_camids = np.array([1, 0, 2])
        metrics = evaluate_rankings(distmat, query_ids, gallery_ids, query_camids, gallery_camids, max_rank=3)
        self.assertAlmostEqual(metrics["rank1"], 1.0)
        self.assertAlmostEqual(metrics["mAP"], 1.0)


if __name__ == "__main__":
    unittest.main()
