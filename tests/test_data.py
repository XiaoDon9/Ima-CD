import tempfile
from pathlib import Path
import unittest

import numpy as np
import scipy.io as sio
import torch

from imacd import config
from imacd.data import _read_segments
from imacd.pairing import (PairDataset, QueryDataset, SplitData, normalize_pair,
                          reference_partition, separate_references)
from baselines.data import DomainBalancedBatchSampler


class DataTests(unittest.TestCase):
    def test_healthy_normalization(self):
        query = np.arange(32, dtype=np.float32) * 4
        refs = [np.arange(32, dtype=np.float32) + i for i in range(3)]
        original = config.INPUT_NORMALIZATION
        with config.data_settings("temporary-data", "healthy"):
            scaled_q, scaled_refs = normalize_pair(query, refs)
            scale = np.concatenate(refs).std() + 1e-6
            np.testing.assert_allclose(scaled_q, (query - query.mean()) / scale)
            np.testing.assert_allclose(scaled_refs[0], (refs[0] - refs[0].mean()) / scale)
        self.assertEqual(config.INPUT_NORMALIZATION, original)
        with config.data_settings("temporary-data", "window"):
            scaled_q, _ = normalize_pair(query, refs)
            self.assertAlmostEqual(float(scaled_q.std()), 1, places=5)

    def test_disjoint_reference_pool_and_random_replacement(self):
        values = np.stack([np.full(32, i, dtype=np.float32) for i in range(20)])
        domains = np.repeat([1, 2], 10)
        split = SplitData(values, {0: values, 1: values + 100},
                          domains, {0: domains, 1: domains})
        split = separate_references(split)
        self.assertEqual(len(split.healthy_pool), 4)
        self.assertEqual(len(split.class_segments[0]), 16)
        self.assertFalse(set(split.healthy_pool[:, 0]) & set(split.class_segments[0][:, 0]))
        for dataset in (PairDataset(split, 10, normalize=False), QueryDataset(split, normalize=False)):
            for i in range(len(dataset)):
                refs, query, label = dataset[i]
                domain = int(query[0, 0] >= (110 if label else 10)) + 1
                self.assertTrue(set(refs[:, 0].tolist()).issubset(
                    set(split.healthy_pool[split.healthy_domain == domain, 0])))
                self.assertLess(len(set(refs[:, 0].tolist())), 3)
                if label == 0:
                    self.assertNotIn(query[0, 0].item(), refs[:, 0].tolist())
        dataset = QueryDataset(split, normalize=False, seed=42)
        first = dataset[3][0].clone()
        dataset[5]
        torch.testing.assert_close(first, dataset[3][0])
        _, query_ids = reference_partition(160, 1045)
        np.testing.assert_array_equal(query_ids, reference_partition(160, 1045)[1])
        broken = SplitData(values[:1], {0: values}, np.array([1]), {0: domains})
        with self.assertRaisesRegex(RuntimeError, "no healthy segments"):
            PairDataset(broken, 10)
        with self.assertRaises(ValueError):
            reference_partition(1, 1)

    def test_mat_column_and_segment_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "signal.MAT"
            raw = np.column_stack([np.zeros(3 * 4096), np.arange(3 * 4096)])
            sio.savemat(path, {"Data": raw})
            segments = _read_segments(str(path), 2)
            self.assertEqual(segments.shape, (2, 4096))
            np.testing.assert_array_equal(segments.ravel(), raw[:8192, 1].astype(np.float32))

    def test_balanced_domain_batches(self):
        domains = np.repeat(np.arange(4), 10)
        sampler = DomainBalancedBatchSampler(domains, batch_size=8, seed=42)
        for indices in sampler:
            np.testing.assert_array_equal(np.bincount(domains[indices]), [2, 2, 2, 2])
        for batch_size in (3, 7, 100):
            with self.assertRaises(ValueError):
                DomainBalancedBatchSampler(domains, batch_size=batch_size)
