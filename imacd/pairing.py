"""Reference/query pairing datasets shared by the WT data pipeline.

Every item is a (normal references, query, label) triple: K healthy reference
segments drawn from the SAME (installation, speed) domain as the query, which
is what lets the model imagine each class's features from that machine's own
healthy signature.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from . import config as CW
from .noise import add_awgn_np


REFERENCE_SPLIT_SEED = 20261018
REFERENCE_FRACTION = 0.2


def reference_partition(count: int, domain: int) -> tuple[np.ndarray, np.ndarray]:
    """Fixed support/query partition, independent of optimization seed."""
    if count < 2:
        raise ValueError("At least two healthy windows per domain are required")
    rng = np.random.default_rng(np.random.SeedSequence([REFERENCE_SPLIT_SEED, int(domain)]))
    reference = np.sort(rng.choice(count, max(1, int(count * REFERENCE_FRACTION)), replace=False))
    query = np.setdiff1d(np.arange(count), reference)
    return reference, query


def separate_references(split: "SplitData") -> "SplitData":
    healthy_label = CW.class_to_idx()[CW.HEALTHY_CLASS]
    healthy_queries = split.class_segments[healthy_label]
    if split.healthy_domain is None or split.class_domain is None:
        reference, query = reference_partition(len(healthy_queries), 0)
        split.healthy_pool = healthy_queries[reference].copy()
        split.class_segments[healthy_label] = healthy_queries[query].copy()
        return split
    domains = split.class_domain[healthy_label]
    reference_ids, query_ids = [], []
    for domain in np.unique(domains):
        ids = np.flatnonzero(domains == domain)
        reference, query = reference_partition(len(ids), int(domain))
        reference_ids.extend(ids[reference])
        query_ids.extend(ids[query])
    split.healthy_pool = healthy_queries[reference_ids].copy()
    split.healthy_domain = domains[reference_ids].copy()
    split.class_segments[healthy_label] = healthy_queries[query_ids].copy()
    split.class_domain[healthy_label] = domains[query_ids].copy()
    return split


@dataclass
class SplitData:
    healthy_pool: np.ndarray
    class_segments: dict[int, np.ndarray]
    # Per-segment domain ids (installation*1000+speed), parallel to
    # healthy_pool / class_segments. When both are provided, PairDataset /
    # QueryDataset restrict normal-anchor sampling to the same domain as the
    # query; leaving them None falls back to domain-agnostic sampling.
    healthy_domain: np.ndarray | None = None
    class_domain: dict[int, np.ndarray] | None = None


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-6)


def _normalize_by_refs(q: np.ndarray, picks: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Use the pooled raw healthy-reference standard deviation for all inputs."""
    ref_std = np.concatenate(picks, axis=-1).std() + 1e-6
    return (q - q.mean()) / ref_std, [(p - p.mean()) / ref_std for p in picks]


def normalize_pair(q: np.ndarray, picks: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    if CW.INPUT_NORMALIZATION == "healthy":
        return _normalize_by_refs(q, picks)
    return _zscore(q), [_zscore(p) for p in picks]


def _split_segments(segments: np.ndarray, train_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(segments) == 0:
        return segments, segments
    rng = np.random.default_rng(seed)
    idx = np.arange(len(segments))
    rng.shuffle(idx)
    n_train = max(1, int(len(idx) * train_ratio))
    tr = segments[idx[:n_train]]
    va = segments[idx[n_train:]]
    if len(va) == 0:
        va = tr[:1]
    return tr, va


def _group_indices_by_domain(domain_ids: np.ndarray) -> dict[int, np.ndarray]:
    """{domain_id: positions in the pool belonging to that domain}."""
    groups: dict[int, list[int]] = {}
    for i, d in enumerate(domain_ids):
        groups.setdefault(int(d), []).append(i)
    return {k: np.asarray(v, dtype=np.int64) for k, v in groups.items()}


def _validate_domain_coverage(
    class_domain: dict[int, np.ndarray], healthy_by_domain: dict[int, np.ndarray]
) -> None:
    missing = {}
    for y, doms in class_domain.items():
        gap = sorted(set(int(d) for d in np.unique(doms)) - set(healthy_by_domain.keys()))
        if gap:
            missing[y] = gap
    if missing:
        raise RuntimeError(
            f"Strict same-domain pairing failed: no healthy segments for domains "
            f"required by classes {missing}"
        )


class PairDataset(Dataset):
    def __init__(
        self,
        split_data: SplitData,
        n_samples: int,
        normalize: bool = True,
        seed: int = CW.DEFAULT_SEED,
        n_refs: int = 3,
    ) -> None:
        self.healthy = split_data.healthy_pool
        self.class_segments = split_data.class_segments
        self.n_classes = len(self.class_segments)
        self.normalize = normalize
        self.n_samples = n_samples
        self.n_refs = n_refs
        self.rng = np.random.default_rng(seed)
        self.seed = seed

        empty = [k for k, v in self.class_segments.items() if len(v) == 0]
        if empty:
            raise RuntimeError(f"Empty class segments for classes: {empty}")

        self.domain_matched = split_data.healthy_domain is not None and split_data.class_domain is not None
        if self.domain_matched:
            self.class_domain = split_data.class_domain
            self.healthy_by_domain = _group_indices_by_domain(split_data.healthy_domain)
            _validate_domain_coverage(self.class_domain, self.healthy_by_domain)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        y = idx % self.n_classes
        q_pool = self.class_segments[y]
        j = self.rng.integers(0, len(q_pool))
        q = q_pool[j].copy()
        # Sample n_refs normal segments (with replacement) and stack them;
        # the model averages their features into a single reference.
        if self.domain_matched:
            dom = int(self.class_domain[y][j])
            candidates = self.healthy_by_domain[dom]
            picks = [self.healthy[candidates[self.rng.integers(0, len(candidates))]].copy()
                     for _ in range(self.n_refs)]
        else:
            picks = [self.healthy[self.rng.integers(0, len(self.healthy))].copy()
                     for _ in range(self.n_refs)]
        if self.normalize:
            q, picks = normalize_pair(q, picks)
        q = torch.from_numpy(q).float().unsqueeze(0)
        n = torch.from_numpy(np.stack(picks, axis=0)).float()  # (K, L)
        return n, q, torch.tensor(y, dtype=torch.long)


class QueryDataset(Dataset):
    def __init__(
        self,
        split_data: SplitData,
        normalize: bool = True,
        test_snr_db: float | None = None,
        seed: int = CW.DEFAULT_SEED,
        n_refs: int = 3,
    ) -> None:
        domain_matched = split_data.healthy_domain is not None and split_data.class_domain is not None
        xs = []
        ys = []
        doms = []
        for y, segs in split_data.class_segments.items():
            for i, s in enumerate(segs):
                xs.append(s)
                ys.append(y)
                if domain_matched:
                    doms.append(int(split_data.class_domain[y][i]))
        if not xs:
            raise RuntimeError("No test query segments found.")
        self.x = np.stack(xs, axis=0)
        self.y = np.asarray(ys, dtype=np.int64)
        self.normalize = normalize
        self.healthy = split_data.healthy_pool
        self.test_snr_db = test_snr_db
        self.n_refs = n_refs
        self.rng = np.random.default_rng(seed)
        self.seed = seed

        self.domain_matched = domain_matched
        if domain_matched:
            self.dom = np.asarray(doms, dtype=np.int64)
            self.healthy_by_domain = _group_indices_by_domain(split_data.healthy_domain)
            _validate_domain_coverage(split_data.class_domain, self.healthy_by_domain)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        q = self.x[idx].copy()
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, idx]))
        if self.domain_matched:
            candidates = self.healthy_by_domain[int(self.dom[idx])]
        else:
            candidates = np.arange(len(self.healthy))
        picks = [self.healthy[i].copy() for i in rng.choice(candidates, self.n_refs, replace=True)]
        # At test time both the query and the normal references are noisy
        # (independent AWGN draws at the same target SNR), matching a real
        # deployment where both signals are acquired under the same noisy
        # operating conditions.
        q = add_awgn_np(q, self.test_snr_db, rng)
        picks = [add_awgn_np(p, self.test_snr_db, rng) for p in picks]
        if self.normalize:
            q, picks = normalize_pair(q, picks)
        q = torch.from_numpy(q).float().unsqueeze(0)
        n = torch.from_numpy(np.stack(picks, axis=0)).float()  # (K, L)
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return n, q, y
