"""Single-signal WT loaders with window z-scoring and domain-balanced source batches."""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from . import config as CT
from imacd.noise import add_awgn_np
from imacd.data import _read_segments, _read_segments_angle
from imacd.pairing import reference_partition


def parse_int_list(spec: str | list[int] | tuple[int, ...] | None, default: list[int]) -> list[int]:
    """'20,25' -> [20, 25]; None/'' -> default; lists pass through."""
    if spec is None:
        return list(default)
    if isinstance(spec, (list, tuple)):
        return [int(v) for v in spec]
    s = str(spec).strip()
    if not s:
        return list(default)
    return [int(tok) for tok in s.replace(" ", "").split(",") if tok]


def fmt_int_list(values: list[int]) -> str:
    return "-".join(str(v) for v in values)


def _find_mat(cls_name: str, installation: int, speed: int) -> str:
    d = os.path.join(CT.DATA_ROOT, cls_name, str(installation))
    for pat in (f"*_{speed}.MAT", f"*_{speed}.mat"):
        files = sorted(glob.glob(os.path.join(d, pat)))
        if files:
            return files[0]
    raise FileNotFoundError(
        f"No MAT file for class={cls_name!r} installation={installation} speed={speed} under {d}"
    )


# Data matrix columns: 0/1 vibration sensors (1 is the one used), 2 the
# once-per-revolution tacho pulse, 3 an unused auxiliary channel.
VIB_COL = 1
TACHO_COL = 2


def _domain_id(installation: int, speed: int) -> int:
    """Encode (installation, speed) as a single int key (speeds are <= 55)."""
    return installation * 1000 + speed


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-6)


@dataclass
class FlatData:
    """Windows, labels and contiguous source-domain indices."""
    x: np.ndarray            # (N, L) float32
    y: np.ndarray            # (N,) int64 class label
    domain_id: np.ndarray    # (N,) int64 installation*1000+speed
    domain_index: np.ndarray  # (N,) int64 in [0, n_domains)
    domain_ids_sorted: list[int]

    @property
    def n_domains(self) -> int:
        return len(self.domain_ids_sorted)

    def __len__(self) -> int:
        return len(self.y)


# Reading the MAT files dominates a protocol run: the loader is called three
# times per seed (source / val / target) but only ever with a handful of
# distinct band specs, so one protocol sweep would otherwise re-read the same
# files dozens of times. FlatData is treated as immutable — SegmentDataset
# copies every segment it hands out — so the entries are safe to share.
_FLAT_CACHE: dict[tuple, FlatData] = {}


def clear_cache() -> None:
    """Drop the segment cache (only needed if the files change on disk)."""
    _FLAT_CACHE.clear()


def load_flat(
    installations: list[int],
    speeds: list[int],
    max_segments_per_file: int | None = CT.DEFAULT_MAX_SEGMENTS_PER_FILE,
    seg_domain: str = CT.DEFAULT_SEG_DOMAIN,
) -> FlatData:
    """Every segment of the given installation x speed grid, with labels.

    No train/val split happens here: the protocol's bands are disjoint, so each
    band is simply loaded whole — the same choice the Ima-CD loaders make when
    a held-out val band is supplied.

    Results are memoized per (installations, speeds, cap, seg_domain).
    """
    if seg_domain not in {"time", "angle"}:
        raise ValueError("seg_domain must be time/angle")

    key = (os.path.abspath(CT.DATA_ROOT), tuple(installations), tuple(speeds),
           max_segments_per_file, seg_domain)
    if key in _FLAT_CACHE:
        return _FLAT_CACHE[key]

    cls_map = CT.class_to_idx()
    xs, ys, doms = [], [], []
    for cls_name in CT.CLASSES:
        for inst in installations:
            for sp in speeds:
                mat_path = _find_mat(cls_name, inst, sp)
                if seg_domain == "angle":
                    s = _read_segments_angle(mat_path, max_segments_per_file)
                else:
                    s = _read_segments(mat_path, max_segments_per_file)
                if not len(s):
                    continue
                if cls_map[cls_name] == 0:
                    _, query_ids = reference_partition(len(s), _domain_id(inst, sp))
                    s = s[query_ids]
                xs.append(s)
                ys.append(np.full(len(s), cls_map[cls_name], dtype=np.int64))
                doms.append(np.full(len(s), _domain_id(inst, sp), dtype=np.int64))

    if not xs:
        raise RuntimeError(
            f"No segments found for installations={installations} speeds={speeds}"
        )
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    domain_id = np.concatenate(doms, axis=0)
    ids_sorted = sorted(int(d) for d in np.unique(domain_id))
    lookup = {d: i for i, d in enumerate(ids_sorted)}
    domain_index = np.asarray([lookup[int(d)] for d in domain_id], dtype=np.int64)
    out = FlatData(x=x, y=y, domain_id=domain_id, domain_index=domain_index,
                   domain_ids_sorted=ids_sorted)
    _FLAT_CACHE[key] = out
    print(f"[data] loaded i{installations} s{speeds} ({seg_domain}): "
          f"{len(out)} segments, {out.n_domains} domains")
    return out


class SegmentDataset(Dataset):
    """(segment, label, domain_index) triples with optional AWGN.

    Noise is drawn per __getitem__ from a per-index seeded generator so the
    evaluation set is deterministic regardless of loader order or workers.
    """

    def __init__(
        self,
        data: FlatData,
        normalize: bool = True,
        snr_db: float | None = None,
        seed: int = CT.DEFAULT_SEED,
    ) -> None:
        self.data = data
        self.normalize = normalize
        self.snr_db = snr_db
        self.seed = seed

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        seg = self.data.x[idx].copy()
        if self.snr_db is not None:
            seg = add_awgn_np(seg, self.snr_db, np.random.default_rng(self.seed + idx))
        if self.normalize:
            seg = _zscore(seg)
        return (
            torch.from_numpy(seg).float().unsqueeze(0),  # (1, L)
            torch.tensor(int(self.data.y[idx]), dtype=torch.long),
            torch.tensor(int(self.data.domain_index[idx]), dtype=torch.long),
        )


class DomainBalancedBatchSampler(Sampler[list[int]]):
    """Each batch contains an equal number of windows from every source domain."""

    def __init__(self, domain_index: np.ndarray, batch_size: int, seed: int = 0) -> None:
        if domain_index.size == 0:
            raise ValueError("No source domain samples")
        self.groups = [np.flatnonzero(domain_index == d)
                       for d in range(int(domain_index.max()) + 1)]
        self.groups = [g for g in self.groups if len(g)]
        if not self.groups:
            raise ValueError("no domains in domain_index")
        if batch_size < len(self.groups) or batch_size % len(self.groups):
            raise ValueError("batch_size must be divisible by the number of source domains")
        self.per_domain = batch_size // len(self.groups)
        self.n_batches = min(len(g) for g in self.groups) // self.per_domain
        if not self.n_batches:
            raise ValueError("Not enough samples for one domain-balanced batch")
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        perms = [rng.permutation(g) for g in self.groups]
        for b in range(self.n_batches):
            batch: list[int] = []
            for p in perms:
                batch.extend(int(i) for i in p[b * self.per_domain:(b + 1) * self.per_domain])
            yield batch


def build_source_loader(
    installations: list[int],
    speeds: list[int],
    batch_size: int = CT.DEFAULT_BATCH_SIZE,
    num_workers: int = CT.DEFAULT_NUM_WORKERS,
    max_segments_per_file: int | None = CT.DEFAULT_MAX_SEGMENTS_PER_FILE,
    seg_domain: str = CT.DEFAULT_SEG_DOMAIN,
    seed: int = CT.DEFAULT_SEED,
) -> tuple[DataLoader, FlatData]:
    """Labeled source loader, domain-balanced batches."""
    data = load_flat(installations, speeds, max_segments_per_file, seg_domain)
    ds = SegmentDataset(data, normalize=True, snr_db=None, seed=seed)
    sampler = DomainBalancedBatchSampler(data.domain_index, batch_size, seed=seed)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=num_workers)
    return loader, data


def build_plain_loader(
    installations: list[int],
    speeds: list[int],
    batch_size: int = CT.DEFAULT_BATCH_SIZE,
    num_workers: int = CT.DEFAULT_NUM_WORKERS,
    max_segments_per_file: int | None = CT.DEFAULT_MAX_SEGMENTS_PER_FILE,
    seg_domain: str = CT.DEFAULT_SEG_DOMAIN,
    seed: int = CT.DEFAULT_SEED,
    snr_db: float | None = None,
    shuffle: bool = False,
    drop_last: bool = False,
) -> tuple[DataLoader, FlatData]:
    """Unshuffled loader for validation / test, or a shuffled one for the
    unlabeled target stream."""
    data = load_flat(installations, speeds, max_segments_per_file, seg_domain)
    ds = SegmentDataset(data, normalize=True, snr_db=snr_db, seed=seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, drop_last=drop_last)
    return loader, data


def infinite(loader: DataLoader):
    """Cycle a loader forever, for the unlabeled target stream."""
    while True:
        for batch in loader:
            yield batch
