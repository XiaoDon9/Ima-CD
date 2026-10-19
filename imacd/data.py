"""Data pipeline for the WT planetary gearbox dataset (.MAT files).

Reads raw MAT files directly (column 1 = vibration, column 2 = tacho), cuts
non-overlapping segments either as fixed SEG_LEN time windows or as
tacho-based angle-domain revolutions, and hands them to the PairDataset /
QueryDataset in pairing.py for reference/query pairing.

Directory layout: DATA_ROOT/{class}/{installation}/{prefix}{installation}_{speed}.MAT
"""
from __future__ import annotations

import glob
import os
import warnings
from functools import lru_cache

import numpy as np
import scipy.io as sio
from torch.utils.data import DataLoader

from . import config as CW
from .pairing import PairDataset, QueryDataset, SplitData, _split_segments


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
    d = os.path.join(CW.DATA_ROOT, cls_name, str(installation))
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


@lru_cache(maxsize=80)
def _read_segments(path: str, max_segments_per_file: int | None) -> np.ndarray:
    # Files carry ['Data', 'DataCount', 'ChannelCount', 'SampleRate', 'Interval'];
    # some contain duplicate metadata variables (e.g. "Interval"), which makes
    # scipy emit MatReadWarning ("replacing previous with new"). Only 'Data' is
    # used here, so load it by name and silence the harmless warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mat = sio.loadmat(path, struct_as_record=False, squeeze_me=True,
                          variable_names=["Data"])
        if "Data" not in mat:  # fall back: read everything, pick a 2-D array
            mat = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    raw = None
    if "Data" in mat:
        raw = np.asarray(mat["Data"])
    else:
        for k, v in mat.items():
            if k.startswith("__"):
                continue
            arr = np.asarray(v)
            if arr.ndim >= 2 and arr.shape[1] >= 2:
                raw = arr
                break
    if raw is None or raw.ndim < 2 or raw.shape[1] < 2:
        raise ValueError(f"No [N, >=2] data variable found in {path}")
    sig = raw[:, VIB_COL].astype(np.float32)
    if max_segments_per_file is not None:
        sig = sig[: max_segments_per_file * CW.SEG_LEN]
    n = (len(sig) // CW.SEG_LEN) * CW.SEG_LEN
    if n == 0:
        return np.empty((0, CW.SEG_LEN), dtype=np.float32)
    return sig[:n].reshape(-1, CW.SEG_LEN).copy()


def _rev_starts(tacho: np.ndarray) -> np.ndarray:
    """Rising-edge indices of the once-per-revolution tacho pulse."""
    lo, hi = np.percentile(tacho, [1, 99])
    above = tacho > 0.5 * (lo + hi)
    return np.flatnonzero(~above[:-1] & above[1:]) + 1


def _read_segments_angle(
    path: str,
    max_segments_per_file: int | None,
    n_per_rev: int = CW.ANGLE_N_PER_REV,
    revs_per_seg: int = CW.ANGLE_REVS_PER_SEG,
) -> np.ndarray:
    """Tacho-based angle-domain segmentation.

    Cuts the vibration signal at revolution starts (tacho rising edges) and
    linearly resamples every revolution to n_per_rev points, concatenating
    revs_per_seg consecutive revolutions per segment. Segments are
    non-overlapping and speed-invariant: one segment always covers the same
    rotation angle, and mesh orders land on the same bins at every speed.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mat = sio.loadmat(path, struct_as_record=False, squeeze_me=True,
                          variable_names=["Data"])
    raw = np.asarray(mat["Data"])
    sig = raw[:, VIB_COL].astype(np.float64)
    starts = _rev_starts(raw[:, TACHO_COL])
    n_segs = max((len(starts) - 1) // revs_per_seg, 0)
    if max_segments_per_file is not None:
        n_segs = min(n_segs, max_segments_per_file)
    out = np.empty((n_segs, n_per_rev * revs_per_seg), dtype=np.float32)
    grid = np.arange(n_per_rev, dtype=np.float64)
    for i in range(n_segs):
        for j in range(revs_per_seg):
            a = int(starts[i * revs_per_seg + j])
            b = int(starts[i * revs_per_seg + j + 1])
            xp = np.arange(b - a, dtype=np.float64) * (n_per_rev / (b - a))
            out[i, j * n_per_rev:(j + 1) * n_per_rev] = np.interp(grid, xp, sig[a:b])
    return out


def _domain_id(installation: int, speed: int) -> int:
    """Encode (installation, speed) as a single int key (speeds are <= 55, no collision)."""
    return installation * 1000 + speed


def load_split(
    installations: list[int],
    speeds: list[int],
    split: str,
    train_ratio: float = CW.DEFAULT_TRAIN_RATIO,
    max_segments_per_file: int | None = CW.DEFAULT_MAX_SEGMENTS_PER_FILE,
    seed: int = CW.DEFAULT_SEED,
    seg_domain: str = CW.DEFAULT_SEG_DOMAIN,
) -> SplitData:
    split = split.lower()
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train/val/test")
    if seg_domain not in {"time", "angle"}:
        raise ValueError("seg_domain must be time/angle")
    seg_len = CW.SEG_LEN if seg_domain == "time" else CW.ANGLE_N_PER_REV * CW.ANGLE_REVS_PER_SEG

    cls_map = CW.class_to_idx()
    class_segments: dict[int, np.ndarray] = {}
    class_domain: dict[int, np.ndarray] = {}
    healthy_pool = np.empty((0, seg_len), dtype=np.float32)
    healthy_domain = np.empty((0,), dtype=np.int64)
    empty_seg = np.empty((0, seg_len), dtype=np.float32)
    empty_dom = np.empty((0,), dtype=np.int64)

    for cls_name in CW.CLASSES:
        all_parts, tr_parts, va_parts = [], [], []
        all_dom_parts, tr_dom_parts, va_dom_parts = [], [], []
        for inst in installations:
            for sp in speeds:
                mat_path = _find_mat(cls_name, inst, sp)
                if seg_domain == "angle":
                    s = _read_segments_angle(mat_path, max_segments_per_file)
                else:
                    s = _read_segments(mat_path, max_segments_per_file)
                if not len(s):
                    continue
                dom_id = _domain_id(inst, sp)
                # Split each (installation, speed) domain independently (stratified)
                # so every domain has segments in both train and val, which strict
                # same-domain normal-anchor pairing depends on.
                tr_s, va_s = _split_segments(s, train_ratio=train_ratio, seed=seed + cls_map[cls_name] + dom_id)
                all_parts.append(s)
                tr_parts.append(tr_s)
                va_parts.append(va_s)
                all_dom_parts.append(np.full(len(s), dom_id, dtype=np.int64))
                tr_dom_parts.append(np.full(len(tr_s), dom_id, dtype=np.int64))
                va_dom_parts.append(np.full(len(va_s), dom_id, dtype=np.int64))

        all_seg = np.concatenate(all_parts, axis=0) if all_parts else empty_seg
        tr = np.concatenate(tr_parts, axis=0) if tr_parts else empty_seg
        va = np.concatenate(va_parts, axis=0) if va_parts else empty_seg
        all_dom = np.concatenate(all_dom_parts, axis=0) if all_dom_parts else empty_dom
        tr_dom = np.concatenate(tr_dom_parts, axis=0) if tr_dom_parts else empty_dom
        va_dom = np.concatenate(va_dom_parts, axis=0) if va_dom_parts else empty_dom

        if split == "test":
            chosen, chosen_dom = all_seg, all_dom
        elif split == "train":
            chosen, chosen_dom = tr, tr_dom
        else:
            chosen, chosen_dom = va, va_dom
        class_segments[cls_map[cls_name]] = chosen
        class_domain[cls_map[cls_name]] = chosen_dom
        if cls_name == CW.HEALTHY_CLASS:
            healthy_pool = chosen
            healthy_domain = chosen_dom

    if len(healthy_pool) == 0:
        raise RuntimeError("No healthy segments found for selected installations/speeds/split.")
    from .pairing import separate_references
    return separate_references(SplitData(
        healthy_pool=healthy_pool,
        class_segments=class_segments,
        healthy_domain=healthy_domain,
        class_domain=class_domain,
    ))


def build_train_val_loaders(
    train_installations: list[int],
    train_speeds: list[int],
    batch_size: int = CW.DEFAULT_BATCH_SIZE,
    num_workers: int = CW.DEFAULT_NUM_WORKERS,
    train_ratio: float = CW.DEFAULT_TRAIN_RATIO,
    max_segments_per_file: int | None = CW.DEFAULT_MAX_SEGMENTS_PER_FILE,
    seed: int = CW.DEFAULT_SEED,
    seg_domain: str = CW.DEFAULT_SEG_DOMAIN,
    val_installations: list[int] | None = None,
    val_speeds: list[int] | None = None,
):
    """Train/val loaders.

    Default (val_speeds=None): validation is a random per-domain subset of the
    training speeds — same-domain, so it saturates and barely ranks checkpoints.

    With val_speeds: validation is a HELD-OUT speed band. Both sides then use
    every segment of their own band (no within-domain split is needed, since
    the bands are disjoint), and checkpoint selection becomes a real
    cross-speed generalization signal.
    """
    if val_speeds:
        val_inst = val_installations or train_installations
        overlap = sorted(set(train_speeds) & set(val_speeds))
        if overlap and sorted(set(train_installations) & set(val_inst)):
            print(f"[data][WARNING] train/val speeds overlap on {overlap}: "
                  f"validation reuses trained segments -> selection will be optimistic.")
        tr_data = load_split(train_installations, train_speeds, "test", train_ratio,
                             max_segments_per_file, seed, seg_domain=seg_domain)
        va_data = load_split(val_inst, val_speeds, "test", train_ratio,
                             max_segments_per_file, seed, seg_domain=seg_domain)
    else:
        tr_data = load_split(train_installations, train_speeds, "train", train_ratio, max_segments_per_file, seed,
                             seg_domain=seg_domain)
        va_data = load_split(train_installations, train_speeds, "val", train_ratio, max_segments_per_file, seed,
                             seg_domain=seg_domain)

    n_min_train = max(len(v) for v in tr_data.class_segments.values())
    n_min_val = min(len(v) for v in va_data.class_segments.values())
    tr_ds = PairDataset(tr_data, n_samples=max(1000, n_min_train * CW.NUM_CLASSES), normalize=True, seed=seed)
    va_ds = QueryDataset(va_data, normalize=True, seed=seed + 1)

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return tr_loader, va_loader


def build_test_loader(
    test_installations: list[int],
    test_speeds: list[int],
    batch_size: int = CW.DEFAULT_BATCH_SIZE,
    num_workers: int = CW.DEFAULT_NUM_WORKERS,
    max_segments_per_file: int | None = CW.DEFAULT_MAX_SEGMENTS_PER_FILE,
    seed: int = CW.DEFAULT_SEED,
    test_snr_db: float | None = None,
    seg_domain: str = CW.DEFAULT_SEG_DOMAIN,
):
    te_data = load_split(test_installations, test_speeds, "test", CW.DEFAULT_TRAIN_RATIO, max_segments_per_file, seed,
                         seg_domain=seg_domain)
    te_ds = QueryDataset(te_data, normalize=True, test_snr_db=test_snr_db, seed=seed)
    return DataLoader(te_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
