"""Additive white Gaussian noise at a target SNR (dB).

Vendored from cross_domain_fsl/noise.py so this package stays self-contained.
Convention: noise_power = signal_power / 10^(snr_db/10), where signal_power is
the per-segment mean square along the last axis. ``snr_db=None`` means no
noise (clean signal passthrough).
"""
from __future__ import annotations

import numpy as np


def add_awgn_np(x: np.ndarray, snr_db: float | None,
                rng: np.random.Generator | None = None) -> np.ndarray:
    """x: [..., L]. Returns x + noise at ``snr_db`` (None -> unchanged)."""
    if snr_db is None:
        return x
    if rng is None:
        rng = np.random.default_rng()
    sig_power = np.maximum(np.mean(x ** 2, axis=-1, keepdims=True), 1e-12)
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = rng.standard_normal(size=x.shape) * np.sqrt(noise_power)
    return (x + noise).astype(x.dtype, copy=False)


__all__ = ["add_awgn_np"]
