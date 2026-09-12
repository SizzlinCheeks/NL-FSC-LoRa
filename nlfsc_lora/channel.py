"""Simple channel impairments for comparing linear vs. nonlinear chirps."""

from typing import Optional

import numpy as np


def awgn(x: np.ndarray, snr_db: float, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    sig_power = np.mean(np.abs(x) ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape))
    return x + noise


def apply_cfo(x: np.ndarray, cfo_hz: float, fs: float) -> np.ndarray:
    """Carrier frequency offset: a constant, uncompensated frequency error."""
    n = np.arange(len(x))
    return x * np.exp(1j * 2 * np.pi * cfo_hz * n / fs)


def apply_timing_offset(x: np.ndarray, offset_samples: int) -> np.ndarray:
    """Integer-sample timing error, modeled as a cyclic shift within the symbol."""
    return np.roll(x, offset_samples)
