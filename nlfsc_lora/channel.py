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


def apply_doppler_scale(x: np.ndarray, alpha: float) -> np.ndarray:
    """Wideband ("true") Doppler: time-scale the whole burst, r(t) = x(alpha*t),
    instead of approximating Doppler as a constant frequency shift (apply_cfo).

    x is treated as a finite-duration pulse. Once alpha*t runs past the end of x
    (alpha > 1: a compressed, foreshortened echo), the observation is padded with
    zeros rather than extrapolated -- the transmitter has simply stopped emitting
    by then, so there is nothing there to interpolate into.
    """
    n = len(x)
    idx = alpha * np.arange(n)
    valid = idx <= (n - 1)
    grid = np.arange(n)
    out = np.zeros(n, dtype=complex)
    out[valid] = np.interp(idx[valid], grid, x.real) + 1j * np.interp(idx[valid], grid, x.imag)
    return out
