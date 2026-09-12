"""Chirp generation via a phase-accumulator: f(t) -> phi(t) -> x(t) = exp(j*phi(t)).

Mirrors LoRa's chirp-spread-spectrum (CSS) construction: a base up-chirp sweeps
the full bandwidth B once per symbol period T, and each of the M = 2**SF
symbols is a cyclic time-shift of that base chirp. The only thing this module
generalizes is the trajectory: instead of a fixed linear sweep f0 + k*t, the
instantaneous frequency is f0 + B*g(t/T) for an arbitrary normalized shape g.

Because g is shared between transmitter and receiver, the matched-filter
concept behind dechirping still applies (see receiver.py) -- what changes is
whether the cheap FFT-bin trick LoRa relies on still works, which is exactly
the tradeoff this project studies.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class ChirpConfig:
    sf: int  # spreading factor; alphabet size M = 2**sf
    bandwidth: float  # B, Hz
    sample_rate: float  # Fs, Hz -- should be an integer multiple of B (oversampling)
    g: Callable[[np.ndarray], np.ndarray]  # normalized trajectory, g(0)=0, g(1)=1
    f_center: float = 0.0  # baseband center frequency

    @property
    def M(self) -> int:
        return 1 << self.sf

    @property
    def symbol_duration(self) -> float:
        return self.M / self.bandwidth

    @property
    def n_samples(self) -> int:
        return int(round(self.symbol_duration * self.sample_rate))

    @property
    def oversampling(self) -> float:
        return self.sample_rate / self.bandwidth


def base_frequency(cfg: ChirpConfig) -> np.ndarray:
    """Instantaneous frequency of the base (m=0) chirp over one symbol period."""
    n = cfg.n_samples
    u = np.arange(n) / n
    return cfg.f_center - cfg.bandwidth / 2.0 + cfg.bandwidth * cfg.g(u)


def phase_accumulator(f: np.ndarray, fs: float) -> np.ndarray:
    """Integrate instantaneous frequency into phase: phi[n] = phi[n-1] + 2*pi*f[n]/fs."""
    dphi = 2 * np.pi * f / fs
    phi = np.cumsum(dphi)
    return phi - dphi[0]  # phi[0] = 0


def base_waveform(cfg: ChirpConfig) -> np.ndarray:
    """Complex baseband samples of the base up-chirp, x[n] = exp(j*phi[n])."""
    phi = phase_accumulator(base_frequency(cfg), cfg.sample_rate)
    return np.exp(1j * phi)


def symbol_waveform(cfg: ChirpConfig, m: int) -> np.ndarray:
    """Symbol m: the base chirp cyclically time-shifted by m*T/M, LoRa-style.

    This is the discrete-sample equivalent of s_m(t) = s_0((t + m*T/M) mod T),
    including the phase discontinuity ("wrap glitch") that real LoRa symbols
    exhibit at the fold-over point -- the shift is applied to the sampled base
    waveform, not resynthesized from scratch.
    """
    base = base_waveform(cfg)
    n = cfg.n_samples
    shift = int(round(m * n / cfg.M)) % n
    return np.roll(base, -shift)


def all_symbol_waveforms(cfg: ChirpConfig) -> np.ndarray:
    """All M symbol waveforms stacked as an (M, n_samples) array."""
    base = base_waveform(cfg)
    n = cfg.n_samples
    return np.stack([np.roll(base, -(int(round(m * n / cfg.M)) % n)) for m in range(cfg.M)])
