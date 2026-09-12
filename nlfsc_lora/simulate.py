"""Experiment harness: hold everything but the trajectory constant and measure.

Implements the comparison framework from section 16 of the design writeup --
same bandwidth, symbol duration, sample rate, and channel, only g changes --
and reports symbol error rate (SER) rather than eyeballing a spectrogram.
"""

from dataclasses import replace
from typing import Callable, Optional, Sequence

import numpy as np

from .channel import apply_cfo, apply_doppler_scale, apply_timing_offset, awgn
from .chirp import ChirpConfig, symbol_waveform
from .receiver import fft_demod, matched_filter_bank_demod
from .sync import joint_cfo_symbol_demod

Demod = Callable[[np.ndarray, ChirpConfig], int]

_DEMODS = {"fft": fft_demod, "mfbank": matched_filter_bank_demod, "cfo_search": joint_cfo_symbol_demod}


def ser_vs_snr(
    cfg: ChirpConfig,
    snr_db_range: Sequence[float],
    n_symbols: int = 200,
    demod: str = "mfbank",
    seed: Optional[int] = None,
) -> np.ndarray:
    """Symbol error rate at each SNR point, over random symbols and noise draws."""
    rng = np.random.default_rng(seed)
    decode = _DEMODS[demod]
    ser = np.empty(len(snr_db_range))
    for i, snr_db in enumerate(snr_db_range):
        errors = 0
        for _ in range(n_symbols):
            m = int(rng.integers(0, cfg.M))
            tx = symbol_waveform(cfg, m)
            rx = awgn(tx, snr_db, rng)
            if decode(rx, cfg) != m:
                errors += 1
        ser[i] = errors / n_symbols
    return ser


def ser_vs_cfo(
    cfg: ChirpConfig,
    cfo_hz_range: Sequence[float],
    snr_db: float,
    n_symbols: int = 200,
    demod: str = "mfbank",
    seed: Optional[int] = None,
) -> np.ndarray:
    """Symbol error rate as a function of an uncompensated carrier frequency offset."""
    rng = np.random.default_rng(seed)
    decode = _DEMODS[demod]
    ser = np.empty(len(cfo_hz_range))
    for i, cfo_hz in enumerate(cfo_hz_range):
        errors = 0
        for _ in range(n_symbols):
            m = int(rng.integers(0, cfg.M))
            tx = symbol_waveform(cfg, m)
            tx = apply_cfo(tx, cfo_hz, cfg.sample_rate)
            rx = awgn(tx, snr_db, rng)
            if decode(rx, cfg) != m:
                errors += 1
        ser[i] = errors / n_symbols
    return ser


def ser_vs_doppler_scale(
    cfg: ChirpConfig,
    alpha_range: Sequence[float],
    snr_db: float,
    n_symbols: int = 200,
    demod: str = "mfbank",
    seed: Optional[int] = None,
) -> np.ndarray:
    """Symbol error rate under wideband ("true") Doppler: the transmitted symbol is
    time-scaled by alpha (channel.apply_doppler_scale) rather than shifted in frequency
    (contrast with ser_vs_cfo). Runs the same cyclic-shift symbol construction and
    dechirp-by-conjugate-reference LoRa uses -- what differs is only which decoder can
    read the result back out (see receiver.py: fft_demod only works for a linear g)."""
    rng = np.random.default_rng(seed)
    decode = _DEMODS[demod]
    ser = np.empty(len(alpha_range))
    for i, alpha in enumerate(alpha_range):
        errors = 0
        for _ in range(n_symbols):
            m = int(rng.integers(0, cfg.M))
            tx = symbol_waveform(cfg, m)
            tx = apply_doppler_scale(tx, alpha)
            rx = awgn(tx, snr_db, rng)
            if decode(rx, cfg) != m:
                errors += 1
        ser[i] = errors / n_symbols
    return ser


def ser_vs_timing_offset(
    cfg: ChirpConfig,
    offset_samples_range: Sequence[int],
    snr_db: float,
    n_symbols: int = 200,
    demod: str = "mfbank",
    seed: Optional[int] = None,
) -> np.ndarray:
    """Symbol error rate as a function of an integer-sample timing error."""
    rng = np.random.default_rng(seed)
    decode = _DEMODS[demod]
    ser = np.empty(len(offset_samples_range))
    for i, offset in enumerate(offset_samples_range):
        errors = 0
        for _ in range(n_symbols):
            m = int(rng.integers(0, cfg.M))
            tx = symbol_waveform(cfg, m)
            tx = apply_timing_offset(tx, offset)
            rx = awgn(tx, snr_db, rng)
            if decode(rx, cfg) != m:
                errors += 1
        ser[i] = errors / n_symbols
    return ser


def ser_vs_model_mismatch(
    cfg_tx: ChirpConfig,
    g_rx_variants: Sequence[Callable[[np.ndarray], np.ndarray]],
    snr_db: float,
    n_symbols: int = 200,
    demod: str = "mfbank",
    seed: Optional[int] = None,
) -> np.ndarray:
    """SER when the receiver's assumed trajectory g_rx differs from the transmitter's g_tx.

    Models section 9's concern: the receiver's reference must match f(t) precisely,
    or the accumulated phase error Delta_phi(t) = 2*pi*cumsum(f_tx - f_rx)/fs degrades
    the correlation peak (see metrics.phase_error for the underlying quantity).
    """
    rng = np.random.default_rng(seed)
    decode = _DEMODS[demod]
    ser = np.empty(len(g_rx_variants))
    for i, g_rx in enumerate(g_rx_variants):
        cfg_rx = replace(cfg_tx, g=g_rx)
        errors = 0
        for _ in range(n_symbols):
            m = int(rng.integers(0, cfg_tx.M))
            tx = symbol_waveform(cfg_tx, m)
            rx = awgn(tx, snr_db, rng)
            if decode(rx, cfg_rx) != m:
                errors += 1
        ser[i] = errors / n_symbols
    return ser
