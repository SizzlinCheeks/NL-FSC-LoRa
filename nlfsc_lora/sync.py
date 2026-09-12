"""CFO estimation and correction.

Two approaches, one that doesn't work well and one that does -- both kept
here because the first result is informative in its own right.

spectral_band_edges/estimate_cfo implement a real idea: every LoRa-style
symbol here has identical |FFT(symbol)| regardless of which of the M
symbols was sent (a cyclic time shift only adds a linear phase ramp in
frequency, never changes which frequencies are present -- shift theorem),
so in principle the received signal's occupied band is centered on
cfg.f_center plus whatever CFO is present, *independent of the unknown
symbol index*. Measured against real data, though, this blind single-symbol
estimate is dominated by the chirp's own spectral leakage: a rectangular-
windowed chirp's magnitude spectrum has on the order of 1-2% of its energy
spread outside the nominal swept band (turn-on/turn-off transients, plus a
phase discontinuity at the cyclic-shift wrap point for m != 0), and that
leakage is asymmetric enough to bias a 5%/95%-threshold or energy-centroid
estimate by several hundred Hz -- comparable to or larger than the very CFO
values (~100 Hz) this project cares about. It is kept here, with tests that
pin down exactly how large that bias is, as a documented negative result:
"look at the spectral occupancy of one received symbol" does not by itself
give a usable CFO estimate for this waveform family.

joint_cfo_symbol_search is what actually works: since the receiver already
knows the full set of M candidate reference symbols, searching a small grid
of candidate CFOs, correcting for each, and taking whichever (CFO, symbol)
pair gives the strongest matched-filter response is a working (if more
expensive) joint estimator -- effectively a coarse delay/Doppler ambiguity
search restricted to the frequency axis. See examples/run_experiments.py's
09_lora_cfo_correction.png for the resulting SER improvement.
"""

from typing import Sequence, Tuple

import numpy as np

from .chirp import ChirpConfig, all_symbol_waveforms


def spectral_band_edges(rx: np.ndarray, fs: float, energy_fraction: float = 0.05) -> Tuple[float, float]:
    """Frequencies where cumulative spectral energy crosses energy_fraction and
    1-energy_fraction -- an estimate of the occupied band's bottom and top edges."""
    n = len(rx)
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fs))
    power = np.fft.fftshift(np.abs(np.fft.fft(rx)) ** 2)
    cum = np.cumsum(power)
    cum /= cum[-1]
    f_low = float(np.interp(energy_fraction, cum, freqs))
    f_high = float(np.interp(1.0 - energy_fraction, cum, freqs))
    return f_low, f_high


def estimate_cfo(rx: np.ndarray, cfg: ChirpConfig, energy_fraction: float = 0.05) -> float:
    """CFO estimate: how far the received band's center sits from cfg.f_center."""
    f_low, f_high = spectral_band_edges(rx, cfg.sample_rate, energy_fraction)
    return (f_low + f_high) / 2.0 - cfg.f_center


def correct_cfo(rx: np.ndarray, cfo_est: float, fs: float) -> np.ndarray:
    n = np.arange(len(rx))
    return rx * np.exp(-1j * 2 * np.pi * cfo_est * n / fs)


def joint_cfo_symbol_search(rx: np.ndarray, cfg: ChirpConfig, cfo_candidates: Sequence[float]) -> Tuple[int, float, float]:
    """Try every candidate CFO correction, decode against all M references at each, and
    keep whichever (symbol, CFO) pair gives the strongest normalized matched-filter score.

    O(len(cfo_candidates) * M) correlations per symbol -- far more expensive than plain
    matched_filter_bank_demod, but it is what actually recovers a symbol once CFO exceeds
    the half-bin threshold that breaks the CFO-blind receiver (see joint_cfo_symbol_demod
    and 09_lora_cfo_correction.png).
    """
    refs = all_symbol_waveforms(cfg)
    norms = np.linalg.norm(refs, axis=1)
    best_score, best_m, best_cfo = -1.0, 0, 0.0
    for cfo in cfo_candidates:
        corrected = correct_cfo(rx, cfo, cfg.sample_rate)
        corr = np.abs(refs.conj() @ corrected) / (norms * np.linalg.norm(corrected) + 1e-15)
        m = int(np.argmax(corr))
        if corr[m] > best_score:
            best_score, best_m, best_cfo = float(corr[m]), m, float(cfo)
    return best_m, best_cfo, best_score


def joint_cfo_symbol_demod(rx: np.ndarray, cfg: ChirpConfig, cfo_span: float = 1500.0, cfo_step: float = 50.0) -> int:
    """Symbol-only wrapper around joint_cfo_symbol_search, matching the demod(rx, cfg)
    signature the rest of the codebase uses (simulate.py's _DEMODS, receiver.py's decoders)."""
    cfo_candidates = np.arange(-cfo_span, cfo_span + cfo_step, cfo_step)
    m, _cfo, _score = joint_cfo_symbol_search(rx, cfg, cfo_candidates)
    return m
