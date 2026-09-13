"""Cheap symbol/CFO estimation from a single local instantaneous-chirp-rate measurement.

The idea this module tests: a linear chirp's instantaneous rate df/dt is the
same constant everywhere, so measuring it tells you nothing about *where* in
the symbol you are -- there is no local information to exploit, which is
exactly why receiver.py needs either a full-record FFT (fft_demod, and only
because of the shift-invariance special to linear chirps) or a full
correlation against every reference (matched_filter_bank_demod). A
nonlinear trajectory's rate varies with position in a *known* way, and
critically, adding a constant CFO shifts frequency without changing the
rate of change at all -- so the local rate alone identifies the symbol
independent of CFO, and once the symbol (hence the expected CFO-free
frequency) is known, the CFO falls out by comparison. No correlation
against any reference is needed at all.

Measured cost/accuracy tradeoff (see examples/run_experiments.py's
10_local_rate_estimator.png): roughly 50-90x cheaper per symbol than
matched_filter_bank_demod, but only reliable at much higher SNR (tens of dB
higher) -- there is no equivalent of LoRa's despreading gain from
correlating over the whole record, only whatever averaging the local
window buys. That averaging is itself capped: the window must stay clear
of the cyclic-shift wrap glitch (see chirp.py), whose location depends on
the very symbol index being estimated, so widening the window past a
certain point risks straddling it and corrupting the fit outright rather
than gradually degrading.
"""

from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import brentq

from .channel import awgn
from .chirp import ChirpConfig, symbol_waveform


def local_freq_and_rate(rx: np.ndarray, n_center: int, half_win: int, fs: float) -> Tuple[float, float]:
    """Fit unwrapped phase over a window to a cubic in sample index (exact for a
    quadratic g; a good local approximation otherwise) and read off the
    instantaneous frequency and chirp rate at n_center.

    The cubic order matters, not just a linear/quadratic-phase fit: a nonlinear
    g's rate itself changes across the window, and an under-fit polynomial
    lets that curvature leak into a systematic bias on the frequency term.
    """
    n = len(rx)
    idx = (np.arange(n_center - half_win, n_center + half_win + 1)) % n
    phase = np.unwrap(np.angle(rx[idx]))
    k = np.arange(-half_win, half_win + 1)
    d, c, b, a = np.polyfit(k, phase, 3)
    freq = b * fs / (2 * np.pi)
    rate = 2 * c * fs**2 / (2 * np.pi)
    return freq, rate


def _invert_rate_to_u(dg: Callable[[np.ndarray], np.ndarray], rate_target: float) -> float:
    """Numerically invert dg/du at rate_target, assuming dg is monotonic on (0, 1)."""
    lo, hi = dg(np.array([1e-6]))[0], dg(np.array([1 - 1e-6]))[0]
    target = float(np.clip(rate_target, min(lo, hi), max(lo, hi)))
    f = lambda u: dg(np.array([u]))[0] - target
    return brentq(f, 1e-6, 1 - 1e-6)


def local_rate_estimate(
    rx: np.ndarray,
    cfg: ChirpConfig,
    dg: Callable[[np.ndarray], np.ndarray],
    n_center: Optional[int] = None,
    half_win: int = 100,
) -> Tuple[int, float]:
    """Estimate (symbol, CFO) from one local rate/frequency measurement.

    dg must be the trajectory's rate function (dg/du) -- a closed form from
    nlfsc_lora.trajectories, or numerical_derivative(g, ...) as a fallback.
    Requires dg to be non-constant (a genuinely nonlinear trajectory): for a
    linear chirp dg/du is the same everywhere and the inversion is degenerate.
    """
    n = cfg.n_samples
    if n_center is None:
        n_center = n // 2
    freq_meas, rate_meas = local_freq_and_rate(rx, n_center, half_win, cfg.sample_rate)

    rate_target = rate_meas * cfg.symbol_duration / cfg.bandwidth  # = dg/du at the true u
    u_est = _invert_rate_to_u(dg, rate_target)

    # chirp.py's phase accumulator associates f[k] with the increment *entering* sample k
    # (a forward-difference convention), a half-sample offset from a centered rate/phase fit.
    shift_est = round(u_est * n - n_center - 0.5) % n
    m_est = round(shift_est * cfg.M / n) % cfg.M

    shift_for_m = round(m_est * n / cfg.M) % n
    u_expected = ((n_center + shift_for_m + 0.5) % n) / n
    f_expected = cfg.f_center - cfg.bandwidth / 2.0 + cfg.bandwidth * cfg.g(np.array([u_expected]))[0]
    cfo_est = freq_meas - f_expected
    return m_est, cfo_est


def local_rate_demod(rx: np.ndarray, cfg: ChirpConfig, dg: Callable[[np.ndarray], np.ndarray], half_win: int = 100) -> int:
    """Symbol-only wrapper matching the demod(rx, cfg) signature used elsewhere."""
    m_est, _cfo_est = local_rate_estimate(rx, cfg, dg, half_win=half_win)
    return m_est


def local_rate_ser_vs_snr(
    cfg: ChirpConfig,
    dg: Callable[[np.ndarray], np.ndarray],
    snr_db_range: Sequence[float],
    n_symbols: int = 200,
    half_win: int = 100,
    seed: Optional[int] = None,
) -> np.ndarray:
    """SER of local_rate_demod vs SNR -- mirrors simulate.ser_vs_snr's shape, kept separate
    because this decoder needs the trajectory's own dg, which the generic demod(rx, cfg)
    dispatch used elsewhere (simulate._DEMODS) doesn't carry."""
    rng = np.random.default_rng(seed)
    ser = np.empty(len(snr_db_range))
    for i, snr_db in enumerate(snr_db_range):
        errors = 0
        for _ in range(n_symbols):
            m = int(rng.integers(0, cfg.M))
            tx = symbol_waveform(cfg, m)
            rx = awgn(tx, snr_db, rng)
            if local_rate_demod(rx, cfg, dg, half_win=half_win) != m:
                errors += 1
        ser[i] = errors / n_symbols
    return ser
