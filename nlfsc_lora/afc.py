"""Dual-edge automatic frequency control (AFC): track and correct a drifting CFO
across a sequence of bursts using cheap local rate/frequency measurements at
both ends of the swept bandwidth, instead of a one-shot per-symbol correction.

This builds on local_rate.py's core insight (a nonlinear trajectory's local
rate varies with position in a way a constant CFO never disturbs, so it is a
CFO-independent position sensor) but applies it differently. local_rate.py
tries to *decode the symbol* from one local measurement, and pays for it with
a hard noise floor and a structural error band where the cyclic-shift wrap
glitch falls inside the fixed measurement window (examples/output/
10_local_rate_estimator.png). Here, the symbol comes from the receiver's own
already-robust decode (fft_correlation_demod, ~O(N log N), exact-agreement
with the brute-force decoder -- see receiver.py) -- rate/frequency sensing is
used only to *measure the residual CFO* once the symbol is already known, not
to guess it blindly. That sidesteps both of local_rate.py's weaknesses: the
edge measurement only has to be locally accurate, not globally decode-grade,
and a glitch-corrupted edge is caught by the internal consistency check below
(measured rate vs. the rate the known symbol predicts at that position)
rather than silently corrupting the estimate.

Why two edges instead of one center window (local_rate.py's choice): the
wrap glitch sits at one fixed sample position per symbol, so it can corrupt
at most one edge at a time. Two widely-separated measurement windows nearly
eliminates simultaneous corruption, and the consistency check below lets a
clean edge outvote (via a quality-weighted average) a glitch-corrupted one.

Why tracking across bursts rather than a one-shot correction: a receiver
sitting behind a *drifting* Doppler shift (the realistic case for anything
where Doppler is large enough to matter -- e.g. a LoRa-over-LEO-satellite
uplink) needs to keep re-centering as the raw CFO grows, not just correct
once. A first-order exponential loop filter (the same structure used in
classical AFC/PLL loops) accumulates noisy per-burst measurements into a
slowly-updated correction that follows the drift while damping noise -- see
examples/output/12_dual_edge_afc.png for the tradeoff between loop gain
(tracking speed) and residual jitter.

This loop has a real, discovered-not-assumed limitation: it can only track
a CFO that is *already* within fft_correlation_demod's own half-bin capture
range, because the burst it measures from is the one it just corrected and
decoded -- if the raw CFO exceeds that range, the decode itself is wrong,
which feeds a garbage measurement back into the loop and it diverges rather
than converging (confirmed directly: a naive loop started cold against a
1000 Hz offset at SF7, half-bin=488 Hz, ran away to tens of kHz within 50
bursts instead of homing in). This is the standard "pull-in range" problem
in AFC/PLL design, not a bug in the measurement itself -- the fix used here
is the standard fix: a one-time coarse acquisition (sync.py's
joint_cfo_symbol_search, the wide-grid search built earlier in this project)
gets the loop within capture range, then this cheap per-burst loop takes
over for ongoing tracking of drift. See run_afc_sequence's `acquire` option.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from .channel import apply_cfo, awgn
from .chirp import ChirpConfig, symbol_waveform
from .receiver import fft_correlation_demod
from .sync import correct_cfo, joint_cfo_symbol_search


def _edge_measurement(rx: np.ndarray, cfg: ChirpConfig, dg: Callable[[np.ndarray], np.ndarray],
                       m_hat: int, n_pos: int, half_win: int) -> Tuple[float, float]:
    """CFO estimate and a relative rate-mismatch (0 = perfectly consistent with m_hat,
    large = probably a glitch-corrupted window or a wrong m_hat) from one local window."""
    n = cfg.n_samples
    idx = (np.arange(n_pos - half_win, n_pos + half_win + 1)) % n
    phase = np.unwrap(np.angle(rx[idx]))
    k = np.arange(-half_win, half_win + 1)
    d, c, b, a = np.polyfit(k, phase, 3)
    freq_meas = b * cfg.sample_rate / (2 * np.pi)
    rate_meas = 2 * c * cfg.sample_rate ** 2 / (2 * np.pi)

    shift = round(m_hat * n / cfg.M) % n
    u_expected = ((n_pos + shift + 0.5) % n) / n
    f_expected = cfg.f_center - cfg.bandwidth / 2.0 + cfg.bandwidth * cfg.g(np.array([u_expected]))[0]
    rate_expected = (cfg.bandwidth / cfg.symbol_duration) * dg(np.array([u_expected]))[0]

    cfo_est = freq_meas - f_expected
    mismatch = abs(rate_meas - rate_expected) / (abs(rate_expected) + 1e-9)
    return cfo_est, mismatch


def dual_edge_cfo_estimate(
    rx: np.ndarray,
    cfg: ChirpConfig,
    dg: Callable[[np.ndarray], np.ndarray],
    m_hat: int,
    edge_half_win: int = 40,
    edge_margin: int = 40,
    mismatch_reject: float = 0.3,
) -> Tuple[Optional[float], float, float]:
    """Combine both edges' CFO estimates, quality-weighted by rate-mismatch (a clean
    edge downweights a glitch-corrupted or inconsistent one rather than being averaged
    down by it). Returns (cfo_estimate_or_None, mismatch_start, mismatch_end); the
    estimate is None if both edges exceed mismatch_reject (burst untrustworthy --
    e.g. m_hat is probably wrong, or both edges got unlucky)."""
    n = cfg.n_samples
    n_start = edge_margin + edge_half_win
    n_end = n - 1 - edge_margin - edge_half_win

    cfo_a, mismatch_a = _edge_measurement(rx, cfg, dg, m_hat, n_start, edge_half_win)
    cfo_b, mismatch_b = _edge_measurement(rx, cfg, dg, m_hat, n_end, edge_half_win)

    weight_a = 1.0 / (mismatch_a + 1e-3) if mismatch_a < mismatch_reject else 0.0
    weight_b = 1.0 / (mismatch_b + 1e-3) if mismatch_b < mismatch_reject else 0.0
    if weight_a == 0.0 and weight_b == 0.0:
        return None, mismatch_a, mismatch_b
    cfo_est = (weight_a * cfo_a + weight_b * cfo_b) / (weight_a + weight_b)
    return cfo_est, mismatch_a, mismatch_b


@dataclass
class AFCLoop:
    """First-order exponential loop filter: cfo_tracked chases each new trusted
    measurement by `gain` of the way there. gain=0 never adapts; gain=1 snaps
    straight to the latest measurement with no smoothing."""
    gain: float = 0.3
    cfo_tracked: float = 0.0

    def update(self, cfo_measured: Optional[float]) -> float:
        if cfo_measured is not None:
            self.cfo_tracked += self.gain * (cfo_measured - self.cfo_tracked)
        return self.cfo_tracked


def run_afc_sequence(
    cfg: ChirpConfig,
    dg: Callable[[np.ndarray], np.ndarray],
    true_symbols: Sequence[int],
    true_cfo_sequence: Sequence[float],
    snr_db: float,
    gain: float = 0.3,
    edge_half_win: int = 40,
    edge_margin: int = 40,
    acquire: bool = True,
    acquire_span: float = 3000.0,
    acquire_step: float = 50.0,
    seed: Optional[int] = None,
):
    """Simulate receiving one burst per (symbol, true CFO) pair, correcting each with
    the *current* tracked estimate before decoding, then updating the tracker from
    that burst's dual-edge measurement (using the decoded symbol, not the true one --
    a real receiver doesn't know the true symbol either).

    If acquire is True, the first burst is handled specially: fft_correlation_demod
    only works once the residual CFO is already within its own half-bin capture
    range, so a cold start against a large offset needs a wide-search acquisition
    step first (sync.py's joint_cfo_symbol_search) to get the loop within range --
    see the module docstring for what happens without this (divergence, not slow
    convergence). Every burst after the first uses only the cheap dual-edge loop.

    Returns three arrays of length len(true_symbols): decoded symbols, the tracked
    CFO estimate *before* each burst was corrected (what the receiver actually used),
    and the residual CFO each burst was decoded under (true - tracked).
    """
    rng = np.random.default_rng(seed)
    loop = AFCLoop(gain=gain)
    decoded = np.empty(len(true_symbols), dtype=int)
    tracked_history = np.empty(len(true_symbols))
    residual_history = np.empty(len(true_symbols))
    acquire_candidates = np.arange(-acquire_span, acquire_span + acquire_step, acquire_step)

    for i, (m_true, cfo_true) in enumerate(zip(true_symbols, true_cfo_sequence)):
        cfo_tracked = loop.cfo_tracked
        tracked_history[i] = cfo_tracked
        residual_history[i] = cfo_true - cfo_tracked

        tx = apply_cfo(symbol_waveform(cfg, m_true), cfo_true, cfg.sample_rate)
        rx = awgn(tx, snr_db, rng)

        if i == 0 and acquire:
            m_hat, cfo_acquired, _score = joint_cfo_symbol_search(rx, cfg, acquire_candidates)
            decoded[i] = m_hat
            loop.cfo_tracked = cfo_acquired
            continue

        rx_corrected = correct_cfo(rx, cfo_tracked, cfg.sample_rate)
        m_hat = fft_correlation_demod(rx_corrected, cfg)
        decoded[i] = m_hat

        cfo_residual_est, _mismatch_a, _mismatch_b = dual_edge_cfo_estimate(
            rx_corrected, cfg, dg, m_hat, edge_half_win=edge_half_win, edge_margin=edge_margin
        )
        # cfo_residual_est measures (true - tracked) from the corrected signal; feed the
        # loop the implied absolute CFO so it tracks true_cfo, not the residual alone.
        new_measurement = None if cfo_residual_est is None else cfo_tracked + cfo_residual_est
        loop.update(new_measurement)

    return decoded, tracked_history, residual_history
