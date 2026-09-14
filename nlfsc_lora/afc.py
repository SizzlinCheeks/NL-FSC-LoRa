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

A second, separately discovered limitation, found only by testing runs much
longer than the original 200-burst demo: dual_edge_cfo_estimate's own
mismatch-based quality gate occasionally (~1 in 400, measured directly) lets
through a measurement that is wildly wrong (1-4.5 kHz off) while still
reporting a passing mismatch score -- a noisy cubic fit's rate coefficient
can look self-consistent even when its frequency coefficient is badly off.
Over a couple hundred bursts that is unlikely to bite; over a multi-thousand-
burst run (any realistic satellite or UAV pass) it is close to certain to
happen at least once, and without a second line of defense the loop has no
way to tell a genuine correction from a corrupted one -- it trusts the
outlier, the next decode fails because the residual now exceeds
fft_correlation_demod's own capture range, that failure corrupts the
following measurement too, and the loop never recovers for the rest of the
run. Both AFCLoop (`max_jump_hz`) and KalmanAFCLoop (`innovation_gate`) now
gate outlier measurements before they reach the tracked state, which fixes
this directly (see each class's docstring).

Two more things this project asked of the design were tested directly rather
than assumed: whether a Kalman filter tracking CFO *and* its rate jointly
(KalmanAFCLoop, below) actually reduces AFCLoop's documented steady-state
lag under a constant drift (examples/output/17_kalman_vs_expfilter_ramp.png
-- a real but modest improvement once gated correctly), and whether the same
two-edge measurement holds up against a Doppler that reverses sign mid-run
-- a UAV or drone passing near the receiver, approaching then receding,
rather than a satellite's roughly-monotonic ramp (examples/output/
18_uav_flyover_afc.png and 19_afc_rate_of_change_limit.png). It does, with a
large margin: both trackers hold ~1.0 decode accuracy up to a peak
instantaneous Doppler rate of roughly 55 Hz/burst, a real cliff found by
sweeping rather than assumed, but one that sits many orders of magnitude
past any physically realistic satellite or UAV Doppler acceleration.
"""

from collections import Counter
from dataclasses import dataclass, field
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
    straight to the latest measurement with no smoothing.

    `max_jump_hz` gates outlier measurements before they ever reach the filter.
    dual_edge_cfo_estimate's own mismatch-based quality check occasionally
    misses a bad edge: measured directly, about 1 in 400 measurements land
    1-4.5 kHz from the truth while still reporting a passing mismatch score
    (the rate coefficient a noisy cubic fit produces can look self-consistent
    even when the frequency coefficient is badly wrong). At ~200 Hz measurement
    noise std, a genuine measurement essentially never lands beyond a few
    hundred Hz of the current estimate -- over a couple hundred bursts that
    1-in-400 rate is unlikely to bite, but over a multi-thousand-burst run
    (any realistic satellite or UAV pass) it is close to certain to happen at
    least once, and with no gate, AFCLoop has no way to tell a genuine large
    correction from a corrupted one: it trusts the outlier, the next decode
    fails because the residual now exceeds fft_correlation_demod's own
    capture range, that failure corrupts the following measurement too, and
    the loop never recovers for the rest of the run. Default 1200 Hz is about
    6 measurement-noise standard deviations, comfortably above real jitter and
    below the outlier magnitudes actually observed."""
    gain: float = 0.3
    cfo_tracked: float = 0.0
    max_jump_hz: Optional[float] = 1200.0

    def update(self, cfo_measured: Optional[float]) -> float:
        if cfo_measured is not None:
            if self.max_jump_hz is None or abs(cfo_measured - self.cfo_tracked) <= self.max_jump_hz:
                self.cfo_tracked += self.gain * (cfo_measured - self.cfo_tracked)
        return self.cfo_tracked


@dataclass
class KalmanAFCLoop:
    """Constant-velocity Kalman filter tracking CFO and CFO *rate* jointly, one
    burst per step, instead of AFCLoop's single fixed-gain exponential filter.

    AFCLoop only ever chases the latest measurement by a fixed fraction `gain`,
    which is why a linearly drifting CFO (Chapter 6's satellite-pass scenario)
    settles into a steady-state lag proportional to drift_rate/gain -- a classic
    type-1 tracking-loop error, not measurement noise. This tracker instead keeps
    an explicit running estimate of the rate itself and predicts forward with it
    every burst, so a *constant* drift rate is tracked with (in principle) no
    steady-state lag at all, and a *changing* drift rate -- a UAV whose Doppler
    swings from positive to negative as it passes overhead, rather than a
    satellite's roughly-monotonic ramp -- is followed as the rate estimate itself
    updates, instead of only ever reacting to where the CFO already is.

    This is the textbook generalization of two other fixes for the same lag
    (an integral term added to the loop filter; a larger gain during fast drift,
    smaller once converged): a tuned constant-velocity Kalman filter contains
    both as special cases of its own gain schedule, computed automatically from
    the noise parameters below rather than hand-tuned. The cost is two noise
    parameters to set instead of one gain, and, like any linear tracker, no
    guarantee of catching a rate change sharper than `process_noise_rate` allows
    for -- a real discontinuity is still smoothed over a handful of bursts.

    `innovation_gate` rejects statistically implausible measurements before
    they update the state -- the standard fix (see e.g. Kalman-filter-based GPS
    carrier tracking loops) for the same outlier problem AFCLoop's max_jump_hz
    addresses: dual_edge_cfo_estimate occasionally (~1 in 400, measured) passes
    its own mismatch check while still being off by 1-4.5 kHz, and over a long
    enough run that is close to certain to happen at least once. Comparing the
    squared innovation against `gate * S` (S being this filter's own predicted
    measurement variance, so the gate adapts to the filter's current
    confidence rather than using one fixed Hz threshold) is the Kalman-native
    version of the same check; a gated-out measurement still contributes a
    predict-only step so the state doesn't stall waiting for the next one.
    """
    process_noise_cfo: float = 1.0        # (Hz)^2 per burst: unmodeled CFO jitter beyond the rate model
    process_noise_rate: float = 0.2       # (Hz/burst)^2 per burst: how fast the rate itself can change
    # Measured, not assumed: dual_edge_cfo_estimate's own noise at SF7/10dB SNR is
    # ~184 Hz std (~33,800 Hz^2) -- an initial guess of 400 Hz^2 here (~20 Hz std)
    # was off by ~85x and made the filter overtrust individual measurements.
    measurement_noise: float = 34000.0    # (Hz)^2: variance of one dual-edge CFO measurement
    innovation_gate: Optional[float] = 25.0  # reject if innovation^2 > gate * S (~5 sigma)
    cfo_tracked: float = 0.0
    rate_tracked: float = 0.0
    _P: np.ndarray = field(default_factory=lambda: np.diag([1.0e8, 1.0e8]))

    def update(self, cfo_measured: Optional[float]) -> float:
        F = np.array([[1.0, 1.0], [0.0, 1.0]])  # cfo[i+1] = cfo[i] + rate[i]; rate[i+1] = rate[i]
        Q = np.diag([self.process_noise_cfo, self.process_noise_rate])
        x = F @ np.array([self.cfo_tracked, self.rate_tracked])
        P = F @ self._P @ F.T + Q

        if cfo_measured is not None:
            H = np.array([1.0, 0.0])
            innovation = cfo_measured - H @ x
            S = H @ P @ H.T + self.measurement_noise
            if self.innovation_gate is None or innovation ** 2 <= self.innovation_gate * S:
                K = (P @ H) / S
                x = x + K * innovation
                P = P - np.outer(K, H) @ P

        self.cfo_tracked, self.rate_tracked = x
        self._P = P
        return self.cfo_tracked


def _combine_acquisition(results: Sequence[Tuple[int, float]]) -> float:
    """Combine several bursts' independent joint_cfo_symbol_search results into
    one acquired CFO. Majority-vote the decoded symbol first (robust to an
    occasional single-burst false peak -- searching many CFO candidates against
    one noisy burst is itself less reliable than a plain single-hypothesis
    decode, discovered directly: at -15dB SNR a single acquisition burst had a
    27% symbol error rate where plain decoding had 0%), then take the *median*
    CFO among bursts agreeing with that majority symbol -- median rather than
    mean for the same reason max_jump_hz/innovation_gate use robust statistics
    elsewhere in this module: it isn't dragged far by one remaining outlier the
    majority vote didn't already exclude."""
    m_values = [m for m, _ in results]
    majority_m = Counter(m_values).most_common(1)[0][0]
    agreeing_cfos = [cfo for m, cfo in results if m == majority_m]
    return float(np.median(agreeing_cfos))


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
    acquire_bursts: int = 1,
    seed: Optional[int] = None,
    loop: Optional[object] = None,
):
    """Simulate receiving one burst per (symbol, true CFO) pair, correcting each with
    the *current* tracked estimate before decoding, then updating the tracker from
    that burst's dual-edge measurement (using the decoded symbol, not the true one --
    a real receiver doesn't know the true symbol either).

    If acquire is True, the first `acquire_bursts` bursts are handled specially:
    fft_correlation_demod only works once the residual CFO is already within its
    own half-bin capture range, so a cold start against a large offset needs a
    wide-search acquisition step first (sync.py's joint_cfo_symbol_search) to get
    the loop within range -- see the module docstring for what happens without
    this (divergence, not slow convergence). Every burst after the acquisition
    window uses only the cheap dual-edge loop.

    `acquire_bursts` > 1 (e.g. a multi-symbol preamble) runs the search
    independently on each of the first `acquire_bursts` bursts and combines them
    with `_combine_acquisition` rather than trusting a single burst: searching
    many CFO candidates is itself more failure-prone at a given SNR than a plain
    decode (more hypotheses tested, more chances for a noise-induced false
    peak), so a single-burst acquisition needs meaningfully higher SNR to be
    reliable than steady-state tracking does. Combining several independent
    acquisition bursts (exactly what a multi-symbol preamble is for) closes
    most of that gap -- measured directly, majority-voting 8 bursts dropped a
    27% single-burst acquisition error rate at -15dB SNR to 0%.

    `loop` lets the tracker itself be swapped in (e.g. a KalmanAFCLoop instead of
    the default AFCLoop(gain=gain)) without duplicating this whole pipeline; any
    object exposing a mutable `.cfo_tracked` attribute and an `.update(measurement)`
    method works. Defaults to `AFCLoop(gain=gain)`, matching prior behavior exactly.

    Returns three arrays of length len(true_symbols): decoded symbols, the tracked
    CFO estimate *before* each burst was corrected (what the receiver actually used),
    and the residual CFO each burst was decoded under (true - tracked).
    """
    rng = np.random.default_rng(seed)
    loop = AFCLoop(gain=gain) if loop is None else loop
    decoded = np.empty(len(true_symbols), dtype=int)
    tracked_history = np.empty(len(true_symbols))
    residual_history = np.empty(len(true_symbols))
    acquire_candidates = np.arange(-acquire_span, acquire_span + acquire_step, acquire_step)
    acquire_bursts = max(1, acquire_bursts) if acquire else 0
    acquire_results = []

    for i, (m_true, cfo_true) in enumerate(zip(true_symbols, true_cfo_sequence)):
        cfo_tracked = loop.cfo_tracked
        tracked_history[i] = cfo_tracked
        residual_history[i] = cfo_true - cfo_tracked

        tx = apply_cfo(symbol_waveform(cfg, m_true), cfo_true, cfg.sample_rate)
        rx = awgn(tx, snr_db, rng)

        if acquire and i < acquire_bursts:
            m_hat, cfo_hat, _score = joint_cfo_symbol_search(rx, cfg, acquire_candidates)
            decoded[i] = m_hat
            acquire_results.append((m_hat, cfo_hat))
            if i == acquire_bursts - 1:
                loop.cfo_tracked = _combine_acquisition(acquire_results)
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
