"""Automatic frequency control (AFC): track and correct a drifting CFO across a
sequence of bursts using a per-burst residual-CFO measurement, instead of a
one-shot per-symbol correction.

The module's default per-burst measurement is full_symbol_cfo_estimate (below),
which dechirps the whole received symbol against its known decoded shape and
reads the residual CFO off the resulting tone's FFT peak -- full coherent gain
over the whole symbol, the same processing gain fft_correlation_demod itself
gets. This module started with a different measurement, dual_edge_cfo_estimate
(also still here, and still what the paragraphs immediately below describe),
which fits an 81-sample local window at each end of the symbol instead. That
design is kept for its own documented properties, but is no longer what
run_afc_sequence uses by default -- see the "negative-SNR" paragraph below for
why, and full_symbol_cfo_estimate's own docstring for the fix and its numbers.

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

A third, more fundamental limitation surfaced directly against this
project's own core requirement: LoRa's whole premise is decoding under the
noise floor, so a tracking mechanism that stops working at moderate SNR
defeats the point. Measured directly: dual_edge_cfo_estimate's 81-sample
cubic phase fit returns None on ~85-90% of bursts at -10 dB SNR -- even with
a held-still, zero-residual signal, nothing to track at all -- because it
never benefits from the ~27 dB of coherent processing gain
(~10*log10(n_samples) at SF7/500kHz) that fft_correlation_demod itself gets
from correlating the *whole* symbol; a small local window just doesn't have
enough samples to average the noise down. This is why decoding itself
survives to -20 to -30 dB while the tracking measurement was unusable below
roughly 5-8 dB -- not a subtle tuning gap, a missing factor of the same
processing gain the rest of the receiver already relies on. The fix,
full_symbol_cfo_estimate (below), gets that gain back by dechirping the
*whole* symbol against its known shape and reading the residual CFO off an
FFT peak instead of a local phase-derivative fit -- measured directly, this
holds a ~100 Hz std with zero outliers in 1000 trials at -10 dB SNR, and
degrades gracefully (sparser but still clean measurements, not silently
wrong ones) well below that. run_afc_sequence uses this by default now;
dual_edge_cfo_estimate is kept for its own documented properties (see
above) but no longer receives per-burst measurements in the main pipeline.

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


def full_symbol_cfo_estimate(
    rx: np.ndarray,
    cfg: ChirpConfig,
    m_hat: int,
    search_hz: Optional[float] = None,
    peak_ratio_reject: float = 5.0,
) -> Tuple[Optional[float], float]:
    """Per-burst residual-CFO measurement using the *whole* symbol's coherent
    gain, not a small local window -- the fix for a real limit discovered
    directly while pushing this project's own negative-SNR requirement (LoRa's
    entire premise is decoding under the noise floor): dual_edge_cfo_estimate's
    81-sample cubic phase fit throws away nearly all of the ~10*log10(n_samples)
    (~27 dB at SF7/500kHz's n_samples=512) processing gain that fft_correlation_demod
    itself gets "for free" from correlating the full symbol -- which is exactly why
    decoding survives to -20 to -30 dB SNR while the local edge measurement was
    unusable (its own mismatch gate rejecting ~85-90% of bursts, measured directly)
    below roughly 5-8 dB.

    The idea: once m_hat is known (from the receiver's own robust decode), multiply
    the received, already-CFO-corrected symbol by the conjugate of its exact expected
    waveform. Ideally this leaves a pure tone at the *residual* CFO (the deterministic
    nonlinear phase of the trajectory itself cancels exactly, by construction), buried
    in noise -- a textbook single-tone-in-noise frequency estimation problem, solved
    here by an FFT over the full symbol (the same coherent-integration length the
    decoder itself uses) with parabolic interpolation for sub-bin precision. Measured
    directly: at SF7/500kHz, zero-residual, this holds a ~100 Hz std with *zero*
    catastrophic outliers in 1000 trials at -10 dB SNR, where the old edge measurement
    was returning unusable garbage on the rare bursts it didn't reject outright.

    Below roughly -12 to -15 dB SNR this estimator hits the standard FFT-based
    frequency estimator "threshold effect": noise elsewhere in the spectrum
    occasionally outscores the true tone and the peak search locks onto the wrong
    bin, a large and clearly-wrong error rather than a small one. `search_hz` bounds
    the peak search to a window around zero (the residual CFO is always small after
    tracking corrects for the last known estimate) which already removes most
    false-bin picks; `peak_ratio_reject` (the winning bin's magnitude relative to the
    spectrum's mean, a cheap and effective proxy for "is this actually a tone or just
    the loudest noise bin") gates out the rest -- measured directly, a threshold of
    5.0 lets through essentially every good measurement from -10 dB up (0 bad among
    1000 accepted at -10 dB) while still accepting an honest, if sparser, trickle of
    good measurements down to -20 dB (3/1000 accepted, all good) rather than a hard
    wall. Defaults to one FFT bin's width (`cfg.bandwidth / cfg.M`) for `search_hz`,
    matching fft_correlation_demod's own native capture tolerance.

    Returns (cfo_estimate_or_None, peak_ratio); None means the peak wasn't
    convincing enough to trust (burst untrustworthy at this SNR, not a signal
    that anything is wrong with m_hat or the loop)."""
    n = cfg.n_samples
    ref = symbol_waveform(cfg, m_hat)
    dechirped = rx * np.conj(ref)
    spectrum_mag = np.abs(np.fft.fft(dechirped))

    if search_hz is None:
        search_hz = cfg.bandwidth / cfg.M
    max_k = int(np.ceil(search_hz * n / cfg.sample_rate))
    candidates = np.concatenate([np.arange(0, max_k + 1), np.arange(n - max_k, n)])
    k0 = candidates[np.argmax(spectrum_mag[candidates])]

    k_minus, k_plus = (k0 - 1) % n, (k0 + 1) % n
    y0, y_minus, y_plus = spectrum_mag[k0], spectrum_mag[k_minus], spectrum_mag[k_plus]
    denom = y_minus - 2 * y0 + y_plus
    delta = 0.5 * (y_minus - y_plus) / denom if denom != 0 else 0.0
    k_est = k0 + delta
    if k_est > n / 2:
        k_est -= n

    peak_ratio = y0 / (np.mean(spectrum_mag) + 1e-12)
    if peak_ratio < peak_ratio_reject:
        return None, peak_ratio
    return k_est * cfg.sample_rate / n, peak_ratio


@dataclass
class AFCLoop:
    """First-order exponential loop filter: cfo_tracked chases each new trusted
    measurement by `gain` of the way there. gain=0 never adapts; gain=1 snaps
    straight to the latest measurement with no smoothing.

    `max_jump_hz` gates outlier measurements before they ever reach the filter.
    Originally found against dual_edge_cfo_estimate (its mismatch-based quality
    check occasionally misses a bad edge -- about 1 in 400 measurements, measured
    directly, land 1-4.5 kHz from the truth while still reporting a passing
    mismatch score). The module's default estimator is now full_symbol_cfo_estimate,
    whose own failure mode is different (an FFT threshold effect at very low SNR,
    largely caught by its own `peak_ratio_reject` gate already -- see its
    docstring), but this gate is kept as a second line of defense regardless of
    which estimator feeds it: with no gate at all, a genuine measurement lands
    within a few hundred Hz of the current estimate essentially always, so over a
    multi-thousand-burst run (any realistic satellite or UAV pass) even a rare
    outlier is close to certain to happen at least once, and without a gate
    AFCLoop has no way to tell a genuine large correction from a corrupted one:
    it trusts the outlier, the next decode fails because the residual now exceeds
    fft_correlation_demod's own capture range, that failure corrupts the
    following measurement too, and the loop never recovers for the rest of the
    run. Default 1200 Hz sits comfortably above real jitter from either estimator
    and below the outlier magnitudes actually observed."""
    gain: float = 0.3
    cfo_tracked: float = 0.0
    max_jump_hz: Optional[float] = 1200.0

    def update(self, cfo_measured: Optional[float]) -> float:
        if cfo_measured is not None:
            if self.max_jump_hz is None or abs(cfo_measured - self.cfo_tracked) <= self.max_jump_hz:
                self.cfo_tracked += self.gain * (cfo_measured - self.cfo_tracked)
        return self.cfo_tracked

    def set_acquired(self, cfo: float) -> None:
        """Seed the tracker with a trusted acquisition result -- full trust,
        not blended in via `update`'s partial gain (that's what causes the
        cold-start divergence the module docstring documents in the first
        place). AFCLoop carries no other hidden state, so this is just the
        assignment; KalmanAFCLoop overrides it to also reset its covariance."""
        self.cfo_tracked = cfo


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
    # Measured, not assumed, and re-measured after switching the module's default
    # per-burst measurement from dual_edge_cfo_estimate to full_symbol_cfo_estimate:
    # the old value (34000, ~184Hz std) was dual_edge_cfo_estimate's noise at SF7/
    # 10dB SNR. full_symbol_cfo_estimate is far more precise (its whole point --
    # full coherent gain over the symbol instead of an 81-sample window), so that
    # value badly overstated the new estimator's noise, making the filter under-
    # trust good measurements right when trusting them matters most: at -10dB SNR
    # through a Doppler reversal, this cost real decode accuracy (0.81 vs AFCLoop's
    # 0.94 on an otherwise-identical run) purely from being too slow to keep up with
    # the fast part of the reversal. Re-measured at SF7/125kHz/-10dB (the negative-
    # SNR regime this whole fix targets, not the 10dB point the old value used):
    # ~27 Hz std (~720 Hz^2).
    measurement_noise: float = 720.0      # (Hz)^2: variance of one full_symbol CFO measurement
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

    def set_acquired(self, cfo: float) -> None:
        """Seed the filter with a trusted acquisition result and correspondingly
        reduced uncertainty, instead of leaving `_P` at its pre-acquisition
        (near-infinite) default. Without this, the first few post-acquisition
        measurements get absorbed with a near-total Kalman gain (P >> R), which
        can drag the tracked value away from a good acquired estimate right at
        the acquisition-to-tracking handoff -- discovered directly while
        building an end-to-end packet test: KalmanAFCLoop's SER plateaued well
        above AFCLoop's under otherwise-identical conditions until this was
        fixed. CFO variance is set to `measurement_noise` itself (one
        steady-state measurement's worth of uncertainty -- conservative, since
        an acquisition combining several preamble bursts is usually better
        than that); rate variance starts at a moderate, not-infinite prior
        since acquisition carries no rate information at all."""
        self.cfo_tracked = cfo
        self.rate_tracked = 0.0
        self._P = np.diag([self.measurement_noise, 100.0])


def measurement_noise_for(cfg: ChirpConfig) -> float:
    """A configuration-scaled default for KalmanAFCLoop.measurement_noise, in place
    of one fixed number. full_symbol_cfo_estimate's noise is an FFT-peak-location
    estimate, so at fixed SNR its variance scales with the *square* of the FFT bin
    width (cfg.bandwidth / cfg.M) -- confirmed empirically, not assumed: measured at
    two very different configurations at the same SNR (SF7, -10dB), 125kHz gave
    ~720 Hz^2 (bin width 976.6Hz) and 500kHz gave ~11700 Hz^2 (bin width 3906.25Hz),
    a ~16.3x ratio against a predicted (3906.25/976.6)^2 = 16x -- a near-exact match.
    KalmanAFCLoop's dataclass default (720.0) is only correct for the 125kHz
    configuration chapter 6's examples and tests use; this function is the right
    default for any other configuration (in particular examples/packet_experiments.py's
    500kHz packets), calibrated at -10dB SNR, this project's target negative-SNR
    operating point.

    Discovered as a real bug, not a hypothetical one: without this, KalmanAFCLoop's
    500kHz packet-level Doppler-reversal test showed a real accuracy gap against
    AFCLoop at -10 to -14dB SNR (e.g. 37% vs 0% payload symbol errors at -10dB) that
    had nothing to do with the Kalman filter's tracking algorithm itself -- it
    was 720's ~16x-too-small variance making the filter overtrust individual
    500kHz measurements it should have been more skeptical of. Using this function's
    properly-scaled value closed the gap almost entirely (0.2% vs 0% at -12dB)."""
    bin_hz = cfg.bandwidth / cfg.M
    return 7.6e-4 * bin_hz ** 2


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
    search_hz: Optional[float] = None,
    peak_ratio_reject: float = 5.0,
    acquire: bool = True,
    acquire_span: float = 3000.0,
    acquire_step: float = 50.0,
    acquire_bursts: int = 1,
    seed: Optional[int] = None,
    loop: Optional[object] = None,
):
    """Simulate receiving one burst per (symbol, true CFO) pair, correcting each with
    the *current* tracked estimate before decoding, then updating the tracker from
    that burst's full_symbol_cfo_estimate measurement (using the decoded symbol, not
    the true one -- a real receiver doesn't know the true symbol either).

    Uses full_symbol_cfo_estimate, not dual_edge_cfo_estimate, for the per-burst
    measurement -- switched directly in response to this project's own negative-SNR
    requirement (LoRa's whole premise is decoding under the noise floor): the
    dual-edge small-window fit was found to be unusable below roughly 5-8 dB SNR
    (its own quality gate rejecting ~85-90% of bursts even with nothing to track),
    while the full-symbol version, which gets the same coherent-integration gain the
    decoder itself enjoys, holds a ~100 Hz measurement std with no observed outliers
    down to -10 dB and degrades gracefully (sparser but still clean measurements)
    below that -- see full_symbol_cfo_estimate's own docstring for the numbers.
    `dg` is accepted for backward-compatible call signatures and is no longer used by
    this function directly (dual_edge_cfo_estimate, which does need it, is still
    available to call directly -- see its own docstring for why it's kept in the
    module despite being superseded here).

    If acquire is True, the first `acquire_bursts` bursts are handled specially:
    fft_correlation_demod only works once the residual CFO is already within its
    own half-bin capture range, so a cold start against a large offset needs a
    wide-search acquisition step first (sync.py's joint_cfo_symbol_search) to get
    the loop within range -- see the module docstring for what happens without
    this (divergence, not slow convergence). Every burst after the acquisition
    window uses only the cheap per-burst measurement above.

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
    method works, and one optionally implementing `.set_acquired(cfo)` gets it
    called once at the end of acquisition instead of `.cfo_tracked` being poked
    directly -- AFCLoop's version is a plain assignment, but a loop carrying
    hidden state beyond cfo_tracked (KalmanAFCLoop's covariance) needs the
    chance to reset that state too, not just the visible estimate; skipping
    this for Kalman was a real bug, not a hypothetical one (see
    KalmanAFCLoop.set_acquired's docstring). Defaults to `AFCLoop(gain=gain)`,
    matching prior behavior exactly.

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
                acquired_cfo = _combine_acquisition(acquire_results)
                if hasattr(loop, "set_acquired"):
                    loop.set_acquired(acquired_cfo)
                else:
                    loop.cfo_tracked = acquired_cfo
            continue

        rx_corrected = correct_cfo(rx, cfo_tracked, cfg.sample_rate)
        m_hat = fft_correlation_demod(rx_corrected, cfg)
        decoded[i] = m_hat

        cfo_residual_est, _peak_ratio = full_symbol_cfo_estimate(
            rx_corrected, cfg, m_hat, search_hz=search_hz, peak_ratio_reject=peak_ratio_reject
        )
        # cfo_residual_est measures (true - tracked) from the corrected signal; feed the
        # loop the implied absolute CFO so it tracks true_cfo, not the residual alone.
        new_measurement = None if cfo_residual_est is None else cfo_tracked + cfo_residual_est
        loop.update(new_measurement)

    return decoded, tracked_history, residual_history
