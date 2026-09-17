"""Pins for examples/duty_cycle_experiments.py's two findings: (1) real
between-packet Doppler drift under EU868-style 1% duty cycling almost always
exceeds this project's own half-bin decode-capture tolerance (so a stale,
uncorrected CFO estimate carried from one packet to the next essentially
never works -- every packet needs its own acquisition), and (2) for the
configurations where drift also approaches or exceeds a fixed acquisition
span, re-centering that same-width search on a rate-based prediction
(reusing KalmanAFCLoop's own predict-only step) recovers cases a
zero-centered search misses, at no extra search cost.
"""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
from duty_cycle_experiments import acquire, make_cfg, predicted_center


def test_predicted_center_matches_real_rate_times_elapsed_time():
    """predicted_center's own unit contract: rate0 must be Hz/burst (the
    tracker's native unit), not Hz/s -- pinned directly so a future caller
    can't silently reintroduce the unit-conversion bug this module's own
    exploration hit (a 244x error from passing Hz/s straight through)."""
    cfg = make_cfg(9, 125e3)
    true_rate_hz_per_s = 640.0
    gap_s = 46.33
    n_equiv_bursts = gap_s / cfg.symbol_duration
    rate_per_burst = true_rate_hz_per_s * cfg.symbol_duration

    predicted = predicted_center(0.0, rate_per_burst, n_equiv_bursts)
    expected = true_rate_hz_per_s * gap_s  # real elapsed drift, computed independently
    # predicted_center rounds n_equiv_bursts to the nearest whole burst (it steps the
    # tracker's own discrete, one-burst-per-step model), so it matches the continuous
    # expectation only up to that sub-burst quantization -- one burst's worth of rate,
    # not a numerical-precision tolerance.
    assert predicted == pytest.approx(expected, abs=rate_per_burst)


def test_predicted_center_is_off_by_the_unit_error_when_rate_not_converted():
    """The negative case, pinned explicitly: passing a Hz/s rate directly
    (skipping the symbol_duration conversion) does NOT give the right
    answer -- confirms the conversion in the test above is load-bearing, not
    cosmetic."""
    cfg = make_cfg(9, 125e3)
    true_rate_hz_per_s = 640.0
    gap_s = 46.33
    n_equiv_bursts = gap_s / cfg.symbol_duration

    wrong = predicted_center(0.0, true_rate_hz_per_s, n_equiv_bursts)  # unconverted -- wrong on purpose
    correct = true_rate_hz_per_s * gap_s
    assert abs(wrong - correct) > 100 * abs(correct)  # off by orders of magnitude, not a rounding difference


def test_rate_predicted_acquisition_recovers_jump_a_blind_search_misses():
    """The headline result: a jump that lands outside a zero-centered
    acquisition span (so a blind search cannot find it at all) is
    recoverable once the same-width search is re-centered on the
    rate-based prediction -- checked directly at a realistic SNR, not
    assumed from the mean prediction being correct in principle.

    span=1500Hz, not a wide +/-25kHz span: joint_cfo_symbol_search jointly
    searches (symbol, CFO), and at this SF/BW (bin width 244Hz) a wide span
    covers roughly 100 bin-widths, giving ~100 (symbol, CFO) alias
    hypotheses that can outscore the true one -- confirmed directly, not
    assumed: a wide span reproducibly locked onto a wrong symbol 2 bins away
    from the truth *even at 40dB SNR*, a systematic aliasing failure, not a
    noise problem (see examples/duty_cycle_experiments.py's own module
    docstring for the full story). A narrow, alias-safe span removes it, and
    is also the more realistic test: it's exactly the accuracy a correctly
    re-centered search actually needs. true_rate=637.3 (not a round 640) for
    the same reason -- a round rate x round gap landed the true CFO exactly
    on the blind search's own grid in early testing, an artifact, not a
    finding."""
    cfg = make_cfg(9, 125e3)
    half_bin = cfg.bandwidth / cfg.M / 2
    # step=50Hz, not 250Hz: at this SF/BW, half_bin=122Hz, so a 250Hz step exceeds this
    # project's own established "step must stay under half the half-bin tolerance" rule
    # (tests/test_afc.py::test_acquisition_step_must_stay_under_half_bin_or_it_silently_fails)
    # -- caught directly while building this experiment, not assumed safe by copying a step
    # size from a different (SF7/500kHz) configuration's own convention.
    span, step, n_bursts = 1500.0, 50.0, 8
    true_rate_per_burst = 637.3 * cfg.symbol_duration
    gap_s = 15.0
    n_equiv_bursts = gap_s / cfg.symbol_duration
    true_jump = true_rate_per_burst * n_equiv_bursts  # ~9.6kHz, outside a 1500Hz span

    assert true_jump > span, "test setup should exercise a jump outside the blind search's span"

    center_pred = predicted_center(0.0, true_rate_per_burst, n_equiv_bursts)
    snr_db = -10.0
    n_trials = 20

    ok_blind = ok_pred = 0
    for t in range(n_trials):
        acq_blind = acquire(cfg, true_jump, center=0.0, span=span, step=step, n_bursts=n_bursts, snr_db=snr_db, seed=t)
        acq_pred = acquire(cfg, true_jump, center=center_pred, span=span, step=step, n_bursts=n_bursts, snr_db=snr_db, seed=t)
        ok_blind += int(abs(acq_blind - true_jump) < half_bin)
        ok_pred += int(abs(acq_pred - true_jump) < half_bin)

    assert ok_blind == 0, "a jump outside the span should never be found by a zero-centered blind search"
    # Measured directly: a real, substantial recovery, not perfect (the prediction's own
    # accuracy plus this SNR's acquisition noise together leave some trials short). The
    # bar here is calibrated with margin for n_trials=20's sampling noise, not rounded up
    # to make the story cleaner than it is.
    assert ok_pred >= 0.3 * n_trials, "the rate-predicted center should recover a real fraction of jumps a blind search finds none of"
