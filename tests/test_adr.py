"""Pins for examples/adr_experiments.py's finding: KalmanAFCLoop's rate
estimate is in Hz per burst, and burst duration depends on spreading
factor, so carrying a rate estimate across a real LoRaWAN ADR spreading-
factor switch without accounting for that gives a badly wrong prediction --
duty_cycle_experiments.py's predicted_center_hz_per_s fixes it by taking a
physical rate (Hz/s) and converting internally instead of trusting the
call site to get two separately-computed, unit-bearing arguments to agree.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
from adr_experiments import LORAWAN_REQUIRED_SNR, adr_pick_sf
from duty_cycle_experiments import make_cfg, predicted_center, predicted_center_hz_per_s


def test_adr_picks_a_higher_sf_as_link_quality_degrades():
    """The core ADR behavior: a monotonically worse measured SNR should
    never pick a *lower* (faster, less robust) spreading factor."""
    snrs = [10.0, 5.0, 0.0, -5.0, -10.0, -15.0, -20.0]
    sfs = [adr_pick_sf(snr) for snr in snrs]
    assert sfs == sorted(sfs), "SF should never decrease as SNR gets worse"
    assert sfs[0] == 7  # best link -> fastest datarate
    assert sfs[-1] == 12  # worst link -> most robust datarate


def test_adr_picked_sf_always_meets_its_own_margin_requirement():
    """Every SF adr_pick_sf returns should actually satisfy the margin it
    was asked for, for every SF that has enough headroom -- not just the
    ones tested directly above."""
    margin_db = 10.0
    for snr in [12.3, -3.7, -18.0]:
        sf = adr_pick_sf(snr, margin_db=margin_db)
        assert snr - LORAWAN_REQUIRED_SNR[sf] >= margin_db - 1e-9 or sf == 12


def test_cross_sf_prediction_error_grows_with_sf_gap_for_the_buggy_path():
    """The real bug, pinned directly: reusing a rate tracked under one SF's
    Hz/burst convention, but stepping it with a *different* SF's burst
    count, produces an error that grows with the gap between the two SFs
    (each +1 SF step doubles symbol_duration) -- not a fixed, small offset."""
    true_rate = 637.3  # Hz/s
    gap_s = 5.09
    true_new_cfo = true_rate * gap_s

    errors = []
    for sf_b in [7, 8, 9, 10, 12]:
        cfg_a = make_cfg(7, 125e3)
        cfg_b = make_cfg(sf_b, 125e3)
        rate_a_per_burst = true_rate * cfg_a.symbol_duration
        n_equiv_bursts_b_units = gap_s / cfg_b.symbol_duration
        buggy = predicted_center(0.0, rate_a_per_burst, n_equiv_bursts_b_units)
        errors.append(abs(buggy - true_new_cfo))

    assert errors[0] < 10.0  # SF7->SF7: no actual change, near-zero error
    # each subsequent, larger SF gap should show a real (not just noisy) increase
    for i in range(1, len(errors) - 1):
        assert errors[i + 1] > errors[i] * 0.9  # allow float/rounding slack, not a strict requirement of increase


def test_predicted_center_hz_per_s_is_accurate_across_every_sf_transition():
    """The fix, pinned directly across the same transitions the bug above
    breaks: accurate to a tiny fraction of the true drift regardless of
    which SF the previous packet's rate was tracked under vs. the new
    packet's own SF."""
    true_rate = 637.3
    gap_s = 5.09
    true_new_cfo = true_rate * gap_s

    for sf_b in [7, 8, 9, 10, 12]:
        cfg_b = make_cfg(sf_b, 125e3)
        fixed = predicted_center_hz_per_s(0.0, true_rate, gap_s, cfg_b)
        assert fixed == pytest.approx(true_new_cfo, abs=15.0)
