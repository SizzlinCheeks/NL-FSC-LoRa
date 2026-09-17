"""Pins for examples/interference_experiments.py's findings: a real LoRaWAN
receiver (lora_phy) shows the capture effect (a sufficiently dominant
desired packet survives a same-strength or weaker interferer) and real
spreading-factor quasi-orthogonality (a different-SF interferer is far
less disruptive than a same-SF one, even much stronger). Skips cleanly if
lora_phy isn't installed.
"""
import os
import sys

import pytest

pytest.importorskip("lora_phy")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
from interference_experiments import collision_trial


def test_no_interferer_always_decodes():
    n_trials = 15
    fails = 0
    for t in range(n_trials):
        ok = collision_trial(7, 7, sir_db=999.0, snr_db=15.0, offset_frac=0.0, seed=t)
        fails += int(not ok)
    assert fails == 0


def test_same_sf_capture_requires_desired_to_be_stronger():
    """Same-SF, full overlap: a desired packet noticeably weaker than the
    interferer should essentially never be captured, and one noticeably
    stronger should essentially always be."""
    n_trials = 20
    weaker_ok = sum(collision_trial(7, 7, sir_db=-10.0, snr_db=15.0, offset_frac=0.0, seed=t) for t in range(n_trials))
    stronger_ok = sum(collision_trial(7, 7, sir_db=10.0, snr_db=15.0, offset_frac=0.0, seed=t) for t in range(n_trials))
    assert weaker_ok == 0
    assert stronger_ok == n_trials


def test_different_sf_interferer_is_far_less_disruptive_than_same_sf():
    """The real SF quasi-orthogonality property, checked directly: at an
    SIR where a same-SF interferer already defeats capture completely, a
    different-SF interferer at the *same* SIR should still let the desired
    packet through most of the time."""
    sir_db = -8.0  # same-SF should fail here (0% observed well below this in exploration)
    n_trials = 20
    same_sf_ok = sum(collision_trial(7, 7, sir_db, 15.0, 0.0, seed=t) for t in range(n_trials))
    diff_sf_ok = sum(collision_trial(7, 9, sir_db, 15.0, 0.0, seed=t) for t in range(n_trials))
    assert same_sf_ok <= 2  # essentially never captured
    assert diff_sf_ok >= 0.6 * n_trials  # captured most of the time despite the same nominal SIR


def test_a_strong_interferer_breaks_decoding_as_long_as_it_reaches_the_tail():
    """The overlap finding, pinned as actually measured (not as first
    guessed): since the interferer here is the same duration as the desired
    packet, starting it anywhere from the very beginning up through roughly
    90% of the way into the desired packet still means it overlaps all the
    way through to the desired packet's own tail -- and a clearly-dominant
    interferer breaks decoding for every one of those starting points, not
    just a full-overlap one. Capture only recovers once the interferer
    starts late enough that it *stops* reaching the tail (the CRC-bearing
    symbols and the final interleaved block), which for equal-length
    packets only happens in the last ~10% of possible starting points."""
    sir_db = -3.0
    n_trials = 20
    mid_overlap_ok = sum(collision_trial(7, 7, sir_db, 15.0, offset_frac=0.5, seed=t) for t in range(n_trials))
    near_tail_overlap_ok = sum(collision_trial(7, 7, sir_db, 15.0, offset_frac=0.90, seed=t) for t in range(n_trials))
    past_tail_ok = sum(collision_trial(7, 7, sir_db, 15.0, offset_frac=0.97, seed=t) for t in range(n_trials))
    assert mid_overlap_ok == 0  # still reaches the tail -- breaks it
    assert near_tail_overlap_ok == 0  # still just barely reaches the tail -- still breaks it
    assert past_tail_ok >= 0.8 * n_trials  # starts late enough to miss the tail entirely -- recovers
