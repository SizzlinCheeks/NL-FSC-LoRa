import os
import sys

import numpy as np
import pytest

from nlfsc_lora.channel import awgn
from nlfsc_lora.chirp import ChirpConfig
from nlfsc_lora.trajectories import TRAJECTORIES, hyperbolic_center_freq
from nlfsc_lora.trajectory_mux import demux_stream, mix_streams

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
from trajectory_mimo_experiments import (
    SHAPES,
    _add_noise_unit_ref_power,
    cross_trajectory_leakage,
    make_cfg as make_shared_band_cfg,
)


def make_cfg(traj="linear", sf=7, bw=1000.0, os=4, f_center=None):
    """A shared, non-zero f_center is required for genuine multiplexing:
    all streams must occupy the exact same band to be a meaningful test of
    trajectory-domain separation (a stream parked at a different center
    frequency would trivially separate by filtering, proving nothing).
    Defaults to hyperbolic_center_freq(bw) so hyperbolic's own band-must-
    not-cross-zero constraint is satisfied by the *shared* band every
    other shape sits in too, not by shifting hyperbolic off on its own."""
    g, _ = TRAJECTORIES[traj]
    fc = hyperbolic_center_freq(bw) if f_center is None else f_center
    return ChirpConfig(sf=sf, bandwidth=bw, sample_rate=os * bw, g=g, f_center=fc)


def test_mix_streams_rejects_mismatched_symbol_durations():
    cfg_a = make_cfg("linear", sf=7)
    cfg_b = make_cfg("hyperbolic", sf=9)
    with pytest.raises(ValueError):
        mix_streams([cfg_a, cfg_b], [0, 0])


def test_two_stream_mux_recovers_both_symbols_noiseless():
    """The basic correctness check: two different-shaped streams sharing
    one band and time slot, mixed at equal power, should each demux back
    to their own true symbol with no noise present -- for every symbol
    pair, not just a lucky one."""
    cfg_lin = make_cfg("linear")
    cfg_hyp = make_cfg("hyperbolic")
    rng = np.random.default_rng(0)
    for _ in range(30):
        m_lin, m_hyp = rng.integers(0, cfg_lin.M, size=2)
        mixed = mix_streams([cfg_lin, cfg_hyp], [m_lin, m_hyp])
        assert demux_stream(mixed, cfg_lin) == m_lin
        assert demux_stream(mixed, cfg_hyp) == m_hyp


def test_four_stream_mux_recovers_every_symbol_noiseless():
    """Scaling up to four differently-shaped streams at once, equal power,
    no noise: every stream should still demux to its own true symbol.
    (Five equal-power streams at this toy SF/bandwidth already shows an
    occasional noiseless error -- examples/trajectory_mimo_experiments.py
    characterizes exactly where that capacity edge sits, rather than this
    test asserting perfection past the point it actually holds.)"""
    shapes = ["linear", "quadratic", "sigmoid", "exponential"]
    cfgs = [make_cfg(s) for s in shapes]
    rng = np.random.default_rng(1)
    for _ in range(20):
        symbols = rng.integers(0, cfgs[0].M, size=len(cfgs))
        mixed = mix_streams(cfgs, symbols)
        for cfg, m_true in zip(cfgs, symbols):
            assert demux_stream(mixed, cfg) == m_true


def test_mux_survives_moderate_noise():
    """Not just a noiseless correctness check: at a modest per-stream SNR,
    both streams of a two-stream mux should still decode correctly on
    average across many trials."""
    cfg_lin = make_cfg("linear")
    cfg_hyp = make_cfg("hyperbolic")
    rng = np.random.default_rng(2)
    fails = 0
    n_trials = 60
    for t in range(n_trials):
        m_lin, m_hyp = rng.integers(0, cfg_lin.M, size=2)
        mixed = mix_streams([cfg_lin, cfg_hyp], [m_lin, m_hyp])
        rx = awgn(mixed, snr_db=10.0, rng=rng)
        fails += int(demux_stream(rx, cfg_lin) != m_lin)
        fails += int(demux_stream(rx, cfg_hyp) != m_hyp)
    assert fails <= 0.1 * (2 * n_trials)


def test_leakage_matrix_diagonal_is_one_and_off_diagonal_is_well_separated():
    """Pins the headline behind 33_trajectory_mimo_leakage_matrix.png: a
    pure shape-i signal always looks perfectly like itself to its own
    receiver (diagonal == 1.0 exactly, by construction), and looks
    substantially less like a valid symbol to every *other* shape's
    receiver -- well below 1.0, not just marginally."""
    cfgs = {s: make_shared_band_cfg(s) for s in SHAPES}
    rng = np.random.default_rng(0)
    for si in SHAPES:
        assert cross_trajectory_leakage(cfgs[si], cfgs[si], n_symbols=5, rng=rng) == pytest.approx(1.0)
    for si in SHAPES:
        for sj in SHAPES:
            if si == sj:
                continue
            leak = cross_trajectory_leakage(cfgs[si], cfgs[sj], n_symbols=20, rng=rng)
            assert leak < 0.65


def test_five_streams_hit_an_interference_floor_two_streams_do_not():
    """Pins the headline behind 34_trajectory_mimo_capacity_and_nearfar.png's
    left panel: at a good SNR (well past where a single stream's own noise
    floor matters), two simultaneous streams still decode essentially
    perfectly, but five equal-power streams settle into a real, nonzero
    SER floor driven by mutual interference, not noise -- more SNR alone
    does not fix it, because there isn't enough of that at this stream
    count."""
    good_snr_db = 6.0
    n_trials = 150

    def mean_ser(k):
        cfgs = [make_shared_band_cfg(s) for s in SHAPES[:k]]
        rng = np.random.default_rng(42)
        errors = 0
        for _ in range(n_trials):
            symbols = rng.integers(0, cfgs[0].M, size=k)
            mixed = mix_streams(cfgs, symbols)
            rx = _add_noise_unit_ref_power(mixed, good_snr_db, rng)
            for cfg, m_true in zip(cfgs, symbols):
                errors += int(demux_stream(rx, cfg) != m_true)
        return errors / (n_trials * k)

    assert mean_ser(2) == 0.0
    assert mean_ser(5) > 0.003


def test_near_far_victim_survives_balanced_power_and_fails_when_swamped():
    """Pins the headline behind 34_trajectory_mimo_capacity_and_nearfar.png's
    right panel: a victim stream at a comfortable SNR decodes correctly
    when the aggressor sharing its slot is no stronger than it is, but
    fails almost every time once the aggressor is swamping it by a wide
    margin -- trajectory-domain separation degrades under real power
    imbalance rather than being immune to it."""
    cfg_victim = make_shared_band_cfg("linear")
    cfg_aggressor = make_shared_band_cfg("hyperbolic")
    victim_snr_db = 10.0
    n_trials = 100

    def victim_ser(ratio_db):
        rng = np.random.default_rng(7)
        aggressor_amp = 10 ** (ratio_db / 20.0)
        errors = 0
        for _ in range(n_trials):
            m_victim, m_aggr = rng.integers(0, cfg_victim.M, size=2)
            mixed = mix_streams([cfg_victim, cfg_aggressor], [m_victim, m_aggr],
                                 amplitudes=[1.0, aggressor_amp])
            rx = _add_noise_unit_ref_power(mixed, victim_snr_db, rng)
            errors += int(demux_stream(rx, cfg_victim) != m_victim)
        return errors / n_trials

    assert victim_ser(0.0) == 0.0
    assert victim_ser(20.0) >= 0.9
