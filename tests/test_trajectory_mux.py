import numpy as np
import pytest

from nlfsc_lora.channel import awgn
from nlfsc_lora.chirp import ChirpConfig
from nlfsc_lora.trajectories import TRAJECTORIES, hyperbolic_center_freq
from nlfsc_lora.trajectory_mux import demux_stream, mix_streams


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
