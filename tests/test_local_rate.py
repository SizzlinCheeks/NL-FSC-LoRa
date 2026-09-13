import numpy as np
import pytest

from nlfsc_lora.channel import apply_cfo
from nlfsc_lora.chirp import ChirpConfig, symbol_waveform
from nlfsc_lora.local_rate import local_freq_and_rate, local_rate_demod, local_rate_estimate
from nlfsc_lora.trajectories import TRAJECTORIES


def make_cfg(traj="quadratic", sf=7, bw=125e3, os=4):
    g, _ = TRAJECTORIES[traj]
    return ChirpConfig(sf=sf, bandwidth=bw, sample_rate=os * bw, g=g)


@pytest.mark.parametrize("m_true,cfo_true", [(0, 0.0), (5, 300.0), (33, 300.0), (100, -150.0), (127, 500.0)])
def test_noiseless_symbol_and_cfo_recovered_exactly_for_quadratic(m_true, cfo_true):
    cfg = make_cfg("quadratic")
    _, dg = TRAJECTORIES["quadratic"]
    tx = apply_cfo(symbol_waveform(cfg, m_true), cfo_true, cfg.sample_rate)
    m_est, cfo_est = local_rate_estimate(tx, cfg, dg)
    assert m_est == m_true
    assert cfo_est == pytest.approx(cfo_true, abs=1.0)


def test_noiseless_symbol_recovered_for_hyperbolic():
    """m=20, half_win=50: unlike quadratic's g (a pure polynomial, for which the cubic
    local phase fit is exact at any window size -- see the quadratic test above, which
    passes for m up to 127 with the default half_win=100), hyperbolic's phase law is
    transcendental (an integral of 1/(1-beta*t)), so the cubic fit is only a good local
    approximation for a small enough window; a half_win=100 window picks up enough
    curvature mismatch to bias the estimate by a full symbol (verified separately: it
    stays correct through half_win=50 and starts misjudging the symbol at half_win=75)."""
    from nlfsc_lora.trajectories import hyperbolic_center_freq

    bw = 125e3
    f_center = hyperbolic_center_freq(bw)
    g, dg = TRAJECTORIES["hyperbolic"]
    cfg = ChirpConfig(sf=7, bandwidth=bw, sample_rate=4 * bw, g=g, f_center=f_center)
    tx = apply_cfo(symbol_waveform(cfg, 20), 200.0, cfg.sample_rate)
    m_est, cfo_est = local_rate_estimate(tx, cfg, dg, half_win=50)
    assert m_est == 20
    assert cfo_est == pytest.approx(200.0, abs=5.0)


def test_default_window_placement_fails_for_a_real_band_of_symbols():
    """Documents the practical consequence of the glitch-straddling failure mode: with
    the default n_center (n_samples/2) and half_win=100, roughly the middle third of
    all symbol values (by shift) put the glitch inside the window, and the estimate is
    wrong -- not degraded, just wrong -- regardless of noise. A real implementation
    would need multiple candidate windows (e.g. a majority vote across a few spread-out
    n_center choices) to work reliably across the whole symbol alphabet."""
    cfg = make_cfg("quadratic")
    _, dg = TRAJECTORIES["quadratic"]
    failures = 0
    for m in range(cfg.M):
        tx = symbol_waveform(cfg, m)
        m_est, _ = local_rate_estimate(tx, cfg, dg)
        if m_est != m:
            failures += 1
    assert failures > cfg.M // 6  # a real, non-trivial band of the alphabet, not a fluke


def test_linear_trajectory_rate_carries_no_positional_information():
    """The whole method depends on dg/du varying with position; a linear chirp's dg/du
    is a constant (dg=1 everywhere), so there is nothing to invert -- confirming this
    is not usable for a linear chirp, which is exactly why a nonlinear g was needed."""
    cfg = make_cfg("linear")
    _, dg = TRAJECTORIES["linear"]
    u = np.linspace(0, 1, 50)
    assert np.allclose(dg(u), dg(u)[0])  # dg/du is constant: no information in the rate


def test_window_straddling_the_wrap_glitch_corrupts_the_fit():
    """A large enough window crosses the cyclic-shift discontinuity and the fit breaks,
    rather than degrading gracefully -- a real structural limit, not just added noise."""
    cfg = make_cfg("quadratic")
    _, dg = TRAJECTORIES["quadratic"]
    m_true = 33
    tx = symbol_waveform(cfg, m_true)
    n_center = cfg.n_samples // 2
    m_small_window, _ = local_rate_estimate(tx, cfg, dg, n_center=n_center, half_win=50)
    assert m_small_window == m_true

    shift = round(m_true * cfg.n_samples / cfg.M) % cfg.n_samples
    glitch_index = (cfg.n_samples - shift) % cfg.n_samples
    # a window centered at n_center that comfortably straddles the glitch
    assert abs(glitch_index - n_center) < 150
    m_big_window, _ = local_rate_estimate(tx, cfg, dg, n_center=n_center, half_win=150)
    assert m_big_window != m_true
