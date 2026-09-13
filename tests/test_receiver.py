import numpy as np
import pytest

from nlfsc_lora.channel import awgn
from nlfsc_lora.chirp import ChirpConfig, symbol_waveform
from nlfsc_lora.receiver import dechirp, fft_correlation_demod, fft_demod, matched_filter_bank_demod
from nlfsc_lora.trajectories import TRAJECTORIES, hyperbolic_center_freq


def make_cfg(traj="linear", sf=5, bw=1000.0, os=4):
    g, _ = TRAJECTORIES[traj]
    return ChirpConfig(sf=sf, bandwidth=bw, sample_rate=os * bw, g=g)


def test_dechirp_cancels_matched_base_symbol():
    cfg = make_cfg("linear")
    tx = symbol_waveform(cfg, 0)
    d = dechirp(tx, cfg)
    assert np.allclose(d, 1.0, atol=1e-9)


def test_fft_demod_exact_for_linear_chirp_noiseless():
    cfg = make_cfg("linear")
    for m in range(cfg.M):
        tx = symbol_waveform(cfg, m)
        assert fft_demod(tx, cfg) == m


@pytest.mark.parametrize("traj", TRAJECTORIES.keys())
def test_matched_filter_bank_exact_noiseless(traj):
    cfg = make_cfg(traj)
    for m in range(cfg.M):
        tx = symbol_waveform(cfg, m)
        assert matched_filter_bank_demod(tx, cfg) == m


def test_fft_demod_degrades_for_strongly_nonlinear_chirp():
    """The FFT-bin shortcut relies on a linear chirp's shift-invariant df/dt;
    it is not expected to hold up for a sharply nonlinear trajectory."""
    g, _ = TRAJECTORIES["sigmoid"]
    cfg = ChirpConfig(sf=6, bandwidth=1000.0, sample_rate=4000.0, g=g)
    errors = sum(1 for m in range(cfg.M) if fft_demod(symbol_waveform(cfg, m), cfg) != m)
    assert errors > 0


def test_fft_demod_fails_for_properly_embedded_hyperbolic_chirp():
    """HFM is nonlinear (f(t+tau)-f(t) is not constant in t) exactly like sigmoid or
    quadratic, so it loses the FFT-bin shortcut too -- independent of Doppler, and
    independent of whether f_center is placed correctly for HFM to be well-posed."""
    bw = 1000.0
    g, _ = TRAJECTORIES["hyperbolic"]
    cfg = ChirpConfig(sf=6, bandwidth=bw, sample_rate=4 * bw, g=g, f_center=hyperbolic_center_freq(bw))
    errors = sum(1 for m in range(cfg.M) if fft_demod(symbol_waveform(cfg, m), cfg) != m)
    assert errors > 0


@pytest.mark.parametrize("traj", TRAJECTORIES.keys())
def test_fft_correlation_demod_exact_noiseless(traj):
    """Unlike fft_demod, this reads the symbol off a correlation *lag*, not a
    dechirped frequency, so it doesn't depend on dechirping producing a pure tone --
    should work for every trajectory, not just linear."""
    cfg = make_cfg(traj)
    for m in range(cfg.M):
        tx = symbol_waveform(cfg, m)
        assert fft_correlation_demod(tx, cfg) == m


@pytest.mark.parametrize("traj", ["linear", "quadratic", "hyperbolic"])
def test_fft_correlation_demod_matches_matched_filter_bank_exactly(traj):
    """Same decision every time, not just similar accuracy -- it's the identical
    computation (correlation against all M references), just done via the
    correlation theorem instead of M separate dot products."""
    bw = 1000.0
    g, _ = TRAJECTORIES[traj]
    f_center = hyperbolic_center_freq(bw) if traj == "hyperbolic" else 0.0
    cfg = ChirpConfig(sf=6, bandwidth=bw, sample_rate=4 * bw, g=g, f_center=f_center)
    rng = np.random.default_rng(0)
    for _ in range(200):
        m = int(rng.integers(0, cfg.M))
        rx = awgn(symbol_waveform(cfg, m), -10.0, rng)
        assert fft_correlation_demod(rx, cfg) == matched_filter_bank_demod(rx, cfg)
