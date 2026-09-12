import numpy as np
import pytest

from nlfsc_lora.chirp import ChirpConfig, symbol_waveform
from nlfsc_lora.receiver import dechirp, fft_demod, matched_filter_bank_demod
from nlfsc_lora.trajectories import TRAJECTORIES


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
