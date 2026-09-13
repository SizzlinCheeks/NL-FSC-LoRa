import numpy as np
import pytest

from nlfsc_lora.chirp import ChirpConfig, base_frequency, symbol_waveform
from nlfsc_lora.metrics import autocorrelation, peak_to_sidelobe_ratio_db, phase_error
from nlfsc_lora.trajectories import TRAJECTORIES


def make_cfg(traj="linear", sf=6, bw=1000.0, os=4):
    g, _ = TRAJECTORIES[traj]
    return ChirpConfig(sf=sf, bandwidth=bw, sample_rate=os * bw, g=g)


@pytest.mark.parametrize("traj", TRAJECTORIES.keys())
def test_autocorrelation_peaks_at_zero_lag(traj):
    cfg = make_cfg(traj)
    x = symbol_waveform(cfg, 0)
    ac = autocorrelation(x)
    center = len(ac) // 2
    assert np.argmax(np.abs(ac)) == center
    assert np.abs(ac[center]) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("traj", TRAJECTORIES.keys())
def test_peak_to_sidelobe_ratio_is_positive(traj):
    cfg = make_cfg(traj)
    x = symbol_waveform(cfg, 0)
    ac = autocorrelation(x)
    assert peak_to_sidelobe_ratio_db(ac) > 0


def test_phase_error_zero_for_matched_trajectory():
    cfg = make_cfg("quadratic")
    f = base_frequency(cfg)
    err = phase_error(f, f, cfg.sample_rate)
    assert np.allclose(err, 0.0)


def test_phase_error_grows_for_mismatched_exponent():
    cfg_tx = make_cfg("quadratic")
    g_rx = TRAJECTORIES["cubic"][0]
    f_tx = base_frequency(cfg_tx)
    f_rx = cfg_tx.f_center - cfg_tx.bandwidth / 2 + cfg_tx.bandwidth * g_rx(np.arange(cfg_tx.n_samples) / cfg_tx.n_samples)
    err = phase_error(f_tx, f_rx, cfg_tx.sample_rate)
    assert np.max(np.abs(err)) > 0
