import numpy as np
import pytest

from nlfsc_lora.chirp import ChirpConfig
from nlfsc_lora.doppler import doppler_scale_response, peak_response
from nlfsc_lora.trajectories import TRAJECTORIES, hyperbolic_center_freq


def make_cfg(traj, sf=6, bw=1000.0, os=4, f_center=0.0):
    g, _ = TRAJECTORIES[traj]
    return ChirpConfig(sf=sf, bandwidth=bw, sample_rate=os * bw, g=g, f_center=f_center)


@pytest.mark.parametrize("traj", TRAJECTORIES.keys())
def test_unscaled_replica_gives_near_unity_peak(traj):
    """alpha=1 (no Doppler) should reproduce the reference almost exactly, for any shape."""
    f_center = hyperbolic_center_freq(1000.0) if traj == "hyperbolic" else 0.0
    cfg = make_cfg(traj, f_center=f_center)
    peak, lag = peak_response(cfg, alpha=1.0)
    assert peak == pytest.approx(1.0, abs=1e-6)
    assert lag == 0


def test_hyperbolic_requires_nonzero_center_to_be_well_posed():
    """q = f_low / f_high must be in (0, 1); centering at 0 Hz makes f_low negative."""
    q = 1.0 / 3.0
    f_center = hyperbolic_center_freq(1000.0, q=q)
    f_low, f_high = f_center - 500.0, f_center + 500.0
    assert f_low > 0
    assert f_low / f_high == pytest.approx(q)


def test_hyperbolic_more_doppler_scale_tolerant_than_linear():
    """The core HFM claim: under time-scaling (not a constant frequency shift), HFM's
    matched-filter peak should degrade less than a linear chirp's at the same alpha."""
    bw = 1000.0
    f_center = hyperbolic_center_freq(bw)
    cfg_lin = make_cfg("linear", sf=7, bw=bw, f_center=f_center)
    cfg_hfm = make_cfg("hyperbolic", sf=7, bw=bw, f_center=f_center)

    alpha_range = [1.05]
    peaks_lin, _ = doppler_scale_response(cfg_lin, alpha_range)
    peaks_hfm, _ = doppler_scale_response(cfg_hfm, alpha_range)

    assert peaks_hfm[0] > peaks_lin[0]


def test_peak_magnitude_bounded():
    cfg = make_cfg("linear", f_center=0.0)
    for alpha in [0.95, 1.0, 1.05]:
        peak, _ = peak_response(cfg, alpha)
        assert 0.0 <= peak <= 1.0 + 1e-9
