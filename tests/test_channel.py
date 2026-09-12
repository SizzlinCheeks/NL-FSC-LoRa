import numpy as np

from nlfsc_lora.channel import apply_doppler_scale
from nlfsc_lora.chirp import ChirpConfig, base_waveform
from nlfsc_lora.trajectories import TRAJECTORIES


def test_doppler_scale_identity_at_alpha_one():
    g, _ = TRAJECTORIES["linear"]
    cfg = ChirpConfig(sf=6, bandwidth=1000.0, sample_rate=4000.0, g=g)
    x = base_waveform(cfg)
    assert np.allclose(apply_doppler_scale(x, 1.0), x, atol=1e-9)


def test_doppler_scale_compression_zero_pads_the_tail():
    """alpha > 1: the echo of a finite-duration burst finishes early: past that point
    there is nothing left to interpolate into, so the tail should be exactly zero."""
    g, _ = TRAJECTORIES["linear"]
    cfg = ChirpConfig(sf=6, bandwidth=1000.0, sample_rate=4000.0, g=g)
    x = base_waveform(cfg)
    y = apply_doppler_scale(x, alpha=1.2)
    assert y[-1] == 0.0
    assert np.any(y != 0.0)  # not the whole thing


def test_doppler_scale_stretch_uses_only_the_first_part_of_x():
    """alpha < 1: every sample maps inside the original support, no padding needed."""
    g, _ = TRAJECTORIES["linear"]
    cfg = ChirpConfig(sf=6, bandwidth=1000.0, sample_rate=4000.0, g=g)
    x = base_waveform(cfg)
    y = apply_doppler_scale(x, alpha=0.8)
    assert np.all(np.abs(y) > 0.0)
