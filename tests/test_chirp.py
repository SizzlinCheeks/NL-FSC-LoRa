import numpy as np
import pytest

from nlfsc_lora.chirp import ChirpConfig, base_frequency, base_waveform, symbol_waveform
from nlfsc_lora.trajectories import TRAJECTORIES


def make_cfg(traj="linear", sf=5, bw=1000.0, os=4):
    g, _ = TRAJECTORIES[traj]
    return ChirpConfig(sf=sf, bandwidth=bw, sample_rate=os * bw, g=g)


@pytest.mark.parametrize("traj", TRAJECTORIES.keys())
def test_base_frequency_spans_bandwidth(traj):
    cfg = make_cfg(traj)
    f = base_frequency(cfg)
    assert f.min() >= -cfg.bandwidth / 2 - 1e-9
    assert f.max() <= cfg.bandwidth / 2 + 1e-9
    assert f[0] == pytest.approx(-cfg.bandwidth / 2, abs=1e-6)


@pytest.mark.parametrize("traj", TRAJECTORIES.keys())
def test_waveform_is_unit_envelope(traj):
    cfg = make_cfg(traj)
    x = base_waveform(cfg)
    assert np.allclose(np.abs(x), 1.0, atol=1e-9)


def test_symbol_zero_equals_base():
    cfg = make_cfg("linear")
    assert np.array_equal(symbol_waveform(cfg, 0), base_waveform(cfg))


@pytest.mark.parametrize("traj", TRAJECTORIES.keys())
def test_symbol_waveform_is_cyclic_shift_of_base(traj):
    cfg = make_cfg(traj)
    base = base_waveform(cfg)
    m = 3
    shift = int(round(m * cfg.n_samples / cfg.M))
    expected = np.roll(base, -shift)
    assert np.array_equal(symbol_waveform(cfg, m), expected)


def test_all_symbols_distinct():
    cfg = make_cfg("quadratic")
    seen = {tuple(np.round(symbol_waveform(cfg, m), 6)) for m in range(cfg.M)}
    assert len(seen) == cfg.M
