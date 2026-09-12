import numpy as np
import pytest

from nlfsc_lora.channel import apply_cfo
from nlfsc_lora.chirp import ChirpConfig, symbol_waveform
from nlfsc_lora.sync import correct_cfo, estimate_cfo, joint_cfo_symbol_demod, joint_cfo_symbol_search
from nlfsc_lora.trajectories import TRAJECTORIES


def make_cfg(sf=7, bw=125e3, os=4):
    g, _ = TRAJECTORIES["linear"]
    return ChirpConfig(sf=sf, bandwidth=bw, sample_rate=os * bw, g=g)


def test_estimate_cfo_has_a_symbol_independent_bias_at_zero_offset():
    """Even with no CFO at all, the raw band-edge estimate isn't 0: the chirp's
    rectangular-windowed truncation makes the occupied band very slightly
    asymmetric, biasing the 5%/95% threshold by about half a frequency bin. That
    bias is the same for every symbol (consistent with |FFT| being symbol-
    independent -- see test_chirp.py), so it *looks* calibratable by subtracting
    the zero-CFO estimate as a constant offset -- but that calibration does not
    generalize to nonzero CFO or survive noise (see the bias test below), so it
    is characterized here rather than treated as a usable fix."""
    cfg = make_cfg()
    bias_values = {m: estimate_cfo(symbol_waveform(cfg, m), cfg) for m in [0, 1, 33, 100, 127]}
    values = list(bias_values.values())
    assert max(values) - min(values) < 1e-6  # identical across symbols
    assert abs(values[0]) > 100.0  # but not zero -- a real, non-negligible artifact


def test_estimate_cfo_band_edge_bias_is_large_documented_negative_result():
    """Documents the negative result from the design investigation: the blind
    band-edge estimate is biased by hundreds of Hz on a 100 Hz true CFO, because
    the chirp's own truncation leakage (~1-2% of energy outside the nominal band)
    dominates a shift this small. This is a real limitation, not a bug -- see
    nlfsc_lora/sync.py's module docstring."""
    cfg = make_cfg()
    tx = symbol_waveform(cfg, 33)
    rx = apply_cfo(tx, 100.0, cfg.sample_rate)
    est = estimate_cfo(rx, cfg)
    assert abs(est - 100.0) > 50.0  # the estimate is nowhere near usable at this scale


def test_joint_search_recovers_symbol_and_cfo_noiseless():
    cfg = make_cfg()
    true_cfo = 700.0
    tx = symbol_waveform(cfg, 33)
    rx = apply_cfo(tx, true_cfo, cfg.sample_rate)
    cfo_candidates = np.arange(-1500, 1501, 50)
    m, cfo_est, score = joint_cfo_symbol_search(rx, cfg, cfo_candidates)
    assert m == 33
    assert cfo_est == pytest.approx(true_cfo, abs=25.0)  # within half the grid step
    assert score == pytest.approx(1.0, abs=1e-6)


def test_joint_cfo_symbol_demod_beats_plain_mfbank_past_half_bin_cfo():
    """At a CFO well past the half-bin threshold where the CFO-blind receiver
    collapses (see 04_ser_vs_cfo.png), the joint search should still recover
    the correct symbol noiselessly."""
    from nlfsc_lora.receiver import matched_filter_bank_demod

    cfg = make_cfg()
    tx = symbol_waveform(cfg, 33)
    rx = apply_cfo(tx, 900.0, cfg.sample_rate)  # ~ almost 1 full bin at SF7
    assert matched_filter_bank_demod(rx, cfg) != 33
    assert joint_cfo_symbol_demod(rx, cfg) == 33


def test_correct_cfo_round_trip():
    cfg = make_cfg()
    tx = symbol_waveform(cfg, 10)
    rx = apply_cfo(tx, 250.0, cfg.sample_rate)
    corrected = correct_cfo(rx, 250.0, cfg.sample_rate)
    assert np.allclose(corrected, tx, atol=1e-9)
