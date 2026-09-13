import numpy as np

from nlfsc_lora.chirp import ChirpConfig
from nlfsc_lora.simulate import ser_vs_doppler_scale
from nlfsc_lora.trajectories import TRAJECTORIES, hyperbolic_center_freq


def test_hfm_lora_beats_linear_lora_near_the_half_bin_threshold():
    """The ambiguity-function-level HFM advantage (nlfsc_lora.doppler) does NOT
    straightforwardly carry over to full M-ary LoRa-style decoding: LoRa's fine,
    closely-spaced symbol hypotheses are dominated by the Doppler-induced *lag* drift
    (near-identical for every trajectory -- see 07_doppler_scale_tolerance.png's "peak
    lag" panel), which exceeds half a symbol bin at a much smaller alpha than where
    HFM's magnitude-preservation would matter. Both trajectories decode perfectly
    within that half-bin tolerance and both eventually saturate to 100% error well
    beyond it; there is a real, reproducible window just past the threshold, before
    full saturation, where HFM's surviving correlation magnitude measurably still
    outperforms linear's collapsed one (verified stable across 8 seeds during
    development -- see the project history for the sweep that found this window).
    """
    bw = 1000.0
    f_center = hyperbolic_center_freq(bw)
    g_lin, _ = TRAJECTORIES["linear"]
    g_hfm, _ = TRAJECTORIES["hyperbolic"]
    cfg_lin = ChirpConfig(sf=6, bandwidth=bw, sample_rate=4 * bw, g=g_lin, f_center=f_center)
    cfg_hfm = ChirpConfig(sf=6, bandwidth=bw, sample_rate=4 * bw, g=g_hfm, f_center=f_center)

    alpha_range = [1.012]
    ser_lin = ser_vs_doppler_scale(cfg_lin, alpha_range, snr_db=15, n_symbols=150, demod="mfbank", seed=0)
    ser_hfm = ser_vs_doppler_scale(cfg_hfm, alpha_range, snr_db=15, n_symbols=150, demod="mfbank", seed=0)

    assert ser_hfm[0] < ser_lin[0]


def test_ser_vs_doppler_scale_perfect_at_alpha_one():
    bw = 1000.0
    g, _ = TRAJECTORIES["linear"]
    cfg = ChirpConfig(sf=5, bandwidth=bw, sample_rate=4 * bw, g=g)
    ser = ser_vs_doppler_scale(cfg, [1.0], snr_db=15, n_symbols=40, demod="mfbank", seed=0)
    assert ser[0] == 0.0
