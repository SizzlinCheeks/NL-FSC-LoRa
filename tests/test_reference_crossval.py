"""Cross-validates this project's linear-chirp (g=linear) baseline against a
real, independent LoRa PHY reference implementation, `lora_phy`
(pyLoRaPHY, https://github.com/myzhang1029/pyLoRaPHY, MIT, a Python port of
the widely-used jkadbear/LoRaPHY) -- optional (pip install .[refcheck], pinned
to 0.2.0, see pyproject.toml), so every test here skips cleanly if it isn't
installed. Every other test in this project checks the transmitter and
receiver against *themselves*: self-consistency, not real-world compatibility.
This file is the one place that checks against an independently-implemented
system, not this project's own conventions.

**A real, characterized finding, not just a pass/fail.** The base (m=0)
up-chirp matches lora_phy's own reference chirp to numerical precision (see
test_base_waveform_matches_reference_exactly) once a benign, fully-explained
half-sample-grid convention difference is corrected for: lora_phy evaluates
instantaneous frequency at each sample's grid position offset by
`bandwidth / (2*n_samples)` relative to this project's own u = n/n_samples
convention -- a constant, symbol-independent frequency offset, not a shape
difference. Confirmed directly: it accounts for the *entire* correlation gap
measured before correcting for it, via the standard CFO-loss sinc formula
this project already uses elsewhere (NARRATIVE.md Sec5.1).

Cyclically-shifted symbols (m != 0) are a different story, and a genuinely
interesting one: this project's own `symbol_waveform` builds symbol m by
*cyclically rolling the sampled base waveform* (chirp.py's own docstring is
explicit that this is "a description of the model in this project, not a
claim about the internals of any particular commercial LoRa chipset"). Real
LoRa (and lora_phy) instead generates each symbol as a *closed-form chirp
with a shifted starting frequency* that wraps mid-symbol. These are NOT
bit-identical waveforms once m != 0 -- checked directly
(test_symbol_waveform_diverges_from_reference_away_from_m_zero): correlation
between the two constructions dips to about 0.92 near m=M/2 (a real,
symbol-dependent effect around the wrap point, not noise), which is a
roughly 0.7dB matched-filter loss relative to this project's own idealized
model. That said -- and this is the actual headline result --
test_full_alphabet_noiseless_cross_decode_is_exact confirms this loss never
flips a decoding decision: every symbol, decoded by the *other* system's
receiver, comes back correct, at every m, noiselessly. The wrap-construction
difference costs some matched-filter magnitude; it does not cost symbol
correctness, because the loss is shared closely enough across neighboring
symbol hypotheses that the argmax decision is unaffected.
"""
import numpy as np
import pytest

lora_phy = pytest.importorskip("lora_phy")
from lora_phy import LoRaReceiver
from lora_phy import common as lp_common

from nlfsc_lora.channel import awgn
from nlfsc_lora.chirp import ChirpConfig, base_waveform, symbol_waveform
from nlfsc_lora.receiver import fft_correlation_demod
from nlfsc_lora.trajectories import TRAJECTORIES

SF = 7
BW = 125e3
SR = 4 * BW  # matches this project's own OVERSAMPLING=4 convention


def make_cfg():
    g_lin, _ = TRAJECTORIES["linear"]
    return ChirpConfig(sf=SF, bandwidth=BW, sample_rate=SR, g=g_lin, f_center=0.0)


def half_bin_cfo(cfg):
    """The benign, constant frequency-grid convention offset described in
    this module's docstring: lora_phy's chirp(start_freq_offset=m, cfo=X)
    with X=this matches this project's own symbol_waveform(cfg, m) to
    numerical precision at m=0."""
    return cfg.bandwidth / (2 * cfg.n_samples)


def reference_waveform(cfg, m):
    """lora_phy's own construction of symbol m, on this project's own
    frequency-grid convention (half_bin_cfo-corrected)."""
    return lp_common.chirp(True, cfg.sf, cfg.bandwidth, cfg.sample_rate,
                            start_freq_offset=m, cfo=half_bin_cfo(cfg))


def decode_with_reference(cfg, rx_signal):
    """Decode one symbol-length burst with lora_phy's own dechirp, undoing
    the same half_bin_cfo grid convention shift on the way in."""
    n = cfg.n_samples
    t = np.arange(n) / cfg.sample_rate
    shifted = rx_signal * np.exp(-1j * 2 * np.pi * half_bin_cfo(cfg) * t)
    receiver = LoRaReceiver(rf_freq=868e6, spreading_factor=cfg.sf, bandwidth=cfg.bandwidth,
                             sample_rate=cfg.sample_rate, resample_to_2x=False)
    _height, peak_idx = receiver._dechirp(shifted, 0, is_up=True)
    return receiver._peak_idx_to_symbol(peak_idx, 0)


def test_base_waveform_matches_reference_exactly():
    """m=0, no cyclic shift, no wrap point in play -- the cleanest possible
    comparison. Should match to numerical precision once the benign
    half-bin-grid convention (this module's docstring) is corrected for."""
    cfg = make_cfg()
    ours = base_waveform(cfg)
    theirs = reference_waveform(cfg, 0)
    corr = np.vdot(theirs, ours) / (np.linalg.norm(ours) * np.linalg.norm(theirs))
    assert abs(corr) == pytest.approx(1.0, abs=1e-9)
    assert np.max(np.abs(ours - theirs)) < 1e-9


def test_symbol_waveform_diverges_from_reference_away_from_m_zero():
    """The real, characterized wrap-construction difference this module's
    docstring describes: cyclic sample-roll (this project) vs closed-form
    shifted-start chirp (real LoRa/lora_phy) are not the same waveform once
    m != 0. Pinned as a real, bounded effect (not "should be close to 1",
    which would be false) rather than silently assumed away."""
    cfg = make_cfg()
    worst = 1.0
    for m in range(0, cfg.M, 4):
        ours = symbol_waveform(cfg, m)
        theirs = reference_waveform(cfg, m)
        corr = abs(np.vdot(theirs, ours)) / (np.linalg.norm(ours) * np.linalg.norm(theirs))
        worst = min(worst, corr)
    # m=64 (mid-alphabet) measured at ~0.9239 in exploration; assert a bounded
    # real gap exists (rules out "it's actually exact and I mismeasured")
    # without over-fitting the test to the exact float.
    assert 0.85 < worst < 0.98


def test_full_alphabet_noiseless_cross_decode_is_exact():
    """The headline result: despite the wrap-construction shape difference
    above, decoding never flips. Bidirectional -- their waveform through our
    decoder, and our waveform through their decoder -- across the entire
    symbol alphabet, noiseless."""
    cfg = make_cfg()
    for m in range(cfg.M):
        theirs = reference_waveform(cfg, m)
        assert fft_correlation_demod(theirs, cfg) == m

        ours = symbol_waveform(cfg, m)
        assert decode_with_reference(cfg, ours) == m


def test_cross_decode_ser_matches_self_consistent_ser_at_moderate_snr():
    """Does the wrap-construction loss cost anything once there's real noise,
    not just a noiseless argmax? Compare self-consistent SER (this project's
    own tx and rx) against cross SER (lora_phy's tx, this project's rx) at a
    moderate SNR where the gap, if any, should be visible. Checked directly
    rather than assumed: the two should track closely -- this is the
    practical form of "does the difference matter," not just "does it
    exist" (the previous test already answered that)."""
    cfg = make_cfg()
    snr_db = -10.0
    n_trials = 400
    rng_self = np.random.default_rng(0)
    rng_cross = np.random.default_rng(0)

    errs_self = 0
    errs_cross = 0
    for _ in range(n_trials):
        m_true = int(rng_self.integers(0, cfg.M))
        rx_self = awgn(symbol_waveform(cfg, m_true), snr_db, rng_self)
        if fft_correlation_demod(rx_self, cfg) != m_true:
            errs_self += 1

        m_true_c = int(rng_cross.integers(0, cfg.M))
        rx_cross = awgn(reference_waveform(cfg, m_true_c), snr_db, rng_cross)
        if fft_correlation_demod(rx_cross, cfg) != m_true_c:
            errs_cross += 1

    ser_self = errs_self / n_trials
    ser_cross = errs_cross / n_trials
    # Not "identical" (different noise draws, different underlying
    # waveforms) -- but should be in the same regime, not a different one.
    assert abs(ser_self - ser_cross) < 0.05
