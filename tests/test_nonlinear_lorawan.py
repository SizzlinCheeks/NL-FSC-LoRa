"""Pins for examples/nonlinear_lorawan_experiments.py's findings: real
LoRaWAN protocol framing (Hamming FEC, interleaving, whitening, Gray
coding, CRC16 -- lora_phy's encode()/decode()) works identically whether
the physical waveform underneath is a linear chirp (what real hardware
uses) or this project's own hyperbolic trajectory (what no real radio
can). Skips cleanly if lora_phy isn't installed.
"""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("lora_phy")
from lora_phy import LoRaReceiver, LoRaTransmitter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
from nonlinear_lorawan_experiments import (
    SF,
    BANDWIDTH,
    make_cfg,
    make_cfg_pair,
    run_nonlinear_lorawan_packet,
    run_wideband_nonlinear_lorawan_packet,
)
from nlfsc_lora.afc import KalmanAFCLoop, measurement_noise_for
from nlfsc_lora.channel import apply_cfo, awgn
from nlfsc_lora.chirp import symbol_waveform
from nlfsc_lora.sync import joint_cfo_symbol_search
from nlfsc_lora.afc import _combine_acquisition


def _tx_rx(shape):
    cfg = make_cfg(shape)
    tx = LoRaTransmitter(SF, BANDWIDTH, cfg.sample_rate, has_header=True, coding_rate=4, enable_crc=True)
    rx = LoRaReceiver(rf_freq=868e6, spreading_factor=SF, bandwidth=BANDWIDTH, sample_rate=cfg.sample_rate)
    return cfg, tx, rx


@pytest.mark.parametrize("shape", ["linear", "hyperbolic"])
def test_narrowband_packet_passes_crc_at_high_snr(shape):
    """A real Hamming/CRC-coded packet, physically carried on this
    project's own trajectory (linear or hyperbolic), under a real 20kHz
    CFO, tracked by this project's own AFC pipeline: should pass CRC every
    time at high SNR, for either shape -- the basic end-to-end proof that
    real LoRaWAN protocol semantics don't care which trajectory carries
    them."""
    cfg, tx, rx = _tx_rx(shape)
    payload_len = 12
    n_total = 8 + tx.encode(np.zeros(payload_len, dtype=np.uint8)).shape[0]
    seq = np.full(n_total, 20000.0)
    fails = 0
    n_trials = 15
    for t in range(n_trials):
        ok, _b, _s = run_nonlinear_lorawan_packet(cfg, tx, rx, payload_len, seq, snr_db=40.0, seed=t)
        fails += int(not ok)
    assert fails == 0


def test_decoded_bytes_match_transmitted_for_hyperbolic_phy():
    """CRC passing should mean what it says for the nonlinear PHY too: the
    decoded bytes should actually equal what was sent."""
    cfg, tx, rx = _tx_rx("hyperbolic")
    payload_len = 12
    n_total = 8 + tx.encode(np.zeros(payload_len, dtype=np.uint8)).shape[0]
    seq = np.full(n_total, 20000.0)
    true_payload = np.random.default_rng(0 + 500_000).integers(0, 256, payload_len, dtype=np.uint8)
    ok, decoded_bytes, _syms = run_nonlinear_lorawan_packet(cfg, tx, rx, payload_len, seq, snr_db=40.0, seed=0)
    assert ok
    assert np.array_equal(decoded_bytes, true_payload)


def test_acquisition_is_more_reliable_for_hyperbolic_than_linear_under_cfo():
    """The real, striking, mechanistically-grounded finding behind why the
    hyperbolic PHY's packet-pass-rate curve sits above linear's in
    30_nonlinear_lorawan_pass_rate.png: joint_cfo_symbol_search's
    (symbol, CFO) acquisition is far more failure-prone for a linear chirp
    at the same SNR. Linear FM's ambiguity function has well-known
    range-Doppler coupling (a CFO error can closely mimic a symbol-shift
    error) -- exactly the classical reason sonar/radar literature favors
    HFM (Kroszczyński 1969 onward, already cited in NARRATIVE.md); HFM's
    own ambiguity function does not have that coupling. Measured directly
    at SF7/500kHz, 20kHz true CFO, 8-burst combined acquisition: linear
    fails roughly half the time at -14dB where hyperbolic essentially never
    does."""
    cfg_lin = make_cfg("linear")
    cfg_hyp = make_cfg("hyperbolic")
    true_cfo = 20000.0
    span, step = 25000.0, 250.0
    candidates = np.arange(-span, span + step, step)
    snr_db = -14.0
    n_trials = 40

    def acquisition_failures(cfg):
        rng = np.random.default_rng(0)
        fails = 0
        for _ in range(n_trials):
            results = []
            for _b in range(8):
                tx_burst = apply_cfo(symbol_waveform(cfg, 0), true_cfo, cfg.sample_rate)
                rx_burst = awgn(tx_burst, snr_db, rng)
                m_hat, cfo_hat, _s = joint_cfo_symbol_search(rx_burst, cfg, candidates)
                results.append((m_hat, cfo_hat))
            acquired = _combine_acquisition(results)
            if abs(acquired - true_cfo) > 2000.0:
                fails += 1
        return fails

    linear_fails = acquisition_failures(cfg_lin)
    hyperbolic_fails = acquisition_failures(cfg_hyp)
    assert hyperbolic_fails <= 5  # essentially always succeeds
    assert linear_fails >= 3 * max(hyperbolic_fails, 1)  # substantially, not marginally, worse


def test_wideband_scale_packet_recovered_only_when_corrected():
    """The strongest result: a Doppler time-scale (alpha=1.05) no real
    LoRaWAN radio could survive at all -- a real, Hamming/CRC-coded packet
    physically carried on this project's own hyperbolic trajectory should
    pass CRC every time once corrected with paired_sweep's DHFM technique,
    and fail essentially every time uncorrected."""
    q = 1.0 / 3.0
    cfg_up, cfg_down = make_cfg_pair(q)
    tx = LoRaTransmitter(SF, BANDWIDTH, cfg_up.sample_rate, has_header=True, coding_rate=4, enable_crc=True)
    rx = LoRaReceiver(rf_freq=868e6, spreading_factor=SF, bandwidth=BANDWIDTH, sample_rate=cfg_up.sample_rate)
    alpha_true = 1.05
    payload_len = 12
    n_trials = 15

    fails_corrected = fails_uncorrected = 0
    for t in range(n_trials):
        ok_c, _b, _a = run_wideband_nonlinear_lorawan_packet(cfg_up, cfg_down, tx, rx, payload_len, alpha_true, q,
                                                               snr_db=40.0, seed=t, correct=True)
        ok_u, _b, _a = run_wideband_nonlinear_lorawan_packet(cfg_up, cfg_down, tx, rx, payload_len, alpha_true, q,
                                                               snr_db=40.0, seed=t, correct=False)
        fails_corrected += int(not ok_c)
        fails_uncorrected += int(not ok_u)
    assert fails_corrected == 0
    assert fails_uncorrected >= 0.9 * n_trials
