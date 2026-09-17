"""Pytest-level pins for examples/lorawan_experiments.py's headline finding:
this project's own Doppler tracking (afc.py), applied to a real, Hamming
FEC + interleaved + whitened + Gray-coded + CRC16-checked LoRaWAN packet
(via the optional `lora_phy` reference, see pyproject.toml's "refcheck"
extra), survives intact -- not just at the symbol level examples/
packet_experiments.py already checks, but at the packet-CRC level a real
receiver actually cares about. Skips cleanly if lora_phy isn't installed.

Small trial counts here (this is a correctness pin, not the full SNR-sweep
characterization -- see examples/lorawan_experiments.py's own figure for
that); the point is a fast, permanent regression check, not a
publication-quality curve.
"""
import numpy as np
import pytest

pytest.importorskip("lora_phy")
from lora_phy import LoRaReceiver, LoRaTransmitter

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
from lorawan_experiments import AFCLoop, KalmanAFCLoop, make_cfg, measurement_noise_for, run_lorawan_packet

SF = 7
BANDWIDTH = 500e3


def make_tx_rx():
    cfg = make_cfg()
    tx = LoRaTransmitter(SF, BANDWIDTH, cfg.sample_rate, has_header=True, coding_rate=4, enable_crc=True)
    rx = LoRaReceiver(rf_freq=868e6, spreading_factor=SF, bandwidth=BANDWIDTH, sample_rate=cfg.sample_rate)
    return cfg, tx, rx


def n_total_bursts(tx, payload_len):
    from lorawan_experiments import N_PREAMBLE
    return N_PREAMBLE + tx.encode(np.zeros(payload_len, dtype=np.uint8)).shape[0]


@pytest.mark.parametrize("tracker_name,factory", [
    ("AFCLoop", lambda cfg: AFCLoop(gain=0.3)),
    ("KalmanAFCLoop", lambda cfg: KalmanAFCLoop(measurement_noise=measurement_noise_for(cfg))),
])
def test_constant_cfo_packet_passes_crc_at_high_snr(tracker_name, factory):
    """A real 20kHz constant CFO (this project's own grounded LEO-pass
    magnitude), acquired from the preamble and tracked through a real
    Hamming/interleaved/whitened/Gray-coded/CRC16 packet: should pass CRC
    every time at a clean, near-noiseless SNR."""
    cfg, tx, rx = make_tx_rx()
    payload_len = 12
    n_total = n_total_bursts(tx, payload_len)
    seq = np.full(n_total, 20000.0)
    fails = 0
    n_trials = 15
    for t in range(n_trials):
        ok, _bytes, _syms = run_lorawan_packet(cfg, tx, rx, payload_len, seq, snr_db=40.0, seed=t, loop=factory(cfg))
        fails += int(not ok)
    assert fails == 0


@pytest.mark.parametrize("tracker_name,factory", [
    ("AFCLoop", lambda cfg: AFCLoop(gain=0.3)),
    ("KalmanAFCLoop", lambda cfg: KalmanAFCLoop(measurement_noise=measurement_noise_for(cfg))),
])
def test_uav_reversal_packet_passes_crc_at_high_snr(tracker_name, factory):
    """The same UAV-style sign-reversing Doppler profile as 18_uav_flyover_afc.png
    and packet_experiments.py's Test 3, through a real coded packet: should
    still pass CRC every time at high SNR despite the sign reversal."""
    cfg, tx, rx = make_tx_rx()
    payload_len = 12
    n_total = n_total_bursts(tx, payload_len)
    t = np.linspace(-n_total / 2, n_total / 2, n_total)
    seq = -3000.0 * t / np.sqrt(t ** 2 + 90.0 ** 2)
    fails = 0
    n_trials = 15
    for trial in range(n_trials):
        ok, _bytes, _syms = run_lorawan_packet(cfg, tx, rx, payload_len, seq, snr_db=40.0, seed=trial, loop=factory(cfg))
        fails += int(not ok)
    assert fails == 0


def test_decoded_payload_bytes_match_transmitted_when_crc_passes():
    """CRC passing should mean what it says: the decoded bytes should
    actually equal what was sent, not just an internally-consistent but
    wrong result."""
    cfg, tx, rx = make_tx_rx()
    payload_len = 12
    n_total = n_total_bursts(tx, payload_len)
    seq = np.full(n_total, 20000.0)

    rng = np.random.default_rng(0)
    true_payload = np.random.default_rng(0 + 500_000).integers(0, 256, payload_len, dtype=np.uint8)
    ok, decoded_bytes, _syms = run_lorawan_packet(cfg, tx, rx, payload_len, seq, snr_db=40.0, seed=0, loop=AFCLoop(gain=0.3))
    assert ok
    assert np.array_equal(decoded_bytes, true_payload)


def test_static_correction_baseline_does_worse_than_tracking_under_uav_reversal():
    """The same "does tracking actually help, not just decode" comparison
    this project makes everywhere else (12_dual_edge_afc.png onward), now at
    the packet-CRC level: a receiver that acquires once and holds that
    estimate through a sign-reversing Doppler profile should fail packets
    that active tracking recovers, at a SNR/profile combination where the
    reversal has moved the CFO well away from its acquired value.

    payload_len=150 (368 total bursts), not the 12-byte payload the other
    tests use: the UAV-style reversal needs enough packet length to actually
    unfold and carry the CFO away from its acquired value (the same reason
    packet_experiments.py's own Test 3 uses a 712-symbol payload instead of
    its Tests 1-2's short one) -- checked directly, a 12-byte payload's ~48
    bursts is too short a window for this profile to diverge far enough to
    break even a static, acquire-once correction."""
    cfg, tx, rx = make_tx_rx()
    payload_len = 150
    n_total = n_total_bursts(tx, payload_len)
    t = np.linspace(-n_total / 2, n_total / 2, n_total)
    seq = -3000.0 * t / np.sqrt(t ** 2 + 90.0 ** 2)

    n_trials = 15
    tracked_fails = static_fails = 0
    for trial in range(n_trials):
        ok_tracked, _b, _s = run_lorawan_packet(cfg, tx, rx, payload_len, seq, snr_db=10.0, seed=trial,
                                                 loop=AFCLoop(gain=0.3), track=True)
        ok_static, _b, _s = run_lorawan_packet(cfg, tx, rx, payload_len, seq, snr_db=10.0, seed=trial, track=False)
        tracked_fails += int(not ok_tracked)
        static_fails += int(not ok_static)
    assert tracked_fails <= static_fails
    assert static_fails > 0
