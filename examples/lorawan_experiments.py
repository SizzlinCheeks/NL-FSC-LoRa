"""Does this project's own Doppler tracking (afc.py) survive real LoRaWAN
channel coding -- not just this project's own simplified symbol-error-rate
counting?

Every other packet-level test in this project (examples/packet_experiments.py)
grades "payload symbol error rate": did the decoded chirp shift match the
transmitted one, symbol by symbol. Real LoRaWAN never ships raw symbols --
each byte of payload goes through whitening, Hamming FEC, diagonal
interleaving, and Gray coding before modulation, and a CRC16 either passes or
fails on the whole packet at the far end. None of that machinery is built into
this project (deliberately -- NARRATIVE.md Chapter 7 is explicit that its own
preamble/payload framing is "not a bit-accurate reproduction of the real
sync-word convention," since the trajectory-shape question this project
studies doesn't depend on it). This file is the one place that machinery gets
exercised for real, via `lora_phy` (pyLoRaPHY, an independent, MIT-licensed
Python port of the widely-used jkadbear/LoRaPHY reference implementation --
optional dependency, pip install .[refcheck], pinned to 0.2.0; see
pyproject.toml for why). Everything here skips cleanly if it isn't installed.

**Scope, stated up front.** This project's core contribution -- nonlinear
trajectories, HFM's wideband-Doppler self-similarity, the paired-sweep
correction -- cannot be tested this way at all: lora_phy's receiver, like
every real LoRa chipset, only ever demodulates linear chirps. There is no
real reference to cross-check a hyperbolic trajectory against, because no
real system generates one. What CAN be tested for real is the narrowband
CFO/Doppler tracking (afc.py's AFCLoop/KalmanAFCLoop/full_symbol_cfo_estimate),
which works identically for any trajectory including plain linear -- so this
file uses g=linear throughout, the one shape with a real-world counterpart.

**How the pieces fit together.** `lora_phy`'s `LoRaTransmitter.encode()` and
`LoRaReceiver.decode()` are used purely as the bit<->symbol FEC/interleave/
whiten/Gray/CRC layer -- turning a real payload into the exact stream of
chirp-shift values (0..M-1) real hardware would transmit, and turning a
receiver's decoded shift values back into payload bytes plus a CRC pass/fail,
exactly as a real LoRaWAN receiver would. Everything *around* that layer --
preamble, Doppler profile, acquisition, per-burst tracking -- reuses this
project's own already-validated pieces (sync.py::joint_cfo_symbol_search,
afc.py::full_symbol_cfo_estimate/AFCLoop/KalmanAFCLoop), the same building
blocks packet_experiments.py's Tests 1-3 use, just fed a real, FEC-coded
symbol stream instead of a random one.

Each burst's *waveform* is generated with `lora_phy.common.chirp` directly
(the real construction, not this project's own symbol_waveform) -- deliberate,
because tests/test_reference_crossval.py already found and quantified a real
difference between the two: this project's own cyclic-sample-roll construction
and real LoRa's closed-form-shifted-start construction are not bit-identical
away from m=0 (correlation dips to ~0.92 near mid-alphabet), though neither
ever flips a noiseless decode. Using the real construction here, not this
project's approximation of it, means this test is a genuine check of that
finding under real Doppler and real channel coding together, not an assumption
carried over from the single-symbol result.

Run with: python examples/lorawan_experiments.py
"""
import os
import time

import matplotlib.pyplot as plt
import numpy as np

try:
    from lora_phy import LoRaReceiver, LoRaTransmitter
    from lora_phy import common as lp_common
    HAVE_LORA_PHY = True
except ImportError:
    HAVE_LORA_PHY = False

from nlfsc_lora.afc import AFCLoop, KalmanAFCLoop, _combine_acquisition, full_symbol_cfo_estimate, measurement_noise_for
from nlfsc_lora.channel import awgn
from nlfsc_lora.channel import apply_cfo
from nlfsc_lora.chirp import ChirpConfig
from nlfsc_lora.receiver import fft_correlation_demod
from nlfsc_lora.sync import correct_cfo, joint_cfo_symbol_search
from nlfsc_lora.trajectories import TRAJECTORIES

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
SF = 7
BANDWIDTH = 500e3
OVERSAMPLING = 4
N_PREAMBLE = 8  # same convention as packet_experiments.py -- not lora_phy's own preamble, see module docstring


def make_cfg() -> ChirpConfig:
    g_lin, _ = TRAJECTORIES["linear"]
    return ChirpConfig(sf=SF, bandwidth=BANDWIDTH, sample_rate=OVERSAMPLING * BANDWIDTH, g=g_lin, f_center=0.0)


def half_bin_cfo(cfg: ChirpConfig) -> float:
    """Same benign frequency-grid convention offset tests/test_reference_crossval.py
    found and corrects for: lora_phy's chirp(cfo=X) with this X lands on this
    project's own frequency-grid convention. Applied here so the only CFO this
    test measures is the deliberately-injected Doppler, not an incidental
    grid-convention artifact."""
    return cfg.bandwidth / (2 * cfg.n_samples)


def run_lorawan_packet(cfg, tx, rx, payload_len_bytes, true_cfo_sequence, snr_db, seed,
                        loop=None, acquire_span=25000.0, acquire_step=250.0, track=True):
    """One real LoRaWAN-framed packet through this project's own Doppler
    tracking pipeline. `track=False` runs a baseline that acquires once from
    the preamble and then holds that single estimate for the whole packet --
    no per-burst correction -- the same "static correction" baseline
    contrasted against tracking throughout this project (12_dual_edge_afc.png
    onward), now applied to a real, CRC-checked packet instead of a symbol
    error count.

    Returns (crc_ok, decoded_data_bytes, decoded_symbols).
    """
    rng = np.random.default_rng(seed)
    payload_rng = np.random.default_rng(seed + 500_000)
    payload = payload_rng.integers(0, 256, payload_len_bytes, dtype=np.uint8)
    data_symbols_true = tx.encode(payload)  # real Hamming/interleave/whiten/Gray-coded stream
    all_true = np.concatenate([np.zeros(N_PREAMBLE, dtype=int), data_symbols_true.astype(int)])
    assert len(all_true) == len(true_cfo_sequence)

    loop = AFCLoop(gain=0.3) if loop is None else loop
    decoded = np.empty(len(all_true), dtype=int)
    acquire_candidates = np.arange(-acquire_span, acquire_span + acquire_step, acquire_step)
    acquire_results = []
    cfo_bin = half_bin_cfo(cfg)
    acquired_cfo = None

    for i, (m_true, cfo_true) in enumerate(zip(all_true, true_cfo_sequence)):
        cfo_tracked = loop.cfo_tracked
        burst = lp_common.chirp(True, cfg.sf, cfg.bandwidth, cfg.sample_rate,
                                 start_freq_offset=int(m_true), cfo=cfo_bin)
        rx_burst = awgn(apply_cfo(burst, cfo_true, cfg.sample_rate), snr_db, rng)

        if i < N_PREAMBLE:
            m_hat, cfo_hat, _score = joint_cfo_symbol_search(rx_burst, cfg, acquire_candidates)
            decoded[i] = m_hat
            acquire_results.append((m_hat, cfo_hat))
            if i == N_PREAMBLE - 1:
                acquired_cfo = _combine_acquisition(acquire_results)
                if hasattr(loop, "set_acquired"):
                    loop.set_acquired(acquired_cfo)
                else:
                    loop.cfo_tracked = acquired_cfo
            continue

        correction = acquired_cfo if not track else cfo_tracked
        rx_corrected = correct_cfo(rx_burst, correction, cfg.sample_rate)
        m_hat = fft_correlation_demod(rx_corrected, cfg)
        decoded[i] = m_hat

        if track:
            cfo_residual_est, _pr = full_symbol_cfo_estimate(rx_corrected, cfg, m_hat)
            new_measurement = None if cfo_residual_est is None else cfo_tracked + cfo_residual_est
            loop.update(new_measurement)

    data_symbols = decoded[N_PREAMBLE:].astype(np.uint16)
    try:
        data, checksum = rx.decode(data_symbols)
        crc_ok = len(checksum) == 2 and list(data[payload_len_bytes:payload_len_bytes + 2]) == checksum
        decoded_bytes = data[:payload_len_bytes]
    except Exception:
        crc_ok, decoded_bytes = False, None
    return bool(crc_ok), decoded_bytes, decoded


def _packet_pass_rate(cfg, tx, rx, payload_len_bytes, cfo_fn, snr_range, n_trials,
                       loop_factory=None, track=True, acquire_span=25000.0, acquire_step=250.0):
    n_total = N_PREAMBLE + tx.encode(np.zeros(payload_len_bytes, dtype=np.uint8)).shape[0]
    # header/payload symbol count is fixed for a given payload_len_bytes/coding_rate/CRC
    # config regardless of payload content -- computed once via a throwaway encode()
    true_cfo_sequence = cfo_fn(n_total)
    pass_rate = np.empty(len(snr_range))
    for i, snr_db in enumerate(snr_range):
        passes = 0
        for t in range(n_trials):
            loop = loop_factory() if loop_factory is not None else None
            ok, _bytes, _syms = run_lorawan_packet(cfg, tx, rx, payload_len_bytes, true_cfo_sequence,
                                                     snr_db, seed=t, loop=loop, track=track,
                                                     acquire_span=acquire_span, acquire_step=acquire_step)
            passes += int(ok)
        pass_rate[i] = passes / n_trials
    return pass_rate


def test_correctness_noiseless():
    """Sanity check before any statistics: at very high SNR, with tracking
    on, a real Hamming/CRC-coded packet under real Doppler should pass its
    CRC every time -- both trackers, both a constant offset and a
    sign-reversing UAV-style profile."""
    cfg = make_cfg()
    tx = LoRaTransmitter(SF, BANDWIDTH, cfg.sample_rate, has_header=True, coding_rate=4, enable_crc=True)
    rx = LoRaReceiver(rf_freq=868e6, spreading_factor=SF, bandwidth=BANDWIDTH, sample_rate=cfg.sample_rate)
    payload_len = 12
    n_total = N_PREAMBLE + tx.encode(np.zeros(payload_len, dtype=np.uint8)).shape[0]

    profiles = {
        "constant 20kHz": np.full(n_total, 20000.0),
        "UAV reversal": (lambda: (lambda t: -3000.0 * t / np.sqrt(t ** 2 + 90.0 ** 2))(
            np.linspace(-n_total / 2, n_total / 2, n_total)))(),
    }
    for name, seq in profiles.items():
        for tracker_name, factory in [("AFCLoop", lambda: AFCLoop(gain=0.3)),
                                       ("KalmanAFCLoop", lambda: KalmanAFCLoop(measurement_noise=measurement_noise_for(cfg)))]:
            fails = 0
            n_trials = 20
            for t in range(n_trials):
                ok, _b, _s = run_lorawan_packet(cfg, tx, rx, payload_len, seq, snr_db=40.0, seed=t, loop=factory())
                fails += int(not ok)
            print(f"[{name} / {tracker_name}] noiseless-ish (40dB): {fails}/{n_trials} CRC failures")
            assert fails == 0, f"{name}/{tracker_name} should pass CRC every time at 40dB"


def experiment_packet_pass_rate_vs_snr():
    """The real deliverable: packet-level CRC pass rate vs SNR, tracking on
    vs off, for both a constant-CFO (satellite-style) and a sign-reversing
    (UAV-style) Doppler profile -- the same two profiles
    packet_experiments.py's Tests 2-3 already characterized at the symbol
    level, now graded by whether a real, Hamming/CRC-coded packet survives
    intact, not by counting individual symbol errors.

    A third panel, UAV reversal at a much longer payload (150 bytes, 368
    total bursts vs. the other two panels' 12-byte/48-burst packets), is
    there for a real reason found while building this, not added for
    variety: at the short, realistic-uplink payload length, tracking and a
    static (acquire-once) correction turned out to be indistinguishable --
    checked directly, not assumed away -- because 48 bursts isn't enough
    packet length for even this reversal to drift far from its acquired
    value (the same reason packet_experiments.py's own Test 3 uses a
    712-symbol payload instead of Tests 1-2's short one). The long-payload
    panel is what actually shows tracking earning its keep at the
    packet-CRC level, and the short-payload panels' near-identical curves
    are themselves an honest, useful finding about when continuous tracking
    matters in practice -- see this file's own module docstring and the
    project write-up for the applicability discussion this connects to."""
    cfg = make_cfg()
    tx = LoRaTransmitter(SF, BANDWIDTH, cfg.sample_rate, has_header=True, coding_rate=4, enable_crc=True)
    rx = LoRaReceiver(rf_freq=868e6, spreading_factor=SF, bandwidth=BANDWIDTH, sample_rate=cfg.sample_rate)
    snr_range = np.arange(-20, 6, 2)
    n_trials = 60

    const_cfo_fn = lambda n: np.full(n, 20000.0)
    uav_cfo_fn = lambda n: (lambda t: -3000.0 * t / np.sqrt(t ** 2 + 90.0 ** 2))(np.linspace(-n / 2, n / 2, n))

    panels = [
        ("constant 20kHz", const_cfo_fn, 12),
        ("UAV reversal", uav_cfo_fn, 12),
        ("UAV reversal (150B payload)", uav_cfo_fn, 150),
    ]

    results = {}
    for profile_name, cfo_fn, payload_len in panels:
        for tracker_name, factory in [("AFCLoop", lambda: AFCLoop(gain=0.3)),
                                       ("KalmanAFCLoop", lambda: KalmanAFCLoop(measurement_noise=measurement_noise_for(cfg)))]:
            t0 = time.time()
            tracked = _packet_pass_rate(cfg, tx, rx, payload_len, cfo_fn, snr_range, n_trials,
                                         loop_factory=factory, track=True)
            print(f"[{profile_name}/{tracker_name}] tracking-on pass-rate sweep done in {time.time()-t0:.1f}s")
            results[(profile_name, tracker_name, True)] = tracked
        t0 = time.time()
        static = _packet_pass_rate(cfg, tx, rx, payload_len, cfo_fn, snr_range, n_trials, track=False)
        print(f"[{profile_name}/static] pass-rate sweep done in {time.time()-t0:.1f}s")
        results[(profile_name, "static (acquire once, no tracking)", False)] = static

    return snr_range, results


def plot_lorawan_packet_pass_rate(snr_range, results):
    os.makedirs(OUT_DIR, exist_ok=True)
    profiles = ["constant 20kHz", "UAV reversal", "UAV reversal (150B payload)"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))
    for ax, profile_name in zip(axes, profiles):
        for (p, tracker_name, _track), pass_rate in results.items():
            if p != profile_name:
                continue
            ax.plot(snr_range, pass_rate, marker="o", ms=4, label=tracker_name)
        ax.set(xlabel="SNR (dB)", ylabel="packet CRC pass rate", ylim=(-0.05, 1.05),
               title=f"{profile_name}")
        ax.invert_xaxis()
        ax.legend(fontsize=8)
    fig.suptitle("Real LoRaWAN-framed packets (Hamming FEC + interleave + whiten + Gray + CRC16, via lora_phy)\n"
                 "under this project's own Doppler tracking: packet-level CRC pass rate, not symbol error rate")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "24_lorawan_packet_pass_rate.png"), dpi=150)
    plt.close(fig)


def main():
    if not HAVE_LORA_PHY:
        print("lora_phy is not installed (pip install .[refcheck]) -- skipping LoRaWAN framing experiments.")
        return
    test_correctness_noiseless()
    snr_range, results = experiment_packet_pass_rate_vs_snr()
    plot_lorawan_packet_pass_rate(snr_range, results)
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
