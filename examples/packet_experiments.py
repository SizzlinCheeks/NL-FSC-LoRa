"""End-to-end packet-level validation, not just single-symbol statistics: build
an actual packet (a short preamble of repeated base symbols, LoRa-style, then a
random payload), send it through the channel, and confirm the receiver -- this
project's dechirp/demodulation plus the dual-edge AFC/Kalman tracking built in
afc.py -- recovers the payload correctly. Three progressively harder tests, as
requested:

  Test 1: no CFO/Doppler at all -- the baseline. Correct decoding, repeated
          many times, then a symbol-error-rate sweep as SNR drops.
  Test 2: a constant CFO/Doppler offset across the whole packet, acquired from
          the preamble (sync.py's joint_cfo_symbol_search) and held by the
          tracking loop through the payload.
  Test 3: a UAV-style Doppler that reverses sign mid-packet (the same
          closest-point-of-approach model as examples/run_experiments.py's
          18_uav_flyover_afc.png), acquired the same way, tracked burst by
          burst through a longer payload -- a flyover needs enough bursts for
          the sign reversal to unfold slowly enough to stay trackable (see
          19_afc_rate_of_change_limit.png's ~55 Hz/burst cliff), which is why
          this test's packet is much longer than tests 1-2's.

SF=7, BW=500 kHz throughout -- a standard LoRa bandwidth option this project
hadn't exercised before (everything else in run_experiments.py uses 125 kHz).
The preamble is a simplified stand-in for real LoRa framing: N_PREAMBLE copies
of the unshifted (m=0) base symbol, the same concept as LoRa's preamble
up-chirps, generalized to any trajectory shape -- not a bit-accurate
reproduction of the real sync-word/SFD convention, which doesn't bear on the
trajectory-shape question this project is actually about.

Reuses nlfsc_lora/afc.py::run_afc_sequence unchanged: a "packet" here is just
a longer true_symbols sequence (preamble zeros + random payload) fed through
the exact same acquire-then-track pipeline already built and tested for
Chapter 6's AFC figures -- no new tracking logic, only a new way of framing
and grading what's fed into it (only the payload portion counts toward SER;
the preamble's own decode is incidental).

Run with: python examples/packet_experiments.py
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np

from nlfsc_lora.afc import AFCLoop, KalmanAFCLoop, run_afc_sequence
from nlfsc_lora.chirp import ChirpConfig
from nlfsc_lora.trajectories import TRAJECTORIES, hyperbolic_center_freq

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
SF = 7
BANDWIDTH = 500e3  # 500 kHz -- new for this project; everything else used 125 kHz
OVERSAMPLING = 4
N_PREAMBLE = 8       # simplified preamble: N_PREAMBLE copies of the base (m=0) symbol
N_PAYLOAD = 30       # payload length for tests 1-2
BIN_HZ = BANDWIDTH / (1 << SF)
HALF_BIN_HZ = BIN_HZ / 2  # native static-capture tolerance at this SF/BW, for reference


def make_cfg(traj: str) -> tuple:
    g, dg = TRAJECTORIES[traj]
    f_center = hyperbolic_center_freq(BANDWIDTH) if traj == "hyperbolic" else 0.0
    cfg = ChirpConfig(sf=SF, bandwidth=BANDWIDTH, sample_rate=OVERSAMPLING * BANDWIDTH, g=g, f_center=f_center)
    return cfg, dg


def run_packet(cfg, dg, n_payload: int, true_cfo_sequence, snr_db: float, seed: int,
                loop=None, acquire_span: float = 10000.0, acquire_step: float = 100.0):
    """Build one packet (preamble + random payload), send it through
    run_afc_sequence exactly as built for Chapter 6, and return only the
    payload's decode result -- the preamble exists to be acquired from, not to
    be graded.

    acquire_bursts=N_PREAMBLE puts the whole preamble to use for acquisition
    (majority-voted, afc.py::_combine_acquisition), not just its first symbol:
    discovered while building this test, a single acquisition burst is
    noticeably less SNR-robust than steady-state tracking (searching many CFO
    candidates against one noisy burst gives more chances for a false peak),
    and this is exactly what a multi-symbol preamble is for."""
    payload_rng = np.random.default_rng(seed)
    payload = payload_rng.integers(0, cfg.M, n_payload)
    symbols = np.concatenate([np.zeros(N_PREAMBLE, dtype=int), payload])

    decoded, tracked, residual = run_afc_sequence(
        cfg, dg, symbols, true_cfo_sequence, snr_db,
        seed=seed + 1_000_000,  # independent stream from the payload rng above
        acquire=True, acquire_span=acquire_span, acquire_step=acquire_step, acquire_bursts=N_PREAMBLE,
        loop=loop,
    )
    return decoded[N_PREAMBLE:], payload, tracked, residual


def _ser_sweep(cfg, dg, n_payload, cfo_fn, snr_range, n_trials, loop_factory=None, acquire_span=10000.0):
    """Mean SER across n_trials packets at each SNR point. cfo_fn(n_total) builds
    the true_cfo_sequence for a packet of that total length (preamble+payload)."""
    n_total = N_PREAMBLE + n_payload
    true_cfo_sequence = cfo_fn(n_total)
    ser = np.empty(len(snr_range))
    for i, snr_db in enumerate(snr_range):
        errors, total = 0, 0
        for t in range(n_trials):
            loop = loop_factory() if loop_factory is not None else None
            decoded, payload, _tr, _res = run_packet(cfg, dg, n_payload, true_cfo_sequence, snr_db, seed=t,
                                                       loop=loop, acquire_span=acquire_span)
            errors += int(np.sum(decoded != payload))
            total += n_payload
        ser[i] = errors / total
    return ser


def test1_no_doppler():
    """Test 1: same frequency, no offset, no Doppler. Send the (simplified)
    sync preamble, then random payload symbols; confirm the payload decodes
    correctly, repeated across many trials, then across a falling-SNR sweep."""
    print("\n=== Test 1: no CFO/Doppler ===")
    cfg, dg = make_cfg("hyperbolic")

    # Correctness first, noiseless, many trials, before any SER statistics.
    zero_cfo = np.zeros(N_PREAMBLE + N_PAYLOAD)
    total_errors = 0
    n_trials_check = 200
    for t in range(n_trials_check):
        decoded, payload, _tr, _res = run_packet(cfg, dg, N_PAYLOAD, zero_cfo, snr_db=40.0, seed=t)
        total_errors += int(np.sum(decoded != payload))
    print(f"noiseless-ish (40dB) correctness check: {total_errors} payload-symbol errors "
          f"out of {n_trials_check * N_PAYLOAD} across {n_trials_check} packets")
    assert total_errors == 0, "Test 1 baseline should decode every payload symbol correctly"

    snr_range = np.arange(-30, -8, 2)
    n_trials = 150
    t0 = time.time()
    ser = _ser_sweep(cfg, dg, N_PAYLOAD, lambda n: np.zeros(n), snr_range, n_trials)
    print(f"SER sweep done in {time.time()-t0:.1f}s")
    return snr_range, ser


def test2_constant_doppler():
    """Test 2: exact same packet setup as test 1, but with a constant CFO/Doppler
    offset (6000 Hz, well beyond this SF/BW's ~1953 Hz native half-bin
    tolerance) across the whole packet, acquired from the preamble and held by
    the tracking loop through the payload. Both AFCLoop and KalmanAFCLoop."""
    print("\n=== Test 2: constant CFO/Doppler ===")
    cfg, dg = make_cfg("hyperbolic")
    const_cfo = 6000.0
    print(f"constant CFO = {const_cfo} Hz (half-bin tolerance at this SF/BW is only {HALF_BIN_HZ:.0f} Hz)")

    cfo_fn = lambda n: np.full(n, const_cfo)

    # Correctness check for each tracker, noiseless-ish.
    for name, factory in [("AFCLoop", lambda: AFCLoop(gain=0.3)), ("KalmanAFCLoop", lambda: KalmanAFCLoop())]:
        total_errors = 0
        n_trials_check = 100
        seq = cfo_fn(N_PREAMBLE + N_PAYLOAD)
        for t in range(n_trials_check):
            decoded, payload, _tr, _res = run_packet(cfg, dg, N_PAYLOAD, seq, snr_db=40.0, seed=t, loop=factory())
            total_errors += int(np.sum(decoded != payload))
        print(f"{name}: noiseless-ish (40dB) correctness check: {total_errors} errors "
              f"out of {n_trials_check * N_PAYLOAD} across {n_trials_check} packets")
        assert total_errors == 0, f"Test 2 baseline should decode correctly with {name}"

    snr_range = np.arange(-30, -8, 2)
    n_trials = 150
    results = {}
    for name, factory in [("AFCLoop", lambda: AFCLoop(gain=0.3)), ("KalmanAFCLoop", lambda: KalmanAFCLoop())]:
        t0 = time.time()
        ser = _ser_sweep(cfg, dg, N_PAYLOAD, cfo_fn, snr_range, n_trials, loop_factory=factory)
        print(f"{name} SER sweep done in {time.time()-t0:.1f}s")
        results[name] = ser
    return snr_range, results


def test3_uav_doppler():
    """Test 3: same idea as test 2, but the Doppler is a UAV-style flyover that
    reverses sign mid-packet (the same closest-point-of-approach model as
    18_uav_flyover_afc.png). A flyover needs enough bursts for the sign
    reversal to unfold slowly enough to stay trackable -- max_cfo/t0 must stay
    well under the ~55 Hz/burst cliff from 19_afc_rate_of_change_limit.png --
    so this test's payload is much longer than tests 1-2's, not an arbitrary
    change: the same t0=90 (peak rate ~33 Hz/burst) already validated safe --
    at SNR=10dB.

    That cliff turns out to be SNR-dependent, discovered directly rather than
    assumed: at this same t0=90, sweeping SNR from -10dB up to 10dB found
    accuracy pinned at a hard floor (~55% payload symbol errors) for
    everything below roughly 6dB, then resolving to perfect by 10dB -- not a
    smooth waterfall like tests 1-2's. Tracing one failing run showed why: the
    first ~300 payload symbols decode perfectly, then accuracy drops to
    exactly 0% right around the reversal's steepest point and never recovers
    for the rest of the packet -- a deterministic loss of lock, not noise
    accumulating gradually. The ~55 Hz/burst cliff in
    19_afc_rate_of_change_limit.png was characterized at a single SNR (10dB);
    a noisier dual-edge measurement makes any given rate harder to track, so
    the true safe-rate ceiling drops as SNR drops, and 33 Hz/burst -- safely
    inside that ceiling at 10dB -- stops being safe below roughly 6-8dB here.
    The SNR range below sweeps across that transition rather than the lower
    range tests 1-2 use, so it actually shows the shape of it instead of
    landing entirely on one side."""
    print("\n=== Test 3: UAV-style (sign-reversing) Doppler ===")
    cfg, dg = make_cfg("hyperbolic")
    max_cfo, t0 = 3000.0, 90.0
    n_payload = 712  # + N_PREAMBLE=8 -> 720 total, matching 18_uav_flyover_afc.png's validated span
    half = (N_PREAMBLE + n_payload) / 2.0
    print(f"flyover: max_cfo={max_cfo}Hz, t0={t0} bursts, peak rate={max_cfo/t0:.1f} Hz/burst "
          f"(well under the ~55 Hz/burst cliff), packet length={N_PREAMBLE + n_payload} bursts")

    def cfo_fn(n):
        t = np.linspace(-half, half, n)
        return -max_cfo * t / np.sqrt(t ** 2 + t0 ** 2)

    for name, factory in [("AFCLoop", lambda: AFCLoop(gain=0.3)), ("KalmanAFCLoop", lambda: KalmanAFCLoop())]:
        total_errors = 0
        n_trials_check = 10
        seq = cfo_fn(N_PREAMBLE + n_payload)
        for t in range(n_trials_check):
            decoded, payload, _tr, _res = run_packet(cfg, dg, n_payload, seq, snr_db=40.0, seed=t, loop=factory())
            total_errors += int(np.sum(decoded != payload))
        print(f"{name}: noiseless-ish (40dB) correctness check: {total_errors} errors "
              f"out of {n_trials_check * n_payload} across {n_trials_check} packets")
        assert total_errors == 0, f"Test 3 baseline should decode correctly with {name}"

    snr_range = np.arange(-2, 14, 2)  # spans the discovered lock-loss transition (~6-10dB), not tests 1-2's range
    n_trials = 20
    results = {}
    for name, factory in [("AFCLoop", lambda: AFCLoop(gain=0.3)), ("KalmanAFCLoop", lambda: KalmanAFCLoop())]:
        t0s = time.time()
        ser = _ser_sweep(cfg, dg, n_payload, cfo_fn, snr_range, n_trials, loop_factory=factory)
        print(f"{name} SER sweep done in {time.time()-t0s:.1f}s")
        results[name] = ser
    return snr_range, results


def plot_all(t1, t2, t3):
    os.makedirs(OUT_DIR, exist_ok=True)
    snr1, ser1 = t1
    snr2, res2 = t2
    snr3, res3 = t3
    floor1 = 1.0 / (150 * N_PAYLOAD)
    floor2 = floor1
    floor3 = 1.0 / (20 * 712)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(snr1, np.maximum(ser1, floor1), marker="o", ms=4)
    axes[0].set(xlabel="SNR (dB)", ylabel="payload symbol error rate", yscale="log",
                title="Test 1: no CFO/Doppler")

    for name, ser in res2.items():
        axes[1].plot(snr2, np.maximum(ser, floor2), marker="o", ms=4, label=name)
    axes[1].set(xlabel="SNR (dB)", yscale="log", title="Test 2: constant CFO (6 kHz)")
    axes[1].legend(fontsize=8)

    for name, ser in res3.items():
        axes[2].plot(snr3, np.maximum(ser, floor3), marker="o", ms=4, label=name)
    axes[2].set(xlabel="SNR (dB)", yscale="log", title="Test 3: UAV-style Doppler reversal")
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.invert_xaxis()  # read left-to-right as a channel degrading over time, matching 14_ser_vs_snr_all_shapes.png

    fig.suptitle("End-to-end packet test: preamble + random payload, SF=7, BW=500kHz, hyperbolic trajectory")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "20_packet_level_validation.png"), dpi=150)
    plt.close(fig)


def main():
    t1 = test1_no_doppler()
    t2 = test2_constant_doppler()
    t3 = test3_uav_doppler()
    plot_all(t1, t2, t3)
    print(f"\nWrote figure to {OUT_DIR}/20_packet_level_validation.png")


if __name__ == "__main__":
    main()
