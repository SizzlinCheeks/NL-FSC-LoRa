"""End-to-end packet-level validation, not just single-symbol statistics: build
an actual packet (a short preamble, LoRa-style, then a random payload), send
it through the channel, and confirm the receiver -- this project's
dechirp/demodulation plus its Doppler-correction machinery (afc.py for
tests 1-3, paired_sweep.py for test 4) -- recovers the payload correctly.
Four progressively harder tests, as requested:

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
  Test 4: a wideband ("true") Doppler time-scale, a different channel model
          from tests 1-3's narrowband (constant-CFO) Doppler entirely --
          acquired not from afc.py's machinery but from
          nlfsc_lora.paired_sweep's DHFM-style paired opposite-sweep preamble
          (one up-sweep burst, one down-sweep burst, both m=0), the real
          active-sonar technique examples/run_experiments.py's
          21_paired_sweep_doppler_correction.png already validated at the
          single-symbol level -- this test checks it holds up in a full
          packet, not just isolated symbols.

SF=7, BW=500 kHz throughout -- a standard LoRa bandwidth option this project
hadn't exercised before (everything else in run_experiments.py uses 125 kHz).
Tests 1-3's preamble is a simplified stand-in for real LoRa framing:
N_PREAMBLE copies of the unshifted (m=0) base symbol, the same concept as
LoRa's preamble up-chirps, generalized to any trajectory shape -- not a
bit-accurate reproduction of the real sync-word/SFD convention, which doesn't
bear on the trajectory-shape question this project is actually about. Test 4
uses a different, shorter preamble (one up-sweep + one down-sweep m=0 burst,
paired_sweep.py's own acquisition unit) -- see its own docstring.

Tests 1-3 reuse nlfsc_lora/afc.py::run_afc_sequence unchanged: a "packet"
there is just a longer true_symbols sequence (preamble zeros + random
payload) fed through the exact same acquire-then-track pipeline already
built and tested for Chapter 6's AFC figures -- no new tracking logic, only
a new way of framing and grading what's fed into it (only the payload
portion counts toward SER; the preamble's own decode is incidental). Test 4
uses nlfsc_lora/paired_sweep.py instead -- a different correction mechanism
for a different (wideband, not narrowband) Doppler model entirely.

Run with: python examples/packet_experiments.py
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np

from nlfsc_lora.afc import AFCLoop, KalmanAFCLoop, measurement_noise_for, run_afc_sequence
from nlfsc_lora.channel import apply_doppler_scale, awgn
from nlfsc_lora.chirp import ChirpConfig, base_waveform, symbol_waveform
from nlfsc_lora.paired_sweep import acquire_doppler_scale, correct_doppler_scale, reversed_trajectory
from nlfsc_lora.receiver import fft_correlation_demod
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


def make_cfg_pair(q: float = 1.0 / 3.0) -> tuple:
    """Up-sweep and down-sweep hyperbolic ChirpConfigs sharing the same band,
    for Test 4's paired-sweep preamble (nlfsc_lora.paired_sweep)."""
    g_up, dg_up = TRAJECTORIES["hyperbolic"]
    g_down = reversed_trajectory(g_up)
    f_center = hyperbolic_center_freq(BANDWIDTH, q)
    cfg_up = ChirpConfig(sf=SF, bandwidth=BANDWIDTH, sample_rate=OVERSAMPLING * BANDWIDTH, g=g_up, f_center=f_center)
    cfg_down = ChirpConfig(sf=SF, bandwidth=BANDWIDTH, sample_rate=OVERSAMPLING * BANDWIDTH, g=g_down, f_center=f_center)
    return cfg_up, cfg_down


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
    for name, factory in [("AFCLoop", lambda: AFCLoop(gain=0.3)), ("KalmanAFCLoop", lambda: KalmanAFCLoop(measurement_noise=measurement_noise_for(cfg)))]:
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
    for name, factory in [("AFCLoop", lambda: AFCLoop(gain=0.3)), ("KalmanAFCLoop", lambda: KalmanAFCLoop(measurement_noise=measurement_noise_for(cfg)))]:
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
    change: t0=90 (peak rate ~33 Hz/burst) is comfortably under that cliff.

    This test used to have a much worse floor: below roughly 5-8dB SNR,
    tracking accuracy collapsed to a flat ~55% error regardless of how low
    SNR went, unlike tests 1-2's smooth waterfall -- unacceptable for a
    protocol whose entire point is decoding under the noise floor. Tracing it
    down (not just re-explaining it) found the real cause was upstream of any
    Doppler-specific tracking dynamics: the per-burst residual-CFO measurement
    this loop depends on, afc.py's original dual_edge_cfo_estimate, fits an
    81-sample local window and gets none of the ~27dB coherent processing
    gain fft_correlation_demod itself gets from correlating the *whole*
    512-sample symbol -- so even with a perfectly-tracked, zero-residual
    signal (nothing to track at all), it returned an unusable None on
    ~85-90% of bursts at -10dB SNR (measured directly). Tests 1-2 never
    exposed this because neither needs a continuous stream of fresh
    measurements (test 1 needs no tracking; test 2's CFO is constant, so
    coasting on a stale value between rare good measurements is harmless).
    Test 3's continuously-moving CFO is the only one of the three that does
    need it, so it's the only one that exposed a floor that was silently
    present the whole time.

    The fix: afc.py::full_symbol_cfo_estimate, which dechirps the *whole*
    symbol against its known decoded shape and reads the residual CFO off an
    FFT peak instead -- the same coherent-integration length the decoder
    itself uses, so it gets the same processing gain. run_afc_sequence uses
    this by default now. Measured directly, that alone took this test from a
    hard floor around 5-8dB down to AFCLoop decoding this exact scenario
    perfectly at -12dB SNR and below. KalmanAFCLoop needed a second fix on
    top of that: the new estimator's noise is roughly 16x smaller here than
    at the 125kHz configuration chapter 6's examples use (FFT-peak-estimator
    variance scales with the square of the FFT bin width, and this test's
    bandwidth -- 500kHz vs 125kHz -- is 4x wider), so KalmanAFCLoop's fixed
    measurement_noise default badly overstated its own noise here and made
    the filter sluggish exactly when the reversal moved fastest. Using
    afc.py::measurement_noise_for(cfg) (config-scaled, not a fixed constant)
    instead of the bare default closes nearly all of that gap.

    Below roughly -14dB SNR both trackers still degrade -- correctly:
    full_symbol_cfo_estimate's own FFT-threshold effect (see its docstring)
    genuinely runs out of usable measurements down there, same as the
    decoder itself eventually would. The SNR range below sweeps across that
    transition instead of parking entirely on one side of it."""
    print("\n=== Test 3: UAV-style (sign-reversing) Doppler ===")
    cfg, dg = make_cfg("hyperbolic")
    max_cfo, t0 = 3000.0, 90.0
    n_payload = 712  # + N_PREAMBLE=8 -> 720 total, matching 18_uav_flyover_afc.png's validated span
    half = (N_PREAMBLE + n_payload) / 2.0
    print(f"flyover: max_cfo={max_cfo}Hz, t0={t0} bursts, peak rate={max_cfo/t0:.1f} Hz/burst "
          f"(well under the ~55 Hz/burst cliff at 10dB), packet length={N_PREAMBLE + n_payload} bursts")

    def cfo_fn(n):
        t = np.linspace(-half, half, n)
        return -max_cfo * t / np.sqrt(t ** 2 + t0 ** 2)

    for name, factory in [("AFCLoop", lambda: AFCLoop(gain=0.3)), ("KalmanAFCLoop", lambda: KalmanAFCLoop(measurement_noise=measurement_noise_for(cfg)))]:
        total_errors = 0
        n_trials_check = 10
        seq = cfo_fn(N_PREAMBLE + n_payload)
        for t in range(n_trials_check):
            decoded, payload, _tr, _res = run_packet(cfg, dg, n_payload, seq, snr_db=40.0, seed=t, loop=factory())
            total_errors += int(np.sum(decoded != payload))
        print(f"{name}: noiseless-ish (40dB) correctness check: {total_errors} errors "
              f"out of {n_trials_check * n_payload} across {n_trials_check} packets")
        assert total_errors == 0, f"Test 3 baseline should decode correctly with {name}"

    snr_range = np.arange(-20, 2, 2)  # spans the new (much lower) transition, now that the measurement floor is fixed
    n_trials = 20
    results = {}
    for name, factory in [("AFCLoop", lambda: AFCLoop(gain=0.3)), ("KalmanAFCLoop", lambda: KalmanAFCLoop(measurement_noise=measurement_noise_for(cfg)))]:
        t0s = time.time()
        ser = _ser_sweep(cfg, dg, n_payload, cfo_fn, snr_range, n_trials, loop_factory=factory)
        print(f"{name} SER sweep done in {time.time()-t0s:.1f}s")
        results[name] = ser
    return snr_range, results


def run_packet_paired_sweep(cfg_up, cfg_down, n_payload: int, alpha_true: float, q: float, snr_db: float, seed: int):
    """One packet under Test 4's channel model: a wideband Doppler time-scale
    alpha_true (channel.apply_doppler_scale), not tests 1-3's narrowband
    constant CFO. Preamble is a single paired-sweep pair (one up-sweep, one
    down-sweep, both m=0) -- already validated robust to well below -10dB SNR
    with just one pair (nlfsc_lora.paired_sweep, 21_paired_sweep_doppler_correction.png),
    unlike sync.py's joint_cfo_symbol_search acquisition which needed several
    preamble bursts averaged together (Test 2). Returns both the paired-sweep-
    corrected and the uncorrected payload decode, so the same packet-level
    comparison 21 already made at the single-symbol level can be checked here
    too."""
    rng = np.random.default_rng(seed)
    rx_pre_up = awgn(apply_doppler_scale(base_waveform(cfg_up), alpha_true), snr_db, rng)
    rx_pre_down = awgn(apply_doppler_scale(base_waveform(cfg_down), alpha_true), snr_db, rng)
    alpha_hat = acquire_doppler_scale(rx_pre_up, rx_pre_down, cfg_up, cfg_down, q)

    payload = rng.integers(0, cfg_up.M, n_payload)
    decoded_corrected = np.empty(n_payload, dtype=int)
    decoded_uncorrected = np.empty(n_payload, dtype=int)
    for i, m_true in enumerate(payload):
        rx = awgn(apply_doppler_scale(symbol_waveform(cfg_up, int(m_true)), alpha_true), snr_db, rng)
        decoded_uncorrected[i] = fft_correlation_demod(rx, cfg_up)
        decoded_corrected[i] = fft_correlation_demod(correct_doppler_scale(rx, alpha_hat), cfg_up)
    return decoded_corrected, decoded_uncorrected, payload


def test4_wideband_doppler_scale():
    """Test 4: a wideband Doppler time-scale (not tests 1-3's narrowband
    constant CFO), corrected with nlfsc_lora.paired_sweep's DHFM-style paired
    opposite-sweep preamble rather than afc.py's machinery -- the real
    active-sonar technique from "How Real HFM Sonar and Radar Systems
    Actually Handle Doppler" / examples/run_experiments.py's
    21_paired_sweep_doppler_correction.png, now checked in a full packet
    rather than isolated symbols.

    alpha_true=1.05 is comfortably past this configuration's own half-bin
    failure threshold (~1.004, from the same n_samples/M relationship
    08_lora_ser_vs_doppler_scale.png documents) -- an uncorrected receiver
    should fail on essentially every payload symbol."""
    print("\n=== Test 4: wideband Doppler scale, paired-sweep correction ===")
    q = 1.0 / 3.0
    cfg_up, cfg_down = make_cfg_pair(q)
    alpha_true = 1.05
    print(f"alpha_true={alpha_true} (half-bin failure threshold at this SF/BW is only "
          f"~{1.0 + 1.0 / (2 * cfg_up.M):.4f})")

    total_corrected, total_uncorrected = 0, 0
    n_trials_check = 100
    for t in range(n_trials_check):
        dec_c, dec_u, payload = run_packet_paired_sweep(cfg_up, cfg_down, N_PAYLOAD, alpha_true, q, snr_db=40.0, seed=t)
        total_corrected += int(np.sum(dec_c != payload))
        total_uncorrected += int(np.sum(dec_u != payload))
    print(f"noiseless-ish (40dB) correctness check: corrected {total_corrected} errors, "
          f"uncorrected {total_uncorrected} errors, out of {n_trials_check * N_PAYLOAD} across {n_trials_check} packets")
    assert total_corrected == 0, "Test 4 baseline should decode correctly once paired-sweep corrected"
    assert total_uncorrected > 0.9 * n_trials_check * N_PAYLOAD, \
        "an uncorrected receiver should fail on essentially every payload symbol at this alpha"

    snr_range = np.arange(-20, 2, 2)
    n_trials = 150
    t0 = time.time()
    ser_corrected = np.empty(len(snr_range))
    ser_uncorrected = np.empty(len(snr_range))
    for i, snr_db in enumerate(snr_range):
        errs_c, errs_u = 0, 0
        for t in range(n_trials):
            dec_c, dec_u, payload = run_packet_paired_sweep(cfg_up, cfg_down, N_PAYLOAD, alpha_true, q, snr_db, seed=t)
            errs_c += int(np.sum(dec_c != payload))
            errs_u += int(np.sum(dec_u != payload))
        total = n_trials * N_PAYLOAD
        ser_corrected[i] = errs_c / total
        ser_uncorrected[i] = errs_u / total
    print(f"SER sweep done in {time.time()-t0:.1f}s")
    return snr_range, {"paired-sweep corrected": ser_corrected, "uncorrected": ser_uncorrected}


def plot_all(t1, t2, t3, t4):
    os.makedirs(OUT_DIR, exist_ok=True)
    snr1, ser1 = t1
    snr2, res2 = t2
    snr3, res3 = t3
    snr4, res4 = t4
    floor1 = 1.0 / (150 * N_PAYLOAD)
    floor2 = floor1
    floor3 = 1.0 / (20 * 712)
    floor4 = 1.0 / (150 * N_PAYLOAD)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
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

    for name, ser in res4.items():
        axes[3].plot(snr4, np.maximum(ser, floor4), marker="o", ms=4, label=name)
    axes[3].set(xlabel="SNR (dB)", yscale="log", title="Test 4: wideband Doppler scale\n(paired-sweep correction)")
    axes[3].legend(fontsize=8)

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
    t4 = test4_wideband_doppler_scale()
    plot_all(t1, t2, t3, t4)
    print(f"\nWrote figure to {OUT_DIR}/20_packet_level_validation.png")


if __name__ == "__main__":
    main()
