"""End-to-end comparison of linear vs. nonlinear chirp trajectories.

Holds bandwidth, symbol duration, sample rate, and channel constant (per
section 16 of the design writeup) and varies only the frequency trajectory
g(t/T). Produces PNGs under examples/output/:

  01_frequency_trajectories.png  - f(t) for each shape, and the resulting df/dt
  02_autocorrelation.png         - autocorrelation magnitude, main lobe + sidelobes
  03_ser_vs_snr.png              - SER vs SNR, FFT-bin decoder vs matched-filter bank
  04_ser_vs_cfo.png              - SER vs carrier frequency offset (narrowband Doppler)
  05_ser_vs_timing_offset.png    - SER vs integer-sample timing error
  06_model_mismatch.png          - SER when the receiver's g doesn't match the transmitter's
  07_doppler_scale_tolerance.png - matched-filter peak vs. wideband (time-scale) Doppler,
                                    linear/quadratic/sigmoid vs. hyperbolic FM (HFM)
  08_lora_ser_vs_doppler_scale.png - the same comparison run through full LoRa-style
                                    M-ary cyclic-shift decoding instead of a single filter
  09_lora_cfo_correction.png     - SER vs CFO, CFO-blind matched-filter-bank decoding vs.
                                    a joint CFO+symbol search receiver
  10_local_rate_estimator.png    - SER vs SNR, matched-filter bank vs. a much cheaper
                                    single-symbol local-chirp-rate decoder
  11_hfm_vs_quadratic_ser.png    - SER vs SNR, linear vs quadratic vs hyperbolic (HFM),
                                    high-statistics waterfall using fft_correlation_demod
  12_dual_edge_afc.png           - dual-edge rate sensing driving a multi-burst AFC loop
                                    that tracks a drifting Doppler, vs a static correction
  13_fft_correlation_demo.png    - fft_correlation_demod laid open: |S[k]| and |R[k]|
                                    (broadband, uninformative on their own), what
                                    conjugating R[k] actually changes (Im{.} flips sign,
                                    Re{.} doesn't), then |IDFT{S[k]*conj(R[k])}[l]|, where
                                    all that broadband energy concentrates into one peak
  14_ser_vs_snr_all_shapes.png   - SER vs SNR for every trajectory shape at once, all
                                    decoded with fft_correlation_demod for a fair comparison
  15_dechirp_linear_vs_hyperbolic.png - symbol 33, linear vs hyperbolic: dechirped
                                    instantaneous frequency and the resulting FFT, showing
                                    linear collapse to one clean tone/peak and hyperbolic not
  16_hyperbolic_symbol33_waveform.png - what the hyperbolic trajectory's symbol-33
                                    cyclic shift actually looks like, f(t) vs t
  17_kalman_vs_expfilter_ramp.png - same 0-3000Hz linear drift as 12_dual_edge_afc.png,
                                    fixed-gain AFCLoop vs a constant-velocity KalmanAFCLoop
  18_uav_flyover_afc.png         - dual-edge AFC through a UAV-style Doppler sign
                                    reversal (approach -> closest approach -> recede),
                                    static vs AFCLoop vs KalmanAFCLoop
  19_afc_rate_of_change_limit.png - decode accuracy vs peak Doppler rate at the sign
                                    crossing: where dual-edge tracking actually breaks,
                                    and how far that is from any realistic UAV/satellite rate
  21_paired_sweep_doppler_correction.png - SER vs Doppler scale, uncorrected vs a real
                                    active-sonar-style (DHFM) paired opposite-sweep
                                    preamble correction (nlfsc_lora.paired_sweep)

Run with: python examples/run_experiments.py
"""

import os
import time
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from nlfsc_lora.afc import AFCLoop, KalmanAFCLoop, run_afc_sequence
from nlfsc_lora.channel import apply_doppler_scale, awgn
from nlfsc_lora.chirp import ChirpConfig, base_frequency, base_waveform, symbol_waveform
from nlfsc_lora.doppler import doppler_scale_response
from nlfsc_lora.local_rate import local_rate_demod, local_rate_ser_vs_snr
from nlfsc_lora.metrics import autocorrelation, instantaneous_chirp_rate
from nlfsc_lora.paired_sweep import acquire_doppler_scale, correct_doppler_scale, reversed_trajectory
from nlfsc_lora.receiver import dechirp, fft_correlation_demod, fft_demod, matched_filter_bank_demod
from nlfsc_lora.simulate import ser_vs_cfo, ser_vs_doppler_scale, ser_vs_model_mismatch, ser_vs_snr, ser_vs_timing_offset
from nlfsc_lora.trajectories import TRAJECTORIES, hyperbolic_center_freq

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
SF = 7
BANDWIDTH = 125e3  # a standard LoRa channel bandwidth
OVERSAMPLING = 4
SEED = 0

SHAPES = ["linear", "quadratic", "sigmoid", "sinusoidal", "exponential"]
TRAJECTORY_PLOT_SHAPES = SHAPES + ["hyperbolic"]


def cfg_for(traj: str) -> ChirpConfig:
    g, _ = TRAJECTORIES[traj]
    f_center = hyperbolic_center_freq(BANDWIDTH) if traj == "hyperbolic" else 0.0
    return ChirpConfig(sf=SF, bandwidth=BANDWIDTH, sample_rate=OVERSAMPLING * BANDWIDTH, g=g, f_center=f_center)


def plot_trajectories():
    # Plotted as f(t) - f_center so every shape (including hyperbolic, which
    # is only well-posed away from 0 Hz) is comparable on the same axes.
    fig, (ax_f, ax_rate) = plt.subplots(1, 2, figsize=(11, 4))
    for traj in TRAJECTORY_PLOT_SHAPES:
        cfg = cfg_for(traj)
        f = base_frequency(cfg)
        t = np.arange(cfg.n_samples) / cfg.sample_rate * 1e3
        ax_f.plot(t, (f - cfg.f_center) / 1e3, label=traj)
        ax_rate.plot(t, instantaneous_chirp_rate(f, cfg.sample_rate) / 1e9, label=traj)
    ax_f.set(xlabel="t (ms)", ylabel="f(t) - f_center (kHz)", title="Instantaneous frequency")
    ax_rate.set(xlabel="t (ms)", ylabel="df/dt (GHz/s)", title="Instantaneous chirp rate")
    ax_f.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "01_frequency_trajectories.png"), dpi=150)
    plt.close(fig)


def plot_hyperbolic_symbol_waveform(m: int = 33):
    """What one actual symbol's cyclic shift looks like, same style as
    01_frequency_trajectories.png (f(t) vs t) but for a single trajectory
    (hyperbolic) at a single symbol (m=33 by default) instead of every
    shape's m=0 base. Symbol m is the base trajectory cyclically time-shifted
    by m*T/M (chirp.py::symbol_waveform); plotting base_frequency itself
    rolled the same way shows that shift exactly, including the wrap
    discontinuity where the trajectory's end folds back to its start.
    """
    cfg = cfg_for("hyperbolic")
    n = cfg.n_samples
    shift = int(round(m * n / cfg.M)) % n
    f_shifted = np.roll(base_frequency(cfg), -shift)
    t = np.arange(n) / cfg.sample_rate * 1e3
    wrap_t = t[(n - shift) % n]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, (f_shifted - cfg.f_center) / 1e3)
    ax.axvline(wrap_t, color="k", linestyle="--", linewidth=1, label="cyclic-shift wrap point")
    ax.set(xlabel="t (ms)", ylabel="f(t) - f_center (kHz)",
           title=f"Hyperbolic trajectory, symbol m={m}: the actual waveform this project decodes")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "16_hyperbolic_symbol33_waveform.png"), dpi=150)
    plt.close(fig)


def plot_autocorrelation():
    fig, ax = plt.subplots(figsize=(7, 4))
    for traj in SHAPES:
        cfg = cfg_for(traj)
        x = symbol_waveform(cfg, 0)
        ac = autocorrelation(x)
        lags = np.arange(-len(ac) // 2, len(ac) // 2)
        ax.plot(lags, 20 * np.log10(np.abs(ac) + 1e-12), label=traj)
    ax.set(xlabel="lag (samples)", ylabel="|R_ss| (dB)", title="Autocorrelation, symbol m=0", ylim=(-40, 5))
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "02_autocorrelation.png"), dpi=150)
    plt.close(fig)


def plot_fft_correlation_demo():
    """Lay open fft_correlation_demod's three steps for one received symbol:
    FFT of the reference, FFT of the received signal, and the correlation
    C[l] = IDFT{S[k]*conj(R[k])} recovered from them -- the same computation
    as nlfsc_lora/receiver.py::fft_correlation_demod, just with the
    intermediate arrays plotted instead of only the final argmax.
    """
    traj = "hyperbolic"
    m_true = 33
    snr_db = 10.0
    g, _ = TRAJECTORIES[traj]
    cfg = ChirpConfig(
        sf=SF, bandwidth=BANDWIDTH, sample_rate=OVERSAMPLING * BANDWIDTH, g=g,
        f_center=hyperbolic_center_freq(BANDWIDTH),
    )
    n = cfg.n_samples
    rng = np.random.default_rng(SEED)

    base = base_waveform(cfg)
    rx = awgn(symbol_waveform(cfg, m_true), snr_db, rng)

    S = np.fft.fft(base)
    R = np.fft.fft(rx)
    corr = np.fft.ifft(S * np.conj(R))
    valid_lags = (np.arange(cfg.M) * n // cfg.M) % n
    m_hat = int(np.argmax(np.abs(corr[valid_lags])))
    tau_true = m_true * n // cfg.M

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    axes[0].plot(np.abs(S))
    axes[0].set(xlabel="k", ylabel="|S[k]|", title="FFT of the reference waveform")

    axes[1].plot(np.abs(R), color="tab:orange")
    axes[1].set(xlabel="k", ylabel="|R[k]|", title=f"FFT of the received signal (symbol {m_true}, {snr_db:.0f} dB SNR)")

    # |conj(R[k])| is identical to |R[k]| -- conjugation never changes magnitude.
    # What it actually does is flip the sign of the imaginary part (equivalently,
    # negate the phase); that's the only part of R[k] this panel needs to show.
    axes[2].plot(np.imag(R), color="tab:orange", alpha=0.6, label="Im{R[k]}")
    axes[2].plot(np.imag(np.conj(R)), color="tab:purple", label="Im{conj(R[k])} = -Im{R[k]}")
    axes[2].axhline(0, color="k", linewidth=0.6)
    axes[2].set(xlabel="k", ylabel="amplitude", title="Conjugating R[k]: Re{·} unchanged, Im{·} flips sign")
    axes[2].legend(fontsize=8, loc="upper right")

    n_over_m = n // cfg.M
    axes[3].plot(np.abs(corr), color="tab:green", label="|C[l]| = |IDFT{S[k]*conj(R[k])}[l]|")
    axes[3].axvline(tau_true, color="k", linestyle="--", linewidth=1)
    axes[3].scatter(valid_lags, np.abs(corr[valid_lags]), s=10, color="tab:red", zorder=3, label="the M valid symbol positions")
    peak_y = np.abs(corr[tau_true])
    axes[3].annotate(
        f"dashed line = peak, at lag {tau_true}\n= symbol {tau_true}/{n_over_m} = {m_hat}",
        xy=(tau_true, peak_y), xytext=(tau_true + 110, peak_y * 0.55),
        fontsize=8, ha="left", arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
    )
    axes[3].set(xlabel=f"lag l (samples 0..{n-1}); symbol m = l / (N/M) = l / {n_over_m}", ylabel="|C[l]|",
                title=f"Correlation after IFFT  (decoded m={m_hat}, true m={m_true})")
    axes[3].legend(fontsize=8, loc="upper right")

    fig.suptitle("The two FFTs are broadband and uninformative alone; the IFFT of their product concentrates into one peak")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "13_fft_correlation_demo.png"), dpi=150)
    plt.close(fig)


def plot_dechirp_linear_vs_hyperbolic():
    """The concrete before/after Chapter 2 of NARRATIVE.md walks through in
    words: dechirp the same symbol (33, noiseless) for linear vs hyperbolic
    and show what's actually different. For linear, dechirping produces a
    flat instantaneous frequency (a tone), so its FFT is one sharp spike at
    bin 33 and fft_demod reads the symbol straight off it. For hyperbolic,
    the dechirped frequency keeps changing, so the FFT smears across many
    bins and fft_demod's argmax lands on the wrong symbol entirely.
    """
    m_true = 33
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for row, traj in enumerate(["linear", "hyperbolic"]):
        g, _ = TRAJECTORIES[traj]
        f_center = hyperbolic_center_freq(BANDWIDTH) if traj == "hyperbolic" else 0.0
        cfg = ChirpConfig(sf=SF, bandwidth=BANDWIDTH, sample_rate=OVERSAMPLING * BANDWIDTH, g=g, f_center=f_center)
        n = cfg.n_samples
        rx = symbol_waveform(cfg, m_true)  # noiseless: isolates the dechirp/FFT mechanism itself
        d = dechirp(rx, cfg)

        f_inst = np.diff(np.unwrap(np.angle(d))) * cfg.sample_rate / (2 * np.pi)
        t = np.arange(len(f_inst)) / cfg.sample_rate * 1e3
        wrap_t = t[(n - int(round(m_true * n / cfg.M))) % n]
        axes[row, 0].plot(t, f_inst / 1e3)
        axes[row, 0].axvline(wrap_t, color="k", linestyle="--", linewidth=1, alpha=0.5, label="cyclic-shift wrap point")
        axes[row, 0].set(xlabel="t (ms)", ylabel="f (kHz)")
        axes[row, 0].set_title(f"{traj}: instantaneous frequency AFTER dechirping (symbol {m_true})", fontsize=10)
        axes[row, 0].legend(fontsize=8)

        spec = np.fft.fft(d)
        decoded = fft_demod(rx, cfg)
        valid_lags = (np.arange(cfg.M) * n // cfg.M) % n
        axes[row, 1].plot(np.abs(spec))
        for lag in valid_lags:
            axes[row, 1].axvline(lag, color="gray", linewidth=0.4, alpha=0.25, zorder=0)
        axes[row, 1].axvline(m_true, color="k", linestyle="--", linewidth=1, label=f"true symbol = {m_true}")
        correct = "correct" if decoded == m_true else "WRONG"
        axes[row, 1].set(xlabel=f"FFT bin k  (N={n} bins; gray lines = the M={cfg.M} valid symbol positions)",
                          ylabel="|spec[k]|")
        axes[row, 1].set_title(f"{traj}: FFT of dechirped signal (decoded m={decoded}, {correct})", fontsize=10)
        axes[row, 1].legend(fontsize=8)

    fig.suptitle("Dechirping symbol 33: linear collapses to a (piecewise-constant) tone; hyperbolic doesn't")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "15_dechirp_linear_vs_hyperbolic.png"), dpi=150)
    plt.close(fig)


def plot_ser_vs_snr():
    # Wide despreading gain at SF7 (N=512 samples/symbol) pushes the MF-bank
    # waterfall down to roughly -30..-15 dB; the FFT-bin decoder for nonlinear
    # trajectories fails even at high SNR, so its curve sits near 1.0 throughout.
    snr_range = np.arange(-30, -8, 2)
    n_symbols = 150
    floor = 1.0 / (2 * n_symbols)  # so a measured SER of exactly 0 still shows on the log axis
    fig, ax = plt.subplots(figsize=(7, 4))
    for traj in SHAPES:
        cfg = cfg_for(traj)
        ser_fft = ser_vs_snr(cfg, snr_range, n_symbols=n_symbols, demod="fft", seed=SEED)
        ser_mf = ser_vs_snr(cfg, snr_range, n_symbols=n_symbols, demod="mfbank", seed=SEED)
        ax.plot(snr_range, np.maximum(ser_fft, floor), "--", label=f"{traj} (FFT)")
        ax.plot(snr_range, np.maximum(ser_mf, floor), "-", label=f"{traj} (MF bank)")
    ax.set(xlabel="SNR (dB)", ylabel="symbol error rate", yscale="log", title="SER vs SNR")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "03_ser_vs_snr.png"), dpi=150)
    plt.close(fig)


def plot_ser_vs_cfo():
    # The decision boundary between adjacent LoRa symbols sits at half a bin
    # (B/M / 2); average over several seeds so the transition band is a real
    # measurement rather than one noisy draw -- a single seed at this symbol
    # count makes trajectories look several hundred Hz apart when the gap is
    # actually within a bin's width of noise (see the mismatch investigation
    # in the project history for how misleading one seed can be here).
    # fft_correlation_demod (exact same decision as mfbank, ~85x cheaper -- see
    # receiver.py) affords much tighter statistics than the original 450/point.
    f_center = hyperbolic_center_freq(BANDWIDTH)
    doppler_shapes = ["linear", "quadratic", "sigmoid", "hyperbolic"]
    bin_hz = BANDWIDTH / (1 << SF)
    cfo_range = np.linspace(0, 1.8 * bin_hz, 19)
    seeds = [0, 1, 2, 3, 4]
    n_symbols = 2000
    fig, ax = plt.subplots(figsize=(7, 4))
    for traj in doppler_shapes:
        g, _ = TRAJECTORIES[traj]
        cfg = ChirpConfig(sf=SF, bandwidth=BANDWIDTH, sample_rate=OVERSAMPLING * BANDWIDTH, g=g, f_center=f_center)
        sers = [ser_vs_cfo(cfg, cfo_range, snr_db=6, n_symbols=n_symbols, demod="fft_corr", seed=s) for s in seeds]
        ax.plot(cfo_range, np.mean(sers, axis=0), marker="o", ms=3, label=traj)
    ax.axvline(bin_hz / 2, color="k", linestyle=":", linewidth=1, label="half bin (B/M/2)")
    ax.set(xlabel="CFO (Hz)", ylabel="symbol error rate",
           title=f"SER vs carrier frequency offset (SNR=6dB, {len(seeds)*n_symbols} symbols/point)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "04_ser_vs_cfo.png"), dpi=150)
    plt.close(fig)


def plot_ser_vs_timing_offset():
    fig, ax = plt.subplots(figsize=(7, 4))
    for traj in SHAPES:
        cfg = cfg_for(traj)
        offsets = np.arange(0, cfg.n_samples // 8, max(1, cfg.n_samples // 64))
        ser = ser_vs_timing_offset(cfg, offsets, snr_db=6, n_symbols=150, demod="mfbank", seed=SEED)
        ax.plot(offsets, ser, label=traj)
    ax.set(xlabel="timing offset (samples)", ylabel="symbol error rate", title="SER vs timing offset (MF bank, SNR=6dB)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "05_ser_vs_timing_offset.png"), dpi=150)
    plt.close(fig)


def plot_model_mismatch():
    """Receiver assumes power(p) with p offset from the transmitter's true exponent."""
    cfg_tx = ChirpConfig(sf=SF, bandwidth=BANDWIDTH, sample_rate=OVERSAMPLING * BANDWIDTH, g=TRAJECTORIES["quadratic"][0])
    p_values = np.linspace(1.96, 2.04, 17)
    g_variants = [partial(_power, p=p) for p in p_values]
    ser = ser_vs_model_mismatch(cfg_tx, g_variants, snr_db=6, n_symbols=150, demod="mfbank", seed=SEED)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(p_values, ser, marker="o")
    ax.axvline(2.0, color="k", linestyle=":", label="true exponent (p=2)")
    ax.set(xlabel="receiver's assumed exponent p", ylabel="symbol error rate",
           title="SER vs transmitter/receiver trajectory mismatch (SNR=6dB)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "06_model_mismatch.png"), dpi=150)
    plt.close(fig)


def _power(u, p):
    return np.asarray(u, dtype=float) ** p


def plot_doppler_scale_tolerance():
    """Wideband ("true") Doppler: r(t) = s(alpha*t), as opposed to the constant-CFO
    approximation used everywhere else in this file (see nlfsc_lora.doppler for why the
    two are different models, and which one HFM's invariance claim is actually about).

    All shapes are embedded at the *same* nonzero f_center so the comparison is apples to
    apples -- scale-Doppler sensitivity depends on the absolute frequency band, not just
    the bandwidth, and hyperbolic FM (HFM) is only well-posed away from 0 Hz in the first
    place (1/f(t) is singular at f=0).
    """
    f_center = hyperbolic_center_freq(BANDWIDTH)
    doppler_shapes = ["linear", "quadratic", "sigmoid", "hyperbolic"]
    alpha_range = np.linspace(0.9, 1.1, 41)

    fig, (ax_mag, ax_lag) = plt.subplots(1, 2, figsize=(11, 4))
    for traj in doppler_shapes:
        g, _ = TRAJECTORIES[traj]
        cfg = ChirpConfig(sf=SF, bandwidth=BANDWIDTH, sample_rate=OVERSAMPLING * BANDWIDTH, g=g, f_center=f_center)
        peaks, lags = doppler_scale_response(cfg, alpha_range)
        ax_mag.plot(alpha_range, 20 * np.log10(peaks + 1e-12), label=traj)
        ax_lag.plot(alpha_range, lags, label=traj)
    ax_mag.set(xlabel="Doppler time-scale factor (alpha)", ylabel="matched-filter peak (dB)",
               title="Peak magnitude vs. time-scale Doppler", ylim=(-15, 1))
    ax_lag.set(xlabel="Doppler time-scale factor (alpha)", ylabel="peak lag (samples)",
               title="Peak location vs. time-scale Doppler")
    ax_mag.legend(fontsize=8)
    fig.suptitle(f"Wideband Doppler tolerance, all shapes swept over [{f_center-BANDWIDTH/2:.0f}, {f_center+BANDWIDTH/2:.0f}] Hz")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "07_doppler_scale_tolerance.png"), dpi=150)
    plt.close(fig)


def plot_lora_ser_vs_doppler_scale():
    """Does HFM's ambiguity-function-level Doppler-scale advantage (07) survive when it's
    actually plugged into LoRa's cyclic-shift M-ary decoding, not just a single matched
    filter? Uses matched_filter_bank_demod throughout -- fft_demod already can't decode a
    hyperbolic-trajectory symbol at all, Doppler or not (see
    tests/test_receiver.py::test_fft_demod_fails_for_properly_embedded_hyperbolic_chirp).

    The half-bin timing tolerance from 04_ser_vs_cfo.png (B/M / 2) reappears here almost
    exactly: for SF7, a Doppler scale of alpha drifts the correlation peak by roughly
    alpha_minus_1 * n_samples samples, and once that exceeds half a symbol bin (n_samples/M/2)
    every trajectory starts making errors, curvature or not -- LoRa's fine, M-ary shift
    alphabet is far more sensitive to a Doppler-induced *lag* than any single waveform's
    correlation-*magnitude* advantage can rescue by itself.

    Uses fft_correlation_demod (exact same decision as mfbank, ~85x cheaper -- see
    receiver.py) to afford much tighter statistics than the original 300/point.
    """
    f_center = hyperbolic_center_freq(BANDWIDTH)
    doppler_shapes = ["linear", "quadratic", "sigmoid", "hyperbolic"]
    n_samples = int(round((1 << SF) / BANDWIDTH * OVERSAMPLING * BANDWIDTH))
    half_bin_alpha = 1.0 + (n_samples / (1 << SF) / 2) / n_samples
    alpha_range = np.linspace(1.0, 2 * half_bin_alpha - 1.0, 21)
    seeds = [0, 1, 2, 3, 4]
    n_symbols = 2000

    fig, ax = plt.subplots(figsize=(7, 4))
    for traj in doppler_shapes:
        g, _ = TRAJECTORIES[traj]
        cfg = ChirpConfig(sf=SF, bandwidth=BANDWIDTH, sample_rate=OVERSAMPLING * BANDWIDTH, g=g, f_center=f_center)
        sers = [ser_vs_doppler_scale(cfg, alpha_range, snr_db=10, n_symbols=n_symbols, demod="fft_corr", seed=s) for s in seeds]
        ax.plot(alpha_range, np.mean(sers, axis=0), marker="o", ms=3, label=traj)
    ax.axvline(half_bin_alpha, color="k", linestyle=":", linewidth=1, label="half-bin threshold")
    ax.set(xlabel="Doppler time-scale factor (alpha)", ylabel="symbol error rate",
           title=f"LoRa-style M-ary SER vs. time-scale Doppler (SNR=10dB, {len(seeds)*n_symbols} symbols/point)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "08_lora_ser_vs_doppler_scale.png"), dpi=150)
    plt.close(fig)


def plot_lora_cfo_correction():
    """A blind, single-symbol CFO estimate from the received spectrum's band edges
    (nlfsc_lora.sync.estimate_cfo) turned out to be a documented negative result --
    its bias from the chirp's own spectral leakage (hundreds of Hz) swamps CFO values
    on the order of a symbol bin. What does work: searching a grid of candidate CFO
    corrections and keeping whichever (CFO, symbol) pair gives the strongest
    matched-filter response (nlfsc_lora.sync.joint_cfo_symbol_demod, "cfo_search" in
    simulate._DEMODS) -- far more expensive per symbol, but it recovers symbols well
    past the half-bin CFO threshold that saturates the CFO-blind receiver in
    04_ser_vs_cfo.png.
    """
    cfg = cfg_for("linear")
    cfo_range = np.linspace(0, 1500, 13)
    seeds = [0, 1]
    n_symbols = 80

    fig, ax = plt.subplots(figsize=(7, 4))
    for demod, label in [("mfbank", "CFO-blind (matched_filter_bank_demod)"), ("cfo_search", "joint CFO+symbol search")]:
        sers = [ser_vs_cfo(cfg, cfo_range, snr_db=6, n_symbols=n_symbols, demod=demod, seed=s) for s in seeds]
        ax.plot(cfo_range, np.mean(sers, axis=0), label=label)
    ax.set(xlabel="CFO (Hz)", ylabel="symbol error rate",
           title=f"LoRa SER vs CFO: blind vs. corrected receiver (SNR=6dB, {len(seeds)*n_symbols} symbols/point)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "09_lora_cfo_correction.png"), dpi=150)
    plt.close(fig)


def plot_local_rate_estimator():
    """Cheaper than either decoder above: nlfsc_lora.local_rate reads the symbol (and
    CFO) off a *single local* chirp-rate measurement, needing only a small phase-polyfit
    window instead of correlating against the whole reference bank -- possible only
    because a nonlinear g's local rate varies with position (a linear chirp's rate is
    constant everywhere and carries no positional information at all).

    The tradeoff is real: no despreading gain from a full-record correlation, only
    whatever the local window averages over, so it needs tens of dB more SNR than
    matched_filter_bank_demod to work at all -- and it has a hard SER floor even at
    infinite SNR, from symbols whose cyclic-shift wrap glitch falls inside the
    (fixed-position) measurement window and corrupts the fit outright.
    """
    cfg = cfg_for("quadratic")
    _, dg = TRAJECTORIES["quadratic"]
    snr_range = np.arange(-10, 41, 5)
    n_symbols = 150

    ser_mf = ser_vs_snr(cfg, snr_range, n_symbols=n_symbols, demod="mfbank", seed=SEED)
    ser_local = local_rate_ser_vs_snr(cfg, dg, snr_range, n_symbols=n_symbols, seed=SEED)

    tx = symbol_waveform(cfg, 33)
    t0 = time.time()
    for _ in range(200):
        matched_filter_bank_demod(tx, cfg)
    mf_ms = (time.time() - t0) / 200 * 1000
    t0 = time.time()
    for _ in range(200):
        local_rate_demod(tx, cfg, dg)
    local_ms = (time.time() - t0) / 200 * 1000

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(snr_range, ser_mf, label=f"matched_filter_bank_demod ({mf_ms:.2f} ms/symbol)")
    ax.plot(snr_range, ser_local, label=f"local_rate_demod ({local_ms:.2f} ms/symbol)")
    ax.set(xlabel="SNR (dB)", ylabel="symbol error rate",
           title="SER vs SNR: full correlation vs. a single local rate measurement")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "10_local_rate_estimator.png"), dpi=150)
    plt.close(fig)


def plot_hfm_vs_quadratic_ser():
    """Does HFM's SNR performance actually differ from quadratic's (or linear's), with
    the receiver that's optimal for all of them (fft_correlation_demod -- exactly
    equivalent to matched_filter_bank_demod, just cheap enough now to afford serious
    statistics: 5000 symbols/point x 3 seeds = 15000 per point, vs. 03_ser_vs_snr.png's
    150)? All three embedded on the same absolute frequency band for a fair comparison.
    """
    f_center = hyperbolic_center_freq(BANDWIDTH)
    shapes = ["linear", "quadratic", "hyperbolic"]
    snr_range = np.arange(-32, -8, 2)
    seeds = [0, 1, 2]
    n_symbols = 5000
    floor = 1.0 / (2 * len(seeds) * n_symbols)  # so a measured SER of exactly 0 still shows on the log axis

    fig, ax = plt.subplots(figsize=(7, 4))
    for traj in shapes:
        g, _ = TRAJECTORIES[traj]
        cfg = ChirpConfig(sf=SF, bandwidth=BANDWIDTH, sample_rate=OVERSAMPLING * BANDWIDTH, g=g, f_center=f_center)
        sers = [ser_vs_snr(cfg, snr_range, n_symbols=n_symbols, demod="fft_corr", seed=s) for s in seeds]
        ax.plot(snr_range, np.maximum(np.mean(sers, axis=0), floor), marker="o", ms=3, label=traj)
    ax.set(xlabel="SNR (dB)", ylabel="symbol error rate", yscale="log",
           title=f"SER vs SNR: linear vs quadratic vs hyperbolic ({len(seeds)*n_symbols} symbols/point)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "11_hfm_vs_quadratic_ser.png"), dpi=150)
    plt.close(fig)


def plot_ser_vs_snr_all_shapes():
    """SER vs SNR for every trajectory shape in the project on one plot, all
    decoded with the same receiver (fft_correlation_demod -- the only one
    that's both fast and correct regardless of shape), so it's a fair,
    apples-to-apples comparison rather than each shape getting whichever
    decoder happens to work for it. All shapes embedded on the same absolute
    frequency band, as in plot_hfm_vs_quadratic_ser above.
    """
    f_center = hyperbolic_center_freq(BANDWIDTH)
    shapes = SHAPES + ["hyperbolic"]
    snr_range = np.arange(-32, -8, 2)
    seeds = [0, 1, 2]
    n_symbols = 3000
    floor = 1.0 / (2 * len(seeds) * n_symbols)  # so a measured SER of exactly 0 still shows on the log axis

    fig, ax = plt.subplots(figsize=(8, 5))
    for traj in shapes:
        g, _ = TRAJECTORIES[traj]
        cfg = ChirpConfig(sf=SF, bandwidth=BANDWIDTH, sample_rate=OVERSAMPLING * BANDWIDTH, g=g, f_center=f_center)
        sers = [ser_vs_snr(cfg, snr_range, n_symbols=n_symbols, demod="fft_corr", seed=s) for s in seeds]
        ax.plot(snr_range, np.maximum(np.mean(sers, axis=0), floor), marker="o", ms=3, label=traj)
    ax.set(xlabel="SNR (dB)", ylabel="symbol error rate", yscale="log",
           title=f"SER vs SNR, every trajectory shape ({len(seeds)*n_symbols} symbols/point, fft_correlation_demod)")
    ax.invert_xaxis()  # read left-to-right as a channel degrading over time: good SNR first, worsening after
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "14_ser_vs_snr_all_shapes.png"), dpi=150)
    plt.close(fig)


def plot_dual_edge_afc():
    """Does measuring rate-of-change at both ends of the swept bandwidth, tracked
    across a sequence of bursts, actually help a receiver "lock back onto center
    frequency" under a drifting Doppler (nlfsc_lora/afc.py)? A single burst's
    residual CFO is measured from two edge windows instead of local_rate.py's one
    center window -- the wrap glitch can corrupt at most one edge at a fixed
    position, so quality-weighting lets a clean edge outvote a corrupted one
    (tests/test_afc.py checks this directly for a symbol whose glitch lands in the
    start-edge window). That measurement feeds a first-order AFC loop, not a
    one-shot correction, because the point is following *drift* (e.g. a
    LoRa-over-LEO-satellite pass), not a single static offset.

    Left panel: one representative run's tracked CFO vs the true (ramping) CFO.
    Right panel: P(correct decode) vs burst index, averaged over many seeds, with
    vs without the tracking loop -- both start from the same one-time acquisition
    (sync.py's joint_cfo_symbol_search; a cold start beyond the decoder's own
    half-bin capture range fails outright without it, see afc.py's module
    docstring and tests/test_afc.py::test_cold_start_beyond_capture_range_needs_acquisition),
    but only the tracked receiver keeps up as the drift continues.
    """
    bw = BANDWIDTH
    g, dg = TRAJECTORIES["hyperbolic"]
    cfg = ChirpConfig(sf=SF, bandwidth=bw, sample_rate=OVERSAMPLING * bw, g=g, f_center=hyperbolic_center_freq(bw))
    n_bursts = 200
    max_cfo = 3000.0  # far past the ~488Hz half-bin static tolerance at SF7
    true_cfo = np.linspace(0, max_cfo, n_bursts)
    n_seeds = 40

    rng0 = np.random.default_rng(0)
    true_symbols_example = rng0.integers(0, cfg.M, n_bursts)
    _decoded, tracked_example, _residual = run_afc_sequence(cfg, dg, true_symbols_example, true_cfo, snr_db=10, seed=0)

    correct_tracked = np.zeros((n_seeds, n_bursts), dtype=bool)
    correct_static = np.zeros((n_seeds, n_bursts), dtype=bool)
    for s in range(n_seeds):
        rng = np.random.default_rng(100 + s)
        true_symbols = rng.integers(0, cfg.M, n_bursts)
        dec_t, _tr, _res = run_afc_sequence(cfg, dg, true_symbols, true_cfo, snr_db=10, gain=0.3, seed=s)
        dec_s, _tr2, _res2 = run_afc_sequence(cfg, dg, true_symbols, true_cfo, snr_db=10, gain=0.0, seed=s)
        correct_tracked[s] = dec_t == true_symbols
        correct_static[s] = dec_s == true_symbols

    fig, (ax_cfo, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_cfo.plot(true_cfo, label="true CFO", lw=1.5)
    ax_cfo.plot(tracked_example, label="tracked CFO", lw=1.0, alpha=0.85)
    ax_cfo.set(xlabel="burst index", ylabel="CFO (Hz)", title="Tracked vs true CFO (one run)")
    ax_cfo.legend(fontsize=8)

    ax_acc.plot(correct_tracked.mean(axis=0), label=f"AFC (gain=0.3)")
    ax_acc.plot(correct_static.mean(axis=0), label="static (acquire once, gain=0)")
    ax_acc.set(xlabel="burst index", ylabel="P(correct decode)",
               title=f"Decode accuracy vs burst index ({n_seeds} seeds)")
    ax_acc.legend(fontsize=8)
    fig.suptitle(f"AFC tracking a 0-{max_cfo:.0f}Hz Doppler drift over {n_bursts} bursts (SNR=10dB)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "12_dual_edge_afc.png"), dpi=150)
    plt.close(fig)


def plot_kalman_vs_expfilter_ramp():
    """AFCLoop's documented steady-state lag under a linear drift (rho per burst)
    is rho/gain -- a type-1 tracking-loop error, not measurement noise (afc.py
    module docstring). KalmanAFCLoop tracks CFO *and* CFO-rate jointly instead of
    chasing the latest measurement by a fixed fraction, so it should settle onto
    a constant drift rate with much less lag once its rate estimate converges.
    Same 0-3000Hz ramp as plot_dual_edge_afc, same everything else, so the two
    trackers' tracked-vs-true curves are directly comparable.
    """
    bw = BANDWIDTH
    g, dg = TRAJECTORIES["hyperbolic"]
    cfg = ChirpConfig(sf=SF, bandwidth=bw, sample_rate=OVERSAMPLING * bw, g=g, f_center=hyperbolic_center_freq(bw))
    n_bursts = 200
    max_cfo = 3000.0
    true_cfo = np.linspace(0, max_cfo, n_bursts)
    n_seeds = 40

    rng0 = np.random.default_rng(0)
    true_symbols_example = rng0.integers(0, cfg.M, n_bursts)
    _d, tracked_exp, _r = run_afc_sequence(cfg, dg, true_symbols_example, true_cfo, snr_db=10, seed=0,
                                            loop=AFCLoop(gain=0.3))
    _d, tracked_kal, _r = run_afc_sequence(cfg, dg, true_symbols_example, true_cfo, snr_db=10, seed=0,
                                            loop=KalmanAFCLoop())

    correct_exp = np.zeros((n_seeds, n_bursts), dtype=bool)
    correct_kal = np.zeros((n_seeds, n_bursts), dtype=bool)
    for s in range(n_seeds):
        rng = np.random.default_rng(100 + s)
        true_symbols = rng.integers(0, cfg.M, n_bursts)
        dec_e, _t, _r = run_afc_sequence(cfg, dg, true_symbols, true_cfo, snr_db=10, seed=s, loop=AFCLoop(gain=0.3))
        dec_k, _t, _r = run_afc_sequence(cfg, dg, true_symbols, true_cfo, snr_db=10, seed=s, loop=KalmanAFCLoop())
        correct_exp[s] = dec_e == true_symbols
        correct_kal[s] = dec_k == true_symbols

    fig, (ax_cfo, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_cfo.plot(true_cfo, label="true CFO", lw=1.5, color="k")
    ax_cfo.plot(tracked_exp, label="AFCLoop (fixed gain=0.3)", lw=1.0, alpha=0.85)
    ax_cfo.plot(tracked_kal, label="KalmanAFCLoop", lw=1.0, alpha=0.85)
    ax_cfo.set(xlabel="burst index", ylabel="CFO (Hz)", title="Tracked vs true CFO (one run)")
    ax_cfo.legend(fontsize=8)

    ax_acc.plot(correct_exp.mean(axis=0), label="AFCLoop (fixed gain=0.3)")
    ax_acc.plot(correct_kal.mean(axis=0), label="KalmanAFCLoop")
    ax_acc.set(xlabel="burst index", ylabel="P(correct decode)",
               title=f"Decode accuracy vs burst index ({n_seeds} seeds)")
    ax_acc.legend(fontsize=8)
    fig.suptitle(f"Fixed-gain vs Kalman tracking, same 0-{max_cfo:.0f}Hz linear drift over {n_bursts} bursts")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "17_kalman_vs_expfilter_ramp.png"), dpi=150)
    plt.close(fig)


def plot_uav_flyover_afc():
    """A satellite pass drifts the CFO in roughly one direction; a low-altitude
    UAV or drone passing near the receiver does not -- it approaches (positive
    Doppler), crosses closest approach (Doppler through zero), then recedes
    (negative Doppler), all within the same short pass. Model this with the
    standard constant-velocity closest-point-of-approach Doppler curve:

        f_d(t) = -max_cfo * t / sqrt(t^2 + t0^2)

    an S-curve saturating to +/-max_cfo far from closest approach, crossing
    zero at t=0, with t0 (bursts) controlling how many bursts the sign flip
    takes. t0=90 here (peak instantaneous rate max_cfo/t0 ~ 33 Hz/burst) is
    comfortably inside the working region found by
    plot_afc_rate_of_change_limit below (the cliff sits around ~55 Hz/burst,
    itself many orders of magnitude past any realistic UAV or satellite
    Doppler acceleration) -- this figure is the positive result: the same
    two-edge measurement this project has used throughout does track a true
    sign reversal, not just a monotonic ramp, once two real bugs found while
    testing this scenario over long sequences are fixed (see afc.py's
    max_jump_hz / innovation_gate and their docstrings: dual_edge_cfo_estimate
    occasionally, ~1 in 400 measured, passes its own quality check while still
    being wildly wrong, and without a sanity gate on the tracker that one bad
    reading was enough to permanently derail the loop for the rest of the run).
    """
    bw = BANDWIDTH
    g, dg = TRAJECTORIES["hyperbolic"]
    cfg = ChirpConfig(sf=SF, bandwidth=bw, sample_rate=OVERSAMPLING * bw, g=g, f_center=hyperbolic_center_freq(bw))
    max_cfo = 3000.0
    t0 = 90.0  # bursts; peak instantaneous rate = max_cfo/t0 ~ 33 Hz/burst, well inside the working region
    half = 360
    n_bursts = 2 * half
    t = np.linspace(-half, half, n_bursts)
    true_cfo = -max_cfo * t / np.sqrt(t ** 2 + t0 ** 2)
    n_seeds = 40

    rng0 = np.random.default_rng(0)
    true_symbols_example = rng0.integers(0, cfg.M, n_bursts)
    _d, tracked_exp, _r = run_afc_sequence(cfg, dg, true_symbols_example, true_cfo, snr_db=10, seed=0,
                                            loop=AFCLoop(gain=0.3))
    _d, tracked_kal, _r = run_afc_sequence(cfg, dg, true_symbols_example, true_cfo, snr_db=10, seed=0,
                                            loop=KalmanAFCLoop())

    correct_static = np.zeros((n_seeds, n_bursts), dtype=bool)
    correct_exp = np.zeros((n_seeds, n_bursts), dtype=bool)
    correct_kal = np.zeros((n_seeds, n_bursts), dtype=bool)
    for s in range(n_seeds):
        rng = np.random.default_rng(100 + s)
        true_symbols = rng.integers(0, cfg.M, n_bursts)
        dec_s, _t, _r = run_afc_sequence(cfg, dg, true_symbols, true_cfo, snr_db=10, seed=s, loop=AFCLoop(gain=0.0))
        dec_e, _t, _r = run_afc_sequence(cfg, dg, true_symbols, true_cfo, snr_db=10, seed=s, loop=AFCLoop(gain=0.3))
        dec_k, _t, _r = run_afc_sequence(cfg, dg, true_symbols, true_cfo, snr_db=10, seed=s, loop=KalmanAFCLoop())
        correct_static[s] = dec_s == true_symbols
        correct_exp[s] = dec_e == true_symbols
        correct_kal[s] = dec_k == true_symbols

    fig, (ax_cfo, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_cfo.plot(true_cfo, label="true CFO (UAV flyover)", lw=1.5, color="k")
    ax_cfo.plot(tracked_exp, label="AFCLoop (fixed gain=0.3)", lw=1.0, alpha=0.85)
    ax_cfo.plot(tracked_kal, label="KalmanAFCLoop", lw=1.0, alpha=0.85)
    ax_cfo.axhline(0, color="gray", linewidth=0.6)
    ax_cfo.set(xlabel="burst index", ylabel="CFO (Hz)", title="Tracked vs true CFO through a Doppler sign reversal")
    ax_cfo.legend(fontsize=8)

    ax_acc.plot(correct_static.mean(axis=0), label="static (acquire once, gain=0)")
    ax_acc.plot(correct_exp.mean(axis=0), label="AFCLoop (fixed gain=0.3)")
    ax_acc.plot(correct_kal.mean(axis=0), label="KalmanAFCLoop")
    ax_acc.set(xlabel="burst index", ylabel="P(correct decode)",
               title=f"Decode accuracy vs burst index ({n_seeds} seeds)")
    ax_acc.legend(fontsize=8)
    fig.suptitle(f"AFC through an approach/recede Doppler reversal (±{max_cfo:.0f}Hz, {n_bursts} bursts)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "18_uav_flyover_afc.png"), dpi=150)
    plt.close(fig)


def plot_afc_rate_of_change_limit():
    """Where does dual-edge tracking actually stop working through a Doppler
    sign reversal? Sweep the flyover curve's steepness (t0, bursts) and plot
    decode accuracy against the resulting peak instantaneous rate (max_cfo/t0,
    Hz/burst) for both trackers. There is a real cliff, found by testing, not
    assumed: both trackers hold ~1.0 accuracy below roughly 55 Hz/burst and
    collapse above roughly 100 Hz/burst. For context, 55 Hz/burst at this
    SF/bandwidth (symbol duration ~1.02 ms) is about 54 kHz/s of Doppler
    *acceleration* -- orders of magnitude past any real satellite or UAV
    scenario, so the cliff itself is not a practical concern, but it is a
    genuine limit of this architecture worth knowing rather than assuming
    away: past it, a receiver correcting with its own most recent estimate
    falls outside fft_correlation_demod's capture range, decodes the wrong
    symbol, and that wrong symbol corrupts the next measurement too.
    """
    bw = BANDWIDTH
    g, dg = TRAJECTORIES["hyperbolic"]
    cfg = ChirpConfig(sf=SF, bandwidth=bw, sample_rate=OVERSAMPLING * bw, g=g, f_center=hyperbolic_center_freq(bw))
    max_cfo = 3000.0
    t0_values = [15, 20, 25, 30, 35, 40, 45, 55, 70, 90, 150, 400]
    n_seeds = 12

    rates, acc_afc, acc_kal = [], [], []
    for t0 in t0_values:
        half = max(150, int(4 * t0))
        n_bursts = 2 * half
        t = np.linspace(-half, half, n_bursts)
        true_cfo = -max_cfo * t / np.sqrt(t ** 2 + t0 ** 2)
        a_accs, k_accs = [], []
        for s in range(n_seeds):
            rng = np.random.default_rng(100 + s)
            true_symbols = rng.integers(0, cfg.M, n_bursts)
            dec_a, _t, _r = run_afc_sequence(cfg, dg, true_symbols, true_cfo, snr_db=10, seed=s, loop=AFCLoop(gain=0.3))
            dec_k, _t, _r = run_afc_sequence(cfg, dg, true_symbols, true_cfo, snr_db=10, seed=s, loop=KalmanAFCLoop())
            a_accs.append((dec_a == true_symbols).mean())
            k_accs.append((dec_k == true_symbols).mean())
        rates.append(max_cfo / t0)
        acc_afc.append(np.mean(a_accs))
        acc_kal.append(np.mean(k_accs))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(rates, acc_afc, marker="o", ms=4, label="AFCLoop (fixed gain=0.3)")
    ax.plot(rates, acc_kal, marker="o", ms=4, label="KalmanAFCLoop")
    ax.axvline(1.0, color="tab:green", linestyle=":", linewidth=2,
               label="generous upper bound for realistic UAV/satellite rates (~1 Hz/burst)")
    ax.set(xlabel="peak instantaneous Doppler rate at sign crossing (Hz/burst)",
           ylabel="mean decode accuracy", xscale="log")
    ax.set_title(f"AFC through a Doppler sign reversal:\nwhere tracking actually breaks ({n_seeds} seeds/point)", fontsize=11)
    ax.set_xlim(0.5, 250.0)  # headroom left of the realistic-rate marker so it's visibly separate from the axis
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "19_afc_rate_of_change_limit.png"), dpi=150)
    plt.close(fig)


def plot_paired_sweep_doppler_correction():
    """08_lora_ser_vs_doppler_scale.png's own finding was that HFM's matched-
    filter Doppler tolerance (07) doesn't survive full M-ary decoding: the
    Doppler-scale-induced lag bias corrupts the decoded symbol directly, so
    SER collapses past roughly the same half-bin threshold every trajectory
    shares. Chapter "How Real HFM Sonar and Radar Systems Actually Handle
    Doppler" asks whether real active sonar's own fix for the analogous
    problem -- transmitting a pair of oppositely-swept pulses and comparing
    their (oppositely-biased) delay estimates (DHFM, Wang et al. 2017) --
    actually fixes this here too. It does, once applied the way real systems
    actually apply it: to a known (m=0) preamble pair, not to an arbitrary
    payload symbol's own cyclic shift directly (the first version of this
    tried the latter and found a real, shift-dependent residual bias from the
    cyclic-shift wrap discontinuity -- see paired_sweep.py's module
    docstring). Acquiring alpha once from a preamble pair and correcting
    every payload burst with it (nlfsc_lora.paired_sweep) before an ordinary
    single-sweep decode restores full accuracy across the same alpha range
    that leaves an uncorrected receiver at ~100% SER.
    """
    q = 1.0 / 3.0
    g_up, _ = TRAJECTORIES["hyperbolic"]
    g_down = reversed_trajectory(g_up)
    f_center = hyperbolic_center_freq(BANDWIDTH, q)
    cfg_up = ChirpConfig(sf=SF, bandwidth=BANDWIDTH, sample_rate=OVERSAMPLING * BANDWIDTH, g=g_up, f_center=f_center)
    cfg_down = ChirpConfig(sf=SF, bandwidth=BANDWIDTH, sample_rate=OVERSAMPLING * BANDWIDTH, g=g_down, f_center=f_center)

    alpha_range = np.linspace(0.85, 1.15, 21)
    snr_db = 10.0
    n_trials = 300
    ser_uncorrected = np.empty(len(alpha_range))
    ser_corrected = np.empty(len(alpha_range))
    for i, alpha in enumerate(alpha_range):
        rng = np.random.default_rng(i)
        rx_pre_up = awgn(apply_doppler_scale(base_waveform(cfg_up), alpha), snr_db, rng)
        rx_pre_down = awgn(apply_doppler_scale(base_waveform(cfg_down), alpha), snr_db, rng)
        alpha_hat = acquire_doppler_scale(rx_pre_up, rx_pre_down, cfg_up, cfg_down, q)

        errs_u, errs_c = 0, 0
        for _ in range(n_trials):
            m_true = int(rng.integers(0, cfg_up.M))
            rx = awgn(apply_doppler_scale(symbol_waveform(cfg_up, m_true), alpha), snr_db, rng)
            if fft_correlation_demod(rx, cfg_up) != m_true:
                errs_u += 1
            if fft_correlation_demod(correct_doppler_scale(rx, alpha_hat), cfg_up) != m_true:
                errs_c += 1
        ser_uncorrected[i] = errs_u / n_trials
        ser_corrected[i] = errs_c / n_trials

    floor = 1.0 / n_trials
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alpha_range, np.maximum(ser_uncorrected, floor), marker="o", ms=4, label="uncorrected (single sweep)")
    ax.plot(alpha_range, np.maximum(ser_corrected, floor), marker="o", ms=4,
            label="paired-sweep acquired + corrected")
    ax.set(xlabel="Doppler time-scale factor (alpha)", ylabel="symbol error rate", yscale="log",
           title=f"DHFM-style paired-sweep Doppler correction (SNR={snr_db:.0f}dB, {n_trials} symbols/point)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "21_paired_sweep_doppler_correction.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_trajectories()
    plot_autocorrelation()
    plot_fft_correlation_demo()
    plot_ser_vs_snr()
    plot_ser_vs_cfo()
    plot_ser_vs_timing_offset()
    plot_model_mismatch()
    plot_doppler_scale_tolerance()
    plot_lora_ser_vs_doppler_scale()
    plot_lora_cfo_correction()
    plot_local_rate_estimator()
    plot_hfm_vs_quadratic_ser()
    plot_ser_vs_snr_all_shapes()
    plot_dechirp_linear_vs_hyperbolic()
    plot_hyperbolic_symbol_waveform()
    plot_dual_edge_afc()
    plot_kalman_vs_expfilter_ramp()
    plot_uav_flyover_afc()
    plot_afc_rate_of_change_limit()
    plot_paired_sweep_doppler_correction()
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
