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

Run with: python examples/run_experiments.py
"""

import os
import time
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from nlfsc_lora.afc import run_afc_sequence
from nlfsc_lora.channel import awgn
from nlfsc_lora.chirp import ChirpConfig, base_frequency, base_waveform, symbol_waveform
from nlfsc_lora.doppler import doppler_scale_response
from nlfsc_lora.local_rate import local_rate_demod, local_rate_ser_vs_snr
from nlfsc_lora.metrics import autocorrelation, instantaneous_chirp_rate
from nlfsc_lora.receiver import dechirp, fft_demod, matched_filter_bank_demod
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

    axes[3].plot(np.abs(corr), color="tab:green", label="|C[l]| = |IDFT{S[k]*conj(R[k])}[l]|")
    axes[3].axvline(tau_true, color="k", linestyle="--", linewidth=1, label=f"true shift  = {tau_true}")
    axes[3].scatter(valid_lags, np.abs(corr[valid_lags]), s=10, color="tab:red", zorder=3, label="the M valid symbol positions")
    axes[3].set(xlabel="lag l", ylabel="|C[l]|", title=f"Correlation after IFFT  (decoded m={m_hat}, true m={m_true})")
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
        rx = symbol_waveform(cfg, m_true)  # noiseless: isolates the dechirp/FFT mechanism itself
        d = dechirp(rx, cfg)

        f_inst = np.diff(np.unwrap(np.angle(d))) * cfg.sample_rate / (2 * np.pi)
        t = np.arange(len(f_inst)) / cfg.sample_rate * 1e3
        axes[row, 0].plot(t, f_inst / 1e3)
        axes[row, 0].set(xlabel="t (ms)", ylabel="f (kHz)",
                          title=f"{traj}: dechirped instantaneous frequency (symbol {m_true})")

        spec = np.fft.fft(d)
        decoded = fft_demod(rx, cfg)
        axes[row, 1].plot(np.abs(spec))
        axes[row, 1].axvline(m_true, color="k", linestyle="--", linewidth=1, label=f"true symbol = {m_true}")
        correct = "correct" if decoded == m_true else "WRONG"
        axes[row, 1].set(xlabel="FFT bin k", ylabel="|spec[k]|")
        axes[row, 1].set_title(f"{traj}: FFT of dechirped signal (decoded m={decoded}, {correct})", fontsize=10)
        axes[row, 1].legend(fontsize=8)

    fig.suptitle("Dechirping symbol 33: linear collapses to a tone (one clean FFT peak); hyperbolic doesn't")
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

    ax_acc.plot(correct_tracked.mean(axis=0), label=f"dual-edge AFC (gain=0.3)")
    ax_acc.plot(correct_static.mean(axis=0), label="static (acquire once, gain=0)")
    ax_acc.set(xlabel="burst index", ylabel="P(correct decode)",
               title=f"Decode accuracy vs burst index ({n_seeds} seeds)")
    ax_acc.legend(fontsize=8)
    fig.suptitle(f"Dual-edge AFC tracking a 0-{max_cfo:.0f}Hz Doppler drift over {n_bursts} bursts (SNR=10dB)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "12_dual_edge_afc.png"), dpi=150)
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
    plot_dual_edge_afc()
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
