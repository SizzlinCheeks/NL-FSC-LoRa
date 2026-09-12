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

Run with: python examples/run_experiments.py
"""

import os
import time
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from nlfsc_lora.chirp import ChirpConfig, base_frequency, symbol_waveform
from nlfsc_lora.doppler import doppler_scale_response
from nlfsc_lora.local_rate import local_rate_demod, local_rate_ser_vs_snr
from nlfsc_lora.metrics import autocorrelation, instantaneous_chirp_rate
from nlfsc_lora.receiver import matched_filter_bank_demod
from nlfsc_lora.simulate import ser_vs_cfo, ser_vs_doppler_scale, ser_vs_model_mismatch, ser_vs_snr, ser_vs_timing_offset
from nlfsc_lora.trajectories import TRAJECTORIES, hyperbolic_center_freq

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
SF = 7
BANDWIDTH = 125e3  # a standard LoRa channel bandwidth
OVERSAMPLING = 4
SEED = 0

SHAPES = ["linear", "quadratic", "sigmoid", "sinusoidal", "exponential"]


def cfg_for(traj: str) -> ChirpConfig:
    g, _ = TRAJECTORIES[traj]
    return ChirpConfig(sf=SF, bandwidth=BANDWIDTH, sample_rate=OVERSAMPLING * BANDWIDTH, g=g)


def plot_trajectories():
    fig, (ax_f, ax_rate) = plt.subplots(1, 2, figsize=(11, 4))
    for traj in SHAPES:
        cfg = cfg_for(traj)
        f = base_frequency(cfg)
        t = np.arange(cfg.n_samples) / cfg.sample_rate * 1e3
        ax_f.plot(t, f / 1e3, label=traj)
        ax_rate.plot(t, instantaneous_chirp_rate(f, cfg.sample_rate) / 1e9, label=traj)
    ax_f.set(xlabel="t (ms)", ylabel="f(t) (kHz)", title="Instantaneous frequency")
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
    bin_hz = BANDWIDTH / (1 << SF)
    cfo_range = np.linspace(0, 1.8 * bin_hz, 14)
    seeds = [0, 1, 2]
    n_symbols = 150
    fig, ax = plt.subplots(figsize=(7, 4))
    for traj in SHAPES:
        cfg = cfg_for(traj)
        sers = [ser_vs_cfo(cfg, cfo_range, snr_db=6, n_symbols=n_symbols, demod="mfbank", seed=s) for s in seeds]
        ax.plot(cfo_range, np.mean(sers, axis=0), label=traj)
    ax.axvline(bin_hz / 2, color="k", linestyle=":", linewidth=1, label="half bin (B/M/2)")
    ax.set(xlabel="CFO (Hz)", ylabel="symbol error rate",
           title=f"SER vs carrier frequency offset (MF bank, SNR=6dB, {len(seeds)*n_symbols} symbols/point)")
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
    """
    f_center = hyperbolic_center_freq(BANDWIDTH)
    doppler_shapes = ["linear", "quadratic", "sigmoid", "hyperbolic"]
    n_samples = int(round((1 << SF) / BANDWIDTH * OVERSAMPLING * BANDWIDTH))
    half_bin_alpha = 1.0 + (n_samples / (1 << SF) / 2) / n_samples
    alpha_range = np.linspace(1.0, 2 * half_bin_alpha - 1.0, 21)
    seeds = [0, 1, 2]
    n_symbols = 100

    fig, ax = plt.subplots(figsize=(7, 4))
    for traj in doppler_shapes:
        g, _ = TRAJECTORIES[traj]
        cfg = ChirpConfig(sf=SF, bandwidth=BANDWIDTH, sample_rate=OVERSAMPLING * BANDWIDTH, g=g, f_center=f_center)
        sers = [ser_vs_doppler_scale(cfg, alpha_range, snr_db=10, n_symbols=n_symbols, demod="mfbank", seed=s) for s in seeds]
        ax.plot(alpha_range, np.mean(sers, axis=0), label=traj)
    ax.axvline(half_bin_alpha, color="k", linestyle=":", linewidth=1, label="half-bin threshold")
    ax.set(xlabel="Doppler time-scale factor (alpha)", ylabel="symbol error rate",
           title=f"LoRa-style M-ary SER vs. time-scale Doppler (MF bank, SNR=10dB, {len(seeds)*n_symbols} symbols/point)")
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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_trajectories()
    plot_autocorrelation()
    plot_ser_vs_snr()
    plot_ser_vs_cfo()
    plot_ser_vs_timing_offset()
    plot_model_mismatch()
    plot_doppler_scale_tolerance()
    plot_lora_ser_vs_doppler_scale()
    plot_lora_cfo_correction()
    plot_local_rate_estimator()
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
