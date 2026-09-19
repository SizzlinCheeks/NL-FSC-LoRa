"""Trajectory-domain multiplexing, characterized rather than assumed: how
well do N differently-shaped streams actually separate when they share one
band and one time slot?

`nlfsc_lora/trajectory_mux.py` explains the mechanism (the same
wrong-trajectory-smears-energy property `fft_correlation_demod` has always
relied on against noise, now facing another stream's *signal* instead).
This file measures three things the mechanism doesn't by itself guarantee:

1. **Cross-trajectory leakage** -- for every pair of shapes, how strongly
   could a pure single-shape signal masquerade as a valid symbol on a
   different shape's receiver? (`33_trajectory_mimo_leakage_matrix.png`)
2. **Capacity scaling** -- as more streams share the same slot at equal
   power, each one's own signal is a smaller fraction of the total
   received power even before any channel noise (K streams means every
   individual stream faces (K-1)/K of the total power as self-interference
   alone). How many streams can this project's own receiver actually pull
   apart before SER rises? (`34_trajectory_mimo_capacity_and_nearfar.png`,
   left panel)
3. **Near-far sensitivity** -- unlike orthogonal codes (CDMA) or antenna-
   separated spatial streams, these trajectories are only *approximately*
   separable. When one stream is much stronger than another sharing the
   same slot, does the weak one still decode? (right panel)

Run with: python examples/trajectory_mimo_experiments.py
"""
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from nlfsc_lora.chirp import ChirpConfig, base_waveform, symbol_waveform
from nlfsc_lora.trajectories import TRAJECTORIES, hyperbolic_center_freq
from nlfsc_lora.trajectory_mux import demux_stream, mix_streams

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
SF = 7
BANDWIDTH = 500e3
OVERSAMPLING = 4
SHAPES = ["linear", "quadratic", "sigmoid", "exponential", "hyperbolic"]


def make_cfg(shape: str) -> ChirpConfig:
    """Every stream shares one band -- hyperbolic_center_freq(BANDWIDTH),
    which also satisfies hyperbolic's own band-must-not-cross-zero
    constraint -- because that shared band is the entire point: a stream
    parked at a different center frequency would trivially separate by
    filtering, proving nothing about trajectory-domain separation."""
    g, _ = TRAJECTORIES[shape]
    return ChirpConfig(sf=SF, bandwidth=BANDWIDTH, sample_rate=OVERSAMPLING * BANDWIDTH, g=g,
                        f_center=hyperbolic_center_freq(BANDWIDTH))


def _add_noise_unit_ref_power(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Every symbol_waveform has unit sample power (|exp(j*phi)| = 1
    exactly), regardless of shape -- so noise referenced to *one* stream's
    own power, not the mixed total, keeps snr_db meaning the same thing
    ("what SNR would this one stream have transmitted alone") as the
    stream count K grows. K-1 other streams then add self-interference on
    top of that fixed noise floor, rather than the noise floor itself
    silently shrinking as K grows -- the real multi-user scenario this
    experiment is about."""
    noise_power = 1.0 / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape))
    return x + noise


def cross_trajectory_leakage(cfg_i: ChirpConfig, cfg_j: ChirpConfig, n_symbols: int = 60,
                              rng: np.random.Generator = None) -> float:
    """Average, over random symbols of stream i, the peak normalized
    correlation of a *pure* shape-i signal against shape-j's own demux
    reference, evaluated at shape-j's M valid symbol lags -- exactly what
    fft_correlation_demod would see if stream i were present alone.
    1.0 = indistinguishable from a true shape-j symbol; near 0 = well
    separated. The diagonal (i == j) is 1.0 by construction."""
    rng = rng or np.random.default_rng(0)
    base_j_fft = np.fft.fft(base_waveform(cfg_j))
    n = cfg_j.n_samples
    valid_lags = (np.arange(cfg_j.M) * n // cfg_j.M) % n
    norm_j_base = np.linalg.norm(base_waveform(cfg_j))
    leak = np.empty(n_symbols)
    for k in range(n_symbols):
        m_i = int(rng.integers(0, cfg_i.M))
        x_i = symbol_waveform(cfg_i, m_i)
        corr = np.fft.ifft(base_j_fft * np.conj(np.fft.fft(x_i)))
        peak = np.max(np.abs(corr[valid_lags]))
        leak[k] = peak / (norm_j_base * np.linalg.norm(x_i))
    return float(np.mean(leak))


def experiment_leakage_matrix():
    cfgs = {s: make_cfg(s) for s in SHAPES}
    rng = np.random.default_rng(0)
    matrix = np.empty((len(SHAPES), len(SHAPES)))
    for i, si in enumerate(SHAPES):
        for j, sj in enumerate(SHAPES):
            matrix[i, j] = cross_trajectory_leakage(cfgs[si], cfgs[sj], rng=rng)
    return matrix


def plot_leakage_matrix(matrix):
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(SHAPES)), SHAPES, rotation=45, ha="right")
    ax.set_yticks(range(len(SHAPES)), SHAPES)
    ax.set(xlabel="receiver's own shape (j)", ylabel="interfering signal's shape (i)",
           title="Cross-trajectory leakage: how much a pure shape-i signal\n"
                 "looks like a valid symbol to a shape-j receiver (1.0 = indistinguishable)")
    for i in range(len(SHAPES)):
        for j in range(len(SHAPES)):
            color = "white" if matrix[i, j] < 0.5 else "black"
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(im, ax=ax, label="leakage (peak normalized correlation)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "33_trajectory_mimo_leakage_matrix.png"), dpi=150)
    plt.close(fig)


def experiment_capacity_scaling():
    cfgs_all = [make_cfg(s) for s in SHAPES]
    snr_range = np.arange(-6, 13, 2)
    n_trials = 300
    results = {}
    for k in [1, 2, 3, 4, 5]:
        cfgs = cfgs_all[:k]
        t0 = time.time()
        ser = np.empty(len(snr_range))
        for si, snr_db in enumerate(snr_range):
            rng = np.random.default_rng(1000 * k + si)
            errors = 0
            for _ in range(n_trials):
                symbols = rng.integers(0, cfgs[0].M, size=k)
                mixed = mix_streams(cfgs, symbols)
                rx = _add_noise_unit_ref_power(mixed, snr_db, rng)
                for cfg, m_true in zip(cfgs, symbols):
                    errors += int(demux_stream(rx, cfg) != m_true)
            ser[si] = errors / (n_trials * k)
        print(f"[K={k}] done in {time.time()-t0:.1f}s")
        results[k] = ser
    return snr_range, results


def experiment_near_far():
    """Two streams (linear + hyperbolic) sharing one slot at a fixed,
    moderate victim SNR (10dB, so the noise floor alone would not explain
    a victim decode failure): sweep the aggressor's power relative to the
    victim's and measure the victim's SER as the imbalance grows."""
    cfg_victim = make_cfg("linear")
    cfg_aggressor = make_cfg("hyperbolic")
    victim_snr_db = 10.0
    power_ratio_db = np.arange(-10, 21, 2)
    n_trials = 300

    ser = np.empty(len(power_ratio_db))
    for i, ratio_db in enumerate(power_ratio_db):
        rng = np.random.default_rng(2000 + i)
        aggressor_amp = 10 ** (ratio_db / 20.0)
        errors = 0
        for _ in range(n_trials):
            m_victim, m_aggr = rng.integers(0, cfg_victim.M, size=2)
            mixed = mix_streams([cfg_victim, cfg_aggressor], [m_victim, m_aggr],
                                 amplitudes=[1.0, aggressor_amp])
            rx = _add_noise_unit_ref_power(mixed, victim_snr_db, rng)
            errors += int(demux_stream(rx, cfg_victim) != m_victim)
        ser[i] = errors / n_trials
    return power_ratio_db, ser


def plot_capacity_and_nearfar(snr_range, capacity_results, power_ratio_db, nearfar_ser):
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    for k, ser in capacity_results.items():
        ax1.plot(snr_range, np.maximum(ser, 1e-4), marker="o", ms=4, label=f"K={k} streams")
    ax1.set(xlabel="per-stream SNR (dB)", ylabel="symbol error rate", yscale="log",
            title="Trajectory-multiplexed capacity:\nSER vs SNR as stream count K grows")
    ax1.legend(fontsize=8)

    ax2.plot(power_ratio_db, nearfar_ser, marker="o", ms=4, color="tab:red")
    ax2.set(xlabel="aggressor power relative to victim (dB)", ylabel="victim symbol error rate",
            title="Near-far sensitivity: victim (linear) SER vs.\n"
                  "aggressor (hyperbolic) power imbalance, victim SNR=10dB")
    ax2.axhline(0, color="gray", lw=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "34_trajectory_mimo_capacity_and_nearfar.png"), dpi=150)
    plt.close(fig)


def main():
    matrix = experiment_leakage_matrix()
    plot_leakage_matrix(matrix)

    snr_range, capacity_results = experiment_capacity_scaling()
    power_ratio_db, nearfar_ser = experiment_near_far()
    plot_capacity_and_nearfar(snr_range, capacity_results, power_ratio_db, nearfar_ser)
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
