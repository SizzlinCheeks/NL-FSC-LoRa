"""Trajectory-domain multiplexing: separating simultaneous streams by
frequency-trajectory *shape* instead of by antenna geometry (classical
spatial MIMO) or by spreading factor (real LoRaWAN's own within-band
multiplexing dimension).

This falls out of a property `receiver.py::fft_correlation_demod` has had
since Chapter 4, proven there for a single stream against noise:
correlating with the *wrong* trajectory leaves substantial residual chirp,
smearing energy across many lags rather than concentrating it in one.
`examples/interference_experiments.py` already measured the same
mechanism operating between different *spreading factors* sharing a real
LoRaWAN channel (SF quasi-orthogonality, `28_lorawan_sf_spectrogram.png`)
-- different SFs sweep the same band at different rates, so they coincide
only briefly. Trajectory shape is a second, independent axis of exactly
that kind of separation: two different-shaped sweeps at the *same* SF
(same sweep rate on average) still diverge in instantaneous frequency
almost everywhere across the symbol, because they trace different curves
through the same band.

No real LoRa chipset generates anything but a linear sweep, so real
hardware's only within-band multiplexing dimension is SF. This project's
own receiver, running its own trajectory family, gets trajectory shape as
a second dimension for free: N differently-shaped streams sharing one
band and one time slot, demultiplexed with N ordinary
`fft_correlation_demod` calls, each against its own stream's `g` -- ordinary
receive processing, not a new decoder.

How well streams separate is not assumed here; `examples/
trajectory_mimo_experiments.py` measures it directly (a cross-trajectory
leakage matrix, multi-stream SER vs. SNR, and a near-far power-imbalance
sweep), the same way SF quasi-orthogonality was measured rather than
assumed.
"""
import numpy as np

from .chirp import ChirpConfig, symbol_waveform
from .receiver import fft_correlation_demod


def mix_streams(cfgs: list, symbols, amplitudes=None) -> np.ndarray:
    """Sum of amplitude_i * symbol_waveform(cfg_i, symbols[i]) for each
    simultaneous stream i, all occupying the same time slot and band.
    Every cfg must share n_samples (same sf, bandwidth, sample_rate) --
    only g (and optionally f_center) may differ between streams."""
    n = cfgs[0].n_samples
    for cfg in cfgs:
        if cfg.n_samples != n:
            raise ValueError("all multiplexed streams must share symbol duration and sample rate")
    amps = np.ones(len(cfgs)) if amplitudes is None else np.asarray(amplitudes, dtype=float)
    mixed = np.zeros(n, dtype=complex)
    for cfg, m, a in zip(cfgs, symbols, amps):
        mixed = mixed + a * symbol_waveform(cfg, int(m))
    return mixed


def demux_stream(rx: np.ndarray, cfg: ChirpConfig) -> int:
    """Recover one stream's symbol from the combined signal: every other
    stream's energy, riding on a different trajectory shape, acts on this
    stream's own correlation the same way broadband interference or noise
    would -- ordinary fft_correlation_demod, nothing stream-count-aware
    about it."""
    return fft_correlation_demod(rx, cfg)
