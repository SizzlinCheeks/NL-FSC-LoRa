"""Wideband ("true") Doppler: time-scaling of the transmitted envelope.

channel.apply_cfo models Doppler the way it actually behaves for LoRa's real
RF link: a bulk carrier-frequency shift, constant over one symbol, because
target velocities are utterly negligible next to the speed of light. That is
the *narrowband* approximation.

The other, exact model is that a moving reflector time-scales the whole
waveform: r(t) = s(alpha * (t - tau)) with alpha = 1 +- v/c (or 2v/c for a
round-trip radar/sonar return). The two coincide only when alpha is close
enough to 1 that the envelope's own time-bandwidth product doesn't notice
the scaling -- true for LoRa's RF Doppler, but not for the regime hyperbolic
FM (HFM) was built for: high time-bandwidth sonar/radar pulses, or targets
where v/c isn't negligible (a diving bat, a fast torpedo). This module tests
that regime directly, independent of whether it's realistic at LoRa's carrier
frequency and bandwidth -- it is a stress test of the waveform, not the link.

The classical HFM property: because its instantaneous frequency satisfies
1/f(t) linear in t, time-scaling f(t) by alpha only reproduces (very nearly)
the same trajectory *shifted in time*, not distorted -- so an unscaled HFM
reference still gives a strong matched-filter response to a Doppler-scaled
echo, merely at a shifted lag. A linear chirp's rate is *not* self-similar
under scaling, so the same test should show its matched-filter peak degrade
(spread and shrink) rather than just shift.
"""

import numpy as np

from .chirp import ChirpConfig, base_waveform, phase_accumulator


def extended_base_waveform(cfg: ChirpConfig, margin: float) -> np.ndarray:
    """Base (m=0) chirp evaluated over margin * symbol_duration, extrapolating g past u=1."""
    n = cfg.n_samples
    n_long = int(np.ceil(n * margin))
    u = np.arange(n_long) / n  # still normalized by ONE symbol's length; u exceeds 1 past n
    f = cfg.f_center - cfg.bandwidth / 2.0 + cfg.bandwidth * cfg.g(u)
    phi = phase_accumulator(f, cfg.sample_rate)
    return np.exp(1j * phi)


def scaled_replica(cfg: ChirpConfig, alpha: float, margin: float = 1.3) -> np.ndarray:
    """Approximate echo r(t) = s0(alpha * t), resampled onto the base chirp's own sample grid."""
    n = cfg.n_samples
    long_wave = extended_base_waveform(cfg, margin)
    idx = alpha * np.arange(n)
    if idx[-1] >= len(long_wave) - 1:
        raise ValueError(f"margin={margin} too small for alpha={alpha}; increase margin")
    grid = np.arange(len(long_wave))
    re = np.interp(idx, grid, long_wave.real)
    im = np.interp(idx, grid, long_wave.imag)
    return re + 1j * im


def peak_response(cfg: ChirpConfig, alpha: float, margin: float = 1.3):
    """Normalized matched-filter peak magnitude and its lag (samples), reference vs. a
    time-scaled replica -- the ambiguity-function-style metric HFM's invariance claim is about."""
    ref = base_waveform(cfg)
    rx = scaled_replica(cfg, alpha, margin=margin)
    xc = np.correlate(rx, ref, mode="full")  # linear (non-circular) cross-correlation
    mag = np.abs(xc) / (np.linalg.norm(ref) * np.linalg.norm(rx) + 1e-15)
    peak_idx = int(np.argmax(mag))
    lag = peak_idx - (len(ref) - 1)
    return float(mag[peak_idx]), lag


def doppler_scale_response(cfg: ChirpConfig, alpha_range, margin: float = 1.3):
    """Peak magnitude and peak lag across a range of Doppler time-scale factors alpha."""
    peaks = np.empty(len(alpha_range))
    lags = np.empty(len(alpha_range))
    for i, alpha in enumerate(alpha_range):
        peaks[i], lags[i] = peak_response(cfg, alpha, margin=margin)
    return peaks, lags
