"""Correlation and spectral-occupancy diagnostics for a chirp waveform.

These answer the questions the writeup poses in sections 7-8: does the
trajectory still produce a sharp correlation peak, how does its energy
distribute across frequency (dwell time ~ dt/df), and how fast does a
transmitter/receiver trajectory mismatch turn into phase error.
"""

import numpy as np


def autocorrelation(x: np.ndarray) -> np.ndarray:
    """Normalized circular autocorrelation, centered so lag 0 is the middle sample."""
    n = len(x)
    spec = np.fft.fft(x)
    ac = np.fft.ifft(spec * np.conj(spec))
    ac = np.fft.fftshift(ac)
    return ac / np.max(np.abs(ac))


def cross_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Normalized circular cross-correlation R_xy, centered so lag 0 is the middle sample."""
    spec_x = np.fft.fft(x)
    spec_y = np.fft.fft(y)
    xc = np.fft.ifft(spec_x * np.conj(spec_y))
    xc = np.fft.fftshift(xc)
    scale = np.sqrt(np.sum(np.abs(x) ** 2) * np.sum(np.abs(y) ** 2))
    return xc / scale


def peak_to_sidelobe_ratio_db(ac: np.ndarray) -> float:
    """Main-lobe peak vs. highest sidelobe, in dB."""
    mag = np.abs(ac)
    center = np.argmax(mag)
    side = np.delete(mag, center)
    return 20 * np.log10(mag[center] / np.max(side))


def phase_error(f_tx: np.ndarray, f_rx: np.ndarray, fs: float) -> np.ndarray:
    """Accumulated phase mismatch: Delta_phi(t) = 2*pi * cumsum(f_tx - f_rx) / fs."""
    return 2 * np.pi * np.cumsum(f_tx - f_rx) / fs


def instantaneous_chirp_rate(f: np.ndarray, fs: float) -> np.ndarray:
    """df/dt via a numerical gradient of the sampled instantaneous frequency."""
    return np.gradient(f) * fs


def frequency_dwell_time(f: np.ndarray, bandwidth: float, n_bins: int = 64):
    """Histogram of samples per frequency bin: how long the trajectory lingers at each frequency."""
    hist, edges = np.histogram(f, bins=n_bins, range=(-bandwidth / 2, bandwidth / 2))
    return hist, edges
