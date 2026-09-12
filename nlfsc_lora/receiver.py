"""Demodulation: dechirp against a reference trajectory, then decide the symbol.

Two decoders are provided on purpose, because they diverge for nonlinear g:

- fft_demod: the standard LoRa trick. Dechirping a *linear* chirp turns a
  cyclic time-shift into a pure frequency shift (because a linear chirp's
  quadratic phase makes df/dt time-shift invariant), so a single FFT reads
  the symbol off as a bin index in O(M log M). This is the "simplicity of
  the standard demodulator" the writeup flags as something you can lose.

- matched_filter_bank_demod: correlate the dechirped signal against every
  one of the M reference symbols and take the argmax. This works for *any*
  trajectory, linear or not, but costs O(M) correlations of length N instead
  of one FFT -- the general fallback once df/dt is no longer constant.

reference_cfg lets the receiver's assumed trajectory differ from the
transmitter's (model mismatch / synchronization-error studies).
"""

import numpy as np

from .chirp import ChirpConfig, base_waveform, all_symbol_waveforms


def dechirp(rx: np.ndarray, reference_cfg: ChirpConfig) -> np.ndarray:
    ref = np.conj(base_waveform(reference_cfg))
    return rx * ref


def fft_demod(rx: np.ndarray, reference_cfg: ChirpConfig) -> int:
    d = dechirp(rx, reference_cfg)
    spec = np.fft.fft(d)
    return int(np.argmax(np.abs(spec))) % reference_cfg.M


def matched_filter_bank_demod(rx: np.ndarray, reference_cfg: ChirpConfig) -> int:
    refs = all_symbol_waveforms(reference_cfg)
    norms = np.linalg.norm(refs, axis=1)
    corr = np.abs(refs.conj() @ rx) / (norms * np.linalg.norm(rx) + 1e-15)
    return int(np.argmax(corr))
