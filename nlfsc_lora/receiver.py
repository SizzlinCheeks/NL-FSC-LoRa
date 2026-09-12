"""Demodulation: dechirp against a reference trajectory, then decide the symbol.

Three decoders, because they diverge for nonlinear g:

- fft_demod: the standard LoRa trick. Dechirping a *linear* chirp turns a
  cyclic time-shift into a pure frequency shift (because a linear chirp's
  quadratic phase makes df/dt time-shift invariant), so a single FFT reads
  the symbol off as a bin index in O(N log N). This is the "simplicity of
  the standard demodulator" the writeup flags as something you can lose --
  it tries to read the symbol off a *frequency*, which only means anything
  if dechirping produced a pure tone in the first place, and only a linear
  chirp guarantees that.

- matched_filter_bank_demod: correlate the received signal against every
  one of the M reference symbols directly (no dechirp needed) and take the
  argmax. Works for *any* trajectory, linear or not, but costs M separate
  length-N dot products -- O(N*M) -- the straightforward fallback once
  df/dt is no longer constant.

- fft_correlation_demod: the same decision as matched_filter_bank_demod,
  computed a completely different (and far cheaper) way. All M references
  are cyclic shifts of one base waveform, so "correlate against all M of
  them" is exactly a circular cross-correlation between rx and the base
  waveform, evaluated at M particular lags -- and the full cross-correlation
  at *every* lag is one FFT pair away via the correlation theorem
  (correlation = IFFT(FFT(rx) * conj(FFT(base)))), true for any two signals
  regardless of chirp structure. This reads the symbol off a *lag* (a time
  shift), never a frequency, so it never runs into the tone-purity problem
  fft_demod has -- it is provably identical to matched_filter_bank_demod
  (tests/test_receiver.py checks exact per-trial agreement, noiseless and
  under noise) at roughly O(N log N) instead of O(N*M): ~85x faster
  measured at SF7. Caches each reference_cfg's base-waveform FFT keyed by
  object identity, holding a *strong* reference to reference_cfg alongside
  the cached entry -- keeping the object alive for as long as it's cached,
  which is what makes identity-based caching safe here: id() reuse after
  garbage collection is the classic hazard with this kind of cache, and
  holding a strong reference is what rules it out (a first version of this
  cache didn't, and briefly aliased two different unrelated ChirpConfigs
  that happened to reuse the same id() across separate tests). This still
  assumes a given ChirpConfig instance's g/bandwidth/etc. aren't mutated
  in place after first use, which is true everywhere in this codebase.

reference_cfg lets the receiver's assumed trajectory differ from the
transmitter's (model mismatch / synchronization-error studies).
"""

import numpy as np

from .chirp import ChirpConfig, base_waveform, all_symbol_waveforms

_base_fft_cache: dict = {}


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


def fft_correlation_demod(rx: np.ndarray, reference_cfg: ChirpConfig) -> int:
    key = id(reference_cfg)
    cached = _base_fft_cache.get(key)
    if cached is None or cached[0] is not reference_cfg:
        n = reference_cfg.n_samples
        base_fft = np.fft.fft(base_waveform(reference_cfg))
        valid_lags = (np.arange(reference_cfg.M) * n // reference_cfg.M) % n
        cached = (reference_cfg, base_fft, valid_lags)  # strong ref keeps id() from being reused
        _base_fft_cache[key] = cached
    _cfg, base_fft, valid_lags = cached

    corr = np.fft.ifft(base_fft * np.conj(np.fft.fft(rx)))
    return int(np.argmax(np.abs(corr[valid_lags])))
