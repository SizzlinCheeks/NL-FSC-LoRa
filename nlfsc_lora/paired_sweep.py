"""Paired opposite-sweep Doppler-bias cancellation, the dual-HFM (DHFM) trick
real active sonar uses to separate true range from Doppler-induced bias
(Wang et al. 2017), adapted here to separate a LoRa-style cyclic-shift symbol
from a wideband Doppler-scale-induced lag bias.

Chapter 5.2's exact self-similarity theorem (doppler.py, NARRATIVE.md/PAPER.md
Sec5.2) proves that a wideband Doppler scale alpha turns an HFM waveform into a
copy of *itself*, shifted by a bias Delta(alpha) -- never distorted, only
shifted. That is a genuine advantage for detecting a Doppler-scaled echo at
all, but it is also exactly why full M-ary decoding doesn't get the advantage
for free: the bias Delta(alpha) adds directly onto the correlation-theorem
lag this project's decoder reads the symbol off (receiver.py::fft_correlation_demod),
so an unknown alpha corrupts the decoded symbol even though the matched-filter
peak itself stays sharp (see 08_lora_ser_vs_doppler_scale.png).

Real active sonar resolves the equivalent problem -- an HFM matched filter's
time-of-arrival is Doppler-biased (Murray et al. 2019) -- not by measuring
anything from within one pulse, but by transmitting a *pair* of oppositely-swept
pulses and comparing their two (oppositely-biased) delay estimates. This module
builds the same idea for this project's hyperbolic trajectory:

Definitions. Let g be the "up" hyperbolic law (TRAJECTORIES["hyperbolic"], period
1/f(t) linear in t, sweeping f_low -> f_high) with band ratio q = f_low/f_high.
The "down" sweep is g's own shape traversed backward in time, g_down(u) = g(1-u)
-- reversed_trajectory() below -- which preserves the period-linear-in-time
structure (confirmed directly: both directions show the same ~0.9+ matched-filter
peak preservation under a +/-10% Doppler scale, not just the "up" direction).

The exact relationship between the two directions' Doppler-induced lag bias,
derived from Chapter 5.2's own Delta(alpha) formula applied to the time-reversed
law and confirmed numerically to sub-sample precision (in the sign convention
raw_lag_estimate below actually uses -- ifft(base_fft * conj(fft(rx))), which
runs the opposite direction from a plain np.correlate(rx, ref); Chapter 5.2's
own Delta(alpha) = (1-alpha)*n/(alpha*(1-q)) is stated in that other, np.correlate
convention, not this module's):

    Delta_up(alpha)   = (alpha - 1) * n / (alpha * (1 - q))
    Delta_down(alpha) = -q * Delta_up(alpha)

not a naive equal-and-opposite pair (that only holds in the degenerate q=1
limit) -- the down sweep's bias is q times smaller and opposite in sign, a
direct consequence of q = f_low/f_high being the same ratio that sets how much
of the hyperbola's curvature falls on each half of the band. Two oppositely-
biased measurements of the same underlying (true shift, alpha) pair is exactly
the two-equations-two-unknowns structure DHFM sonar exploits; solving it here
gives a shift estimate immune to the Doppler bias (to first order, exact for
the idealized/noiseless case) and a genuine alpha estimate as a byproduct --
see combine_paired_lags().

This is deliberately scoped to the hyperbolic law specifically, matching real
DHFM's own HFM-only scope: the closed form above depends on the period-linear-
in-time structure Chapter 5.2 proved exactly. reversed_trajectory() works for
any g, but the weighted combination's exactness is not claimed for shapes
other than hyperbolic.

How this is actually used -- a real lesson, not just an implementation detail:
the first version of this module tried to apply the paired-sweep trick
directly to an arbitrary *payload* symbol's own cyclic shift (send symbol m as
both an up- and down-sweep burst, decode the pair together). That does not
work cleanly: symbol_waveform's cyclic shift introduces its own wrap
discontinuity into the burst, and channel.py::apply_doppler_scale (which
resamples the finite, already-wrapped sample array from a fixed window origin
at n=0, not from the underlying continuous phase law's own origin) picks up an
extra, shift-dependent bias term Chapter 5.2's Delta(alpha) doesn't account
for -- confirmed directly: decode error stays near zero close to m=0 and grows
to many samples' worth of error by mid-band, a systematic pattern, not noise.

Real active sonar and real LoRa's own up/down-chirp trick both sidestep this
the same way: apply the paired-sweep measurement to a *known, unshifted*
reference (a preamble, m=0 both directions) to estimate the channel parameter
(alpha here; CFO+STO for real LoRa's up/down chirp; range+velocity for DHFM
sonar) once, then use that estimate to *correct* ordinary payload bursts
before decoding them with the standard single-sweep decoder -- never trying to
decode an arbitrary payload shift from the paired measurement directly. That
is the pattern acquire_doppler_scale()/correct_doppler_scale() below implement,
and it is exact (the wrap-discontinuity problem above never arises for an
unshifted m=0 reference) and, measured directly, restores full decode accuracy
across a +/-15% Doppler scale sweep that an uncorrected receiver fails
completely outside alpha=1.0 (see 21_paired_sweep_doppler_correction.png).
paired_sweep_decode() (below) is kept for the direct-decode idea it started
from, with the limitation above documented on it rather than hidden.
"""

from typing import Callable, Tuple

import numpy as np

from .chirp import ChirpConfig, base_waveform
from .channel import apply_doppler_scale


def reversed_trajectory(g: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    """The same shape traversed backward in time: g_down(u) = g(1-u).

    Not the same as amplitude-flipping (1-g(u)), which for a nonlinear g is a
    *different* curve that does not preserve Chapter 5.2's self-similarity
    structure -- confirmed directly: for the hyperbolic law, 1-g(u) shows
    badly degraded (~0.2-0.5, vs ~0.9+ for g(1-u)) matched-filter peak
    preservation under the same Doppler scale that g(1-u) tolerates just as
    well as g itself. For a symmetric law like the plain linear chirp the two
    constructions coincide (1-u either way), which is why the distinction
    never shows up there.
    """
    def g_down(u):
        return g(1.0 - np.asarray(u, dtype=float))
    return g_down


def raw_lag_estimate(rx: np.ndarray, reference_cfg: ChirpConfig) -> float:
    """Sub-sample circular correlation-peak lag between rx and reference_cfg's
    base (m=0) waveform, via the same correlation-theorem FFT pair
    receiver.py::fft_correlation_demod uses, but returning the full-resolution
    peak position (parabolic-interpolated) instead of quantizing to one of the
    M per-symbol bins -- the raw quantity combine_paired_lags() needs, since a
    Doppler bias generally does not land exactly on a valid per-symbol lag.

    Returned in (-n/2, n/2] samples (the "which way did it wrap" branch used
    throughout this function and combine_paired_lags() -- see the module
    docstring's caveat about shifts near that branch cut).
    """
    n = reference_cfg.n_samples
    base_fft = np.fft.fft(base_waveform(reference_cfg))
    corr = np.fft.ifft(base_fft * np.conj(np.fft.fft(rx)))
    mag = np.abs(corr)

    k0 = int(np.argmax(mag))
    k_minus, k_plus = (k0 - 1) % n, (k0 + 1) % n
    y0, y_minus, y_plus = mag[k0], mag[k_minus], mag[k_plus]
    denom = y_minus - 2 * y0 + y_plus
    delta = 0.5 * (y_minus - y_plus) / denom if denom != 0 else 0.0
    lag = k0 + delta
    if lag > n / 2:
        lag -= n
    return lag


def combine_paired_lags(lag_up: float, lag_down: float, q: float, n_samples: int) -> Tuple[float, float]:
    """Solve the two-equations-two-unknowns system
    lag_up = shift + Delta_up, lag_down = shift - q*Delta_up
    for (shift_hat, Delta_up_hat), the DHFM-style decoupling this whole module
    is for. Algebra: q*lag_up + lag_down = shift*(1+q) exactly (the q*Delta_up
    terms cancel), giving the closed forms below directly -- not an
    approximation or a fit, the exact solution of that linear system.

    raw_lag_estimate wraps each of lag_up/lag_down independently into
    (-n/2, n/2], which is fine for either alone but breaks this combination
    whenever the true shift sits close to that branch cut: up and down can
    then land on *opposite* sides of it even though they're measuring the
    same underlying shift, corrupting the linear combination by close to a
    full n_samples (confirmed directly -- this was the dominant error source
    before this unwrap step existed, not the closed form itself, which is
    exact). Since |Delta_up| is always small relative to n_samples for any
    realistic Doppler scale, lag_down can only be a wrapped copy of a value
    near lag_up -- so re-wrapping it to whichever representative is closest
    to lag_up resolves the ambiguity robustly."""
    diff = lag_down - lag_up
    diff -= n_samples * round(diff / n_samples)
    lag_down = lag_up + diff

    shift_hat = (q * lag_up + lag_down) / (1.0 + q)
    delta_up_hat = (lag_up - lag_down) / (1.0 + q)
    return shift_hat, delta_up_hat


def doppler_scale_from_bias(delta_up: float, n_samples: int, q: float) -> float:
    """Invert this module's Delta_up(alpha) = (alpha-1)*n/(alpha*(1-q)) (module
    docstring; note this is the negative of Chapter 5.2's own Delta(alpha),
    stated in a different sign convention -- see the module docstring) for
    alpha, turning the bias estimate combine_paired_lags() produces into a
    genuine wideband Doppler-scale estimate -- the direct analogue of DHFM
    sonar's own velocity-from-bias inversion."""
    return n_samples / (n_samples - delta_up * (1.0 - q))


def paired_sweep_decode(
    rx_up: np.ndarray, rx_down: np.ndarray, cfg_up: ChirpConfig, cfg_down: ChirpConfig, q: float
) -> Tuple[int, float]:
    """Decode a symbol directly from a pair of bursts -- the same payload
    symbol sent as both an up-sweep (cfg_up) and a down-sweep (cfg_down, built
    with reversed_trajectory(cfg_up.g)) burst.

    Scope, found directly rather than assumed: exact for m at or near 0 (no
    cyclic-shift wrap discontinuity inside the burst), but carries a real,
    shift-dependent residual error for general m -- confirmed directly, error
    grows roughly linearly with the shift and flips sign past mid-band, from
    the same window-origin effect the module docstring's "how this is
    actually used" section describes. For decoding real payload symbols under
    an unknown Doppler scale, prefer acquire_doppler_scale() on a m=0
    reference pair followed by correct_doppler_scale() + an ordinary
    single-sweep decode -- that pattern has no such limitation. This function
    is kept for the direct-decode idea it started from and for cases (m
    already known to be near 0, e.g. a symbol used only for acquisition) where
    its exactness already holds.

    Returns (m_hat, alpha_hat): the decoded symbol and, as a byproduct of the
    same combination, an estimate of the wideband Doppler scale itself."""
    n = cfg_up.n_samples
    lag_up = raw_lag_estimate(rx_up, cfg_up)
    lag_down = raw_lag_estimate(rx_down, cfg_down)
    shift_hat, delta_up_hat = combine_paired_lags(lag_up, lag_down, q, n)
    m_hat = int(round(shift_hat * cfg_up.M / n)) % cfg_up.M
    alpha_hat = doppler_scale_from_bias(delta_up_hat, n, q)
    return m_hat, alpha_hat


def acquire_doppler_scale(
    rx_preamble_up: np.ndarray, rx_preamble_down: np.ndarray, cfg_up: ChirpConfig, cfg_down: ChirpConfig, q: float
) -> float:
    """Estimate the wideband Doppler scale alpha from a pair of *unshifted*
    (m=0) preamble bursts -- one up-sweep, one down-sweep -- the validated,
    general-purpose entry point this module is actually for (see the module
    docstring's "how this is actually used"). Exact for the idealized case and
    measured directly to ~1e-4 relative accuracy at 0dB SNR, ~5e-4 down to
    -10dB SNR (the same full-symbol coherent gain that fixed afc.py's own
    negative-SNR floor applies here too, for the same reason: raw_lag_estimate
    correlates the whole symbol, not a small window)."""
    n = cfg_up.n_samples
    lag_up = raw_lag_estimate(rx_preamble_up, cfg_up)
    lag_down = raw_lag_estimate(rx_preamble_down, cfg_down)
    _shift_hat, delta_up_hat = combine_paired_lags(lag_up, lag_down, q, n)
    return doppler_scale_from_bias(delta_up_hat, n, q)


def correct_doppler_scale(rx: np.ndarray, alpha_hat: float) -> np.ndarray:
    """Undo an estimated wideband Doppler scale by resampling with its inverse
    -- the wideband-Doppler analogue of sync.py::correct_cfo. Pair with
    acquire_doppler_scale(): acquire alpha_hat once from a preamble, then
    correct every subsequent payload burst before decoding it normally."""
    return apply_doppler_scale(rx, 1.0 / alpha_hat)
