"""Normalized frequency-trajectory shapes g(u).

Every shape maps the unit interval to itself with g(0) = 0 and g(1) = 1, so it
can be plugged into ChirpConfig as f(u) = f_center - B/2 + B * g(u) without
changing the total swept bandwidth B. Swapping g changes only how the
instantaneous frequency moves through that range as a function of time,
i.e. the shape of df/dt rather than the overall excursion.

Each shape is exposed as a plain callable g(u) plus, where the derivative has
a closed form, a matching dg(u) = dg/du for analysis (instantaneous chirp
rate, dwell-time plots). dg is optional: chirp.py integrates f(t) numerically
via a phase accumulator and never needs dg to *generate* a waveform.
"""

from functools import partial

import numpy as np


def linear(u):
    return u


def dlinear(u):
    return np.ones_like(u)


def power(u, p=2.0):
    """f(t) = f0 + k*t**p family. p=1 reduces to the standard linear chirp."""
    return np.asarray(u, dtype=float) ** p


def dpower(u, p=2.0):
    u = np.asarray(u, dtype=float)
    return p * np.power(u, p - 1.0, where=u > 0, out=np.zeros_like(u))


def sigmoid(u, a=10.0, b=0.5):
    """Slow-fast-slow trajectory: lingers near the band edges, sweeps quickly through the middle."""
    u = np.asarray(u, dtype=float)
    raw = 1.0 / (1.0 + np.exp(-a * (u - b)))
    lo = 1.0 / (1.0 + np.exp(-a * (0.0 - b)))
    hi = 1.0 / (1.0 + np.exp(-a * (1.0 - b)))
    return (raw - lo) / (hi - lo)


def dsigmoid(u, a=10.0, b=0.5):
    u = np.asarray(u, dtype=float)
    lo = 1.0 / (1.0 + np.exp(-a * (0.0 - b)))
    hi = 1.0 / (1.0 + np.exp(-a * (1.0 - b)))
    s = 1.0 / (1.0 + np.exp(-a * (u - b)))
    return (a * s * (1.0 - s)) / (hi - lo)


def sinusoidal_perturbed(u, alpha=0.1):
    """Linear sweep with a sinusoidal perturbation. Monotonic for alpha < 1/(2*pi)."""
    u = np.asarray(u, dtype=float)
    return u + alpha * np.sin(2 * np.pi * u)


def dsinusoidal_perturbed(u, alpha=0.1):
    u = np.asarray(u, dtype=float)
    return 1.0 + alpha * 2 * np.pi * np.cos(2 * np.pi * u)


def exponential(u, a=3.0):
    """Monotonically accelerating sweep: g(u) = (exp(a*u) - 1) / (exp(a) - 1).

    Unlike quadratic/cubic (power(p>1)), curvature is controlled by a single
    rate constant `a` rather than a polynomial degree; a -> 0 approaches the
    linear trajectory, larger a concentrates more of the bandwidth sweep into
    the final fraction of the symbol.
    """
    u = np.asarray(u, dtype=float)
    return (np.exp(a * u) - 1.0) / (np.exp(a) - 1.0)


def dexponential(u, a=3.0):
    u = np.asarray(u, dtype=float)
    return a * np.exp(a * u) / (np.exp(a) - 1.0)


def hyperbolic(u, q=1.0 / 3.0):
    """Hyperbolic / linear-period FM (HFM), the bio-sonar / bat-call chirp law.

    Defined by making the instantaneous *period* 1/f(t) linear in t (instead
    of f(t) itself, as in a linear chirp). q = f_low/f_high sets how many
    octaves the sweep spans (q=1/3 spans log2(3)=1.58 octaves) and must be
    strictly between 0 and 1 -- unlike every other shape here, HFM is only
    defined for a band that does not cross zero Hz, since 1/f(t) is singular
    at f=0. Use hyperbolic_center_freq() to place ChirpConfig.f_center so the
    swept band [f_center-B/2, f_center+B/2] actually has this ratio q.
    """
    u = np.asarray(u, dtype=float)
    d = 1.0 - u * (1.0 - q)
    return (1.0 / d - 1.0) / (1.0 / q - 1.0)


def dhyperbolic(u, q=1.0 / 3.0):
    u = np.asarray(u, dtype=float)
    d = 1.0 - u * (1.0 - q)
    return q / d**2


def hyperbolic_center_freq(bandwidth, q=1.0 / 3.0):
    """f_center placing a bandwidth-B sweep so f_low/f_high = q (both positive)."""
    return bandwidth / 2.0 * (1.0 + q) / (1.0 - q)


def piecewise_slow_fast_slow(u, u1=0.2, u2=0.8, edge_frac=0.2):
    """Explicit three-segment trajectory: slow / linear / slow, each segment linear in u."""
    u = np.asarray(u, dtype=float)
    g = np.empty_like(u)

    slow1 = edge_frac * (u / u1)
    mid_rate = (1.0 - 2.0 * edge_frac) / (u2 - u1)
    slow2 = 1.0 - edge_frac + edge_frac * (u - u2) / (1.0 - u2)

    m0 = u <= u1
    m1 = (u > u1) & (u <= u2)
    m2 = u > u2

    g[m0] = slow1[m0]
    g[m1] = edge_frac + mid_rate * (u[m1] - u1)
    g[m2] = slow2[m2]
    return g


# Registry of ready-to-use (g, dg) pairs. dg is None where no closed form is used.
TRAJECTORIES = {
    "linear": (linear, dlinear),
    "quadratic": (partial(power, p=2.0), partial(dpower, p=2.0)),
    "cubic": (partial(power, p=3.0), partial(dpower, p=3.0)),
    "sigmoid": (sigmoid, dsigmoid),
    "sinusoidal": (sinusoidal_perturbed, dsinusoidal_perturbed),
    "exponential": (exponential, dexponential),
    "hyperbolic": (hyperbolic, dhyperbolic),
    "piecewise": (piecewise_slow_fast_slow, None),
}


def numerical_derivative(g, u, h=1e-6):
    """Central-difference dg/du, for shapes without a closed-form derivative."""
    u = np.asarray(u, dtype=float)
    return (g(np.clip(u + h, 0, 1)) - g(np.clip(u - h, 0, 1))) / (2 * h)
