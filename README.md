# NL-FSC-LoRa

Nonlinear frequency-shift chirp spread spectrum, modeled on LoRa's chirp
spread spectrum (CSS).

Standard LoRa spreads each symbol over a chirp whose instantaneous frequency
sweeps the channel bandwidth linearly:

```
f(t)  = f0 + k*t
phi(t) = 2*pi * integral(f(t))
s(t)   = exp(j*phi(t))
```

This project replaces the linear sweep with an arbitrary, deliberately
nonlinear trajectory

```
f(t) = f0 + B * g(t/T),      g(0) = 0, g(1) = 1
```

while keeping the receiver's reference built from the *same* g, so
dechirping (`s(t) * conj(s_ref(t))`) still cancels the swept phase. The
question this code exists to answer isn't "does the spectrogram look
different" -- it's: which of LoRa's useful properties survive when the
trajectory stops being a straight line, and which don't.

## Layout

- `nlfsc_lora/trajectories.py` -- normalized shapes `g(u)`: `linear`,
  `quadratic`/`cubic` (`f0 + k*t**p`), `sigmoid` (slow-fast-slow),
  `exponential` (monotonically accelerating), `hyperbolic` (linear-period
  FM / HFM, the bio-sonar chirp law -- see `doppler.py` below), a
  `sinusoidal`-perturbed linear sweep, and a three-segment `piecewise` shape.
- `nlfsc_lora/chirp.py` -- generates chirps via a phase accumulator
  (`phi[n] = phi[n-1] + 2*pi*f[n]/Fs`) rather than a closed-form phase
  formula, so any `g` "just works" numerically. `symbol_waveform(cfg, m)` is
  the base chirp cyclically time-shifted by `m*T/M`, exactly like LoRa's
  symbol construction -- wrap glitch and all.
- `nlfsc_lora/receiver.py` -- two decoders:
  - `fft_demod`: the standard LoRa trick (dechirp, then FFT the bin index
    off in one shot). This only works because a *linear* chirp's quadratic
    phase makes `df/dt` time-shift invariant, turning a cyclic shift into a
    pure frequency shift after dechirping. It degrades badly, even at
    infinite SNR, once `g` is nonlinear (`tests/test_receiver.py` checks
    this directly).
  - `matched_filter_bank_demod`: correlate against all `M` reference symbols
    and take the argmax. Works for any `g`, linear or not, at the cost of
    `O(M)` correlations instead of one FFT -- the general fallback once the
    cheap demodulator stops applying.
- `nlfsc_lora/channel.py` -- AWGN, carrier frequency offset, integer timing
  offset.
- `nlfsc_lora/metrics.py` -- autocorrelation/cross-correlation,
  peak-to-sidelobe ratio, accumulated phase error between a transmit and
  receive trajectory (`Delta_phi(t) = 2*pi * cumsum(f_tx - f_rx) / Fs`), and
  instantaneous chirp rate / frequency dwell time.
- `nlfsc_lora/simulate.py` -- the actual experiment: hold bandwidth, symbol
  duration, sample rate and channel fixed, vary only `g`, and measure SER vs
  SNR / CFO / timing offset / transmitter-receiver trajectory mismatch.
- `nlfsc_lora/doppler.py` -- the *other* Doppler model: wideband time-scaling,
  `r(t) = s(alpha*t)`, as opposed to `channel.apply_cfo`'s constant-shift
  approximation. This is the regime hyperbolic FM (HFM) is actually designed
  for; see the `04_ser_vs_cfo.png` vs `07_doppler_scale_tolerance.png`
  writeups below for why they give different answers about "curved chirps
  and Doppler."

## Running it

```
pip install -e .[dev]
pytest                       # correctness checks (dechirp, demod, metrics)
python examples/run_experiments.py   # writes comparison plots to examples/output/
```

`examples/run_experiments.py` reproduces the comparison framework described
above for `linear`, `quadratic`, `sigmoid`, `sinusoidal`, and `exponential`
trajectories at SF7 / 125 kHz (a standard LoRa configuration):

- **`01_frequency_trajectories.png`** -- `f(t)` and `df/dt` per shape. Linear
  has constant chirp rate by construction; the others don't.
- **`02_autocorrelation.png`** -- sidelobe structure differs sharply by
  shape; `sigmoid`'s slow edges cost it several dB of peak-to-sidelobe ratio
  versus `linear`.
- **`03_ser_vs_snr.png`** -- the central result. `linear` decodes correctly
  with either demodulator, with the FFT decoder trailing the matched-filter
  bank by several dB (the cost of the cheap trick even when it applies). The
  nonlinear shapes track the matched-filter-bank waterfall almost exactly
  with `matched_filter_bank_demod`, but the plain FFT decoder never recovers
  them at any SNR in range -- nonlinear trajectories need the general
  receiver.
- **`04_ser_vs_cfo.png`** -- SER vs. an uncompensated carrier frequency
  offset (the same model that applies to Doppler over one symbol, since
  Doppler barely changes within a ~ms LoRa symbol). Constant envelope makes
  the *correlation-magnitude loss against the correct symbol* provably
  identical across every trajectory in this file -- it reduces to
  `integral(exp(j*2*pi*cfo*t)) dt`, which doesn't depend on `g`. What can
  still differ is confusability with the *neighboring* symbol hypothesis,
  averaged here over multiple seeds because a single noisy run overstates
  the gap: `quadratic`, `sinusoidal`, and `exponential` track `linear`'s
  threshold closely, while `sigmoid` alone shows a real, reproducible softer
  knee (spending most of the symbol at low chirp rate near the two edges,
  where a one-symbol cyclic shift changes frequency the least). It is a
  modest edge, not a large one -- curving the trajectory is not, by itself,
  a reliable way to buy Doppler tolerance.
- **`05_ser_vs_timing_offset.png`** -- tolerance to an integer-sample timing
  error, per shape.
- **`06_model_mismatch.png`** -- SER as the receiver's assumed exponent `p`
  drifts away from the transmitter's true `p=2`. The transition from
  error-free to unusable happens within about 1.5% of `p` -- a concrete
  demonstration of section 9's point that the receiver's trajectory has to
  match the transmitter's precisely, because the mismatch phase error
  `Delta_phi(t)` accumulates over the whole symbol.
- **`07_doppler_scale_tolerance.png`** -- the *wideband* Doppler test:
  instead of a constant frequency shift, the transmitted symbol is
  literally time-scaled (`r(t) = s(alpha*t)`, `alpha` in `[0.9, 1.1]`) and
  matched-filtered against the original, unscaled reference -- the
  ambiguity-function-style test that hyperbolic FM's Doppler-invariance
  claim is actually about. `linear`, `quadratic`, `sigmoid`, and
  `hyperbolic` are all embedded on the *same* absolute frequency band here
  (via `hyperbolic_center_freq`), which matters because scale-Doppler
  sensitivity depends on the absolute frequencies swept, not just the
  bandwidth, and because HFM's law (`1/f(t)` linear in `t`) is only
  well-posed away from 0 Hz. Result: all four trajectories shift their
  correlation peak in time by almost the same amount (left/right panel,
  "peak lag") -- that part is generic to any single-lobed chirp -- but
  `hyperbolic` holds its peak *magnitude* within about 1 dB across the
  whole +-10% sweep, while `linear` and `sigmoid` lose more than 10 dB at
  the edges and `quadratic` sits in between. This is the real, specific
  effect the design writeup was gesturing at ("absorbs target velocity
  scaling factors into a predictable time shift"): HFM's `1/f(t)`-linear
  construction makes a time-scaled copy of itself nearly coincide with a
  *time-shifted* copy of itself, so an unscaled reference still recognizes
  it; a linear or sigmoid chirp's time-scaled copy is a genuinely different
  waveform (different chirp rate), so the same reference partially
  decorrelates against it.

  This is a different question from `04_ser_vs_cfo.png`, and they have
  different answers on purpose. LoRa's actual RF Doppler is essentially all
  bulk carrier-frequency shift (target velocities are negligible next to
  the speed of light at a sub-GHz carrier), which is exactly the regime
  where `04` showed curvature buys little. The time-scaling regime this
  plot tests matters when the envelope's own time-bandwidth product is
  large enough, or `alpha` deviates from 1 enough (high-speed sonar,
  wideband radar, bat echolocation), for the *within-symbol* envelope
  distortion to be non-negligible -- not really a concern for LoRa's
  bandwidth/symbol-duration/carrier combination, but a real and separate
  effect that HFM specifically exploits, reproduced numerically here.

## Extending it

Adding a new trajectory is one function: `g(u)` on `[0, 1]` with `g(0)=0`,
`g(1)=1` (a closed-form `dg` is optional, used only for the chirp-rate
plot). Register it in `nlfsc_lora.trajectories.TRAJECTORIES` and it is
immediately usable everywhere else -- `ChirpConfig`, both demodulators, the
metrics, and the SER sweeps all treat `g` as an opaque callable.
