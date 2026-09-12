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
  `quadratic`/`cubic` (`f0 + k*t**p`), `sigmoid` (slow-fast-slow), a
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

## Running it

```
pip install -e .[dev]
pytest                       # correctness checks (dechirp, demod, metrics)
python examples/run_experiments.py   # writes comparison plots to examples/output/
```

`examples/run_experiments.py` reproduces the comparison framework described
above for `linear`, `quadratic`, `sigmoid`, and `sinusoidal` trajectories at
SF7 / 125 kHz (a standard LoRa configuration):

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
- **`04_ser_vs_cfo.png`** / **`05_ser_vs_timing_offset.png`** -- tolerance to
  an uncompensated carrier offset and to timing error, per shape.
- **`06_model_mismatch.png`** -- SER as the receiver's assumed exponent `p`
  drifts away from the transmitter's true `p=2`. The transition from
  error-free to unusable happens within about 1.5% of `p` -- a concrete
  demonstration of section 9's point that the receiver's trajectory has to
  match the transmitter's precisely, because the mismatch phase error
  `Delta_phi(t)` accumulates over the whole symbol.

## Extending it

Adding a new trajectory is one function: `g(u)` on `[0, 1]` with `g(0)=0`,
`g(1)=1` (a closed-form `dg` is optional, used only for the chirp-rate
plot). Register it in `nlfsc_lora.trajectories.TRAJECTORIES` and it is
immediately usable everywhere else -- `ChirpConfig`, both demodulators, the
metrics, and the SER sweeps all treat `g` as an opaque callable.
