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
  SNR / CFO / timing offset / transmitter-receiver trajectory mismatch /
  Doppler time-scale (`ser_vs_doppler_scale`) -- all through the same
  cyclic-shift-symbol + dechirp-by-conjugate-reference pipeline LoRa uses,
  just with `g` swapped out.
- `nlfsc_lora/doppler.py` -- the *other* Doppler model: wideband time-scaling,
  `r(t) = s(alpha*t)`, as opposed to `channel.apply_cfo`'s constant-shift
  approximation. This is the regime hyperbolic FM (HFM) is actually designed
  for; see the `04_ser_vs_cfo.png` vs `07_doppler_scale_tolerance.png`
  writeups below for why they give different answers about "curved chirps
  and Doppler."
- `nlfsc_lora/sync.py` -- CFO estimation/correction. `estimate_cfo` is a
  documented *negative* result (a blind, single-symbol band-edge estimate
  is swamped by the chirp's own spectral leakage); `joint_cfo_symbol_search`
  /`joint_cfo_symbol_demod` are what actually works -- see
  `09_lora_cfo_correction.png` below.

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
- **`08_lora_ser_vs_doppler_scale.png`** -- does `07`'s single-waveform
  advantage survive once HFM is actually plugged into LoRa's own decoding
  scheme (cyclic-shift symbols, dechirp by the conjugate reference,
  `matched_filter_bank_demod`)? `fft_demod` is not an option here at all --
  it can't decode a `hyperbolic` symbol correctly even at infinite SNR, for
  the same reason it fails for `sigmoid`/`quadratic`/etc.
  (`tests/test_receiver.py::test_fft_demod_fails_for_properly_embedded_hyperbolic_chirp`):
  the FFT-bin trick needs `f(t+tau)-f(t)` constant in `t`, which only a
  *linear* chirp satisfies.

  Only partially. LoRa's `M = 2**SF` symbols are separated by just
  `n_samples/M` samples (4, at SF7/4x oversampling), so a Doppler-induced
  correlation-peak *lag* -- which turns out to be almost identical across
  every trajectory (see `07`'s "peak lag" panel) -- exceeds half a symbol
  bin at a far smaller `alpha` than where HFM's magnitude-preservation
  would start to matter. Below that half-bin threshold every trajectory
  decodes perfectly; not far past it, every trajectory is fully saturated
  at 100% error. There is a real, reproducible (checked across 3 seeds x
  100 symbols/point) window in between where `hyperbolic` and `quadratic`
  hold on a little longer than `linear`/`sigmoid` before saturating -- but
  it is a narrow shift in *where* the SER waterfall's cliff sits, not the
  dramatic, essentially lossless tolerance `07` showed at the single-symbol
  level. Bottom line: HFM's Doppler-invariance is real (`07` proves it
  numerically), but LoRa's own M-ary shift-code granularity is the
  bottleneck once you build a full modem out of it -- getting HFM's full
  benefit back would need a receiver that jointly searches delay *and*
  Doppler scale, not just `M` fixed-lag symbol hypotheses.
- **`09_lora_cfo_correction.png`** -- a receiver-side fix for `04`'s CFO
  cliff, tried two ways. The first idea -- estimate CFO blindly from a
  single received symbol by finding the "top and bottom of the occupied
  bandwidth" -- has a real mathematical basis (cyclic-shifting a symbol
  can't change which frequencies are present, only when they occur, so
  `|FFT(symbol)|` is identical for all `M` symbols; see
  `test_chirp.py`/`sync.py`'s module docstring) but does not work in
  practice: a rectangular-windowed chirp's own truncation leaks ~1-2% of
  its energy outside the nominal band, asymmetrically enough to bias a
  band-edge or spectral-centroid estimate by several hundred Hz -- more
  than the CFO values (~100s of Hz) this project cares about, and it gets
  worse, not better, once noise is added (`nlfsc_lora/sync.py`'s
  `estimate_cfo`, `tests/test_sync.py` pins down exactly how biased it is).
  What does work: `joint_cfo_symbol_search` tries a grid of candidate CFO
  corrections, decodes against all `M` references at each, and keeps
  whichever (CFO, symbol) pair correlates strongest -- effectively a coarse
  frequency-axis ambiguity search using the receiver's own reference bank
  instead of blind spectral inspection. It is ~50x more expensive per
  symbol than `matched_filter_bank_demod`, but the plot shows why that can
  be worth it: it holds 0% SER across the entire CFO sweep tested (out to
  1500 Hz, 3x past where the CFO-blind receiver is already fully
  saturated), at the same SNR.

## Extending it

Adding a new trajectory is one function: `g(u)` on `[0, 1]` with `g(0)=0`,
`g(1)=1` (a closed-form `dg` is optional, used only for the chirp-rate
plot). Register it in `nlfsc_lora.trajectories.TRAJECTORIES` and it is
immediately usable everywhere else -- `ChirpConfig`, both demodulators, the
metrics, and the SER sweeps all treat `g` as an opaque callable.
