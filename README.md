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
- `nlfsc_lora/receiver.py` -- three decoders:
  - `fft_demod`: the standard LoRa trick (dechirp, then FFT the bin index
    off in one shot). This only works because a *linear* chirp's quadratic
    phase makes `df/dt` time-shift invariant, turning a cyclic shift into a
    pure frequency shift after dechirping. It degrades badly, even at
    infinite SNR, once `g` is nonlinear (`tests/test_receiver.py` checks
    this directly) -- it is trying to read the symbol off a *frequency*,
    which only means anything if dechirping produced a pure tone, and only
    a linear chirp guarantees that.
  - `matched_filter_bank_demod`: correlate against all `M` reference symbols
    directly and take the argmax. Works for any `g`, linear or not, at the
    cost of `O(N*M)` (`M` length-`N` dot products) -- the straightforward
    fallback once the cheap demodulator stops applying.
  - `fft_correlation_demod`: the *same decision* as `matched_filter_bank_demod`
    (tests check exact per-trial agreement, noiseless and under noise), computed
    a completely different way and roughly 85x faster at SF7. All `M`
    references are cyclic shifts of one base waveform, so correlating against
    all of them is exactly a circular cross-correlation between the received
    signal and the base waveform -- and the correlation theorem gets the
    *entire* cross-correlation (every possible lag) from one FFT pair
    (`correlation = IFFT(FFT(rx) * conj(FFT(base)))`), an identity true for
    any two signals, not something specific to chirps. This reads the symbol
    off a correlation *lag* (a time shift), never a frequency, so it never
    runs into the tone-purity problem `fft_demod` has -- it sidesteps that
    question entirely rather than solving it. Not a tradeoff like the other
    ideas explored below: same answer, `O(N log N)` instead of `O(N*M)`.
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
  `09_lora_cfo_correction.png` below. Its inner per-candidate decode uses
  `fft_correlation_demod`'s same correlation-theorem trick instead of a
  brute-force `O(N*M)` search, so the whole CFO grid search is cheaper by
  roughly the same factor at every candidate (~4x measured end to end,
  less than `fft_correlation_demod`'s standalone ~85x because of per-call
  FFT/Python overhead repeated at every grid point).
- `nlfsc_lora/local_rate.py` -- a much cheaper decoder possible only for a
  nonlinear `g`: a single local instantaneous-chirp-rate measurement
  identifies the symbol (CFO-independent, since a constant CFO doesn't
  change a rate of change) and, once the symbol is known, the CFO too --
  no correlation against any reference needed at all. Real speedup, real
  and serious limitations; see `10_local_rate_estimator.png` below.
- `nlfsc_lora/afc.py` -- puts the rate-of-change idea to a different use than
  `local_rate.py`: rather than decoding a symbol from it, measure the
  residual CFO at both ends of the swept bandwidth for a symbol *already*
  decoded by `fft_correlation_demod`, and track it across a sequence of
  bursts with a first-order AFC loop -- so the receiver keeps re-centering
  as a Doppler shift drifts, rather than correcting once. See
  `12_dual_edge_afc.png` below.

## Running it

```
pip install -e .[dev]
pytest                       # correctness checks (dechirp, demod, metrics)
python examples/run_experiments.py   # writes comparison plots to examples/output/
```

`examples/run_experiments.py` reproduces the comparison framework described
above for `linear`, `quadratic`, `sigmoid`, `sinusoidal`, and `exponential`
trajectories at SF7 / 125 kHz (a standard LoRa configuration):

- **`01_frequency_trajectories.png`** -- `f(t)` and `df/dt` per shape,
  including `hyperbolic` (plotted as a deviation from its own center
  frequency, since HFM isn't well-posed centered at 0 Hz). Linear has
  constant chirp rate by construction; the others don't.
- **`02_autocorrelation.png`** -- sidelobe structure differs sharply by
  shape; `sigmoid`'s slow edges cost it several dB of peak-to-sidelobe ratio
  versus `linear`.
- **`16_hyperbolic_symbol33_waveform.png`** -- what one actual symbol's
  cyclic shift looks like: the hyperbolic trajectory shifted to `m=33`,
  same style as `01_frequency_trajectories.png` but for a single shape and
  symbol instead of every shape's `m=0` base. The wrap point where the
  trajectory's end folds back to its start is marked directly.
- **`15_dechirp_linear_vs_hyperbolic.png`** -- the same symbol (33),
  dechirped, linear vs. hyperbolic. Linear's dechirped instantaneous
  frequency is piecewise-constant (two flat levels, one full bandwidth `B`
  apart, stepping at the cyclic-shift wrap point) so its FFT is one clean
  spike at bin 33 and `fft_demod` reads the symbol straight off it.
  Hyperbolic's dechirped frequency never settles into a constant at all --
  it keeps curving even after the wrap, because hyperbolic's own chirp
  *rate* increases over time (see `01_frequency_trajectories.png`) -- so its
  FFT smears and `fft_demod` confidently decodes the wrong symbol (17) on a
  noiseless signal. The FFT panels' gray gridlines mark the `M=128` valid
  symbol positions among the `N=512` total bins (`SF=7` symbols, 4x
  oversampled).
- **`13_fft_correlation_demo.png`** -- `fft_correlation_demod` laid open on
  one received symbol (hyperbolic, symbol 33, 10 dB SNR): `|S[k]|` and
  `|R[k]|`, the two FFT magnitudes; what conjugating `R[k]` actually changes
  (`Im{.}` flips sign, `Re{.}` doesn't -- magnitude is unchanged, so a
  magnitude plot of the conjugate would just repeat the previous panel);
  then `|IDFT{S[k]*conj(R[k])}[l]|`, the correlation recovered from their
  product -- a single sharp peak at the true shift, versus the noise floor
  and the two broad, individually uninformative spectra it was built from.
  The peak's x-position is a *lag* (132 samples), not the symbol index
  directly -- the annotation on the plot spells out the conversion
  (`132 / (N/M) = 132/4 = symbol 33`), matching `chirp.py`'s
  `τ_m = round(m*N/M)`.
- **`14_ser_vs_snr_all_shapes.png`** -- every trajectory shape's SER-vs-SNR
  waterfall on one plot, all decoded with `fft_correlation_demod` so the
  comparison is fair. The curves sit almost on top of each other --
  `sigmoid` trails slightly -- confirming that trajectory shape by itself
  buys essentially nothing for plain-noise tolerance once decoding is no
  longer the bottleneck; whatever nonlinear shape is good for, it isn't this.
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
  still differ is confusability with the *neighboring* symbol hypothesis.
  Now run at 10000 symbols/point (`fft_correlation_demod`'s ~85x speedup
  makes that cheap -- the original version of this plot used 450), the
  gap is unambiguous: `sigmoid`'s 50%-SER threshold is ~555 Hz versus
  ~504-510 Hz for `linear`/`quadratic`/`hyperbolic` (half a symbol bin is
  488 Hz), spending most of the symbol at low chirp rate near the two
  edges, where a one-symbol cyclic shift changes frequency the least. It
  is a real, reproducible, but still modest edge (~10%, not a large one)
  -- curving the trajectory is not, by itself, a reliable way to buy CFO
  tolerance, and notably `hyperbolic` (HFM) is *not* the standout here --
  see `08` below for the regime where it is.
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
  `fft_correlation_demod`)? `fft_demod` is not an option here at all -- it
  can't decode a `hyperbolic` symbol correctly even at infinite SNR, for
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
  at 100% error. Now run at 10000 symbols/point (up from the original
  300), the window in between is a real, precisely-measured effect: 50%-SER
  lands at `alpha`=1.00464 for `hyperbolic` and 1.00448 for `quadratic`,
  versus 1.00393 for `linear` and 1.00375 for `sigmoid` -- HFM and quadratic
  measurably outlast linear/sigmoid, but it is a ~20% shift in *where* the
  SER waterfall's cliff sits, not the dramatic, essentially lossless
  tolerance `07` showed at the single-symbol level. Notably, `sigmoid` --
  the standout trajectory for narrowband CFO tolerance in `04` -- is tied
  for *worst* here: the two Doppler regimes reward opposite trajectory
  shapes, which is why this project treats them as genuinely different
  questions rather than one "Doppler tolerance" number. Bottom line: HFM's
  Doppler-invariance is real (`07` proves it numerically) and does carry a
  measurable, reproducible edge into the full modem, but LoRa's own M-ary
  shift-code granularity caps how much of it survives -- getting HFM's full
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
- **`10_local_rate_estimator.png`** -- the opposite kind of receiver-side
  idea: instead of *more* correlation to buy CFO tolerance, use *less* of
  it to save power. A linear chirp's instantaneous rate `df/dt` is a
  constant everywhere, so measuring it locally tells you nothing about
  where in the symbol you are (which is exactly why the FFT-bin shortcut
  needs the whole record, and why `04`/`09` needed a full correlation or a
  grid search). A nonlinear trajectory's local rate varies with position in
  a *known* way, and a constant CFO shifts frequency without touching the
  rate of change at all -- so a single small window's local rate identifies
  the symbol independent of CFO, and once the symbol (hence the expected
  CFO-free frequency) is known, the CFO estimate falls straight out.
  `nlfsc_lora/local_rate.py` implements this via a local cubic phase fit
  (exact for `quadratic`'s polynomial law; other trajectories with a
  transcendental phase law like `hyperbolic` need a smaller window to keep
  the same fit accurate -- see the two `tests/test_local_rate.py` cases).

  It really is far cheaper -- roughly 10-90x fewer operations per symbol
  than `matched_filter_bank_demod` in testing here -- but the plot shows
  why it isn't simply a faster drop-in replacement. There is no equivalent
  of a full-record correlation's despreading gain, only whatever a small
  window averages over, so it needs tens of dB more SNR to work at all.
  Worse, it never actually reaches a usable error rate: it plateaus at
  roughly 30-40% SER even at 40 dB SNR (effectively noiseless), because a
  real fraction of the symbol alphabet has its cyclic-shift wrap glitch
  fall inside the fixed-position measurement window, which corrupts the
  local phase fit outright rather than just adding noise
  (`tests/test_local_rate.py::test_default_window_placement_fails_for_a_real_band_of_symbols`
  measures this directly: more than 1/6 of all `M` symbols fail this way,
  regardless of noise). So: a real, meaningful compute saving, but not
  currently an "effective" standalone decoder -- it would need a fix for
  the glitch-straddling failure (e.g. a small number of candidate windows
  at different positions with a majority vote) to be usable as one, which
  this project doesn't implement or test.
- **`11_hfm_vs_quadratic_ser.png`** -- does trajectory shape actually change
  SNR performance, with the receiver that's optimal for all of them?
  `fft_correlation_demod`'s ~85x speedup (see `receiver.py` above) makes it
  cheap enough to run 15000 symbols/point instead of `03`'s 150, so this is
  a far more statistically powered answer than the early SNR sweep gave.
  Result: no -- `linear`, `quadratic`, and `hyperbolic` sit on top of each
  other across the entire waterfall (-32 to -10 dB), all embedded on the
  same absolute frequency band for a fair comparison. This matches the
  theory from earlier in the project: constant envelope makes the
  correlation-magnitude physics the same for any trajectory, and with the
  receiver that's actually optimal (full correlation, not the broken
  FFT-bin shortcut), shape differences that showed up elsewhere in this
  project (CFO tolerance, autocorrelation sidelobes, Doppler-scale
  tolerance) don't translate into an SNR-performance difference at all.
  Trajectory curvature is not a free lunch, but it is not a tax on raw
  noise tolerance either -- the tradeoffs it does carry (receiver
  complexity, CFO/Doppler behavior) are the real story, not SNR.
- **`12_dual_edge_afc.png`** -- a working answer to "can the receiver check
  rate-of-change at each end of the bandwidth and use that to lock back onto
  center frequency": yes, with two design choices that turned out to matter.
  First, use it to *track* CFO across a sequence of bursts, not decode a
  symbol from it -- `local_rate.py` tried the latter and paid for it with a
  hard noise floor and a wrap-glitch failure band; here the symbol comes
  from the already-robust `fft_correlation_demod`, and the two edges'
  quality-weighted measurement only has to refine a residual CFO estimate
  once the symbol is known. Second, measuring at *both* edges rather than
  one center window means the fixed-position wrap glitch (which depends on
  the symbol, so it can land anywhere) can corrupt at most one edge at a
  time; a rate-mismatch check (measured rate vs. what the decoded symbol
  predicts at that position) downweights a corrupted edge automatically --
  confirmed directly for a symbol whose glitch sits inside the start-edge
  window (`tests/test_afc.py::test_dual_edge_estimate_exact_given_correct_symbol_noiseless[100--200.0]`).

  The plot's scenario is a drifting Doppler (0-3000 Hz over 200 bursts at
  SF7, where the static half-bin tolerance is only ~488 Hz) -- the realistic
  case for anything where Doppler is large enough to need correcting at all,
  such as a LoRa-over-LEO-satellite pass. Averaged over 40 seeds: the
  dual-edge AFC loop stays at 85-100% decode accuracy through the entire
  drift, while a receiver that only corrects once (acquires, then never
  updates) collapses to 0% by burst ~40, once the drift moves past wherever
  it was originally acquired.

  One real, discovered-not-assumed limitation, documented rather than tuned
  away: a cold start against an offset beyond `fft_correlation_demod`'s own
  half-bin capture range fails outright -- the first decode is wrong, which
  feeds a garbage measurement back into the loop and it diverges rather than
  converging (`nlfsc_lora/afc.py`'s module docstring has the numbers). The
  fix is the standard one from AFC/PLL design: a one-time coarse acquisition
  (`sync.py`'s `joint_cfo_symbol_search`, already built for `09`) gets the
  loop within range, then the cheap per-burst loop takes over. A second,
  related limitation worth knowing before relying on this: a plain
  proportional loop has a textbook steady-state lag tracking a *ramp*
  (`~drift_rate/gain`), not just noise jitter -- push the drift rate too far
  relative to the loop gain and bin size and the lag alone can exceed the
  tolerance (`tests/test_afc.py`'s docstring works the numbers for one such
  case). `17_kalman_vs_expfilter_ramp.png` below follows up on that lag with
  a second-order tracker; `18_uav_flyover_afc.png` and
  `19_afc_rate_of_change_limit.png` test this same loop against a Doppler
  that reverses sign rather than just a monotonic ramp.
- **`17_kalman_vs_expfilter_ramp.png`** -- does a smarter tracker actually
  reduce `AFCLoop`'s steady-state ramp lag? `KalmanAFCLoop` tracks CFO *and*
  its rate jointly (a constant-velocity Kalman filter) instead of chasing
  the latest measurement by a fixed fraction, which is the textbook fix for
  a type-1 lag. On the same drift as `12_dual_edge_afc.png`: a real but
  modest improvement, not dramatic -- the fixed-gain loop was already
  reasonably matched to a smooth, near-constant drift. Tuning this mattered,
  and mattered twice: an initial, untested guess for the measurement-noise
  parameter was off by ~85x from the actual measured noise, and a later
  switch in *how* the per-burst measurement itself works (see below)
  changed that true noise value by another ~16x depending on configuration
  -- both times, using the untuned/stale value made the filter *worse* than
  the simple loop, a reminder that a Kalman filter is only as good as its
  noise parameters, and those are worth measuring for the actual
  configuration in use, not assumed to carry over.
- **`18_uav_flyover_afc.png`** and **`19_afc_rate_of_change_limit.png`** --
  a satellite pass drifts the CFO in roughly one direction; a low-altitude
  UAV or drone approaching, passing closest approach, then receding produces
  a Doppler that reverses sign within the same run. Modeled with the
  standard constant-velocity closest-point-of-approach curve
  (`f_d(t) = -f_max * t / sqrt(t^2 + t0^2)`, an S-curve through zero),
  the same per-burst measurement this project has used throughout does track
  the full reversal (`18`), with a real caveat found only by testing much
  longer sequences than the original 200-burst demo: the original
  measurement's (`dual_edge_cfo_estimate`) quality check occasionally
  (~1/400, measured) passes a value that's wildly wrong anyway, and without
  a second check, one such event permanently derailed the loop for the rest
  of an 8000-burst run -- not a Doppler-rate problem, a missing outlier
  gate. Both trackers now reject implausible measurements before trusting
  them (`AFCLoop.max_jump_hz`, `KalmanAFCLoop.innovation_gate`), which fixes
  it. `19` sweeps the reversal's steepness to find the *real* rate-of-change
  limit rather than assume one: `AFCLoop` holds ~1.0 accuracy up to
  ~55 Hz/burst and collapses above ~100; `KalmanAFCLoop` clears a
  meaningfully higher ~75 Hz/burst with a softer falloff beyond that -- a
  gap that wasn't visible until the measurement upgrade below made it
  possible to see (a fixed-gain filter's rate ceiling is set mainly by its
  gain; a filter tracking rate explicitly has more headroom once its
  measurements are precise enough to use it). Even `AFCLoop`'s lower
  ceiling is ~54 kHz/s of Doppler acceleration at this configuration's
  symbol duration, several orders of magnitude past any physically
  realistic satellite or UAV scenario.
- **The negative-SNR measurement floor, and the fix.** Everything above was
  tested at 10dB SNR -- deliberately isolating tracking dynamics from
  measurement noise, but that sidesteps LoRa's whole premise of decoding
  under the noise floor. Pushed to negative SNR (see the packet test
  below), `dual_edge_cfo_estimate`'s 81-sample local phase fit turned out to
  be the actual bottleneck: it never gets the ~27dB of coherent processing
  gain `fft_correlation_demod` itself gets from correlating the *whole*
  ~512-sample symbol, so even a held-still, zero-residual signal (nothing to
  track) returns nothing usable on ~85-90% of bursts at -10dB SNR (measured
  directly) -- which is why decoding survives to -20 to -30dB while this
  measurement didn't. `nlfsc_lora/afc.py::full_symbol_cfo_estimate` fixes
  this by dechirping the *whole* symbol against its known decoded shape and
  reading the residual CFO off an FFT peak instead -- the same coherent
  gain the decoder already gets. Measured: ~100Hz std with zero outliers at
  -10dB SNR. `run_afc_sequence` uses this by default now;
  `dual_edge_cfo_estimate` stays in the module for its own documented
  properties (above) but no longer feeds the tracking loop. This also
  explains why `18`/`19`'s outlier-gating fix and this fix aren't the same
  thing: outlier-gating protects against a *rare* bad reading among mostly
  good ones (the 1/400 case above, at 10dB); this fixes the opposite
  situation, where almost nothing coming out of the measurement is
  trustworthy in the first place -- a gate can't manufacture good
  information out of bad, it can only correctly reject the bad. Switching
  estimators also exposed a second, config-dependent bug:
  `KalmanAFCLoop.measurement_noise` had been tuned against the old
  estimator at one bandwidth (125kHz); the new estimator's noise scales
  with the *square* of the FFT bin width, so at the packet test's 500kHz
  (4x the bin width) the true noise is ~16x higher than the stale tuned
  value, and using it unmodified made the Kalman filter badly overtrust
  measurements. `nlfsc_lora/afc.py::measurement_noise_for(cfg)` scales the
  default to the actual configuration instead of a fixed constant.

## End-to-end packet test

Everything above is checked per symbol or per burst. `examples/packet_experiments.py`
combines it into an actual packet: a simplified preamble (`N_PREAMBLE`
copies of the base `m=0` symbol -- the same idea as LoRa's own preamble
up-chirps, not a bit-accurate sync-word/SFD reproduction) followed by a
random payload, decoded and graded only on the payload. Three tests at
SF=7, BW=500 kHz -- a bandwidth this project hadn't used elsewhere (125
kHz everywhere above) -- each with a noiseless correctness check first,
then an SER-vs-SNR sweep:

- **`20_packet_level_validation.png`**, left panel, no CFO/Doppler:
  reproduces `14`'s waterfall shape in a full packet context, confirming
  the preamble/payload framing itself doesn't change anything.
- Middle panel, a constant 6 kHz CFO (past this configuration's ~1953 Hz
  half-bin tolerance) acquired from the preamble and held through the
  payload. Building this surfaced two real, fixed issues: (1) acquisition
  itself needs more SNR margin than plain decoding -- searching many CFO
  candidates against one noisy burst gives more chances for a false peak
  (measured: 27% single-burst acquisition error at -15dB where plain
  decoding had 0%), fixed by `run_afc_sequence(..., acquire_bursts=N)`,
  which majority-votes acquisition across `N` preamble bursts instead of
  trusting one (27% -> 0%, measured); and (2) `KalmanAFCLoop` needs its
  covariance reset on acquisition, not just its point estimate -- handing
  the acquired CFO to the tracker via direct assignment left the Kalman
  filter's covariance at its near-infinite pre-acquisition default, so the
  first post-acquisition measurement got absorbed with a near-total gain,
  able to drag the estimate off a good value. `KalmanAFCLoop.set_acquired`
  fixes this; with both fixes, `AFCLoop` and `KalmanAFCLoop` track
  identically here.
- Right panel, a UAV-style sign-reversing Doppler (same
  closest-point-of-approach curve as `18`/`19`). This test originally held
  up at 10dB but fell apart below roughly 5-8dB SNR -- a flat ~55% error
  rate no matter how much further SNR dropped, unacceptable against this
  project's own premise of decoding under the noise floor. Tracing it down
  (not just re-describing the symptom): the tracked estimate was *frozen*
  at its acquired value, not drifting to a wrong one, and the cause was
  `dual_edge_cfo_estimate` itself, unrelated to the Doppler reversal --
  fed a held-still, zero-residual signal at -10dB SNR it still rejected
  ~85-90% of bursts, because its local window fit never gets
  `fft_correlation_demod`'s ~27dB of coherent gain from the whole symbol.
  `max_jump_hz`/`innovation_gate` correctly rejected the rare measurement
  that did sneak through (itself garbage), but a gate can't manufacture
  good information out of bad, and at this SNR there was almost nothing
  else on offer. Test 1 doesn't need tracking and Test 2's CFO is constant
  (so coasting on a stale value is harmless), which is why only Test 3
  exposed this floor.

  With the fix above (`full_symbol_cfo_estimate` + `measurement_noise_for`),
  this test now resolves to a genuine waterfall instead of a flat floor:
  `AFCLoop` decodes this scenario correctly down to roughly -12dB SNR, and
  `KalmanAFCLoop` -- once its measurement noise is scaled to this 500kHz
  configuration -- tracks closely behind. Below roughly -14dB both
  trackers degrade, correctly: `full_symbol_cfo_estimate` has its own
  FFT-threshold effect at very low SNR, the expected limit of any
  coherent-integration estimator, just roughly twenty decibels lower than
  where the original measurement gave out.

Run with `python examples/packet_experiments.py`.

## How this compares to the real world

Two honest comparisons, written up in full in `NARRATIVE.md`/`PAPER.md`
rather than duplicated here:

- **Standard LoRa.** Where nonlinear trajectories help (structural,
  ongoing per-burst CFO tracking a linear chirp can't do at all) and
  where they don't (baseline SNR, narrowband CFO) -- see NARRATIVE.md's
  "Is This Actually Better Than Standard LoRa?" / PAPER.md §8.
- **Real HFM sonar and radar.** This project's dual-edge/full-symbol
  tracking is *not* how real active sonar or radar systems handle
  Doppler with HFM waveforms, and it's worth being explicit about why:
  those systems get Doppler-tolerant detection for free from a single
  fixed matched filter (Kroszczyński 1969), then resolve the resulting
  time/range bias (Murray et al. 2019) with a pair of oppositely-swept
  pulses compared against each other (Wang et al. 2017) -- structurally
  the same trick as standard LoRa's own up-chirp/down-chirp CFO
  resolution, not the same trick as this project's per-burst measurement.
  This project's mechanism works only because a LoRa symbol carries a
  decodable payload a sonar ping doesn't -- see NARRATIVE.md's "How Real
  HFM Sonar and Radar Systems Actually Handle Doppler" / PAPER.md §9.
- **Building the real thing.** The paired opposite-sweep idea above was
  then actually implemented (`nlfsc_lora/paired_sweep.py`) and tested
  against this project's own hyperbolic trajectory --
  `21_paired_sweep_doppler_correction.png` -- including a first attempt
  that didn't work (applying it directly to an arbitrary payload symbol's
  cyclic shift, which has a real, characterized shift-dependent bias) and
  the fix that did (applying it to a known preamble and correcting
  payload bursts afterward, the way real systems actually use it). See
  NARRATIVE.md's "Building the Same Thing: Paired-Sweep Doppler
  Correction" / PAPER.md §10.

## Extending it

Adding a new trajectory is one function: `g(u)` on `[0, 1]` with `g(0)=0`,
`g(1)=1` (a closed-form `dg` is optional, used only for the chirp-rate
plot). Register it in `nlfsc_lora.trajectories.TRAJECTORIES` and it is
immediately usable everywhere else -- `ChirpConfig`, both demodulators, the
metrics, and the SER sweeps all treat `g` as an opaque callable.
