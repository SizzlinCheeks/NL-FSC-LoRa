# Dissertation Outline: Nonlinear Frequency-Shift Chirp Spread Spectrum for Doppler-Tolerant LoRa

Working outline mapping the `nlfsc_lora` codebase, its tests, and its experiment
outputs onto a dissertation structure. Sections marked **[VALIDATED]** report
results already produced and verified in this repository (code + tests +
`examples/output/*.png`). Sections marked **[PROPOSED]** describe ideas that
are well-motivated by the validated results but have not been implemented or
tested here, and should be written up as such — as a stated hypothesis, not a
result — unless and until they are built.

---

## Chapter 1 — Introduction

- **Motivation.** LPWAN/IoT context; why Doppler tolerance is worth studying
  for a chirp-spread-spectrum system. Terrestrial LoRa Doppler is negligible
  (derivable from `v/c` at a sub-GHz carrier — worth stating explicitly and
  showing the calculation rather than asserting it). The stronger, more
  defensible motivating scenario is **LoRa-over-LEO-satellite** (e.g. Swarm,
  Lacuna, and the academic literature on Doppler compensation for
  satellite-IoT uplinks), where satellite velocities (~7.5 km/s) produce CFO
  values large enough to matter. Lead with this rather than generic
  "mobile LoRa."
- **Problem statement.** Standard LoRa's linear chirp and FFT-bin
  demodulation is cheap, but the receiver's Doppler/CFO tolerance is capped
  by symbol-bin granularity (`B/M/2`). Investigate whether reshaping the
  chirp's instantaneous-frequency trajectory — and exploiting the resulting
  rate-of-change structure at the receiver — can improve this.
- **Research questions:**
  - **RQ1** — Does a nonlinear trajectory preserve reliable symbol detection
    under AWGN, relative to standard (linear) LoRa? **[VALIDATED]**
  - **RQ2** — Does trajectory shape affect narrowband (constant-frequency-shift)
    Doppler/CFO tolerance? **[VALIDATED]**
  - **RQ3** — Does trajectory shape affect wideband (time-scale) Doppler
    tolerance, and does any single-waveform advantage survive full M-ary
    decoding? **[VALIDATED]**
  - **RQ4** — Can a receiver exploit local rate-of-change measurements at the
    two edges of the swept bandwidth, accumulated across multiple bursts, to
    self-correct CFO cheaply? **[VALIDATED]** — see `nlfsc_lora/afc.py` and
    Chapter 5 below.

## Chapter 2 — Background & Related Work

- LoRa/CSS fundamentals. Primary citation: Vangelista, "Frequency Shift Chirp
  Modulation: The LoRa Modulation" (2017).
- Chirp and ambiguity-function theory: Woodward's radar ambiguity function;
  linear-period/hyperbolic FM (HFM) — Kroszczynski (1969) is the classical
  reference; bio-sonar/bat-echolocation literature (Simmons, Altes) for the
  waveform's natural-world motivation.
- LoRa synchronization and AFC literature; LoRa-over-LEO Doppler-compensation
  papers specifically — the closest prior art to RQ4. A thorough literature
  search here is important: a committee will expect you to know whether an
  edge-rate/burst-tracking AFC scheme already exists in the satellite-IoT
  literature before claiming novelty.

## Chapter 3 — Methodology

Largely already implemented; this chapter documents the codebase.

- **Generalized chirp model** (`nlfsc_lora/chirp.py`): `f(t) = f0 + B*g(t/T)`,
  phase-accumulator generation (`phi[n] = phi[n-1] + 2*pi*f[n]/Fs`). Standard
  LoRa is the special case `g(u) = u`. LoRa-style M-ary symbols are cyclic
  time-shifts of one base waveform, wrap glitch included.
- **Trajectory library** (`nlfsc_lora/trajectories.py`): linear, quadratic/
  cubic, sigmoid, exponential, sinusoidal-perturbed, piecewise, and
  hyperbolic (HFM, `hyperbolic_center_freq` for its required nonzero-`f_center`
  embedding).
- **Receiver architectures and why they diverge for nonlinear `g`**
  (`nlfsc_lora/receiver.py`):
  - `fft_demod` — the standard LoRa trick; requires `f(t+tau) - f(t)`
    constant in `t`, true only for linear `f`. Proof and empirical
    confirmation: `tests/test_receiver.py::test_fft_demod_fails_for_properly_embedded_hyperbolic_chirp`.
  - `matched_filter_bank_demod` — general but `O(N*M)`.
  - `fft_correlation_demod` — general *and* cheap, via the correlation
    theorem (`correlation = IFFT(FFT(rx) * conj(FFT(base)))`); proven exactly
    equivalent to the brute-force decoder
    (`tests/test_receiver.py::test_fft_correlation_demod_matches_matched_filter_bank_exactly`),
    ~85x faster at SF7. Worth a subsection as a standalone technical
    contribution — it's non-obvious and was verified empirically before being
    trusted.
- **Channel models** (`nlfsc_lora/channel.py`, `nlfsc_lora/doppler.py`): AWGN,
  narrowband CFO (`apply_cfo`), integer timing offset, and wideband Doppler
  time-scaling (`apply_doppler_scale`) — explicitly two different physical
  models, not two views of the same impairment; justify the distinction
  (narrowband: `r(t) = s(t)*exp(j*2*pi*cfo*t)`; wideband: `r(t) = s(alpha*t)`).
- **Validation methodology.** State explicitly: exact per-trial decoder
  agreement tests (not just similar accuracy); multi-seed statistical
  averaging, including the documented case where a single-seed measurement
  was caught overstating an effect and corrected by re-running with more
  seeds/symbols. This is worth reporting as evidence of methodological rigor,
  not omitted as a false start.

## Chapter 4 — Results

One section per finding, in this narrative order:

1. **Baseline SNR performance is trajectory-independent** under the correct
   receiver. `examples/output/11_hfm_vs_quadratic_ser.png` (15,000
   symbols/point; linear, quadratic, hyperbolic overlap across the entire
   waterfall).
2. **Narrowband CFO tolerance across trajectories.**
   `examples/output/04_ser_vs_cfo.png` (10,000 symbols/point). Sigmoid shows
   a real but modest ~10% wider 50%-SER threshold (~555 Hz vs. ~504-510 Hz
   for linear/quadratic/hyperbolic; half-bin = 488 Hz at SF7/125 kHz).
   Constant envelope makes correlation-magnitude loss against the *correct*
   symbol provably trajectory-independent; the differences that do appear
   are from neighbor-symbol confusability.
3. **Wideband (time-scale) Doppler tolerance, single waveform.**
   `examples/output/07_doppler_scale_tolerance.png`. HFM holds its
   correlation peak within ~1 dB across a +-10% time-scale sweep vs. >10 dB
   loss for linear/sigmoid; derive why from `1/f(t)` being linear in `t` for
   HFM (self-similarity of a time-scaled copy under the correlation with an
   unscaled reference).
4. **Does that survive full M-ary decoding?**
   `examples/output/08_lora_ser_vs_doppler_scale.png` (10,000 symbols/point).
   Partially: HFM and quadratic measurably outlast linear/sigmoid
   (50%-SER `alpha` = 1.00464/1.00448 vs. 1.00393/1.00375), but the effect is
   far smaller than the single-waveform result, bottlenecked by LoRa's own
   `B/M` symbol-bin granularity.
5. **Receiver-side CFO correction.** `examples/output/09_lora_cfo_correction.png`.
   Report the band-edge blind CFO estimate
   (`nlfsc_lora/sync.py::estimate_cfo`) as a rigorous **negative result**:
   mathematically motivated (cyclic shift can't change `|FFT(symbol)|`,
   shift theorem) but empirically biased by hundreds of Hz from the chirp's
   own spectral leakage — bigger than the CFO being measured
   (`tests/test_sync.py` pins the bias down exactly). Then the working
   alternative, `joint_cfo_symbol_search`: 0% SER out to 1500 Hz CFO (3x past
   where the CFO-blind receiver saturates), at ~50x the per-symbol cost of
   `matched_filter_bank_demod` (reduced to ~4x once it was rebuilt on
   `fft_correlation_demod`'s cheaper primitive).
6. **Local rate-of-change decoding.** `examples/output/10_local_rate_estimator.png`
   (`nlfsc_lora/local_rate.py`). The closest existing work to RQ4: a
   nonlinear trajectory's local instantaneous rate varies with position in a
   known way (a linear chirp's doesn't), and a constant CFO doesn't change a
   rate of change — so one local measurement can identify the symbol
   independent of CFO, then the CFO falls out. Real speedup (10-90x fewer
   operations than full correlation). Two real, measured limitations to
   report honestly: (a) no despreading gain, so it needs tens of dB more SNR
   to work at all; (b) a structural SER floor (~30-40% even at 40 dB SNR)
   from the cyclic-shift wrap glitch falling inside the fixed measurement
   window for a real fraction (>1/6, measured directly) of the symbol
   alphabet regardless of noise. This sets up Chapter 5 precisely.

## Chapter 5 — Dual-Edge, Multi-Burst AFC **[VALIDATED]**

Implemented in `nlfsc_lora/afc.py`, tested in `tests/test_afc.py` (9 cases),
demonstrated in `examples/output/12_dual_edge_afc.png`. Addresses the two
failure modes measured in Chapter 4.6 by design, not by accident:

- **Noise sensitivity → multi-burst accumulation, not symbol decoding.** The
  key design decision: use rate/frequency sensing only to *refine a residual
  CFO estimate for a symbol the receiver already decoded* (via
  `fft_correlation_demod`, the already-validated, noise-robust decoder from
  Chapter 3), never to guess the symbol itself the way `local_rate.py` does.
  This sidesteps `local_rate.py`'s hard noise floor entirely — the edge
  measurement only has to be locally accurate, not decode-grade. The
  per-burst estimate is then accumulated with a first-order exponential loop
  filter (`AFCLoop`, the same structure as a classical AFC/PLL loop:
  `cfo_tracked += gain * (measurement - cfo_tracked)`), so the correction
  follows a *drifting* Doppler rather than correcting once.
- **Wrap-glitch collision → dual-edge placement with quality weighting.**
  Measuring at both ends of the swept bandwidth means the glitch (which sits
  at one fixed sample position per symbol) can corrupt at most one edge at a
  time. Each edge's measured rate is checked against the rate the decoded
  symbol predicts at that position; a large mismatch (from a corrupted
  window, or an occasional wrong decode) downweights that edge automatically
  via `dual_edge_cfo_estimate`'s quality weighting, rather than needing
  hard-coded knowledge of which symbols are risky. Confirmed directly, not
  just argued: `tests/test_afc.py::test_dual_edge_estimate_exact_given_correct_symbol_noiseless[100--200.0]`
  is a case where the start-edge window does straddle that symbol's glitch
  (mismatch ≈1.06) and the combined estimate is still accurate to within 2 Hz
  because the end edge is clean and dominates the weighted average.

**Result** (`12_dual_edge_afc.png`, 40-seed average): tracking a 0-3000 Hz
Doppler drift over 200 bursts at SF7 (static half-bin tolerance ≈488 Hz), the
dual-edge AFC loop holds 85-100% decode accuracy throughout, while a receiver
that acquires once and never updates collapses to 0% once the drift moves
past its original lock point (~burst 40). This is the chapter's central
figure and should be presented as such — it is the clearest demonstration in
the whole project of a receiver "correcting itself after a set of bursts to
get right on the center frequency again," in the terms the idea was
originally proposed in.

**Two real limitations, discovered rather than assumed, and worth their own
subsection:**
- *Cold-start pull-in range.* A first decode against an offset beyond
  `fft_correlation_demod`'s own half-bin capture range is wrong, which feeds
  a garbage measurement back into the loop and it diverges (verified: a
  naive cold start against 1000 Hz at SF7 ran to tens of kHz within 50
  bursts). The fix used here is the standard AFC/PLL fix — a one-time coarse
  acquisition (`sync.py::joint_cfo_symbol_search`, already built for Chapter
  4.5) before the cheap loop takes over.
- *Steady-state ramp lag.* A plain proportional (first-order) loop has a
  textbook steady-state lag tracking a ramp, `≈ drift_rate / gain` (a
  control-theory type-1 system's velocity error) — not just noise jitter.
  Push the drift rate too far relative to the gain and bin size and this
  systematic lag alone can exceed the tolerance
  (`tests/test_afc.py::test_tracking_survives_a_drift_that_exceeds_the_static_capture_range`'s
  docstring works a specific case that fails this way at a faster drift
  rate). A proportional-integral ("type-2") loop filter would remove this
  lag entirely and is the natural next refinement — **not implemented here**,
  a legitimate item for this dissertation's own future-work section, or a
  follow-up chapter if time permits building and validating it.

## Chapter 6 — Discussion

- Direct synthesis: trajectory curvature alone gives a real but modest CFO
  tolerance edge (Ch. 4.2); the largest practical Doppler-tolerance gain
  found in this work came from receiver architecture, in two forms — the
  joint CFO+symbol search (Ch. 4.5, one-shot, expensive, unbounded range)
  and the dual-edge AFC loop (Ch. 5, cheap, tracks drift, needs the joint
  search once to acquire). Neither strictly requires a nonlinear
  trajectory to exist as a technique, but nonlinearity is what makes
  rate-based sensing (Ch. 4.6, Ch. 5) possible at all, since a linear
  chirp's rate carries no positional information. Candidate thesis
  statement: nonlinear trajectories do not win on raw Doppler tolerance
  alone, they open a distinct *class* of receiver technique (rate-based
  sensing, and the cheap tracking loop it enables) unavailable to standard
  linear LoRa.
- Ch. 5 in particular is the strongest direct answer to the dissertation's
  original motivating scenario (LoRa-over-LEO, Ch. 1): a continuously
  drifting Doppler is exactly the case a one-shot correction cannot handle
  and a tracking loop can, and `12_dual_edge_afc.png` demonstrates that
  gap directly rather than by argument.
- Limitations: SF7-only parameter space; simulation-only (no RF/hardware
  validation); no legacy-LoRa-receiver interoperability story for a
  nonlinear-trajectory PHY; the AFC loop is first-order only (Ch. 5's
  steady-state ramp lag is a known, quantified, unaddressed limitation).

## Chapter 7 — Conclusion & Future Work

- Summarize RQ1-RQ4 findings against their validation status (all four now
  **[VALIDATED]**).
- Future work: a proportional-integral ("type-2") AFC loop to remove Ch. 5's
  steady-state ramp lag; extend beyond SF7; hardware/SDR validation;
  interoperability strategy for a nonlinear-trajectory PHY.

---

## Appendix: repository map

| Topic | Code | Tests | Output |
|---|---|---|---|
| Chirp generation, symbol construction | `nlfsc_lora/chirp.py` | `tests/test_chirp.py` | `01_frequency_trajectories.png` |
| Trajectory library | `nlfsc_lora/trajectories.py` | `tests/test_trajectories.py` | — |
| Autocorrelation / metrics | `nlfsc_lora/metrics.py` | `tests/test_metrics.py` | `02_autocorrelation.png` |
| Decoders (FFT-bin, MF-bank, FFT-correlation) | `nlfsc_lora/receiver.py` | `tests/test_receiver.py` | `03_ser_vs_snr.png`, `11_hfm_vs_quadratic_ser.png` |
| Narrowband CFO channel | `nlfsc_lora/channel.py` | `tests/test_channel.py` | `04_ser_vs_cfo.png` |
| Timing offset | `nlfsc_lora/channel.py` | `tests/test_channel.py` | `05_ser_vs_timing_offset.png` |
| Trajectory mismatch | `nlfsc_lora/simulate.py` | `tests/test_simulate.py` | `06_model_mismatch.png` |
| Wideband Doppler (single waveform) | `nlfsc_lora/doppler.py` | `tests/test_doppler.py` | `07_doppler_scale_tolerance.png` |
| Wideband Doppler (full M-ary) | `nlfsc_lora/simulate.py` | `tests/test_simulate.py` | `08_lora_ser_vs_doppler_scale.png` |
| CFO estimation/correction | `nlfsc_lora/sync.py` | `tests/test_sync.py` | `09_lora_cfo_correction.png` |
| Local rate-of-change decoding | `nlfsc_lora/local_rate.py` | `tests/test_local_rate.py` | `10_local_rate_estimator.png` |
| SNR-independence of trajectory shape | `nlfsc_lora/receiver.py` | `tests/test_receiver.py` | `11_hfm_vs_quadratic_ser.png` |
| Dual-edge AFC (Ch. 5) | `nlfsc_lora/afc.py` | `tests/test_afc.py` | `12_dual_edge_afc.png` |
| Experiment driver | `examples/run_experiments.py` | — | all of `examples/output/*.png` |

Regenerate all figures with `python examples/run_experiments.py`
(`pip install -e .[dev]` first); run `pytest` for the full test suite.
