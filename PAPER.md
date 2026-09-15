# Detection Theory for Nonlinear Chirp Spread Spectrum

A self-contained derivation of the mathematics behind this project: why LoRa's
standard demodulator requires a linear chirp, the correlation-theorem-based
decoder that works for any chirp shape at FFT-like cost, why hyperbolic FM
(HFM) is exactly self-similar under Doppler time-scaling, and the dual-edge
frequency-tracking loop. Each result links to the code that implements it and
the tests that verify it, so every equation here has a numerical check
sitting next to it in the repository.

> **Looking for the intuitive version?** [`NARRATIVE.md`](NARRATIVE.md) tells
> this same story in a more readable, example-driven format for readers who
> already know standard LoRa but want the "why" behind each step before the
> equations. This document is the dense reference version — precise
> derivations, proofs, and code/test citations for lookup.

## Notation

| Symbol | Meaning |
|---|---|
| $B$ | swept bandwidth (Hz) |
| $T$ | symbol duration (s) |
| $SF$ | spreading factor; $M = 2^{SF}$ is the symbol alphabet size |
| $N$ | samples per symbol, $N = T \cdot F_s$ |
| $F_s$ | sample rate (Hz) |
| $g(u)$ | normalized trajectory shape, $u \in [0,1]$, $g(0)=0$, $g(1)=1$ |
| $f(t)$ | instantaneous frequency |
| $\phi(t)$ | instantaneous phase, $\phi(t) = 2\pi\int_0^t f(\tau)d\tau$ |
| $s[n]$ | the base ($m=0$) reference waveform, $s[n] = e^{j\phi(n/F_s)}$ |
| $\tau_m$ | the cyclic sample shift encoding symbol $m$: $\tau_m = \mathrm{round}(mN/M)$ |
| $X[k]$ | the discrete Fourier transform (DFT) of $x[n]$ |

---

## 1. The chirp model

Every waveform in this project is generated from one formula:

$$
f(t) = f_0 + B \cdot g(t/T), \qquad \phi(t) = 2\pi\int_0^t f(\tau)d\tau, \qquad s(t) = e^{j\phi(t)}
$$

Standard LoRa is the special case $g(u) = u$ (a straight ramp — constant
chirp rate). This project's whole subject is what happens when $g$ is
something else: quadratic, sigmoid, hyperbolic, etc. Implementation:
`nlfsc_lora/chirp.py` (`base_frequency`, `base_waveform`); the numerical
phase integral is a discrete phase accumulator,
$\phi[n] = \phi[n-1] + 2\pi f[n]/F_s$.

Each of the $M = 2^{SF}$ symbols is a **cyclic time shift** of the same base
waveform:

$$
s_m[n] = s[(n + \tau_m) \bmod N], \qquad \tau_m = \mathrm{round}(mN/M)
$$

This is exactly how real LoRa encodes a symbol, and it is the fact everything
below hinges on: there are not $M$ independent waveforms, there is **one**
waveform and $M$ shifts of it.

---

## 2. Why the standard LoRa decoder needs a *linear* chirp

The standard trick — dechirp, then read one FFT bin — is: multiply the
received signal by the conjugate of the reference, and look at what's left.

$$
d(t) = s_m(t)\cdot \overline{s(t)} = e^{j\phi(t+\tau_m)}\cdot e^{-j\phi(t)} = e^{j[\phi(t+\tau_m)-\phi(t)]}
$$

This is a **pure tone** — a single spike in the FFT, the thing a
frequency-domain peak-finder can read a bin index off of — exactly when its
instantaneous frequency doesn't depend on $t$. That instantaneous frequency
is

$$
\frac{d}{dt}\Big[\phi(t+\tau_m)-\phi(t)\Big] = f(t+\tau_m) - f(t)
$$

so the question is: for which $f$ is this constant in $t$?

**Claim.** $f(t+\tau)-f(t)$ is constant in $t$ for every shift $\tau$ if and
only if $f$ is affine, $f(t) = f_0 + kt$.

*Proof.* ($\Leftarrow$) If $f(t) = f_0+kt$, then $f(t+\tau)-f(t) = k(t+\tau) - kt = k\tau$, independent of $t$. ($\Rightarrow$) If $f(t+\tau)-f(t) = c(\tau)$ for all $t$, differentiate both sides with respect to $t$: $f'(t+\tau) - f'(t) = 0$ for every $t,\tau$, so $f'$ is constant, so $f$ is affine. $\blacksquare$

So the FFT-bin shortcut is not a general property of chirps — it is a
one-off algebraic coincidence of the straight line. For any curved $g$, the
dechirped signal $d(t)$ is itself a (smaller) chirp, not a tone: its FFT is
smeared across many bins instead of concentrated in one, and there is no
single bin whose index reliably encodes $m$.

Implementation and verification: `nlfsc_lora/receiver.py::fft_demod`;
`tests/test_receiver.py::test_fft_demod_fails_for_properly_embedded_hyperbolic_chirp`
confirms this fails for HFM specifically, not just approximately. A direct
before/after picture — the dechirped instantaneous frequency staying flat
for a linear chirp but sweeping for HFM, and the resulting FFT going from
one sharp peak to a smeared double-hump — is in the project's chat history
and can be regenerated from `nlfsc_lora/receiver.py::dechirp`.

---

## 3. A decoder that works for any chirp: brute-force correlation

Since one FFT can't be trusted for a curved $g$, the direct fallback is to
correlate the received signal against **every** candidate symbol waveform
and keep the best match:

$$
\mathrm{score}_m = \sum_{n=0}^{N-1} \overline{s_m[n]} \cdot r[n] = \sum_{n=0}^{N-1} \overline{s[(n+\tau_m)\bmod N]} \cdot r[n], \qquad \hat m = \arg\max_m |\mathrm{score}_m|
$$

This is correct for any $g$ — implemented as
`nlfsc_lora/receiver.py::matched_filter_bank_demod` — but it is $M$ separate
length-$N$ sums: $O(NM)$ work per received symbol.

---

## 4. The fast decoder: the correlation theorem

### 4.1 One function standing in for all $M$ scores

Define the **circular cross-correlation** of the base waveform $s$ and the
received signal $r$, as a function of an arbitrary lag $l$ (not just the $M$
special lags $\tau_m$):

$$
C[l] = \sum_{n=0}^{N-1} s[n] \cdot \overline{r[(n-l)\bmod N]}
$$

**Claim.** $|\mathrm{score}_m| = |C[\tau_m]|$ for every $m$.

*Proof.* Substitute $k = (n+\tau_m)\bmod N$ (so $n = (k-\tau_m)\bmod N$) into $\mathrm{score}_m$:

$$
\mathrm{score}_m = \sum_{k=0}^{N-1} \overline{s[k]} \cdot r[(k-\tau_m)\bmod N] = \overline{C[\tau_m]}
$$

which has the same magnitude as $C[\tau_m]$. $\blacksquare$

So the $M$ separate brute-force scores are just $M$ particular samples of
**one** function $C[l]$. If $C$ can be computed cheaply for every lag at
once, decoding becomes a lookup instead of a search.

### 4.2 The correlation theorem

Let $S[k]$ and $R[k]$ be the DFTs of $s[n]$ and $r[n]$:

$$
S[k] = \sum_{n=0}^{N-1} s[n] \cdot e^{-j2\pi kn/N}, \qquad R[k] = \sum_{n=0}^{N-1} r[n] \cdot e^{-j2\pi kn/N}
$$

**Theorem.** $C[l] = \mathrm{IDFT}\big(S[k] \cdot \overline{R[k]}\big)[l]$.

*Proof.* Take the DFT of $C$ with respect to $l$ and substitute $m=(n-l)\bmod N$:

$$
\begin{aligned}
\mathrm{DFT}(C)[k] &= \sum_{l=0}^{N-1} \left(\sum_{n=0}^{N-1} s[n] \cdot \overline{r[(n-l)\bmod N]}\right) e^{-j2\pi kl/N} \\
&= \sum_{n=0}^{N-1} s[n] \sum_{m=0}^{N-1} \overline{r[m]} \cdot e^{-j2\pi k(n-m)/N} \\
&= \underbrace{\left(\sum_n s[n] \cdot e^{-j2\pi kn/N}\right)}_{S[k]} \cdot \overline{\left(\sum_m r[m] \cdot e^{-j2\pi km/N}\right)} \\
&= S[k] \cdot \overline{R[k]}
\end{aligned}
$$

Inverting the DFT gives the claim. $\blacksquare$

This is a general fact about discrete signals — the same identity underlies
fast convolution and cross-correlation everywhere in signal processing; it
has nothing to do with chirps specifically. What makes it applicable *here*
is section 4.1: the $M$ candidates being cyclic shifts of one waveform is
what lets a single correlation function stand in for all of them.

### 4.3 The decoder

$$
\boxed{\hat m = \arg\max_{m} \left| \mathrm{IDFT}\big(S[k] \cdot \overline{R[k]}\big)[\tau_m] \right|}
$$

Compute $S$ once per reference (cache it — it never changes), compute $R$
once per received symbol, multiply, inverse-transform, and read off the $M$
positions $\tau_m$. Implementation:
`nlfsc_lora/receiver.py::fft_correlation_demod`.

```python
corr = np.fft.ifft(base_fft * np.conj(np.fft.fft(rx)))   # C[l] for every l at once
m_hat = int(np.argmax(np.abs(corr[valid_lags])))          # look up the M known positions
```

**Correctness.** `tests/test_receiver.py::test_fft_correlation_demod_matches_matched_filter_bank_exactly`
checks bit-for-bit agreement with the brute-force decoder (section 3) across
every trajectory in the project, noiseless and under heavy noise — not
merely similar accuracy, the identical decision on every trial.

### 4.4 Complexity

| Decoder | Cost | Works for nonlinear $g$? |
|---|---|---|
| `fft_demod` (§2) | $O(N\log N)$ | No |
| `matched_filter_bank_demod` (§3) | $O(NM)$ | Yes |
| `fft_correlation_demod` (§4) | $O(N\log N)$ | Yes |

At $SF=7$ ($N=512$, $M=128$): $NM \approx 65{,}500$ versus
$\sim 3N\log_2 N \approx 13{,}800$ for the three transforms — matching the
measured $\sim 85\times$ speedup at that configuration.

---

## 5. Two Doppler models, and why they give different answers

### 5.1 Narrowband: a constant frequency shift

$$
r(t) = s(t) \cdot e^{j2\pi f_{\Delta} t}
$$

The correct-symbol correlation magnitude loss reduces to
$\left|\int_0^T e^{j2\pi f_\Delta t}dt\right|$ — independent of $g$, since
$|s(t)|=1$ always (constant envelope). What differs between trajectories is
confusability with the *neighboring* symbol hypothesis, which is a smaller,
shape-dependent effect (`examples/output/04_ser_vs_cfo.png`).

### 5.2 Wideband: time-scaling, and HFM's exact self-similarity

The physically exact Doppler model is time-scaling, not a frequency shift:

$$
r(t) = s(\alpha t), \qquad \alpha = 1 \pm v/c
$$

**Hyperbolic FM** (linear-period modulation) is the trajectory whose
instantaneous frequency has $1/f(t)$ linear in $t$:

$$
f(t) = \frac{f_{\text{low}}}{1-\beta t}, \qquad \phi(t) = -\frac{2\pi f_{\text{low}}}{\beta}\ln(1-\beta t)
$$

**Theorem (exact self-similarity under time-scaling).** For any $\alpha>0$,

$$
\phi(\alpha t) = \phi(t - \Delta) + c, \qquad \Delta = \frac{1-\alpha}{\alpha\beta}, \quad c = -\frac{2\pi f_{\text{low}}}{\beta}\ln\alpha
$$

*Proof.* The algebraic identity $1-\beta\alpha t = \alpha\big(1-\beta(t-\Delta)\big)$ holds exactly for $\Delta = (1-\alpha)/(\alpha\beta)$ (expand the right side: $\alpha - \alpha\beta t + \alpha\beta\Delta = \alpha - \alpha\beta t + (1-\alpha) = 1-\alpha\beta t$, which is the left side). Taking $-\frac{2\pi f_{\text{low}}}{\beta}\ln(\cdot)$ of both sides:

$$
\phi(\alpha t) = -\frac{2\pi f_{\text{low}}}{\beta}\Big[\ln\alpha + \ln\big(1-\beta(t-\Delta)\big)\Big] = -\frac{2\pi f_{\text{low}}}{\beta}\ln\alpha + \phi(t-\Delta) \qquad \blacksquare
$$

Exponentiating, $s(\alpha t) = e^{jc}\cdot s(t-\Delta)$ — a Doppler-scaled
copy of an HFM waveform is, **exactly**, a constant-phase-rotated,
time-*shifted* copy of the same waveform. A matched filter built for the
unscaled $s$ still recognizes it perfectly; the scaling only moves *where*
the peak appears, not how tall it is. A linear chirp has no such identity —
time-scaling a linear chirp changes its chirp rate outright, producing a
genuinely different waveform, which is exactly why its matched-filter
response degrades under scaling instead of merely shifting
(`examples/output/07_doppler_scale_tolerance.png` shows this numerically:
HFM loses ~1 dB of peak magnitude across a $\pm10\%$ scale sweep versus
$>$10 dB for a linear chirp).

This identity is exact only for the idealized, unbounded-domain continuous
waveform. A real burst has a finite window $[0,T)$, so the shifted copy's
support doesn't line up perfectly with the original observation window at
the edges — which is the real, measured, nonzero loss reported above, and
also why `channel.py::apply_doppler_scale` zero-pads past the end of a
finite burst rather than extrapolating (§5.2 of the code, not this paper).
It is also why this single-waveform advantage does not fully survive being
plugged into LoRa's own fine-grained, $M$-ary cyclic-shift decoding
(`examples/output/08_lora_ser_vs_doppler_scale.png`) — that decoder's
sensitivity to a Doppler-induced *lag* (generic to any trajectory) dominates
before HFM's magnitude-preservation advantage gets a chance to matter.

---

## 6. Dual-edge frequency tracking

### 6.1 Local frequency and rate from a phase fit

Over a short window of $2w{+}1$ samples centered at index $n_0$, fit the
*unwrapped* phase to a cubic in the local sample index $k$:

$$
\angle r[n_0+k] \approx a + bk + ck^2 + dk^3, \qquad k = -w,\dots,w
$$

Reading off the linear and quadratic coefficients gives the local
instantaneous frequency and chirp rate at $n_0$:

$$
\hat f(n_0) = \frac{b F_s}{2\pi}, \qquad \widehat{\dot f}(n_0) = \frac{2c F_s^2}{2\pi}
$$

A cubic is exact for a quadratic trajectory (whose phase law is exactly
cubic in $t$) and a good local approximation otherwise. Implementation:
`nlfsc_lora/local_rate.py::local_freq_and_rate`,
`nlfsc_lora/afc.py::_edge_measurement`.

### 6.2 Why the *rate* identifies position independent of CFO

A constant carrier offset adds a constant to $f(t)$ but leaves $\dot f(t)$
untouched. So, given the decoded symbol $m$ (hence the known, CFO-free
expected rate $\dot f_{\text{exp}}$ at position $n_0$), a residual-CFO
estimate falls out of comparing the *measured* frequency to what that
symbol predicts:

$$
\widehat{\mathrm{CFO}} = \hat f(n_0) - f_{\text{exp}}(n_0 \mid m)
$$

Measuring at **two** positions — near each end of the symbol — and
weighting each by how well its measured rate matches $\dot f_{\text{exp}}$
lets a clean measurement outvote one corrupted by the cyclic-shift phase
discontinuity, which can only ever occur at one fixed position per symbol:

$$
\widehat{\mathrm{CFO}} = \frac{w_A \cdot \widehat{\mathrm{CFO}}_A + w_B \cdot \widehat{\mathrm{CFO}}_B}{w_A+w_B}, \qquad w_i = \frac{1}{\left|\dot f_i - \dot f_{\text{exp},i}\right|/|\dot f_{\text{exp},i}| + \epsilon}
$$

Implementation and the specific verified case where one edge is corrupted
and the other compensates:
`nlfsc_lora/afc.py::dual_edge_cfo_estimate`,
`tests/test_afc.py::test_dual_edge_estimate_exact_given_correct_symbol_noiseless`.

### 6.3 The tracking loop

A first-order exponential filter (the same structure as a classical AFC/PLL
loop) accumulates the per-burst estimate into a slowly-updated correction:

$$
\widehat{\mathrm{CFO}}_{\text{tracked}}[i+1] = \widehat{\mathrm{CFO}}_{\text{tracked}}[i] + \gamma\Big(\widehat{\mathrm{CFO}}[i] - \widehat{\mathrm{CFO}}_{\text{tracked}}[i]\Big)
$$

For a linearly drifting true CFO (rate $\rho$ per burst), this loop has a
textbook steady-state lag of $\rho/\gamma$ — a control-theory type-1 velocity
error, not merely noise jitter; push $\rho$ too far relative to $\gamma$ and
the tolerance can be exceeded by the lag alone, independent of measurement
noise. Implementation: `nlfsc_lora/afc.py::AFCLoop`.

### 6.4 A joint CFO/rate tracker, an outlier-gating fix, and a sign-reversing Doppler

**A second-order tracker.** A constant-velocity Kalman filter carrying state
$x = [\widehat{\mathrm{CFO}}, \widehat{\dot{\mathrm{CFO}}}]^\top$ subsumes
§6.3's fixed-$\gamma$ loop: the predict step advances
$\widehat{\mathrm{CFO}}[i+1] = \widehat{\mathrm{CFO}}[i] + \widehat{\dot{\mathrm{CFO}}}[i]$
and holds $\widehat{\dot{\mathrm{CFO}}}[i+1] = \widehat{\dot{\mathrm{CFO}}}[i]$ each burst
(a constant-velocity model), and a scalar measurement of $\widehat{\mathrm{CFO}}$
alone updates both state components via the Kalman gain, computed from process noise $Q$ and
measurement noise $R$ generalizes both a fixed loop gain and an ad hoc
gain-schedule, computed from the noise statistics rather than hand-tuned.
Measured, not assumed: the per-burst measurement's own noise at SF7/125 kHz/
10 dB SNR is $\approx 27$ Hz std ($R \approx 720$ Hz$^2$; §6.5 revises this
value and explains why it is configuration-dependent, after the per-burst
measurement itself changes). Correctly tuned, it gives a real
but modest reduction in §6.3's steady-state lag under the same linear drift
(`examples/output/17_kalman_vs_expfilter_ramp.png`).
Implementation: `nlfsc_lora/afc.py::KalmanAFCLoop`.

**A discovered outlier-measurement failure mode.** Testing over sequences
much longer than §6.3's 200-burst demonstration surfaced a real bug rather
than a tuning issue: `dual_edge_cfo_estimate`'s mismatch-based quality gate
occasionally ($\approx 1/400$, measured over 2000 trials at a fixed
noiseless residual) passes a measurement 1–4.5 kHz from the truth with a
passing mismatch score — a noisy cubic fit's rate coefficient can look
self-consistent even when its frequency coefficient is badly wrong. Over a
few hundred bursts this rarely triggers; over a multi-thousand-burst run it
is close to certain to occur at least once, and with no second check the
tracker trusts it, the next decode falls outside `fft_correlation_demod`'s
capture range, and the corrupted decode poisons every subsequent
measurement — permanent divergence with no recovery mechanism, confirmed
directly on an 8000-burst run at a *constant* (easily trackable) offset.
Both trackers now gate the innovation before it updates state
(`AFCLoop.max_jump_hz`, a fixed threshold at $\approx 6\sigma$;
`KalmanAFCLoop.innovation_gate`, a $\chi^2$-style test against the filter's
own predicted variance $S$), which eliminates the divergence.

**A sign-reversing Doppler (UAV/drone case).** §5–6 model a monotonic drift
(satellite pass); a low-altitude UAV instead produces a Doppler that
reverses sign — approach, closest approach, recede. Modeled with the
standard constant-velocity closest-point-of-approach curve,

$$
f_d(t) = -f_{d,\max}\cdot\frac{t}{\sqrt{t^2+t_0^2}},
$$

an S-curve saturating to $\pm f_{d,\max}$ with $t_0$ (bursts) setting the
sign-crossing steepness. With the outlier gate in place, both trackers
follow the full reversal at high accuracy while a one-shot static
correction fails once the drift leaves its acquisition point
(`examples/output/18_uav_flyover_afc.png`). Sweeping $t_0$ finds the real
rate-of-change limit rather than assuming one: `AFCLoop` holds accuracy
near 1.0 up to a peak instantaneous rate of $\approx 55$ Hz/burst and
collapses above $\approx 100$ Hz/burst; `KalmanAFCLoop` clears a
meaningfully higher ceiling, near 1.0 up to $\approx 75$ Hz/burst with a
softer falloff beyond it rather than `AFCLoop`'s sharper wall
(`examples/output/19_afc_rate_of_change_limit.png`). The divergence
between the two trackers here is itself new information, not visible
under this project's original measurement/tuning (§6.5): a fixed-gain
filter's rate-tracking ceiling is set mainly by its gain, largely
independent of measurement precision beyond some point, while a filter
that tracks $\widehat{\dot{\mathrm{CFO}}}$ explicitly has more headroom
against a genuinely accelerating drift once its measurements are precise
enough to expose that headroom. At this configuration's symbol duration
($\approx 1.02$ ms), even `AFCLoop`'s lower ceiling of 55 Hz/burst is
$\approx 54$ kHz/s of Doppler *acceleration* — several orders of magnitude
past any physically realistic satellite or UAV scenario, so the cliff is
real but not the binding constraint for either case tested.

This mirrors an independent, established result in the sonar/radar
literature: HFM's Doppler-tolerant matched-filter response is well
documented (Kroszczyński 1969), but it comes with a companion time/range
bias under Doppler, given closed form in Murray et al. (2019) — the same
underlying tradeoff as this project's own Chapter 5 lag finding, observed
independently in a different field.

Implementation: `nlfsc_lora/afc.py::KalmanAFCLoop`, `AFCLoop.max_jump_hz`,
`KalmanAFCLoop.innovation_gate`. Tests:
`tests/test_afc.py::test_afc_loop_gates_a_wild_outlier_measurement`,
`test_kalman_loop_gates_a_wild_outlier_innovation`,
`test_afc_tracks_through_a_doppler_sign_reversal`,
`test_afc_fails_for_an_unrealistically_fast_doppler_reversal`.

### 6.5 The negative-SNR measurement floor, and full-symbol dechirping

Everything above was tested at 10 dB SNR, isolating tracking *dynamics*
from measurement noise. That sidesteps the question that matters most for
a protocol whose premise is decoding under the noise floor: does this
tracking mechanism hold up at negative SNR? It did not, at first, and the
reason is a genuine limit of `dual_edge_cfo_estimate`, not of tracking
itself. Its 81-sample local phase fit never benefits from the coherent
processing gain `fft_correlation_demod` gets by correlating the *whole*
$\approx$512-sample symbol ($\approx 10\log_{10}(512) \approx 27$ dB) —
measured directly, it returns no usable measurement (its own mismatch gate
rejecting both edges) on $\approx$85–90% of bursts at $-10$ dB SNR, even
fed a held-still, zero-residual signal with nothing to track at all. That
gap in processing gain, not a tuning shortfall, is why decoding itself
survives to $-20$ to $-30$ dB while this measurement did not.

The fix keeps §6.2's structure (measure a *residual* against an
already-known symbol, never a blind guess) but replaces the local fit:
dechirp the entire received symbol against its exact expected waveform —
which, for the correct $\hat m$, cancels the trajectory's own phase and
leaves a pure tone at the residual CFO in noise — and read that tone's
frequency off an FFT peak with parabolic sub-bin interpolation, over a
search window bounded to $\pm$one bin (the residual is always small
post-tracking) with a peak-prominence gate replacing the mismatch check.
This recovers the coherent gain the decoder already has: measured
directly, $\approx$100 Hz std with zero outliers among 1000 trials at
$-10$ dB SNR, degrading gracefully rather than catastrophically below
that (`nlfsc_lora/afc.py::full_symbol_cfo_estimate`). `run_afc_sequence`
uses it by default; `dual_edge_cfo_estimate` remains in the module for its
own documented properties but no longer feeds the tracking loop.

This resolves an apparent tension with §6.4's outlier-gating fix, which
was correctly described there as fixing its own problem: a *rare* bad
measurement ($\approx 1/400$) slipping through an otherwise-healthy
stream at 10 dB. Negative SNR is a different regime — not an occasional
bad reading among good ones, but $\approx$85–90% of readings simply
absent — and a gate that correctly rejects untrustworthy measurements
cannot synthesize trustworthy ones in their place. The two fixes address
two distinct failure modes at two different points in the pipeline.

A second, config-dependent bug followed from switching estimators:
`KalmanAFCLoop.measurement_noise` had been tuned once, against
`dual_edge_cfo_estimate` at SF7/125 kHz/10 dB. `full_symbol_cfo_estimate`'s
noise is an FFT-peak estimate, whose variance at fixed SNR scales with the
*square* of the FFT bin width ($\propto (\mathrm{BW}/M)^2$) — confirmed
empirically at two bandwidths 4$\times$ apart, yielding a $\approx 16\times$
variance ratio against a $16\times$ predicted one. §7's 500 kHz packets
inherited a $\approx 16\times$-too-small stale value, making the filter
badly under-trust good measurements exactly when trusting them mattered
most (37% vs `AFCLoop`'s 0% payload symbol error at $-10$ dB, otherwise
identical run). `nlfsc_lora/afc.py::measurement_noise_for(cfg)` replaces
the fixed constant with one scaled to the active configuration; §7 gives
the corrected results.

Implementation: `nlfsc_lora/afc.py::full_symbol_cfo_estimate`,
`measurement_noise_for`. Tests:
`tests/test_afc.py::test_full_symbol_cfo_estimate_accurate_at_negative_snr`,
`test_full_symbol_cfo_estimate_far_less_starved_than_dual_edge_at_negative_snr`,
`test_full_symbol_cfo_estimate_rejects_a_noise_only_peak`,
`test_afc_tracks_a_doppler_reversal_at_negative_snr`.

---

## 7. End-to-end validation: a packet test

Every result above was checked per symbol or per burst. A receiver has to
combine all of it at once: recognize a packet, decode the payload, and,
under Doppler, acquire and track through the whole thing. Three tests
(`examples/packet_experiments.py`), SF=7, BW=500 kHz — untested elsewhere
in this project, which otherwise uses 125 kHz — each sending a simplified
preamble (`N_PREAMBLE` copies of the base $m=0$ symbol, standing in for
LoRa's own preamble up-chirps, not a bit-accurate sync-word/SFD
reproduction) followed by a random payload, graded only on the payload.

**7.1 No Doppler.** Baseline: 200 packets, 6000 payload symbols, zero
errors at 40 dB SNR; the SER-vs-SNR sweep reproduces §4's waterfall shape.

**7.2 Constant CFO, acquired from the preamble.** A 6 kHz offset (past
this configuration's half-bin tolerance, $\approx 1953$ Hz) is acquired
via `sync.py::joint_cfo_symbol_search` on the preamble and held through
the payload by the tracking loop. Two findings, both from testing rather
than assumed:

- *Acquisition needs more SNR margin than decoding.* Searching many CFO
  candidates against one noisy burst gives more chances for a
  noise-induced false peak than a single $M$-ary decode does — measured,
  27% single-burst acquisition symbol error at $-15$ dB SNR versus 0% for
  plain decoding. `run_afc_sequence(..., acquire_bursts=N)` runs
  acquisition independently across the first $N$ preamble bursts and
  combines them (majority-vote the decoded symbol, median CFO among
  bursts agreeing with it) instead of trusting one; measured, $N=8$
  reduced that 27% to 0%.
- *`KalmanAFCLoop` needs its covariance reset on acquisition, not just its
  point estimate.* Handing acquisition's result to the tracker via direct
  assignment (`loop.cfo_tracked = ...`) is correct for `AFCLoop`, whose
  entire state is that one number, but leaves `KalmanAFCLoop`'s covariance
  at its pre-acquisition, near-infinite default — so even after a
  confident acquisition the filter still behaves as if uninformed, and the
  first post-acquisition measurement is absorbed with a near-total Kalman
  gain, able to drag the estimate off a good value. `KalmanAFCLoop.set_acquired`
  fixes this by resetting covariance alongside the point estimate; with it,
  `AFCLoop` and `KalmanAFCLoop` track identically on this test.

**7.3 A sign-reversing (UAV-style) Doppler.** Same closest-point-of-approach
curve as §6.4, $t_0=90$ (peak rate $\approx 33$ Hz/burst), comfortably under
§6.4's rate-of-change cliff. The original version of this test held up at
10 dB but produced a flat $\approx 55\%$ error rate at every point below
roughly 5–8 dB — not a waterfall, and unacceptable against this project's
own premise of decoding under the noise floor. Tracing it down (§6.5):
the tracked estimate, printed burst by burst, was not drifting toward a
wrong value, it was frozen at its acquired value while the true CFO moved
underneath it, and the cause was `dual_edge_cfo_estimate` itself —
independent of the reversal, returning `None` on $\approx$85–90% of
bursts even for a held-still, zero-residual signal at $-10$ dB SNR,
because its local fit never gets `fft_correlation_demod`'s $\approx$27 dB
of coherent gain from the whole symbol. `max_jump_hz`/`innovation_gate`
were correctly rejecting the rare measurement that did pass, itself
garbage — but a gate cannot synthesize a trustworthy measurement out of
an untrustworthy one, and at this SNR there was almost nothing else on
offer. §7.1 was unaffected because it needs no tracking; §7.2 because its
CFO is constant, so coasting on a stale value between rare good
measurements is harmless. §7.3 is the only one of the three whose
correctness depends on a continuous stream of trustworthy per-burst
measurements, so it was the only one to expose a floor that was present,
unnoticed, in every prior test.

With §6.5's fix (`full_symbol_cfo_estimate` plus `measurement_noise_for`
for `KalmanAFCLoop`), this test resolves to a genuine waterfall:
`AFCLoop` decodes correctly down to roughly $-12$ dB SNR, and
`KalmanAFCLoop`, once its measurement noise is scaled to this
configuration, follows closely behind. Below roughly $-14$ dB both
trackers degrade, correctly — `full_symbol_cfo_estimate` has its own
FFT-threshold effect at very low SNR (§6.5), the expected limit of any
coherent-integration estimator, now roughly twenty decibels lower than
where the original local-window measurement gave out. Sweeping the
transition (`examples/output/20_packet_level_validation.png`, right
panel) shows the corrected shape directly.

Implementation: `examples/packet_experiments.py`; `nlfsc_lora/afc.py::run_afc_sequence`
(`acquire_bursts`), `AFCLoop.set_acquired`, `KalmanAFCLoop.set_acquired`,
`full_symbol_cfo_estimate`, `measurement_noise_for`.
Tests: `tests/test_afc.py::test_multi_burst_acquisition_beats_single_burst_at_low_snr`,
`test_kalman_set_acquired_resets_covariance_not_just_cfo`,
`test_afc_tracks_a_doppler_reversal_at_negative_snr`.

---

## Summary

| Result | Section | Code | Test |
|---|---|---|---|
| FFT-bin demod requires linear $f$ | §2 | `receiver.py::fft_demod` | `test_receiver.py::test_fft_demod_fails_for_properly_embedded_hyperbolic_chirp` |
| Brute-force correlation decoder | §3 | `receiver.py::matched_filter_bank_demod` | `test_receiver.py::test_matched_filter_bank_exact_noiseless` |
| Fast correlation-theorem decoder | §4 | `receiver.py::fft_correlation_demod` | `test_receiver.py::test_fft_correlation_demod_matches_matched_filter_bank_exactly` |
| Narrowband Doppler is trajectory-blind at the correct symbol | §5.1 | `channel.py::apply_cfo` | `test_channel.py` |
| HFM's exact self-similarity under time-scaling | §5.2 | `doppler.py`, `trajectories.py::hyperbolic` | `test_doppler.py::test_hyperbolic_more_doppler_scale_tolerant_than_linear` |
| Dual-edge CFO estimation | §6.1-6.2 | `afc.py::dual_edge_cfo_estimate` | `test_afc.py::test_dual_edge_estimate_exact_given_correct_symbol_noiseless` |
| AFC tracking loop | §6.3 | `afc.py::AFCLoop`, `run_afc_sequence` | `test_afc.py::test_tracking_survives_a_drift_that_exceeds_the_static_capture_range` |
| Kalman CFO/rate tracker | §6.4 | `afc.py::KalmanAFCLoop` | `test_afc.py::test_kalman_loop_converges_toward_repeated_measurement` |
| Outlier-measurement divergence and its fix | §6.4 | `afc.py::AFCLoop.max_jump_hz`, `KalmanAFCLoop.innovation_gate` | `test_afc.py::test_afc_loop_gates_a_wild_outlier_measurement`, `test_kalman_loop_gates_a_wild_outlier_innovation` |
| Tracking through a sign-reversing (UAV) Doppler | §6.4 | `afc.py::run_afc_sequence` | `test_afc.py::test_afc_tracks_through_a_doppler_sign_reversal`, `test_afc_fails_for_an_unrealistically_fast_doppler_reversal` |
| Full-symbol coherent CFO measurement (negative-SNR fix) | §6.5 | `afc.py::full_symbol_cfo_estimate`, `measurement_noise_for` | `test_afc.py::test_full_symbol_cfo_estimate_accurate_at_negative_snr`, `test_full_symbol_cfo_estimate_far_less_starved_than_dual_edge_at_negative_snr`, `test_afc_tracks_a_doppler_reversal_at_negative_snr` |
| Multi-burst acquisition (preamble averaging) | §7.2 | `afc.py::run_afc_sequence` (`acquire_bursts`) | `test_afc.py::test_multi_burst_acquisition_beats_single_burst_at_low_snr` |
| Kalman acquisition-covariance fix | §7.2 | `afc.py::KalmanAFCLoop.set_acquired` | `test_afc.py::test_kalman_set_acquired_resets_covariance_not_just_cfo` |
| End-to-end packet decode under Doppler | §7 | `examples/packet_experiments.py` | (SER-vs-SNR sweeps; no dedicated pytest, see script's own correctness assertions) |

## 8. Comparison with standard LoRa, and when this applies

**Where nonlinear trajectories don't help.** §4's SER-vs-SNR sweep across
every trajectory shape shows essentially identical performance to linear
once decoded with `fft_correlation_demod` — trajectory curvature carries
no baseline noise-tolerance advantage. §5.1 shows the correlation-magnitude
loss from a constant CFO is likewise shape-independent, since
$|s(t)|=1$ for every trajectory tested.

**Where the advantage is real but partial.** §5.2's exact self-similarity
theorem gives HFM a genuine single-waveform advantage under wideband
(scaling) Doppler — about 1 dB of matched-filter peak loss across a
$\pm10\%$ scale sweep versus 10 dB+ for linear. §5.2 already documents
that this only partially survives full $M$-ary decoding, since a
Doppler-induced timing lag common to every trajectory shape dominates
first.

**Where the advantage is structural.** A linear chirp's instantaneous
rate $df/dt$ is constant, so there is nothing locally measurable about
*position* within the symbol — which is exactly why standard LoRa's own
CFO estimation is a one-time operation: dechirping a linear chirp yields
a pure tone, so an FFT peak position is a direct frequency measurement,
and pairing an up-chirp preamble symbol with a down-chirp SFD symbol
resolves the resulting CFO/timing-offset ambiguity,
$\widehat{\mathrm{CFO}} \propto (\mathrm{peak}_{\mathrm{up}} +
\mathrm{peak}_{\mathrm{down}})/2$ — cheap and exact, but performed once,
at packet start. A curved trajectory's rate varies with position by
construction, which is what makes §6's dual-edge measurement possible at
all: an ongoing position/CFO sensor built from the payload's own symbols,
requiring no additional reference chirps, for the duration of the packet.
This is not a more expensive version of standard LoRa's mechanism; it is
a capability standard LoRa's linear chirp cannot support at all, regardless
of cost.

**Applicability.** The case for this approach requires, jointly: (i) CFO
that drifts materially within a single packet (long payload, fast drift
rate, or both — if a one-shot preamble correction remains valid for the
whole payload, there is nothing to track); (ii) no exploitable external
model of the motion (no ephemeris for a known orbit, no telemetry
side-channel, an uncooperative or unknown transmitter) — where such a
model exists, precomputed Doppler pre-compensation with a linear chirp is
simpler and at least as effective; and (iii) a preference for one long
packet over repeated short packets with fresh-preamble re-acquisition
(costly under duty-cycle-limited airtime budgets). Outside that
intersection, a linear chirp is the better engineering choice: CFO
estimation is a free byproduct of decoding rather than an additional
measurement, and §4-§5.1 show no compensating loss in baseline noise or
narrowband-CFO performance to offset that simplicity.

---

## References

- Vangelista, L. "Frequency Shift Chirp Modulation: The LoRa Modulation."
  *IEEE Signal Processing Letters*, 2017.
- Woodward, P. M. *Probability and Information Theory, with Applications to
  Radar*. Pergamon Press, 1953. (The ambiguity function.)
- Kroszczyński, J. "Pulse Compression by Means of Linear-Period Modulation."
  *Proceedings of the IEEE*, 1969. (Hyperbolic/linear-period FM.)
- Oppenheim, A. V., Schafer, R. W. *Discrete-Time Signal Processing*.
  (The discrete correlation theorem, §4.2 above.)
- Murray, J. et al. "On the Doppler Bias of Hyperbolic Frequency Modulation
  Matched Filter Time of Arrival Estimates." *IEEE Journal of Oceanic
  Engineering*, 2019. (Closed-form Doppler/range bias for HFM matched
  filtering — the sonar-literature counterpart to this project's own
  Doppler-induced lag finding in §5.2 and §6.4.)

See `DISSERTATION_OUTLINE.md` for how these results map onto a dissertation
structure, and `README.md` for how to regenerate every figure referenced
here.
