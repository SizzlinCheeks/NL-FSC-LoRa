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
Measured, not assumed: `dual_edge_cfo_estimate`'s own noise at SF7/10 dB SNR
is $\approx 184$ Hz std ($R \approx 3.4\times10^4$ Hz$^2$); an initial
untested guess of $R=400$ Hz$^2$ was off by $\sim 85\times$ and made the
filter overtrust individual measurements. Correctly tuned, it gives a real
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
rate-of-change limit rather than assuming one: accuracy holds near 1.0 up
to a peak instantaneous rate of $\approx 55$ Hz/burst and collapses above
$\approx 100$ Hz/burst (`examples/output/19_afc_rate_of_change_limit.png`).
At this configuration's symbol duration ($\approx 1.02$ ms), 55 Hz/burst is
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
