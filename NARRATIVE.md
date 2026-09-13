# Nonlinear Chirp Spread Spectrum: A Narrative Walkthrough

This is a companion to `PAPER.md`, written for a reader who already knows
what LoRa is and wants to understand *why* each result in this project is
true, not just look one up. The math is the same and equally rigorous —
every proof here is the same proof — but the order is different: each idea
is motivated in plain language before the equation that formalizes it, and
each chapter ends by handing the next chapter its problem.

**The story in one line:** define the waveform → show why conventional LoRa
detection fails on it → introduce correlation as the fix → make correlation
computationally practical → ask why you'd want a nonlinear trajectory in
the first place (Doppler) → use the trajectory itself to track where the
signal actually is.

```
1. The Chirp Model
   "What are we transmitting?"
        │
        ▼
2. Why the Standard LoRa Decoder Needs a Linear Chirp
   "Why doesn't normal LoRa demodulation work anymore?"
        │
        ▼
3. A Decoder That Works for Any Chirp
   "Can correlation solve the problem?"
        │
        ▼
4. The Fast Decoder: The Correlation Theorem
   "Can we make correlation computationally practical?"
        │
        ▼
5. Two Doppler Models, and Why They Give Different Answers
   "Why might a nonlinear trajectory be useful in the first place?"
        │
        ▼
6. Dual-Edge Frequency Tracking
   "Once we know the trajectory, how do we track where the
    received signal actually is relative to it?"
```

---

## 1. The Chirp Model

**What exactly is a nonlinear LoRa-like chirp?**

A chirp is a signal whose instantaneous frequency changes with time. In
conventional LoRa, that frequency changes *linearly* — a straight ramp from
the bottom of the channel to the top. This project generalizes the
frequency trajectory so it can follow an arbitrary shape instead of a
straight line:

$$
f(t) = f_0 + B \cdot g(t/T)
$$

$g$ is the thing being changed. $g(u)=u$ reproduces the linear ramp — plain
LoRa. Any other $g$ (curved, S-shaped, whatever) produces a curved
frequency trajectory instead. $g$ is normalized so $g(0)=0$ and $g(1)=1$,
which just means the trajectory always starts at the bottom of the band and
ends at the top, no matter its shape in between.

Frequency integrates to phase, and phase exponentiates to the actual
transmitted waveform:

$$
\phi(t) = 2\pi\int_0^t f(\tau)d\tau, \qquad s(t) = e^{j\phi(t)}
$$

**How does a symbol get encoded?** Each symbol is represented by a
different cyclic shift of the *same* reference chirp — not $M$ unrelated
waveforms, one waveform shifted to $M$ different starting points. That
matters because it means the receiver's job is never "which of $M$
independent signals is this?" — it's always "which shift of *one* signal is
this?" That reframing is what makes Chapters 3 and 4 possible.

$$
s_m[n] = s[(n+\tau_m)\bmod N], \qquad \tau_m = \mathrm{round}(mN/M)
$$

Under the discrete-time representation used in this work, each symbol is a
cyclic sample shift of a common reference waveform, exactly as written
above — this is a description of the model in this project, not a claim
about the internals of any particular commercial LoRa chipset.

---

## 2. Why the Standard LoRa Decoder Needs a Linear Chirp

This is the chapter that explains why any of the rest of this project needs
to exist.

The conventional LoRa receiver multiplies the received chirp by the
conjugate of a reference chirp — "dechirping":

$$
d(t) = r(t) \cdot \overline{s(t)}
$$

Complex multiplication adds phases, so multiplying by a conjugate
*subtracts* the reference's phase:

$$
e^{j\phi_r(t)} \cdot e^{-j\phi_s(t)} = e^{j[\phi_r(t)-\phi_s(t)]}
$$

**What does that signal look like after dechirping?** That's the question
this whole chapter answers. In the ideal case the received signal really is
symbol $m$'s waveform, $\phi_r(t) = \phi(t+\tau_m)$, so the dechirped signal's
*instantaneous frequency* — its rate of phase change — is

$$
f(t+\tau_m) - f(t)
$$

For a linear chirp, $f(t) = f_0 + kt$:

$$
f(t+\tau) - f(t) = k(t+\tau) - kt = k\tau
$$

**— a constant.** A signal whose frequency doesn't change with time is a
pure tone. And a pure tone's FFT is one sharp spike, not a spread-out mess.
The whole standard-LoRa demodulator rests on this chain:

```
linear chirp → dechirp → constant frequency → tone → FFT → one strong bin → symbol
```

The shift $\tau_m$ (i.e. the symbol) survives the entire chain and comes
out the other end as *which bin lit up*. This isn't a coincidence worth
taking for granted — it's worth pinning down exactly when it's true.

**Claim.** $f(t+\tau)-f(t)$ is constant in $t$ for every $\tau$ if and only
if $f$ is affine, $f(t) = f_0+kt$.

*Proof.* ($\Leftarrow$) Shown above. ($\Rightarrow$) If $f(t+\tau)-f(t)=c(\tau)$ for every $t$, differentiate both sides with respect to $t$: $f'(t+\tau)-f'(t)=0$ for all $t,\tau$, so $f'$ is constant, so $f$ is affine. $\blacksquare$

So the FFT-bin trick isn't a general property of chirps at all — it's a
one-off algebraic coincidence that only the straight line has.

**Now the nonlinear case.** For a curved trajectory, the frequency doesn't
advance at a constant rate — it's steeper in some places, shallower in
others. Shifting a *curved* trajectory in time does **not** produce a
constant frequency difference, because the amount the curve has "moved
up or down" by a given time shift depends on how steep it was at that
particular point. So after dechirping, the residual signal's frequency
keeps changing with time — it never settles into a single tone:

```
nonlinear chirp → dechirp → frequency still changes → not a tone
                → FFT energy spreads → no single bin reliably identifies the symbol
```

This is probably the single most important idea in the whole project: it
is *why* everything from here on is necessary. `tests/test_receiver.py::test_fft_demod_fails_for_properly_embedded_hyperbolic_chirp`
confirms this failure directly rather than just arguing for it.

---

## 3. A Decoder That Works for Any Chirp: Brute-Force Correlation

If dechirping no longer produces a tone, we need a different way to figure
out which symbol was transmitted.

The natural question to ask is: **which candidate waveform looks most like
the received waveform?** That's correlation — a way of measuring how well
two signals line up.

$$
\mathrm{score}_m = \sum_{n=0}^{N-1} \overline{s_m[n]} \cdot r[n], \qquad \hat m = \arg\max_m |\mathrm{score}_m|
$$

Because every candidate is a cyclic shift of the same reference waveform
(Chapter 1), this has a very concrete meaning: try every possible shift,
score how well each one matches the received signal, and keep the shift
that matches best.

This works for *any* trajectory, curved or straight — there's no
shift-invariance coincidence being relied on here, just a direct
comparison. The catch is cost: it's $M$ separate length-$N$ sums, $O(NM)$
work for every received symbol. Correlation solves the detection problem,
but calculating every possible correlation separately is expensive — and
that expense is exactly what Chapter 4 removes.

(Implementation: `nlfsc_lora/receiver.py::matched_filter_bank_demod`.)

---

## 4. The Fast Decoder: The Correlation Theorem

Chapter 3 showed that correlation can detect any chirp shape, but doing it
required $M$ separate correlations. Here's the fact that changes that: all
$M$ of those candidate waveforms are shifts of the *same* waveform, so the
$M$ scores were never $M$ unrelated numbers in the first place.

**They are $M$ samples of one circular correlation function.** Define that
function for *every* possible shift $l$, not just the $M$ ones that
correspond to real symbols:

$$
C[l] = \sum_{n=0}^{N-1} s[n] \cdot \overline{r[(n-l)\bmod N]}
$$

Instead of asking for one score at a time, this asks for the correlation at
every possible shift simultaneously. The question is whether $C[l]$ can
actually be computed that way cheaply — and it can, via the correlation
theorem. Let $S[k]$ and $R[k]$ be the ordinary DFTs of $s[n]$ and $r[n]$:

$$
C[l] = \mathrm{IDFT}\big(S[k] \cdot \overline{R[k]}\big)[l]
$$

*(Full derivation and proof of this theorem, plus the reindexing argument
that connects it exactly to $\mathrm{score}_m$, are in `PAPER.md` §4 — kept
there rather than duplicated here, since the point of this document is the
narrative, not re-deriving the same algebra twice.)*

This is precisely the line of code that does the work:

```python
corr = np.fft.ifft(base_fft * np.conj(np.fft.fft(rx)))   # C[l] for every l, at once
m_hat = int(np.argmax(np.abs(corr[valid_lags])))          # look up the M known shifts
```

The mental model for the difference between Chapter 3 and this chapter:

```
Brute force                          FFT correlation

Received                             Received ──→ FFT ──┐
   │                                                     ×
   ▼                                 Reference ─→ FFT ──┘
Compare with shift 0                         │
Compare with shift 1                         ▼
Compare with shift 2                        IFFT
   ...                                       │
Compare with shift M                         ▼
   │                                  ALL shifts at once
   ▼                                         │
Pick largest                                 ▼
                                       Pick largest
```

The right side isn't a mathematical curiosity bolted on for elegance — it's
what turns "check every candidate" from a genuine $O(NM)$ bottleneck into
essentially the cost of three transforms, $O(N\log N)$.

Concretely, at $SF=7$ ($N=512$, $M=128$): $NM \approx 65{,}500$ versus
$\sim 3N\log_2 N \approx 13{,}800$. **Be careful reading that comparison,
though** — those two numbers are a theoretical operation count, not a
timing measurement. The theoretical complexity predicts substantially less
work; the measured implementation separately provides an approximately
$85\times$ runtime improvement for the tested configuration. The two
figures agree in direction but shouldn't be read as the same number
measured two ways — real timing includes constant factors (library
overhead, caching, memory access patterns) the raw operation count doesn't
capture.

**Correctness**, not just speed: `tests/test_receiver.py::test_fft_correlation_demod_matches_matched_filter_bank_exactly`
checks that this decoder gives the *identical* decision as the brute-force
one from Chapter 3 on every trial, not merely similar accuracy.

(Implementation: `nlfsc_lora/receiver.py::fft_correlation_demod`.)

---

## 5. Two Doppler Models, and Why They Give Different Answers

Everything so far has been about detecting a symbol correctly. This
chapter is about *why curving the trajectory in the first place* might
ever help with that — and the honest answer depends entirely on which kind
of Doppler is actually present, because Doppler can affect a signal in two
fundamentally different ways: it can appear as a frequency offset, or it
can act as a time scaling of the waveform. These are not two views of the
same thing — they call for different waveform properties, and this project
found real evidence for both being true.

### 5.1 Narrowband Doppler: a frequency offset

$$
r(t) = s(t) \cdot e^{j2\pi f_\Delta t}
$$

This shifts the *entire* frequency trajectory by a constant amount — every
frequency in the sweep moves up (or down) by the same $f_\Delta$:

```
Expected:   100 → 110 MHz
Received:   101 → 111 MHz
```

The frequency changed, but the *shape* of the trajectory didn't — it's
still the same curve, just relabeled. In particular, the *rate* the
frequency is changing at, $df/dt$, is completely unaffected:

$$
\frac{d}{dt}\big[f(t) + f_\Delta\big] = \frac{df}{dt}
$$

This is worth sitting with, because it's the idea Chapter 6 is built on:
**frequency tells us where the signal is; frequency *rate* tells us whether
it's following the expected trajectory.** A frequency reading can be thrown
off by Doppler; a rate reading, for a curved trajectory, can't.

At the correct-symbol level, curving the trajectory doesn't change how much
correlation magnitude a given $f_\Delta$ costs — that loss reduces to
$\left|\int_0^T e^{j2\pi f_\Delta t}dt\right|$, independent of the
trajectory's shape, because $|s(t)|=1$ always. What *can* differ between
trajectories is confusability with the neighboring symbol — a smaller,
shape-dependent effect:

![Symbol error rate vs. CFO for different trajectories](examples/output/04_ser_vs_cfo.png)

### 5.2 Wideband Doppler: time scaling

A more general — and physically exact — Doppler model isn't a shift at
all, it's a time scaling of the whole waveform:

$$
r(t) = s(\alpha t), \qquad \alpha = 1 \pm v/c
$$

Instead of moving the waveform up or down in frequency, the entire
waveform is compressed or stretched in time. This is where hyperbolic FM
(HFM) — the trajectory whose *period*, $1/f(t)$, is linear in $t$ — earns
its keep. Before the proof, the meaning: **for HFM, time-scaling produces
another copy of the *same* waveform, just shifted in time and rotated by a
constant phase.** Nothing about its shape changes — it isn't distorted into
a different waveform the way a straight-ramp chirp is.

$$
f(t) = \frac{f_{\text{low}}}{1-\beta t}, \qquad \phi(t) = -\frac{2\pi f_{\text{low}}}{\beta}\ln(1-\beta t)
$$

**Theorem (exact).** For any $\alpha>0$,
$\phi(\alpha t) = \phi(t-\Delta) + c$, with
$\Delta = \dfrac{1-\alpha}{\alpha\beta}$ and
$c = -\dfrac{2\pi f_{\text{low}}}{\beta}\ln\alpha$.

*Proof.* The identity $1-\beta\alpha t = \alpha\big(1-\beta(t-\Delta)\big)$ holds exactly for that $\Delta$ (expand the right side and collect terms). Apply $-\frac{2\pi f_{\text{low}}}{\beta}\ln(\cdot)$ to both sides of the identity and the claim follows directly. $\blacksquare$

So $s(\alpha t) = e^{jc} \cdot s(t-\Delta)$ — exactly, not approximately, for the
idealized continuous-time waveform. A matched filter built for the
unscaled $s$ still recognizes a Doppler-scaled copy perfectly; the scaling
only moves *where* the peak lands, never how tall it is. A linear chirp has
no such identity: time-scaling it changes its chirp rate outright into a
genuinely different waveform, which is exactly why a linear chirp's
matched-filter response degrades under scaling rather than merely shifting.
HFM loses about 1 dB of peak magnitude across a $\pm10\%$ scale sweep,
versus more than 10 dB for a linear chirp:

![Matched-filter peak magnitude vs. Doppler time-scale for HFM vs. linear chirp](examples/output/07_doppler_scale_tolerance.png)

The exactness above is a property of the idealized, unbounded-domain
waveform; a real burst has a finite window, so the shifted copy's support
doesn't line up perfectly with the original observation window at the
edges. That's the real, measured, nonzero loss reported above.

And here is the honest, load-bearing caveat of this whole chapter: **HFM
has an important theoretical Doppler property, but that does not mean HFM
automatically solves Doppler for the complete $M$-ary LoRa receiver.**
Plugging HFM into LoRa's own fine-grained, cyclic-shift decoding shows the
advantage only partially surviving — that decoder's sensitivity to a
Doppler-induced *lag*, which turns out to be nearly identical across every
trajectory, sets in before HFM's magnitude-preservation advantage gets a
chance to matter. The single-waveform result is real; it is not, by
itself, the whole story:

![Full LoRa symbol error rate vs. Doppler scale for HFM vs. linear chirp](examples/output/08_lora_ser_vs_doppler_scale.png)

---

## 6. Dual-Edge Frequency Tracking

**If the receiver knows what trajectory to expect, how can it tell where
the received signal actually is relative to that trajectory?**

The intuition is the same one Chapter 5.1 set up. Suppose the expected
signal sweeps:

```
Expected:   100 → 110 MHz
Measured:   101 → 111 MHz
```

The frequency is off by about 1 MHz. But the *rate of change* — how fast
the frequency is climbing — is the same in both cases. That single
observation is the conceptual heart of this chapter: **the frequency tells
us the offset; the frequency rate tells us whether the measured signal is
following the expected trajectory at all.** If the rate matches what the
trajectory predicts, the offset is trustworthy. If it doesn't, something —
noise, a wrong assumption about which symbol this is, a corrupted
measurement — has gone wrong, and the offset shouldn't be trusted.

### 6.1 Local frequency and rate

A short window of samples' *unwrapped* phase can be fit to a cubic in the
local sample index $k$. Before the equations: the **linear** term of that
fit is the frequency at that point; the **quadratic** term is how fast that
frequency is changing (the rate); the **cubic** term captures how the rate
*itself* is curving across the window — needed because, for a genuinely
nonlinear trajectory, even the rate isn't constant over a wide-enough
window.

$$
\angle r[n_0+k] \approx a+bk+ck^2+dk^3, \qquad \hat f(n_0) = \frac{bF_s}{2\pi}, \qquad \widehat{\dot f}(n_0) = \frac{2cF_s^2}{2\pi}
$$

(Implementation: `nlfsc_lora/local_rate.py::local_freq_and_rate`,
`nlfsc_lora/afc.py::_edge_measurement`.)

### 6.2 Why two edges?

A cyclically-shifted symbol (Chapter 1) can contain a phase discontinuity
at one specific location — the point where the shift wraps around. If the
measurement window happens to sit on top of that discontinuity, the
frequency/rate estimate there is corrupted. Rather than trusting a single
measurement and hoping it avoided the discontinuity, measure near *both*
ends of the symbol — the discontinuity is at one fixed position, so it can
corrupt at most one of the two measurements, never both.

The two edges give two CFO estimates, $\widehat{\mathrm{CFO}}_A$ and
$\widehat{\mathrm{CFO}}_B$, and they're combined by how trustworthy each
one looks: if a measured rate closely matches the rate the assumed symbol
predicts at that position, that measurement is probably clean; if it
doesn't, something corrupted it, and it should count for less.

$$
\widehat{\mathrm{CFO}} = \frac{w_A \cdot \widehat{\mathrm{CFO}}_A + w_B \cdot \widehat{\mathrm{CFO}}_B}{w_A+w_B}, \qquad w_i = \frac{1}{\left|\dot f_i - \dot f_{\text{exp},i}\right|/|\dot f_{\text{exp},i}| + \epsilon}
$$

`tests/test_afc.py::test_dual_edge_estimate_exact_given_correct_symbol_noiseless`
verifies this concretely for a case where one edge genuinely does sit on
the discontinuity — the corrupted edge is automatically downweighted, and
the combined estimate stays accurate because the clean edge dominates.

### 6.3 The tracking loop

The receiver doesn't throw away its old estimate and fully replace it with
every new measurement — that would make the estimate as noisy as a single
measurement, defeating the point of measuring repeatedly. Instead, it moves
its running estimate *part of the way* toward the newest measurement:

$$
\widehat{\mathrm{CFO}}_{\text{tracked}}[i{+}1] = \widehat{\mathrm{CFO}}_{\text{tracked}}[i] + \gamma\Big(\widehat{\mathrm{CFO}}[i] - \widehat{\mathrm{CFO}}_{\text{tracked}}[i]\Big)
$$

$\gamma$ controls how far "part of the way" is: a small $\gamma$ gives
smoother, slower tracking (good against noisy measurements, sluggish
against real drift); a large $\gamma$ tracks drift quickly but lets more
measurement noise through. That tradeoff is exactly why a loop like this
can follow a Doppler shift that keeps drifting — a satellite pass, for
instance — while a single one-time correction can't: it just keeps nudging
itself back toward center, burst after burst, rather than committing to
one estimate and hoping it stays valid.

The payoff, demonstrated directly: a receiver tracking a 0-3000 Hz drift
over 200 bursts stays at 85-100% decode accuracy the whole way, while a
receiver that corrects once and never updates collapses to 0% once the
drift moves past where it was originally acquired.

![Decode accuracy under a drifting CFO: dual-edge tracking vs. one-time static correction](examples/output/12_dual_edge_afc.png)

---

## Where to go from here

- `PAPER.md` — the same results with every proof given in full, organized
  as a dense reference rather than a narrative.
- `DISSERTATION_OUTLINE.md` — how these results map onto a dissertation
  chapter structure.
- `README.md` — how to regenerate every figure referenced above.
