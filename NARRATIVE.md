# Nonlinear Chirp Spread Spectrum: A Narrative Walkthrough

This is a companion to `PAPER.md`, written for a reader who already knows
what LoRa is and wants to understand *why* each result in this project is
true, not just look one up. The math is the same and equally rigorous —
every proof here is the same proof — but the order is different: each idea
is motivated in plain language before the equation that formalizes it, and
each chapter ends by handing the next chapter its problem.

The story in one line: define the waveform, show why conventional LoRa
detection fails on it, introduce correlation as the fix, make correlation
computationally practical, ask why you'd want a nonlinear trajectory in the
first place (Doppler), then use the trajectory itself to track where the
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

A chirp is a signal whose instantaneous frequency changes with time. In
conventional LoRa, that frequency changes *linearly*: a straight ramp from
the bottom of the channel to the top. This project generalizes the
frequency trajectory so it can follow an arbitrary shape instead of a
straight line:

$$
f(t) = f_0 + B \cdot g(t/T)
$$

$g$ is the shape being varied. $g(u)=u$ reproduces the linear ramp, plain
LoRa. Any other $g$ — curved, S-shaped, whatever — produces a curved
frequency trajectory instead. $g$ is normalized so $g(0)=0$ and $g(1)=1$,
which just means the trajectory always starts at the bottom of the band
and ends at the top, no matter its shape in between. Here's every shape
this project tests, side by side, including hyperbolic FM (HFM), the
shape Chapter 5 comes back to for its Doppler properties:

![Instantaneous frequency and chirp rate for each trajectory shape: linear, quadratic, sigmoid, sinusoidal, exponential, hyperbolic](pictures/01_frequency_trajectories.png)

The left panel is $f(t)$ itself — this is what "the trajectory" actually
looks like. The right panel is $df/dt$, the *chirp rate*: `linear`'s rate
is a flat line (constant, by construction), while every other shape's rate
visibly changes over the symbol. That difference is what breaks the
standard FFT-bin trick in Chapter 2.

Frequency integrates to phase, and phase exponentiates to the actual
transmitted waveform:

$$
\phi(t) = 2\pi\int_0^t f(\tau)d\tau, \qquad s(t) = e^{j\phi(t)}
$$

A symbol is encoded as a different cyclic shift of the *same* reference
chirp — not $M$ unrelated waveforms, one waveform shifted to $M$ different
starting points. That matters because it means the receiver's job is
never "which of $M$ independent signals is this?" It's always "which
shift of *one* signal is this?" That reframing is what makes Chapters 3
and 4 possible.

$$
s_m[n] = s[(n+\tau_m)\bmod N], \qquad \tau_m = \mathrm{round}(mN/M)
$$

Under the discrete-time representation used in this work, each symbol is a
cyclic sample shift of a common reference waveform, exactly as written
above — a description of the model in this project, not a claim about the
internals of any particular commercial LoRa chipset.

---

## 2. Why the Standard LoRa Decoder Needs a Linear Chirp

This chapter explains why the rest of the project needs to exist.

The conventional LoRa receiver multiplies the received chirp by the
conjugate of a reference chirp, "dechirping":

$$
d(t) = r(t) \cdot \overline{s(t)}
$$

Complex multiplication adds phases, so multiplying by a conjugate
*subtracts* the reference's phase:

$$
e^{j\phi_r(t)} \cdot e^{-j\phi_s(t)} = e^{j[\phi_r(t)-\phi_s(t)]}
$$

In the ideal case the received signal really is symbol $m$'s waveform,
$\phi_r(t) = \phi(t+\tau_m)$, so the dechirped signal's *instantaneous
frequency* — its rate of phase change — is

$$
f(t+\tau_m) - f(t)
$$

For a linear chirp, $f(t) = f_0 + kt$:

$$
f(t+\tau) - f(t) = k(t+\tau) - kt = k\tau
$$

A constant. A signal whose frequency doesn't change with time is a pure
tone, and a pure tone's FFT is one sharp spike, not a spread-out mess. The
whole standard-LoRa demodulator rests on this chain:

```
linear chirp → dechirp → constant frequency → tone → FFT → one strong bin → symbol
```

The shift $\tau_m$ — the symbol — survives the entire chain and comes out
the other end as *which bin lit up*. It's worth pinning down exactly when
that's true rather than taking it for granted.

**Claim.** $f(t+\tau)-f(t)$ is constant in $t$ for every $\tau$ if and only
if $f$ is affine, $f(t) = f_0+kt$.

*Proof.* ($\Leftarrow$) Shown above. ($\Rightarrow$) If $f(t+\tau)-f(t)=c(\tau)$ for every $t$, differentiate both sides with respect to $t$: $f'(t+\tau)-f'(t)=0$ for all $t,\tau$, so $f'$ is constant, so $f$ is affine. $\blacksquare$

So the FFT-bin trick isn't a general property of chirps. It's a one-off
algebraic coincidence that only the straight line has.

For a curved trajectory, the frequency doesn't advance at a constant rate
— it's steeper in some places, shallower in others. Shifting a *curved*
trajectory in time doesn't produce a constant frequency difference,
because how much the curve has moved up or down over a given time shift
depends on how steep it was at that particular point. So after
dechirping, the residual signal's frequency keeps changing with time; it
never settles into a single tone:

```
nonlinear chirp → dechirp → frequency still changes → not a tone
                → FFT energy spreads → no single bin reliably identifies the symbol
```

This is the idea everything from here on depends on. Here it is
concretely, on the same symbol (33), decoded via `fft_demod` for linear
vs. hyperbolic:

![Dechirped instantaneous frequency and the resulting FFT for symbol 33, linear vs. hyperbolic](pictures/15_dechirp_linear_vs_hyperbolic.png)

The left-hand panels are the frequency *after* dechirping, not the raw
transmitted chirp — the raw linear chirp does sweep, as Chapter 1's plot
shows. Dechirping is precisely the operation that cancels that sweep, so
"flat" here isn't a failure to plot the sweep; it's the effect being
demonstrated. Multiplying by the reference's conjugate turns a moving
frequency into a nearly constant one, which is the entire reason a single
FFT can read off a bin index for linear chirps. Top row: the dechirped
frequency sits at one constant value, steps to a *second* constant value
at the wrap point (differing by exactly the swept bandwidth `B`, 125 kHz
here), and stays there. Two flat pieces, not one — but each piece by
itself is still a tone, which is why the FFT on the right still lands one
sharp spike at bin 33 rather than two separate ones.

Bottom row, hyperbolic: the curve after the wrap descends because
hyperbolic's own chirp *rate* is increasing over time (visible in Chapter
1's right-hand `df/dt` panel — hyperbolic's rate climbs fastest near the
end of the sweep). After the wrap, the dechirped signal compares a late,
steep part of the trajectory against an early, shallow part, and that gap
widens as time goes on. That's the downward curve, checked here against
the exact closed-form `f(t+τ)-f(t)`, not just the noisier numerical
derivative. Either way, the frequency never settles into a single value,
so the FFT on the right smears across many bins instead of concentrating
in one, and `fft_demod`'s `argmax` confidently returns the wrong symbol
(17, not 33) — not "less accurate," actually wrong, on a noiseless signal.
That's the exact failure
`tests/test_receiver.py::test_fft_demod_fails_for_properly_embedded_hyperbolic_chirp`
confirms directly rather than just arguing for it.

One more thing the figure shows: the FFT panels run to 512, not 128.
`SF=7` gives `M=128` possible *symbols*, but each symbol is sampled at
`N=512` points (4x oversampling — a standard receiver design choice this
project keeps, not something specific to nonlinear trajectories). The FFT
operates on those 512 time-domain samples, so it has 512 bins; only 128 of
them (the thin gray lines) are actual valid symbol positions the decoder
ever looks at.

---

## 3. A Decoder That Works for Any Chirp: Brute-Force Correlation

If dechirping no longer produces a tone, we need a different way to
figure out which symbol was transmitted.

The natural question is: which candidate waveform looks most like the
received waveform? That's correlation — a way of measuring how well two
signals line up.

$$
\mathrm{score}_m = \sum_{n=0}^{N-1} \overline{s_m[n]} \cdot r[n], \qquad \hat m = \arg\max_m |\mathrm{score}_m|
$$

Because every candidate is a cyclic shift of the same reference waveform
(Chapter 1), this has a concrete meaning: try every possible shift, score
how well each one matches the received signal, and keep the shift that
matches best.

This works for *any* trajectory, curved or straight — there's no
shift-invariance coincidence being relied on here, just a direct
comparison. The catch is cost: $M$ separate length-$N$ sums, $O(NM)$ work
per received symbol. Correlation solves the detection problem, but
calculating every possible correlation separately is expensive, and that
expense is exactly what Chapter 4 removes.

(Implementation: `nlfsc_lora/receiver.py::matched_filter_bank_demod`.)

---

## 4. The Fast Decoder: The Correlation Theorem

Chapter 3 showed that correlation can detect any chirp shape, but doing
it required $M$ separate correlations. Here's the fact that changes that:
all $M$ of those candidate waveforms are shifts of the *same* waveform, so
the $M$ scores were never $M$ unrelated numbers in the first place — they
are $M$ samples of one circular correlation function. Define that
function for *every* possible shift $l$, not just the $M$ ones that
correspond to real symbols:

$$
C[l] = \sum_{n=0}^{N-1} s[n] \cdot \overline{r[(n-l)\bmod N]}
$$

Instead of asking for one score at a time, this asks for the correlation
at every possible shift simultaneously. The question is whether $C[l]$
can actually be computed that way cheaply, and it can, via the
correlation theorem. Let $S[k]$ and $R[k]$ be the ordinary DFTs of $s[n]$
and $r[n]$:

$$
C[l] = \mathrm{IDFT}\big(S[k] \cdot \overline{R[k]}\big)[l]
$$

*(Full derivation and proof of this theorem, plus the reindexing argument
that connects it exactly to $\mathrm{score}_m$, are in `PAPER.md` §4 —
kept there rather than duplicated here, since the point of this document
is the narrative, not re-deriving the same algebra twice.)*

This is the line of code that does the work:

```python
corr = np.fft.ifft(base_fft * np.conj(np.fft.fft(rx)))   # C[l] for every l, at once
m_hat = int(np.argmax(np.abs(corr[valid_lags])))          # look up the M known shifts
```

Here's that computation on one actual received symbol (hyperbolic
trajectory, symbol 33, 10 dB SNR). First, the actual waveform being
decoded — the hyperbolic trajectory cyclically shifted to symbol 33, same
style as the Chapter 1 plot but for this one specific symbol instead of
every shape's `m=0` base:

![Hyperbolic trajectory shifted to symbol 33, showing the cyclic-shift wrap point](pictures/16_hyperbolic_symbol33_waveform.png)

The curve is the same hyperbolic shape from Chapter 1, just started at a
different point and wrapped around — the dashed line marks where the
trajectory's end folds back to its beginning, the "wrap point" Chapter 6
returns to. Now the rest of the computation: `|S[k]|` and `|R[k]|`, the
two FFT magnitudes the code above starts from; what conjugating `R[k]`
actually changes; and `|C[l]|`, the correlation the IFFT recovers from
their product.

![FFT of the reference waveform, FFT of the received signal, what conjugation changes, and the resulting correlation after IFFT, with a clear peak at the true symbol shift](pictures/13_fft_correlation_demo.png)

Neither of the first two panels tells you the symbol by itself. Both
spectra are spread across many frequency bins, an unavoidable property of
a chirp, since it sweeps through a wide range of frequencies over the
symbol. The third panel shows what *conjugating* `R[k]` actually does to
it: not much to its magnitude, since `|conj(R[k])| = |R[k]|` always, so a
magnitude plot of it would just repeat the second panel. Conjugation only
touches the imaginary part, flipping its sign (equivalently, it negates
the phase); the real part is untouched. That's what lets the product
`S[k]·conj(R[k])` add up constructively at the one lag where the two
waveforms actually line up, and mostly cancel elsewhere. The fourth panel
is the payoff: multiplying those two spectra together and taking one IFFT
concentrates all of that spread-out energy into a single sharp spike,
sitting exactly at the true shift (dashed line). The red dots are the $M$
positions `τ_m` that actually correspond to real symbols — `argmax` over
just those $M$ points is the whole decoder, and it lands on the correct
one here.

The fourth panel's x-axis is `lag l`, a sample-domain shift (0 to `N-1` =
511 here), not the symbol index directly, so it's easy to misread. The
peak sits at lag 132, not "33" — that's not a discrepancy, it's Chapter
1's `τ_m = round(m·N/M)` formula showing up again: symbol 33's cyclic
shift is `round(33·512/128) = 132` samples, so the peak *is* symbol 33,
expressed in samples instead of symbol index. To go from a peak lag back
to a symbol, divide by `N/M = 4`: `132 / 4 = 33`. The decoder itself never
does that conversion explicitly — `fft_correlation_demod` only ever looks
at `corr[valid_lags]`, an array already indexed `0..M-1` one entry per
symbol, and `argmax` on that array returns the symbol index directly. The
annotation on the plot spells out the same arithmetic for anyone reading
the lag axis by eye.

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

The right side isn't a mathematical curiosity bolted on for elegance —
it's what turns "check every candidate" from a genuine $O(NM)$ bottleneck
into essentially the cost of three transforms, $O(N\log N)$.

At $SF=7$ ($N=512$, $M=128$): $NM \approx 65{,}500$ versus
$\sim 3N\log_2 N \approx 13{,}800$. Read that comparison carefully, though
— those two numbers are a theoretical operation count, not a timing
measurement. The theoretical complexity predicts substantially less work;
the measured implementation separately provides an approximately
$85\times$ runtime improvement for the tested configuration. The two
figures agree in direction but shouldn't be read as the same number
measured two ways — real timing includes constant factors (library
overhead, caching, memory access patterns) the raw operation count
doesn't capture.

And it's not just faster, it's exactly as correct:
`tests/test_receiver.py::test_fft_correlation_demod_matches_matched_filter_bank_exactly`
checks that this decoder gives the *identical* decision as the
brute-force one from Chapter 3 on every trial, not merely similar
accuracy.

(Implementation: `nlfsc_lora/receiver.py::fft_correlation_demod`.)

Now that every shape has a decoder that's both correct and cheap, a fair
comparison is finally possible: does the trajectory shape itself affect
plain noise tolerance, decoded the same way for everyone?

![Symbol error rate vs. SNR for every trajectory shape, all decoded with fft_correlation_demod](pictures/14_ser_vs_snr_all_shapes.png)

(The x-axis runs from -10 dB down to -32 dB, left to right — read it as a
channel that starts out clean and degrades over time, rather than the
usual "worst to best" waterfall convention.)

Barely. Every shape's curve sits almost on top of the others — `sigmoid`
trails the rest by a small margin near the knee of the curve, but nothing
here is dramatically more or less noise-tolerant than `linear` once the
decoder itself isn't the bottleneck. So curving the trajectory isn't
buying noise tolerance. If it's going to earn its keep, it has to be for
some other reason, which is exactly the question Chapter 5 picks up.

---

## 5. Two Doppler Models, and Why They Give Different Answers

Everything so far has been about detecting a symbol correctly. This
chapter asks why curving the trajectory in the first place might ever
help with that. The answer depends entirely on which kind of Doppler is
actually present, because Doppler can affect a signal in two fundamentally
different ways: it can appear as a frequency offset, or it can act as a
time scaling of the waveform. These aren't two views of the same thing —
they call for different waveform properties, and this project found real
evidence for both being true.

### 5.1 Narrowband Doppler: a frequency offset

$$
r(t) = s(t) \cdot e^{j2\pi f_\Delta t}
$$

This shifts the *entire* frequency trajectory by a constant amount —
every frequency in the sweep moves up (or down) by the same $f_\Delta$:

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

That's the idea Chapter 6 is built on: frequency tells us where the
signal is; frequency *rate* tells us whether it's following the expected
trajectory. A frequency reading can be thrown off by Doppler; a rate
reading, for a curved trajectory, can't.

At the correct-symbol level, curving the trajectory doesn't change how
much correlation magnitude a given $f_\Delta$ costs — that loss reduces to
$\left|\int_0^T e^{j2\pi f_\Delta t}dt\right|$, independent of the
trajectory's shape, because $|s(t)|=1$ always. What can differ between
trajectories is confusability with the neighboring symbol, a smaller,
shape-dependent effect:

![Symbol error rate vs. CFO for different trajectories](pictures/04_ser_vs_cfo.png)

### 5.2 Wideband Doppler: time scaling

A more general, and physically exact, Doppler model isn't a shift at all
— it's a time scaling of the whole waveform:

$$
r(t) = s(\alpha t), \qquad \alpha = 1 \pm v/c
$$

Instead of moving the waveform up or down in frequency, the entire
waveform is compressed or stretched in time. This is where hyperbolic FM
(HFM), the trajectory whose *period*, $1/f(t)$, is linear in $t$, earns
its keep. Before the proof, the meaning: for HFM, time-scaling produces
another copy of the *same* waveform, just shifted in time and rotated by
a constant phase. Nothing about its shape changes; it isn't distorted
into a different waveform the way a straight-ramp chirp is.

$$
f(t) = \frac{f_{\text{low}}}{1-\beta t}, \qquad \phi(t) = -\frac{2\pi f_{\text{low}}}{\beta}\ln(1-\beta t)
$$

**Theorem (exact).** For any $\alpha>0$,
$\phi(\alpha t) = \phi(t-\Delta) + c$, with
$\Delta = \dfrac{1-\alpha}{\alpha\beta}$ and
$c = -\dfrac{2\pi f_{\text{low}}}{\beta}\ln\alpha$.

*Proof.* The identity $1-\beta\alpha t = \alpha\big(1-\beta(t-\Delta)\big)$ holds exactly for that $\Delta$ (expand the right side and collect terms). Apply $-\frac{2\pi f_{\text{low}}}{\beta}\ln(\cdot)$ to both sides of the identity and the claim follows directly. $\blacksquare$

So $s(\alpha t) = e^{jc} \cdot s(t-\Delta)$ — exactly, not approximately,
for the idealized continuous-time waveform. A matched filter built for
the unscaled $s$ still recognizes a Doppler-scaled copy perfectly; the
scaling only moves *where* the peak lands, never how tall it is. A linear
chirp has no such identity: time-scaling it changes its chirp rate
outright into a genuinely different waveform, which is why a linear
chirp's matched-filter response degrades under scaling rather than merely
shifting. HFM loses about 1 dB of peak magnitude across a $\pm10\%$ scale
sweep, versus more than 10 dB for a linear chirp:

![Matched-filter peak magnitude vs. Doppler time-scale for HFM vs. linear chirp](pictures/07_doppler_scale_tolerance.png)

The exactness above is a property of the idealized, unbounded-domain
waveform; a real burst has a finite window, so the shifted copy's support
doesn't line up perfectly with the original observation window at the
edges. That's the real, measured, nonzero loss reported above.

The caveat that matters for the rest of the chapter: HFM has a real
theoretical Doppler advantage, but that doesn't mean HFM automatically
solves Doppler for the complete $M$-ary LoRa receiver. Plugging HFM into
LoRa's own fine-grained, cyclic-shift decoding shows the advantage only
partially surviving — that decoder's sensitivity to a Doppler-induced
*lag*, which turns out to be nearly identical across every trajectory,
sets in before HFM's magnitude-preservation advantage gets a chance to
matter. The single-waveform result is real; it just isn't the whole
story:

![Full LoRa symbol error rate vs. Doppler scale for HFM vs. linear chirp](pictures/08_lora_ser_vs_doppler_scale.png)

---

## 6. Dual-Edge Frequency Tracking

If the receiver knows what trajectory to expect, how can it tell where
the received signal actually is relative to that trajectory? The
intuition is the same one Chapter 5.1 set up. Suppose the expected signal
sweeps:

```
Expected:   100 → 110 MHz
Measured:   101 → 111 MHz
```

The frequency is off by about 1 MHz. But the *rate of change* — how fast
the frequency is climbing — is the same in both cases. That single
observation is the conceptual heart of this chapter: the frequency tells
us the offset; the frequency rate tells us whether the measured signal is
following the expected trajectory at all. If the rate matches what the
trajectory predicts, the offset is trustworthy. If it doesn't, something
— noise, a wrong assumption about which symbol this is, a corrupted
measurement — has gone wrong, and the offset shouldn't be trusted.

### 6.1 Local frequency and rate

A short window of samples' *unwrapped* phase can be fit to a cubic in the
local sample index $k$. Before the equations: the **linear** term of that
fit is the frequency at that point; the **quadratic** term is how fast
that frequency is changing (the rate); the **cubic** term captures how
the rate *itself* is curving across the window, needed because, for a
genuinely nonlinear trajectory, even the rate isn't constant over a
wide-enough window.

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
ends of the symbol — the discontinuity is at one fixed position, so it
can corrupt at most one of the two measurements, never both.

The two edges give two CFO estimates, $\widehat{\mathrm{CFO}}_A$ and
$\widehat{\mathrm{CFO}}_B$, combined by how trustworthy each one looks: if
a measured rate closely matches the rate the assumed symbol predicts at
that position, that measurement is probably clean; if it doesn't,
something corrupted it, and it should count for less.

$$
\widehat{\mathrm{CFO}} = \frac{w_A \cdot \widehat{\mathrm{CFO}}_A + w_B \cdot \widehat{\mathrm{CFO}}_B}{w_A+w_B}, \qquad w_i = \frac{1}{\left|\dot f_i - \dot f_{\text{exp},i}\right|/|\dot f_{\text{exp},i}| + \epsilon}
$$

`tests/test_afc.py::test_dual_edge_estimate_exact_given_correct_symbol_noiseless`
verifies this concretely for a case where one edge genuinely does sit on
the discontinuity — the corrupted edge is automatically downweighted, and
the combined estimate stays accurate because the clean edge dominates.

Put together, 6.1 and 6.2 form one per-burst pipeline, and the order
matters more than it might look at first: decoding comes *before*
measuring, not after. The receiver can't tell "the trajectory looks
different" from "there's a CFO" without already knowing which symbol's
trajectory it's comparing against — trying to measure frequency and rate
*before* decoding is exactly what Chapter 3's brute-force detour and
`local_rate.py` both ran into, and exactly what this design sidesteps by
only ever measuring a *residual*, never a blind guess. Each edge
measurement is a direct cubic fit to the local unwrapped phase, not a
dechirp against a reference waveform (that's Chapter 2's operation, used
here only indirectly, via the `symbol_duration`/`bandwidth`/`g` it takes to
compute what each edge's frequency and rate are *expected* to be for the
now-known symbol):

```
                  burst i arrives (rx)
                           │
                           ▼
    ┌─────────────────────────────────────────────┐
    │ correct_cfo(rx, cfo_tracked)                │  uses the estimate carried
    └─────────────────────────────────────────────┘  over from burst i-1, not
                           │                          anything measured on
                           ▼                          burst i yet
    ┌─────────────────────────────────────────────┐
    │ fft_correlation_demod                       │  decode FIRST -- no
    │ → decoded symbol m_hat                      │  frequency-domain
    └─────────────────────────────────────────────┘  measurement needed here
                           │
                           ▼
    ┌─────────────────────────────────────────────┐
    │ dual_edge_cfo_estimate(..., m_hat)          │  the two-edge measurement
    │ -- only possible now that m_hat is known    │  and quality-weighted
    └─────────────────────────────────────────────┘  combine from §6.1-6.2
                           │
                           ▼
    ┌─────────────────────────────────────────────┐
    │ tracker.update(measurement)                 │
    │ AFCLoop: tracks CFO alone, or               │
    │ KalmanAFCLoop: tracks CFO + CFO-rate (§6.4) │
    │ -- both reject implausible outlier          │
    │ measurements before trusting them           │
    └─────────────────────────────────────────────┘
                           │
                           ▼
                 cfo_tracked (updated)
                           │
                           └───▶ used to correct burst i+1
```

The loop is one burst behind by construction: what corrects burst $i{+}1$
is a measurement made *from* burst $i$, after burst $i$'s own symbol was
already known. There's no path in this design where frequency is measured
before the symbol is decoded — that's not an implementation detail, it's
the whole reason this approach avoids `local_rate.py`'s noise floor and
wrap-glitch failure band in the first place.

### 6.3 The tracking loop

The receiver doesn't throw away its old estimate and fully replace it
with every new measurement — that would make the estimate as noisy as a
single measurement, defeating the point of measuring repeatedly. Instead,
it moves its running estimate *part of the way* toward the newest
measurement:

$$
\widehat{\mathrm{CFO}}_{\text{tracked}}[i{+}1] = \widehat{\mathrm{CFO}}_{\text{tracked}}[i] + \gamma\Big(\widehat{\mathrm{CFO}}[i] - \widehat{\mathrm{CFO}}_{\text{tracked}}[i]\Big)
$$

$\gamma$ controls how far "part of the way" is: a small $\gamma$ gives
smoother, slower tracking, good against noisy measurements but sluggish
against real drift; a large $\gamma$ tracks drift quickly but lets more
measurement noise through. That tradeoff is why a loop like this can
follow a Doppler shift that keeps drifting — a satellite pass, for
instance — while a single one-time correction can't. It just keeps
nudging itself back toward center, burst after burst, rather than
committing to one estimate and hoping it stays valid.

The payoff, demonstrated directly: a receiver tracking a 0-3000 Hz drift
over 200 bursts stays at 85-100% decode accuracy the whole way, while a
receiver that corrects once and never updates collapses to 0% once the
drift moves past where it was originally acquired.

![Decode accuracy under a drifting CFO: dual-edge tracking vs. one-time static correction](pictures/12_dual_edge_afc.png)

### 6.4 Pushing on it: a smarter tracker, and a Doppler that changes sign

Two questions worth asking about a result like this rather than just
accepting it: is the tracking loop itself as good as it can reasonably be,
and does it hold up outside the one scenario it was demonstrated on? A
satellite pass drifts the CFO in roughly one direction for the length of a
burst sequence. A low-altitude UAV or drone passing near the receiver
doesn't — it approaches (positive Doppler), crosses closest approach
(Doppler through zero), then recedes (negative Doppler), all within the
same pass.

On the tracker itself: the exponential loop filter above only ever chases
the latest measurement by a fixed fraction $\gamma$, which is exactly why
it has that steady-state lag under a constant drift. A Kalman filter that
tracks CFO *and* its rate jointly, predicting forward with the rate
estimate every burst instead of only reacting to where the CFO already is,
is the textbook fix for a lag like that — and it's also the standard
approach in the carrier-tracking literature more broadly, used for exactly
this kind of problem in GPS receivers and coherent optical communication
links. Tested directly against the same drift as above:

![Fixed-gain AFCLoop vs a constant-velocity Kalman filter on the same linear Doppler drift](pictures/17_kalman_vs_expfilter_ramp.png)

A real but modest improvement, not a dramatic one — the exponential filter
was already reasonably well-suited to a smooth, roughly-constant drift.
Where the two trackers actually diverge is on the harder question: what
happens when the drift itself reverses?

Modeling a UAV flyover honestly takes the standard constant-velocity
closest-point-of-approach Doppler curve — an S-curve that saturates to
$\pm f_{d,\max}$ far from the pass and crosses zero smoothly at closest
approach, steeper for a faster or lower pass:

$$
f_d(t) = -f_{d,\max} \cdot \frac{t}{\sqrt{t^2 + t_0^2}}
$$

Running that through the same two-edge measurement this chapter has used
throughout — the instantaneous rate of change near the start and the end
of each burst, exactly the "check both ends of the swept bandwidth"
approach this whole tracking idea started from — turned up something
worth being honest about: the very first version of this test, run over a
long sequence, diverged completely partway through, and not near the sign
reversal. Testing it down to the individual measurement found why:
`dual_edge_cfo_estimate`'s own quality check occasionally — about 1 in
400, measured directly — lets through a measurement that's wrong by
1-4.5 kHz while still reporting a passing score. Over a couple hundred
bursts that's unlikely to come up; over a few thousand (any realistic
pass) it's close to certain to happen at least once, and without a second
line of defense, one bad measurement was enough to permanently derail the
loop for the rest of the run — the next decode fails because the residual
now exceeds the decoder's own capture range, that failure corrupts the
next measurement too, and there was nothing to break the cycle. Both
trackers now gate outlier measurements before trusting them (`afc.py`'s
`AFCLoop.max_jump_hz` and `KalmanAFCLoop.innovation_gate`), which fixes it
directly. With that fix in place:

![Dual-edge AFC through a full approach/closest-approach/recede Doppler reversal](pictures/18_uav_flyover_afc.png)

Both trackers follow the reversal cleanly from one side to the other,
while a receiver that only corrects once collapses the moment the drift
carries it away from wherever it first acquired. The same rate-of-change
measurement this project has used from the start does keep working
through a sign change, not just a monotonic ramp.

That still leaves a fair question: *how* fast a reversal can it follow? A
receiver correcting with its own latest estimate needs that estimate to
stay within the decoder's capture range every single burst — push the
reversal fast enough and that stops being true. Sweeping the steepness of
the flyover curve and plotting decode accuracy against the peak
instantaneous rate at the sign crossing finds the actual cliff rather than
guessing at one:

![Decode accuracy vs. peak Doppler rate at the sign crossing, showing the real cliff and how far realistic rates sit from it](pictures/19_afc_rate_of_change_limit.png)

`AFCLoop` holds accuracy near 1.0 up to roughly 55 Hz per burst and
collapses above roughly 100 — a fixed-gain filter's rate-tracking ability
is bounded mainly by its gain, not by how precise any one measurement is,
so this ceiling turns out to be a property of the loop filter itself.
`KalmanAFCLoop` clears meaningfully faster reversals — perfect accuracy
holds up to roughly 75 Hz/burst here, with a softer, more gradual falloff
above that rather than a sharp wall — which makes sense in hindsight: a
filter that tracks the *rate itself*, not just the CFO, should have more
headroom against a fast-changing rate than one that only ever reacts to
where the CFO already is. (§6.5 explains why this gap between the two
trackers wasn't visible in earlier testing.) At this configuration's
symbol duration (~1.02 ms), even `AFCLoop`'s lower ceiling of 55 Hz/burst
works out to about 54 kHz/s of Doppler *acceleration* — and real Doppler
acceleration, for anything from a LEO satellite pass to a fast
low-altitude drone, runs orders of magnitude below that. The cliff is
real, found by testing rather than assumed away, but it isn't close to
being the binding constraint for either scenario this chapter set out to
check.

### 6.5 The negative-SNR floor, and the fix

Everything in this chapter so far was tested at SNR = 10 dB. That was a
reasonable choice for isolating the tracking *dynamics* question — does
the loop follow a drift, does it survive a sign reversal — without also
fighting measurement noise. But it quietly dodged the question that
actually matters most for a protocol whose entire premise is decoding
*under* the noise floor: does this tracking mechanism still work down
there too? Chapter 7 asks that question directly, at negative SNR, and
the honest answer at first was no — not because tracking itself broke,
but because of something more basic underneath it.

`dual_edge_cfo_estimate`'s per-burst measurement (§6.1-6.2) fits an
81-sample local window — a small fraction of the roughly 512-sample
symbol. That local fit never benefits from the coherent processing gain
`fft_correlation_demod` itself gets by correlating the *whole* symbol
against the reference (on the order of 10log₁₀(512) ≈ 27 dB) — which is
exactly why decoding survives down to −20 to −30 dB SNR while this
measurement, checked directly, returns nothing usable on roughly 85-90%
of bursts at −10 dB, even fed a held-still signal with nothing to track
at all. That gap is the real story: not a tuning oversight, but a whole
factor of processing gain the rest of the receiver already relies on and
this one piece of it never had.

The fix keeps the same idea — decode first, then measure only the
*residual* against the known symbol (§6.2's whole rationale still holds)
— but measures it differently: dechirp the entire received symbol against
its exact known shape, which cancels the trajectory's own phase and
leaves (ideally) a pure tone at the residual CFO, buried in noise. Reading
that tone's frequency off an FFT peak, with a touch of parabolic
interpolation for sub-bin precision, gets the same coherent gain the
decoder already enjoys. Measured directly: a noise floor of about 100 Hz
std with no outliers at −10 dB SNR, degrading gracefully rather than
falling off a cliff below that. `run_afc_sequence` uses this
(`afc.py::full_symbol_cfo_estimate`) by default now; `dual_edge_cfo_estimate`
is still in the module, since its own properties (documented in §6.1-6.2)
are worth keeping, but it no longer feeds the tracking loop.

This also explains something that might otherwise look like a
contradiction: §6.4 said outlier-gating (`max_jump_hz`/`innovation_gate`)
"fixes it directly," and that was true — for the problem it was solving.
That fix targets a *rare* bad measurement slipping through an otherwise-
healthy stream, at 10 dB, where good measurements are the norm. Negative
SNR is the opposite situation: not an occasional bad reading among mostly
good ones, but *almost no* trustworthy readings at all (the ~85-90% figure
above). A gate can reject garbage, and it does, correctly — but it cannot
manufacture a good measurement out of a bad one. Two different failure
modes, needing two different fixes: one at the gate, one further upstream
at the measurement itself.

One more real bug came out of pushing this further: `KalmanAFCLoop`'s
`measurement_noise` had been tuned once, against `dual_edge_cfo_estimate`
at one configuration (SF7, 125 kHz, 10 dB). `full_symbol_cfo_estimate`'s
noise is an FFT-peak estimate, so at fixed SNR its variance scales with
the *square* of the FFT bin width — and Chapter 7's packets use 500 kHz,
4× this chapter's bandwidth, so a 16×-different noise floor. Carried
over unchanged, that stale value made `KalmanAFCLoop` badly *under*-trust
good 500 kHz measurements, right when trusting them mattered most: 37%
payload errors at −10 dB where `AFCLoop` had 0%, on an otherwise identical
run. `afc.py::measurement_noise_for(cfg)` replaces the fixed constant
with one scaled to the actual configuration, closing nearly all of that
gap. Chapter 7 tells the full story with the actual numbers.

None of this is a coincidence specific to LoRa. The literature on
hyperbolic FM in sonar and radar has called it the "Doppler-tolerant"
waveform for decades, and it comes with the same tradeoff this project
found on its own in Chapter 5: Doppler insensitivity in the matched-filter
response arrives together with a time/range bias, not for free — see
Kroszczyński's original 1969 paper, and the more recent closed-form
treatment of that bias in Murray et al.'s "On the Doppler Bias of
Hyperbolic Frequency Modulation Matched Filter Time of Arrival Estimates"
(*IEEE Journal of Oceanic Engineering*, 2019). That bias is the same
underlying fact as Chapter 5's *lag*, showing up again under a different
name in a different field — which is a reasonable amount of independent
confirmation that HFM's Doppler behavior here isn't an artifact of this
project's specific simulation.

---

## 7. Putting It All Together: An End-to-End Packet Test

Everything so far has been checked symbol by symbol, or burst by burst in
isolation. A real receiver has to do all of it at once: recognize a
packet, decode a payload correctly, and if there's Doppler, track it
through the whole thing. Four tests, each harder than the last (or, for
the fourth, just a different channel model entirely), all at SF=7 and
500 kHz bandwidth — a bandwidth this project hadn't used before
(everything earlier used 125 kHz).

The packet itself is a simplified stand-in for real LoRa framing: a short
preamble of repeated base ($m=0$) symbols, then a random payload —
the same idea as LoRa's own preamble up-chirps, generalized to any
trajectory shape, not a bit-accurate reproduction of the real sync-word
convention (which doesn't bear on the trajectory-shape question this
project is actually about).

### 7.1 No Doppler at all

Send the preamble, then a random payload, decode it, repeat. At high SNR
every payload symbol comes back correct, every time — 200 packets,
6000 payload symbols, zero errors. Dropping the SNR reproduces the same
waterfall shape established back in Chapter 4:

![Payload symbol error rate vs. SNR for all four packet tests](pictures/20_packet_level_validation.png)

*(Left panel.)* Nothing new here mathematically — it's the same decoder as
always — but it confirms the whole pipeline (preamble, payload framing,
grading only the payload) behaves the way the isolated single-symbol tests
predicted it would.

### 7.2 A constant Doppler shift, acquired from the preamble

Add a constant offset across the whole packet, and let the receiver
actually earn its correction instead of being handed the answer: acquire
the CFO from the preamble (`sync.py`'s `joint_cfo_symbol_search`), then
hold it with the tracking loop through the payload. The offset itself is
20 kHz, not a round test number but this project's own finding from
"Putting Real Numbers on the Applicability Claim": published measurements
put peak Doppler shift for a real 868 MHz LEO pass at roughly that value.

That change in magnitude caught a real gap the moment it was tried: this
test used to run at 6 kHz, comfortably inside the acquisition search's
old ±10 kHz span — accidentally an easier problem than the one motivating
this whole chapter. At the real 20 kHz, that old span doesn't even
contain the true offset, so every single packet fails, at every SNR
tested, not because tracking breaks but because the search is looking in
the wrong place entirely. Widening the span to ±25 kHz (and, since more
candidates cost more time, coarsening the search step from 100 Hz to
250 Hz — checked directly to cost nothing in accuracy) fixes it outright.
Nothing about the acquisition or tracking *logic* was wrong; it had just
never been asked to search as wide as the real problem needs.

Building this test surfaced something worth knowing: acquisition itself
is less SNR-robust than ordinary decoding. Searching many candidate CFOs
against one noisy burst gives more chances for a noise-induced false peak
than deciding among a single burst's own $M$ symbol hypotheses does —
measured directly, a single acquisition burst had a 27% symbol error rate
at −15 dB SNR where plain decoding had 0%. The fix is exactly what a
multi-symbol preamble is *for*: run acquisition independently across
several preamble bursts and combine them (majority-vote the decoded
symbol, then take the median CFO among bursts agreeing with it) instead
of trusting one. Measured directly, that dropped the 27% error rate to 0%.
`afc.py`'s `run_afc_sequence` now takes an `acquire_bursts` argument for
exactly this.

*(Middle panel.)* With that fix, `AFCLoop` and `KalmanAFCLoop` land right
on top of each other — both acquire from the preamble and hold the offset
through the payload equally well.

Getting the two trackers to agree here also caught a real bug worth being
honest about. The first version of this test had `KalmanAFCLoop`
performing noticeably worse than `AFCLoop` — its SER plateaued well above
`AFCLoop`'s at every SNR, on a scenario both had handled equally well
before. The cause: acquisition used to hand the tracker its result by
directly setting `.cfo_tracked`, which is `AFCLoop`'s entire state, so
that's correct for it — but `KalmanAFCLoop` also carries a covariance
matrix describing how *confident* it is, and setting `.cfo_tracked`
directly left that covariance at its pre-acquisition, near-infinite
default. So even right after a confident, preamble-averaged acquisition,
the filter still behaved as if it knew nothing, and the first
post-acquisition measurement got absorbed with a near-total Kalman gain —
enough to drag the estimate away from a good acquired value. The fix,
`KalmanAFCLoop.set_acquired`, resets the covariance along with the
estimate.

With the search-span fix, this test now holds up at the real, grounded
20 kHz magnitude the same way it used to at the arbitrary 6 kHz one:
clean, error-free decoding down to around −12 dB SNR, a transition
through roughly −14 to −20 dB, and both trackers landing on top of each
other throughout — not a smaller or shakier margin for having moved to
the real number, just the same waterfall shape, correctly centered on
the problem this chapter actually set out to solve.

**How much further does it actually go?** Worth pushing rather than
stopping at "the grounded number works" — so `examples/packet_experiments.py`'s
`push_acquisition_magnitude_limit()` walks the constant CFO up from 1 kHz
toward this configuration's own sample rate (2 MHz, so ±1 MHz is where a
complex baseband signal's own aliasing boundary sits) and watches where
it actually breaks.

The first attempt at this chased a red herring worth being honest about.
Scaling the acquisition search's step size up together with its span — a
reasonable-looking way to keep the number of candidates bounded as the
range widens — produced a confusing, non-monotonic failure starting
around 600 kHz. Tracing it down found the real cause: once that step
exceeds roughly half the decoder's own half-bin tolerance (about
1953 Hz here), no candidate in the grid may land close enough to the true
CFO to decode correctly, *no matter how wide or narrow the search span
is*. That's a property of the fixed, absolute precision decoding needs,
not something that should ever scale with how wide the search has to be
— a real bug in the test, not a discovery about the tracker.

Holding the step fixed at a safe, fine value (1500 Hz) instead removes
the problem entirely, and what's left is a clean, almost anticlimactic
result:

![Decode accuracy and acquisition search cost vs. constant CFO magnitude, out to the sample rate's own aliasing boundary](pictures/22_acquisition_magnitude_limit.png)

*(Left panel.)* Flat. Accuracy at each SNR level tested is the same
whether the CFO is 1 kHz or 990 kHz — magnitude alone costs nothing, all
the way out to 99% of ±1 MHz. Past that boundary, values start aliasing
(a CFO of $f$ and one of $f + f_s$ produce identical sampled sequences),
so "pushing further" stops being a meaningful question in this idealized
simulation rather than something that fails — a receiver's real ceiling
would come from front-end RF filtering and ADC sample rate, hardware
choices outside this project's scope, not from any weakness demonstrated
here in the tracking or acquisition math itself.

*(Right panel.)* The real cost of pushing the range is where it actually
shows up: wall-clock search time grows roughly linearly with how wide the
search has to be, since a fixed, fine step means proportionally more
candidates to test. Widening blind acquisition coverage is a genuine,
honest tradeoff against search time — just not against accuracy.

### 7.3 A Doppler that reverses sign

The harder case: a UAV-style approach/closest-approach/recede curve
across the packet, the same model as Chapter 6's flyover test. A sign
reversal needs enough bursts to unfold slowly enough to stay trackable, so
this packet's payload (712 symbols) is much longer than the other two
tests' — not an arbitrary change, but a consequence of the Doppler
profile's own timescale, using the same $t_0=90$ (peak rate about
33 Hz/burst) comfortably under Chapter 6's rate-of-change cliff.

*(Right panel.)* The first version of this test held up fine at 10 dB but
fell apart below roughly 5-8 dB SNR — a flat ~55% error rate no matter how
much further SNR dropped, not the smooth waterfall the other two panels
show, and unacceptable on its own terms: LoRa's whole premise is decoding
*under* the noise floor, so a tracker that stops working well above it
defeats the point. Tracing it down (not just re-explaining the symptom)
found the real cause was upstream of anything Doppler-specific: printing
the tracked estimate burst by burst showed it wasn't drifting to a wrong
value at all, it was *frozen* at its acquired value while the true CFO
moved underneath it, and the reason was `dual_edge_cfo_estimate` itself —
fed a held-still, zero-residual signal (nothing to track, no Doppler
involved at all) at −10 dB SNR, it still returned `None` on ~85-90% of
bursts, because its 81-sample local phase fit never gets the ~27 dB of
coherent processing gain `fft_correlation_demod` gets from correlating the
*whole* symbol. `max_jump_hz`/`innovation_gate` were doing their job
correctly — rejecting the rare measurement that did sneak through, which
was itself garbage — but a gate can only reject bad information, it can't
manufacture good information in its absence, and at this SNR there was
almost nothing else on offer. Test 1 and 2 never exposed this because
neither needs a continuous stream of fresh measurements to stay correct
(Test 1 needs no tracking; Test 2's CFO is constant, so coasting on a
stale value between rare good measurements is harmless) — Test 3's
continuously-moving CFO is the only one of the three whose correctness
actually depends on it.

The fix is Chapter 6.5's: `full_symbol_cfo_estimate`, which dechirps the
*whole* received symbol against its known decoded shape and reads the
residual CFO off an FFT peak, recovering the same coherent gain the
decoder itself already has. `run_afc_sequence` uses this by default now.
`KalmanAFCLoop` needed a second, config-specific fix on top of that: its
`measurement_noise` had been tuned against the old measurement at
Chapter 6's 125 kHz configuration, and the new measurement's noise scales
with the square of the FFT bin width — so at this test's 500 kHz (4× the
bin width), the old tuned value was roughly 16× too small, and using it
unmodified made the filter badly overtrust noisy 500 kHz measurements.
`measurement_noise_for(cfg)` scales the value to the actual configuration
instead of carrying over a fixed constant.

With both fixes in place, the right panel tells a very different story
than the one described above. `AFCLoop` decodes this exact scenario
correctly down to roughly −12 dB SNR, and `KalmanAFCLoop` — once its
measurement noise is scaled correctly — follows closely behind, both
resolving to a genuine waterfall rather than a flat floor. Below roughly
−14 dB both trackers do eventually degrade, and correctly so:
`full_symbol_cfo_estimate` has its own FFT-threshold effect down there
(see its docstring), the same kind of floor any coherent estimator
eventually hits, just now roughly twenty decibels lower than where the
old measurement gave out. That is the answer to the question this test
actually set out to check: given a per-burst measurement that uses the
same processing gain the rest of the receiver already relies on, dual-edge
(now full-symbol) tracking decodes a real packet correctly through a full
Doppler sign reversal, at the same kind of negative SNR this project's
other packet tests already operate at — not stalling out at the edge of
the noise floor the way the original measurement quietly did.

### 7.4 A wideband Doppler scale, corrected the DHFM way

A genuinely different channel model from the first three tests: not a
narrowband constant-CFO offset, but a wideband Doppler *time-scale* —
the effect Chapter 5.2 and the "Building the Same Thing" section further
below are actually about. Correcting it doesn't use `afc.py` at all;
it uses `nlfsc_lora.paired_sweep`'s DHFM-style paired opposite-sweep
preamble, the real active-sonar technique described in "How Real HFM
Sonar and Radar Systems Actually Handle Doppler." That section and
"Building the Same Thing" validated the idea at the single-symbol level
(`21_paired_sweep_doppler_correction.png`); this test checks whether it
holds up once it's just one more piece of a full packet, preamble
framing and all.

It does, and by the same wide margin. At a Doppler scale of $\alpha=1.05$
— past even this configuration's roughly half-bin failure threshold
($\alpha \approx 1.004$) by more than ten times over — an uncorrected
receiver fails on essentially every payload symbol, at every SNR tested.
Acquiring $\alpha$ from a single paired up/down-sweep preamble pair (not
even the multi-burst averaging Test 2's acquisition needed) and
correcting each payload burst before an ordinary decode brings that back
to a clean waterfall, error-free from 40 dB down to about −12 dB, only
degrading below roughly −14 dB as the paired-sweep measurement itself
runs into its own coherent-integration noise floor — the same kind of
floor `full_symbol_cfo_estimate` has, for the same underlying reason.

---

## Is This Actually Better Than Standard LoRa?

Worth answering directly, since it's easy to walk away from a project like
this assuming "curvy chirp, therefore more robust" — and that's mostly not
what was found here.

**Where it isn't better.** Baseline noise tolerance: Chapter 4's SER-vs-SNR
sweep showed every trajectory shape, hyperbolic included, performing
essentially identically to linear once decoded with the right general
decoder. A simple constant CFO: Chapter 5.1 showed the correlation-loss
from an offset is independent of shape too. Curving the trajectory buys
neither.

**Where it's real, but partial.** Under true wideband Doppler — time
*scaling*, not just a frequency shift — HFM's exact self-similarity
theorem holds up: a Doppler-scaled HFM waveform is exactly a shifted,
rotated copy of itself, so a matched filter loses only about 1 dB of peak
magnitude across a ±10% scale sweep versus 10 dB+ for linear. That's a
genuine single-waveform advantage. Chapter 5.2's own honest caveat is that
it only partially survives being plugged into full $M$-ary decoding,
because a Doppler-induced timing lag — nearly identical across every
trajectory shape — dominates before HFM's advantage gets a chance to
matter.

**Where it's real and not partial** is structural, not a property of the
waveform's robustness at all: a linear chirp's instantaneous rate is the
same constant everywhere, so there's nothing local to measure. That's
exactly why standard LoRa's own CFO estimation works the way it does —
dechirping a linear chirp produces a clean tone, so its FFT peak position
*is* a frequency measurement, and pairing an up-chirp preamble symbol with
a down-chirp SFD symbol resolves the CFO/timing-offset ambiguity that
comes with it ($\widehat{\mathrm{CFO}} \propto (\text{peak}_{\text{up}} +
\text{peak}_{\text{down}})/2$). Cheap, and correct — but it only happens
*once*, at the start of the packet. A curved trajectory's rate genuinely
varies with position, which is what makes Chapter 6's dual-edge measurement
possible at all: using the payload's own symbols as an ongoing position/CFO
sensor, with no extra reference chirps, for as long as the packet runs.
Standard LoRa has no equivalent — not a cheaper version of the same idea,
a different mechanism that structurally can't do continuous tracking.

**So when would a system actually need that?** Not just "Doppler exists" —
plenty of real systems handle Doppler without tracking it from the signal
at all. A LEO satellite follows a known orbit, so a ground station with
its ephemeris can precompute the expected Doppler curve and pre-correct
for it; a drone with a telemetry link back to the receiver can do the
same. If that side information is available, linear chirps plus a lookup
table are simpler and better, full stop — this project's own numbers back
that up. The actual case for this approach narrows to wherever a few
things are true together:

- **The Doppler drifts meaningfully within one packet**, not just
  packet-to-packet — a long payload, or a fast drift rate, or both. If a
  one-shot preamble correction stays valid for the whole payload, there's
  nothing to track.
- **No outside knowledge of the motion is available or wanted** — no
  ephemeris, no GPS/telemetry side-channel, an uncooperative or unknown
  transmitter, or a receiver that would rather be self-contained than
  maintain an orbital propagator. Search-and-rescue beacons, tracking tags
  on fast-moving wildlife, a drone swarm without centralized telemetry, or
  a cheap IoT ground segment that skips precise orbit tracking are the
  realistic cases.
- **Sending one long packet beats sending many short, re-synchronized
  ones.** Standard LoRa's fallback under drift with no side information is
  shorter packets and frequent re-acquisition from a fresh preamble —
  which costs airtime and battery under LoRaWAN-style duty-cycle limits.

Outside that intersection, linear chirps win: simpler, and CFO estimation
comes free as a byproduct of decoding rather than as a separate measurement
step. This project's case for nonlinear trajectories was never "always
better" — it's "enables a specific capability standard LoRa structurally
cannot do," which is a narrower, more defensible, and more interesting
claim.

---

## Putting Real Numbers on the Applicability Claim

The list above ("search-and-rescue beacons," "wildlife tracking tags") was
qualitative — plausible-sounding scenarios, not measured ones. Worth
actually checking against real numbers rather than leaving it there,
even though (especially though) some of what turns up complicates the
story this project would rather tell.

**Crystal drift is usually the bigger CFO source, and it doesn't move
within a packet.** Semtech's own guidance for LoRa transceiver reference
clocks recommends a crystal good to about ±10 ppm over −20 to 70°C (worse,
±30 ppm, over the full −40 to 85°C range), with additional aging of a few
ppm over the first year. At 868 MHz that's ±8.7 kHz to ±26 kHz of CFO from
the crystal alone — comparable to, and often larger than, any Doppler
shift this project has tested. But crystal drift is thermally driven, with
time constants of seconds to minutes; a LoRa packet is milliseconds to at
most a few seconds even at SF12. Within *one* packet it is, to good
approximation, a constant offset — which is exactly the case standard
LoRa's own one-shot up/down-chirp correction already handles for free.
Continuous per-burst tracking doesn't help with it at all; the entire
case for tracking rests on whatever *does* move within a packet, which
narrows things to genuine, ongoing relative motion — Doppler, not clock
error.

**Real Doppler magnitude and rate, for the flagship LEO-satellite case.**
Published measurements for 868 MHz LoRa-band LEO satellite links at
roughly 600 km altitude put peak Doppler shift at around ±20 kHz and peak
Doppler *rate* (at closest approach, where it's steepest) in the range of
roughly 270–640 Hz/s, depending on the exact geometry assumed. Converted
into this project's own units — Hz per burst, since that's what the
tracking loop actually has to keep up with — that rate is:

- At SF7/500 kHz (this project's packet-test configuration, symbol
  duration ≈0.256 ms): roughly 0.07–0.16 Hz/burst.
- At SF7/125 kHz (Chapter 6's AFC figures): roughly 0.28–0.66 Hz/burst.

Both are two to three orders of magnitude below the ~55 Hz/burst cliff
`AFCLoop` actually hits (§6.4/6.5), or the ~75 Hz/burst `KalmanAFCLoop`
clears. The "safely past any realistic rate" claim earlier in this
document wasn't wrong, but it's now a specific number instead of an
assertion: on the order of 80–800× of margin, not "a lot."

Worth being honest about the other direction, too: this project's own
test *magnitudes* undersold the real case. Chapter 7's constant-CFO test
used 6 kHz; the UAV flyover used ±3 kHz. A real 868 MHz LEO pass swings a
full ±20 kHz — three to six times larger than anything actually tested
here, and wider than `packet_experiments.py`'s own acquisition search
window (±10 kHz). None of the *rate* conclusions change — the margin above
the cliff is still enormous either way — but a receiver actually built for
this scenario would need a wider acquisition span than this project ever
exercised.

**Real satellite-LoRa systems mostly don't track blindly at all.**
Operators actually flying LoRa-based LEO IoT links (Lacuna Space, and
similar systems described in the satellite-IoT literature) lean on
exactly the alternative this project's own applicability list names first:
since the orbit is known, the Doppler curve is predictable, and the
"fixed offset can be predicted and compensated while decoding" rather than
measured blind. Where they do spend design margin on Doppler, it's mostly
by choosing a wider channel and a more conservative spreading factor (one
real system uses 250 kHz bandwidth and SF11 specifically for Doppler
headroom), not by tracking CFO continuously within a packet. That doesn't
kill this project's case — an *uncooperative or unknown* transmitter,
with no ephemeris to lean on, is still a real gap ephemeris-based
pre-compensation can't fill — but it does mean the single most obvious
motivating example (a cooperative satellite IoT uplink) is, in practice,
mostly solved a different way already.

**Wideband Doppler *scale* — the effect Chapter 5.2 and the paired-sweep
work are actually about — essentially never arises for RF LoRa at all.**
$\alpha$ departs meaningfully from 1 only when $v/c$ is a non-negligible
fraction of a percent; this project's own tests used $\alpha=1.05$, i.e.
$v/c=0.05$, $v\approx15{,}000$ km/s. No real platform gets remotely
close — even a hypersonic vehicle at Mach 10 ($\approx3.4$ km/s) gives
$v/c\approx1.1\times10^{-5}$, an $\alpha$ deviation about 4,400× smaller
than anything tested here, and orders of magnitude below what a receiver
could even resolve. At RF, wideband-scale Doppler is a real, correctly-
modeled effect with no real airtime application this project has found —
which lines up with why it's a *sonar* technique in the literature to
begin with: underwater, $c\approx1500$ m/s, so an ordinary vessel's speed
already is a meaningful fraction of it. Chapter 6/6.5's narrowband
tracker and Chapter 10's paired-sweep correction are both honestly built
and correctly validated; the numbers here are about which one has an
airtime home in RF LoRa specifically, and it's the narrowband one, for
the narrow uncooperative-transmitter case above — not the wideband one.

---

## How Real HFM Sonar and Radar Systems Actually Handle Doppler

Worth being honest about directly, since it's an easy thing to assume:
Chapter 6's dual-edge/full-symbol tracking isn't a version of how real
sonar and radar systems handle Doppler with HFM waveforms. They solve a
related problem, but by a genuinely different route.

Real systems get one part of this essentially for free — the same
self-similarity property behind Chapter 5's finding. Correlate a
Doppler-scaled HFM echo against one fixed, unshifted reference, and the
matched-filter peak stays sharp and strong no matter the target's
velocity ("Doppler-invariant matched filtering," a property known since
Kroszczyński's 1969 paper). A linear-FM system needs a whole bank of
Doppler-shifted references to get the same detection performance against
an unknown velocity; an HFM system needs exactly one. No per-pulse
measurement, no tracking loop — just a fixed correlator that happens not
to care how fast the target is moving.

The cost of that free lunch is the same one this project found on its
own and Murray et al.'s 2019 paper gives a closed form for: the matched
filter's peak lands at a *biased* time, and that bias depends on
velocity. So HFM buys Doppler-tolerant *detection*, but not an accurate
*range*, unless something corrects for the bias — and this is exactly
where real systems and this project part ways.

Real active sonar resolves it with a **pair of pulses swept in opposite
directions** — a down-sweep followed by an up-sweep (empirically the
better-behaved order) — rather than anything measured from within a
single pulse. Range shifts both pulses' delay estimates the same way;
Doppler shifts them in *opposite* directions. Two measurements, two
unknowns, one linear solve, and velocity and range come out decoupled
(Wang et al., 2017) — no bank of Doppler-shifted replicas needed, and no
looking inside a single pulse's own internal structure at all. Later
pings then typically get smoothed by a tracking filter across the dwell.

That paired-pulse trick will sound familiar — it's the same idea, almost
move for move, as standard LoRa's own up-chirp/down-chirp CFO
resolution from the "Is This Actually Better?" section above: two
oppositely-swept references, combined linearly, because reversing the
sweep flips the sign of one error term (Doppler there, CFO's effect on
the dechirped tone here) while leaving the other (range there, timing
offset here) alone. It is *not* the same idea as this project's own
`dual_edge_cfo_estimate`/`full_symbol_cfo_estimate`, which needs no
second, oppositely-swept pulse at all. Those work only because a LoRa
symbol carries something a sonar ping doesn't: a *decodable payload*.
The receiver decodes which symbol it just received, and only then
compares that same burst's own local structure against what the
now-known symbol predicts. A real sonar or radar pulse has no analogous
"payload" to decode first — there's nothing for a real system to check a
pulse against except another, separately-transmitted reference pulse.

So the honest way to put it: this project's tracking design isn't a
simplification or rediscovery of how real HFM systems handle Doppler.
It's a different technique that happens to be available here because
LoRa's chirp-spread-spectrum format hands the receiver a decodable
symbol every single burst — a structural feature real sonar and radar
waveforms don't have, and that real systems solve around with paired
pulses and cross-ping tracking instead.

---

## Building the Same Thing: Paired-Sweep Doppler Correction

Worth actually trying rather than leaving as a paper comparison: does real
active sonar's own DHFM trick — transmit a pair of oppositely-swept
pulses, compare their two oppositely-biased delay estimates — work here
too, adapted to this project's own hyperbolic trajectory?

**Building the down-sweep.** The "down" sweep isn't the up-sweep's
frequency axis flipped ($1-g(u)$) — tried first, and it degrades badly,
losing most of HFM's own Doppler tolerance (peak preservation down
around 0.2–0.5 instead of 0.9+). It's the same shape traversed
*backward in time*: $g_{\text{down}}(u) = g(1-u)$. For a plain linear
chirp the two constructions happen to coincide, which is exactly why
this distinction is easy to miss — it only shows up once the trajectory
is actually curved.

**The bias relationship, derived and then checked.** Chapter 5.2 already
gives the exact Doppler-induced lag for the up-sweep,
$\Delta_{\text{up}}(\alpha)$. Applying the same theorem to the
time-reversed law gives a closed form for the down-sweep's own bias:

$$
\Delta_{\text{down}}(\alpha) = -q \cdot \Delta_{\text{up}}(\alpha)
$$

where $q = f_{\text{low}}/f_{\text{high}}$ — the same band-ratio parameter
that already sets the hyperbolic law's shape. Not a naive equal-and-
opposite pair (that only holds in the degenerate case where the sweep
spans no bandwidth at all); the down sweep's bias is $q$ times smaller,
because $q$ governs how much of the hyperbola's curvature falls on each
half of the band. Checked directly against simulated correlation-peak
lags rather than just trusted: predicted and measured lags agreed to
within about one sample across a wide range of $\alpha$.

Two lag measurements, two unknowns (the true shift and the bias) — the
same two-equations-two-unknowns structure DHFM sonar exploits, solved by
simple substitution once the $q$-weighting is known:

$$
\widehat{\text{shift}} = \frac{q \cdot \text{lag}_{\text{up}} + \text{lag}_{\text{down}}}{1+q}
$$

**The first attempt, and why it didn't work.** The obvious thing to try
first: send an arbitrary *payload* symbol as both an up- and a
down-sweep burst, and decode the pair directly with this formula. That
doesn't work — checked directly, not assumed: decode error stays near
zero for symbols close to $m=0$ and grows to tens of samples' worth of
error by mid-alphabet, a clear systematic pattern, not noise. The cause:
a cyclically-shifted symbol carries its own wrap discontinuity
(Chapter 6.2's "wrap glitch"), and the way this project's Doppler-scale
model resamples an already-shifted, already-wrapped array introduces an
extra bias that depends on the shift itself — something the clean
$\Delta(\alpha)$ formula, derived for an unshifted, unwrapped trajectory,
doesn't account for.

**The fix was to use it the way real systems actually do.** Neither real
active sonar's DHFM pulses nor standard LoRa's own up/down-chirp trick
ever try to decode an arbitrary payload value from the paired
measurement directly — they apply it to a *known, unshifted* reference
(a preamble) to measure a channel parameter once, then use that estimate
to correct ordinary data before decoding it normally. Doing the same
thing here — acquiring $\alpha$ from a paired $m=0$ preamble, then
correcting every payload burst with it before an ordinary single-sweep
decode — sidesteps the wrap-discontinuity problem entirely (there's no
wrap to interact badly with, since the preamble is never shifted) and
works exactly, not just approximately:

![Symbol error rate vs. Doppler scale, uncorrected vs. DHFM-style paired-sweep correction](pictures/21_paired_sweep_doppler_correction.png)

An uncorrected receiver collapses to essentially 100% error the moment
$\alpha$ drifts past roughly half a symbol bin — the same cliff
08_lora_ser_vs_doppler_scale.png already showed. With the paired-sweep
correction, decode accuracy stays pinned at the noise floor across the
entire $\pm15\%$ sweep tested. The $\alpha$ estimate itself is accurate
to a few parts in $10{,}000$ at 10 dB SNR, and — for the same reason
Chapter 6.5's `full_symbol_cfo_estimate` fix works down to very low
SNR — stays accurate to a fraction of a percent even at $-10$ dB, since
this measurement also correlates the *whole* symbol rather than a small
window.

The honest scope: this is a genuine, working adaptation of real active
sonar's own Doppler-handling technique, not a simplified toy version of
it — but note what changed to make it work. The version that decodes an
arbitrary payload shift directly (closer to what a first guess at "the
same thing" might look like) has a real, characterized limitation; the
version that matches how real systems actually deploy the trick — on a
known reference, correcting data afterward — works cleanly. That
distinction was itself worth finding out, not assumed going in.

This held up outside the single-symbol test too. Chapter 7.4 wires the
same acquire-then-correct pattern into a full packet — preamble framing,
an SNR sweep, the whole pipeline — and finds the same result: a
completely failing uncorrected receiver at $\alpha=1.05$ restored to a
clean waterfall down to about $-12$ dB SNR by a single paired-sweep
preamble pair, no multi-burst averaging needed.

### Does it need hyperbolic specifically?

Everything above uses hyperbolic FM because Chapter 5.2's theorem gives an
*exact* closed form for $\Delta_{\text{down}}(\alpha)$ — but the underlying
mechanism the trick actually exploits is simpler than the theorem: two
oppositely-swept measurements of the same unknown shift/bias pair, solved as
a linear system. That mechanism doesn't obviously require an exact
self-similarity theorem to exist at all — worth actually checking on shapes
that don't have one, rather than assuming the technique is hyperbolic-only.

The generalization: replace the theoretical $q=f_{\text{low}}/f_{\text{high}}$
with an empirically fit ratio — a least-squares fit of the same
$\text{lag}_{\text{down}} = -q\cdot\text{lag}_{\text{up}}$ relationship,
measured from an offline, noiseless calibration sweep of the shape's own
waveform (`paired_sweep.fit_bias_ratio`) — and replace the closed-form
$\Delta\to\alpha$ inversion with a lookup against that same calibration
sweep (`paired_sweep.build_alpha_calibration` /
`acquire_doppler_scale_calibrated`). No different in spirit from a sonar
system's own precomputed ambiguity surface — the fit is done once, offline,
against the known waveform, not against anything measured from an unknown
channel.

![Paired-sweep mechanism and calibrated correction for quadratic, sigmoid, and exponential trajectories, vs. hyperbolic](pictures/23_paired_sweep_shape_generalization.png)

*(Left panel.)* The paired measurement laid open, the same way the "Building
the down-sweep" figure never quite was: each shape's own
$\text{lag}_{\text{down}}$ plotted against $\text{lag}_{\text{up}}$ across
the calibration sweep. Hyperbolic sits on a perfectly straight line through
the origin — the exact closed form. Quadratic and exponential are close to
straight but not exactly (the "empirically fit, not exact" distinction —
their fitted $q_{\text{eff}}$ isn't a true constant, just a good
approximation of one). Sigmoid visibly bends away from a straight line near
the extremes of the sweep, which is the direct, visible cause of the other
two panels' finding.

*(Middle and right panels.)* Checked directly, not assumed: quadratic and
exponential restore the same near-0% SER hyperbolic gets, across the whole
tested $\alpha$ range, at the same 10 dB SNR an uncorrected receiver fails
completely at —
`tests/test_paired_sweep.py::test_calibrated_acquisition_restores_full_accuracy_for_well_behaved_shapes`
pins this directly. The calibration-curve substitute loses essentially
nothing for those two shapes relative to hyperbolic's exact closed form.

Sigmoid is the honest exception. Its calibration fit is decent on average,
but under noise, near the edges of the tested $\alpha$ range, a single
preamble-pair acquisition produces a grossly wrong estimate often enough to
show up as outright decode failures (the middle panel's spikes to ~100% SER
at $\alpha=0.90$ and $1.10$) — traced to `trajectories.py`'s own
`sigmoid` docstring: it "lingers near the band edges," so the instantaneous
frequency there is nearly flat, and a nearly-flat frequency carries little
Doppler-*scale* information right where this measurement needs it most.
Multi-burst averaging (the median-combine fix §7.2 used for a different,
purely noise-driven acquisition problem) helps — measured directly, 35 of
200 trials landing more than 0.01 off true $\alpha$ at one preamble burst
falls to 6 of 200 at ten — but doesn't fully close the gap the way it did
there, because this is a structural information-content limit of the
waveform's own shape near $\alpha=0.90$, not a rare false peak among
mostly-good measurements.

The honest scope, updated: the paired-sweep *mechanism* is not
hyperbolic-specific — it generalizes cleanly to at least two other curved
shapes once the closed form is replaced with an empirical fit. What is
specific to hyperbolic is the closed form itself, and the guarantee that
comes with it: an exact answer with no offline calibration step and no
shape-dependent risk of the kind sigmoid shows here.

---

## Testing Against the Real Thing: LoRaWAN Cross-Validation

Everything up to here has been checked against itself: this project's own
transmitter and this project's own receiver, agreeing with each other. That
proves internal consistency, not real-world compatibility. This section
checks against something independent — `lora_phy`
([pyLoRaPHY](https://github.com/myzhang1029/pyLoRaPHY), MIT-licensed, a
well-tested Python port of the widely-used jkadbear/LoRaPHY reference) — a
full, real LoRa PHY implementation: real chirp generation, real preamble and
sync-word framing, real Hamming FEC, interleaving, whitening, Gray coding,
and CRC16. It's an optional dependency (`pip install .[refcheck]`, pinned to
0.2.0 — see `pyproject.toml` for why not latest), and every test that uses it
skips cleanly if it isn't installed.

One thing has to be said up front, because it bounds what "cross-validation"
can even mean here: this project's actual contribution — nonlinear
trajectories, HFM's self-similarity, the paired-sweep correction — cannot be
tested this way at all. Real LoRa hardware, and every real reference
implementation of it, hard-codes the linear chirp. There is no real system
to check a hyperbolic trajectory against, because no real system generates
one. What *can* be tested for real is the narrowband CFO/Doppler tracking
(Chapter 6's `AFCLoop`/`KalmanAFCLoop`/`full_symbol_cfo_estimate`), which
works identically for linear chirps too — so everything below uses
`g=linear`, the one shape with a real-world counterpart.

### Does this project's own math match a real implementation?

The cleanest comparison: this project's base (`m=0`) up-chirp against
`lora_phy`'s own chirp construction. They match to numerical precision —
once a small, fully-explained discrepancy is corrected for first. `lora_phy`
evaluates instantaneous frequency at a sample grid offset by exactly
`bandwidth / (2 * n_samples)` from this project's own `u = n/n_samples`
convention — a constant, symbol-independent frequency offset, not a shape
difference. Checked directly: that offset alone accounts for the *entire*
correlation gap measured before correcting for it, via the same CFO-loss
sinc formula from §5.1. Benign, and now documented
(`tests/test_reference_crossval.py::test_base_waveform_matches_reference_exactly`).

A cyclically-shifted symbol (`m != 0`) is a different, more interesting
story. `chirp.py`'s own docstring has always been explicit that its
construction — roll the *sampled* base waveform — is "a description of the
model in this project, not a claim about the internals of any particular
commercial LoRa chipset." Checked directly against a real implementation,
that disclaimer turns out to matter: real LoRa instead generates each symbol
as a closed-form chirp with a *shifted starting frequency*, wrapping
mid-symbol — not the same waveform as a sample-domain roll. Correlation
between the two constructions dips to about 0.92 near mid-alphabet ($m
\approx M/2$), a real, bounded, ~0.7dB-scale effect around the wrap point,
not noise.

The headline result is what happens next: does that difference ever cost a
decoding decision? Checked directly, across the entire symbol alphabet,
noiselessly, both directions — `lora_phy`'s own waveform decoded by this
project's receiver, and this project's own waveform decoded by `lora_phy`'s
— every symbol comes back correct, every time. The wrap-construction
difference costs matched-filter *magnitude*; it never costs *correctness*,
because the loss is shared closely enough across neighboring symbol
hypotheses that the `argmax` decision doesn't move. Pushed one step further
— does it cost anything once there's real noise, not just a noiseless
`argmax`? — a self-consistent SER (this project's own tx and rx) and a cross
SER (`lora_phy`'s tx, this project's rx) at $-10$dB SNR land within 5% of
each other: the same regime, not a different one
(`test_cross_decode_ser_matches_self_consistent_ser_at_moderate_snr`).

### Real channel coding, not just symbol counting

Every packet test up to Chapter 7 grades "payload symbol error rate": did
the decoded chirp shift match the transmitted one. Real LoRaWAN never ships
raw symbols — each payload byte goes through whitening, Hamming FEC,
diagonal interleaving, and Gray coding before modulation, and a CRC16
either passes or fails on the whole packet at the receiver. None of that
machinery is built into this project by design (Chapter 7's own note: the
preamble/payload framing is "not a bit-accurate reproduction of the real
sync-word convention," since the trajectory-shape question this project
studies doesn't depend on it) — but it's exactly the machinery a real
receiver's success or failure actually turns on, so it's worth checking
against directly rather than assumed to not matter.

`examples/lorawan_experiments.py` wires this project's own Doppler tracking
around `lora_phy`'s `encode()`/`decode()` — the real Hamming/interleave/
whiten/Gray/CRC layer, turning a payload into the exact chirp-shift stream
real hardware would transmit, and turning decoded shifts back into payload
bytes plus a genuine CRC pass/fail. Everything *around* that layer —
preamble, Doppler profile, acquisition, per-burst tracking — reuses this
project's own already-validated pieces from Chapters 6–7. Each burst's
waveform is built with `lora_phy`'s own chirp construction, not this
project's approximation of it, deliberately: this test is a check of the
wrap-construction finding above holding up under real Doppler and real
channel coding together, not an assumption carried over from the
single-symbol result.

![Real LoRaWAN-framed packet CRC pass rate vs SNR: constant-CFO and UAV-reversal Doppler, tracking vs a static one-time correction](pictures/24_lorawan_packet_pass_rate.png)

*(Left and middle panels, 12-byte payload, 48 total bursts.)* Clean
waterfalls, tracking and a static (acquire-once) correction landing right on
top of each other — for *both* the constant-CFO and the sign-reversing
UAV-style profile. That's not a null result to explain away: it's short. A
12-byte LoRaWAN uplink doesn't run long enough for even a full Doppler sign
reversal to drift meaningfully away from its acquired value within the
packet, the same reason Chapter 7's own Test 3 uses a 712-symbol payload
instead of Tests 1–2's short one — this is that same finding, now visible at
the packet-CRC level and tied to a realistic uplink size, not an argument
against tracking.

*(Right panel, 150-byte payload, 368 total bursts, same UAV-reversal
profile.)* The dramatic version of the same finding. Static correction never
passes a single packet's CRC, at *any* SNR tested, all the way up to +5dB —
by the time the reversal has carried the CFO away from its acquired value,
Hamming FEC can't correct enough of the resulting symbol errors and the CRC
fails, packet after packet. Active tracking (`AFCLoop` or `KalmanAFCLoop`)
holds a clean waterfall down to roughly $-8$ to $-10$dB before degrading —
the same kind of transition this project's other packet tests show,
now confirmed to survive real Hamming FEC, interleaving, whitening, and a
CRC16 check, not just a symbol-error count.

### Does tracking mean anything between packets?

Real LoRaWAN doesn't send packets back-to-back. EU868-style sub-bands cap
transmit duty cycle at around 1%, so a device that just finished an uplink
has to stay silent for roughly 99× its own airtime before the next one.
`examples/duty_cycle_experiments.py` puts real numbers on what that gap
costs, using `lora_phy`'s own `time_in_air()` and this project's own
grounded LEO Doppler rate range (270–640 Hz/s, from "Putting Real Numbers on
the Applicability Claim") across SF7–SF12 and two payload sizes:

![Between-packet Doppler drift under a 1% duty cycle vs. this project's own decode thresholds, and rate-predicted acquisition recentering recovering jumps a blind search misses](pictures/25_duty_cycle_reacquisition.png)

*(Left panel.)* Across nearly every configuration tested, the CFO drift
that accumulates during even the *minimum* duty-cycle gap comfortably
exceeds this project's own half-bin decode-capture tolerance for that
configuration. Concretely: a receiver that simply held onto its last
tracked CFO estimate from the previous packet, with no fresh acquisition,
would already fail to decode the very first symbol of the next one, in
nearly every realistic case. Chapters 6–7's continuous, burst-to-burst
tracking is real and validated *within* one packet; it does not, by itself,
extend *across* packets once duty-cycle spacing dominates. Every new packet
needs its own acquisition — exactly how real LoRaWAN receivers already
work (fresh preamble detection and sync per uplink), not a gap this
project's own tracking claim was ever making. For several slower
configurations (large SF, long payload, the faster end of the LEO rate
range), drift also exceeds this project's own 25kHz acquisition search span
outright, so even a *fresh*, blind, zero-centered search would miss the
true CFO entirely.

*(Right panel.)* The one genuinely new capability worth adding, and tested
directly rather than just proposed: does extrapolating the *previous*
packet's own tracked rate across the gap — reusing `KalmanAFCLoop`'s
existing predict-only step, `loop.update(None)`, called once per
gap-equivalent burst, no new tracking math — and re-centering a
*same-width* acquisition search on that prediction recover the cases a
blind, zero-centered search misses? Yes, cleanly: blind search's success
rate collapses to 0% within a few seconds of gap (drift immediately
overruns even a modest, alias-safe search window — see below), while the
rate-predicted center recovers a substantial, consistent fraction of jumps
throughout the tested range, at no extra search cost (same window width,
just moved). The prediction is deliberately not fed the exact true rate — a
$\pm5\%$ relative jitter models the previous packet's own residual
rate-tracking error — so the recovered fraction (30–50%, not the near-100%
a noiseless point prediction would suggest) is the honest cost of that
realistic imperfection, not a best-case number.

**A real methodological pitfall, caught while building this, worth
documenting rather than quietly fixing.** The first version of this
experiment used the same wide $\pm25$kHz acquisition span this project's own
Test 2 validated — and produced a confusing result: a blind search
succeeding *more* often than the correctly re-centered one. The cause
wasn't the prediction; it was the search. `joint_cfo_symbol_search` jointly
searches (symbol, CFO), and at SF9/125kHz's bin width (244Hz), a $\pm25$kHz
span covers roughly 100 bin-widths — 100 (symbol, CFO) *alias* hypotheses,
any of which can score nearly as well as the true one. Confirmed directly,
not inferred: a wide search reproducibly locked onto a wrong symbol two bins
from the truth, *even at 40dB SNR* — a systematic aliasing failure, not a
noise problem. Test 2's own SF7/500kHz configuration never exposed this
because its bin width (3906Hz) packs roughly 16× fewer aliases into the same
nominal span — meaning a fixed acquisition span in Hz was never actually a
shape/SF-independent design choice; it was tuned safe for one configuration
and silently risky for another. The fix, a narrow ($\pm1500$Hz, about 6
bin-widths) alias-safe span, removes the ambiguity — and turns out to make
the actual point better anyway: it's exactly the precision a *correctly
centered* search needs, no wider, which is the whole case for prediction
over blind guessing in the first place.

### Does a real receiver still hear the desired packet at all?

Everything above still assumes one transmitter and thermal noise. Real
LoRaWAN gateways share spectrum with every other device in range, and two
uplinks can genuinely collide on the same channel at the same time. This
was never tested — every earlier test here used a single transmitter.
`examples/interference_experiments.py` builds real overlapping packets
(`lora_phy`'s own `encode`/`modulate`, one "desired" and one "interferer")
and checks whether `lora_phy`'s own full receiver pipeline — real preamble
detection and sync, not this project's instrumented burst slicing — still
recovers the desired one.

![Real LoRaWAN multi-device collision: capture probability vs. SIR and spreading-factor separation, and vs. how much of the packet the interferer actually overlaps](pictures/26_lorawan_interference.png)

*(Left panel.)* Two real, well-known LoRa PHY properties, now directly
measured rather than cited. The **capture effect**: at the same spreading
factor, capture is close to a step function — the desired packet needs to
be within about a decibel of matching or beating the interferer, and below
that threshold it is essentially never recovered. **Spreading-factor
quasi-orthogonality**: an interferer even one SF away from the desired
packet is far less disruptive — the desired packet is still captured down
to roughly $-10$ to $-14$dB SIR (an order of magnitude *weaker* than the
interferer) once the two transmissions use different spreading factors,
because their chirp rates don't correlate well against each other's own
dechirp reference. Both numbers land in the same range the LoRa literature
already reports; this is a direct measurement of it, not a repetition of
the claim.

*(Right panel.)* A finding that took a wrong first guess to get right,
worth being honest about: the naive expectation is that capture probability
degrades gradually the more of the packet a strong interferer overlaps. It
doesn't. At a fixed, clearly-dominant interferer strength ($-3$dB SIR),
capture stays at essentially 0% for *any* starting offset that still
reaches the last 5–10% of the packet, and jumps to recovery only once the
interferer starts late enough to miss that tail entirely. Hamming FEC and
interleaving handle *scattered* errors; they don't have a chance against a
*contiguous* block of corrupted symbols landing on the CRC-bearing block
specifically — a burst error, not a scattered one. The first version of
this experiment swept the overlap fraction evenly across $[0,1]$ and found
an apparently noisy, uninformative curve; probing directly (not adding more
trials and hoping the noise averaged out) found the real transition
concentrated in that narrow, near-full-overlap band, not spread evenly
across the packet at all.

### Does reacquisition survive a real ADR spreading-factor switch?

Real LoRaWAN devices don't transmit at a fixed spreading factor. Adaptive
Data Rate steps a device to a higher (more robust, slower) SF as its link
degrades, and back down as it improves — always the fastest datarate the
current link margin supports. That means the cross-packet reacquisition
idea above has to survive something harder than a same-SF gap: consecutive
packets from the *same* device routinely land at *different* spreading
factors. `examples/adr_experiments.py` checks directly whether it does.

The problem this creates isn't hypothetical. `KalmanAFCLoop`'s own tracked
rate is in Hz *per burst*, and a burst's duration depends on spreading
factor — doubling with every SF step up. Carry a rate estimate from one
packet across an SF change the natural way (look up "the current config" at
each step, which is exactly what stepping the predict-only loop by the
*new* packet's own burst count does) without separately accounting for
which SF the rate was *tracked* under, and the result is wrong by close to
the ratio of the two symbol durations — confirmed directly, not assumed: a
real, plausible SF7-to-SF9 ADR move (two steps, a $4\times$ symbol-duration
ratio) turns a 3.2kHz drift prediction into 0.8kHz, off by nearly the full
$4\times$.

![Cross-SF rate-extrapolation error (buggy vs. Hz/s-safe) and the resulting reacquisition success rate after a real ADR spreading-factor switch](pictures/27_adr_reacquisition.png)

*(Left panel.)* The fix is the same kind as the earlier Hz/burst-vs-Hz/s
bug: don't trust two separately-computed, unit-bearing numbers (a rate and
a burst count) to agree on which config's symbol duration they're expressed
in. `predicted_center_hz_per_s` takes a physical rate (Hz/s, config-
independent) and a real elapsed gap (seconds), converting to whichever
config is actually in play internally rather than leaving that to the call
site. Accurate to well under 1% of the true drift across every spreading-
factor transition tested, from no change at all up to a dramatic SF7-to-
SF12 jump ($32\times$ the symbol-duration ratio) — where the unconverted
version is off by essentially the entire signal.

*(Right panel.)* The practical cost of the bug, not just the size of the
error: at SF7-to-SF9 and SF7-to-SF10, the unconverted prediction is bad
enough that reacquisition at the new spreading factor fails completely, no
better than a blind, zero-centered search — while the Hz/s-safe version
recovers a real, substantial fraction of attempts at every transition
tested. One honest nuance: at the smallest step (SF7-to-SF8), the bug's own
error happens to still fit inside that configuration's own alias-safe
search span, so the unconverted version does almost as well as the fixed
one there — the failure is real and severe, but it shows up once the SF gap
is large enough, not on every ADR move regardless of size.

---

## Where to go from here

- `PAPER.md` — the same results with every proof given in full, organized
  as a dense reference rather than a narrative.
- `DISSERTATION_OUTLINE.md` — how these results map onto a dissertation
  chapter structure.
- `README.md` — how to regenerate every figure referenced above.
