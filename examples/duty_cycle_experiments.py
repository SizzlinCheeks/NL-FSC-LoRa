"""Does this project's within-packet CFO tracking (afc.py) mean anything
*between* packets under real LoRaWAN duty-cycle limits -- or does every new
packet just need its own acquisition, the way real LoRaWAN receivers already
work?

Real numbers first (lora_phy's own tx.time_in_air, this project's own
already-published LEO Doppler rate range of 270-640 Hz/s from "Putting Real
Numbers on the Applicability Claim"), swept across SF7-SF12 at 125/500kHz and
two payload sizes, comparing the resulting Doppler drift during a EU868-style
1% duty-cycle gap (gap >= 99x time-on-air) against two thresholds: this
project's own half-bin decode-capture tolerance (can a *stale*, uncorrected
estimate from the previous packet even land close enough to decode the next
packet's first symbol) and its own 25kHz acquire_span (can the *existing*,
zero-centered acquisition search from packet_experiments.py's Test 2 even
find the true CFO at all).

Finding, checked directly rather than assumed: across nearly every
configuration tested, drift during even the *minimum* duty-cycle gap
comfortably exceeds the half-bin tolerance -- meaning a receiver relying on a
stale CFO estimate from the previous packet would already fail to decode the
new packet's very first symbol. Continuous, burst-to-burst tracking (this
project's central contribution, Chapters 6-7) is real and validated *within*
one packet; it does not, by itself, extend *across* packets once duty-cycle
spacing dominates -- every new packet needs its own acquisition, exactly as
real LoRaWAN receivers already do it (fresh preamble detection and sync per
uplink). For several slower configurations (large SF, long payload, fast
Doppler), drift also exceeds the existing 25kHz acquire_span outright, so
even a *fresh*, zero-centered acquisition search would miss the true CFO
entirely.

The one thing genuinely worth adding, and tested directly below
(`experiment_predicted_vs_blind_acquisition`): does extrapolating the
*previous* packet's tracked CFO and rate (KalmanAFCLoop already carries both)
forward across the gap -- reusing its own constant-velocity predict step,
`loop.update(None)`, called once per gap-equivalent burst duration, no new
tracking math -- and re-centering the *same-width* acquisition search on that
prediction recover the cases a blind, zero-centered search misses, without
costing any extra search time? Yes, cleanly, for exactly the drift magnitudes
this section's own real numbers say actually occur.

Run with: python examples/duty_cycle_experiments.py
"""
import copy
import os

import matplotlib.pyplot as plt
import numpy as np

from nlfsc_lora.afc import KalmanAFCLoop, _combine_acquisition
from nlfsc_lora.channel import apply_cfo, awgn
from nlfsc_lora.chirp import ChirpConfig, symbol_waveform
from nlfsc_lora.sync import joint_cfo_symbol_search
from nlfsc_lora.trajectories import TRAJECTORIES

try:
    from lora_phy import LoRaTransmitter
    HAVE_LORA_PHY = True
except ImportError:
    HAVE_LORA_PHY = False

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
DUTY_CYCLE = 0.01  # EU868-style 1% default sub-band
LEO_RATE_RANGE = (270.0, 640.0)  # Hz/s, this project's own grounded range


def make_cfg(sf, bw):
    g_lin, _ = TRAJECTORIES["linear"]
    return ChirpConfig(sf=sf, bandwidth=bw, sample_rate=4 * bw, g=g_lin, f_center=0.0)


def real_numbers_table():
    """The grounding pass: time-on-air, duty-cycle-mandated gap, and the
    resulting Doppler drift during that gap, across a realistic configuration
    sweep -- not just this project's own one packet-test configuration."""
    if not HAVE_LORA_PHY:
        print("lora_phy not installed -- skipping (time-on-air needs a real LoRaWAN framing implementation).")
        return []
    rows = []
    for sf in [7, 9, 10, 12]:
        for bw in [125e3, 500e3]:
            cfg = make_cfg(sf, bw)
            tx = LoRaTransmitter(sf, bw, cfg.sample_rate, has_header=True, coding_rate=4, enable_crc=True)
            half_bin = bw / (1 << sf) / 2
            for payload_len in [12, 51]:
                on_air_s = tx.time_in_air(payload_len) / 1000.0
                gap_s = on_air_s * (1.0 / DUTY_CYCLE - 1.0)
                for rate in LEO_RATE_RANGE:
                    drift = rate * gap_s
                    rows.append(dict(sf=sf, bw=bw, payload_len=payload_len, on_air_s=on_air_s,
                                      gap_s=gap_s, rate=rate, drift=drift, half_bin=half_bin,
                                      exceeds_half_bin=drift > half_bin, exceeds_span=drift > 25000.0))
                    print(f"SF{sf:2d} BW{int(bw/1e3):3d}k {payload_len:2d}B  on-air={on_air_s*1e3:7.1f}ms  "
                          f"gap={gap_s:8.2f}s  drift@{rate:.0f}Hz/s={drift:9.1f}Hz  "
                          f"half_bin={half_bin:6.1f}Hz  [{'EXCEEDS' if drift>half_bin else 'within'} half-bin]  "
                          f"[{'EXCEEDS' if drift>25000 else 'within'} 25kHz span]")
    return rows


def predicted_center(cfo0, rate0, n_equiv_bursts, process_noise_cfo=1.0, process_noise_rate=0.2):
    """Extrapolate a KalmanAFCLoop's own state forward across a gap by calling
    its existing predict-only step (update(None)) once per gap-equivalent
    burst duration -- no new tracking math, the same constant-velocity model
    already validated within a packet, just run with no measurements coming
    in. Returns the predicted CFO (the mean prediction is exact under this
    linear model regardless of how large the propagated uncertainty grows,
    since process noise only inflates the covariance, not the mean).

    `rate0` is in the tracker's own native units, Hz *per burst* (its model
    is cfo[i+1] = cfo[i] + rate[i], one burst per step) -- a real-world rate
    in Hz/s must be multiplied by cfg.symbol_duration before being passed in
    here, and n_equiv_bursts (gap_seconds / cfg.symbol_duration) converts the
    real elapsed gap into that same burst-indexed unit."""
    loop = KalmanAFCLoop(cfo_tracked=cfo0, rate_tracked=rate0,
                          process_noise_cfo=process_noise_cfo, process_noise_rate=process_noise_rate)
    for _ in range(int(round(n_equiv_bursts))):
        loop.update(None)
    return loop.cfo_tracked


def acquire(cfg, true_cfo, center, span, step, n_bursts, snr_db, seed):
    """N_PREAMBLE-burst acquisition (sync.py::joint_cfo_symbol_search,
    afc.py::_combine_acquisition -- the exact machinery packet_experiments.py's
    Test 2 already validated), candidates centered on `center` instead of the
    usual 0."""
    rng = np.random.default_rng(seed)
    candidates = center + np.arange(-span, span + step, step)
    results = []
    for _ in range(n_bursts):
        tx = apply_cfo(symbol_waveform(cfg, 0), true_cfo, cfg.sample_rate)
        rx = awgn(tx, snr_db, rng)
        m_hat, cfo_hat, _score = joint_cfo_symbol_search(rx, cfg, candidates)
        results.append((m_hat, cfo_hat))
    return _combine_acquisition(results)


def experiment_predicted_vs_blind_acquisition():
    """The direct test: sweep the duty-cycle gap duration itself, holding the
    true Doppler rate fixed at a realistic LEO number -- so the true new CFO
    (rate x gap) and the rate-based prediction grow *together*, exactly as
    they would in reality, rather than being swept as independent,
    artificially decoupled quantities. Compares a blind search (centered at
    0, this project's existing default, correct only for the zero-gap limit)
    against one re-centered on the rate-based extrapolation -- same span
    width, same search cost, different center.

    Two real bugs were caught building this, worth being honest about since
    both would have silently produced a misleading figure:

    1. step=50Hz, not packet_experiments.py's own 250Hz: this project's own
       established rule (that file's push_acquisition_magnitude_limit
       finding, tests/test_afc.py::test_acquisition_step_must_stay_under_half_bin_or_it_silently_fails)
       is that the acquisition step must stay under half the half-bin
       tolerance, or no candidate can land close enough to succeed
       regardless of span. 250Hz was copied from a different (SF7/500kHz,
       half_bin=1953Hz) configuration without rechecking it against this
       one's much tighter half_bin=122Hz.

    2. span=1500Hz, not a wide +/-25kHz span: joint_cfo_symbol_search jointly
       searches (symbol, CFO), and a CFO error near a multiple of this
       config's own bin spacing (bandwidth/M = 244Hz here) can alias with a
       *different* symbol hypothesis almost as strongly as the true
       (symbol, CFO) pair does -- confirmed directly, not assumed: a wide
       +/-25kHz search (about 100 bin-widths) reproducibly locked onto a
       wrong symbol 2 bins away from the truth, *even at 40dB SNR*, a
       systematic aliasing failure, not a noise problem. The number of such
       aliases scales with span/bin_width, which is why
       packet_experiments.py's own Test 2 (SF7/500kHz, bin width 3906Hz,
       ~6 aliases in the same nominal 25kHz) never ran into this: a fixed
       span in Hz is not actually a shape/SF-independent design choice, and
       this project's own established 25kHz figure happens to be safe only
       for the configuration it was tuned on. A narrow, alias-safe span
       (about 6 bin-widths each side here) removes the problem -- and turns
       out to make the real point better anyway: it's exactly the accuracy a
       *correctly re-centered* search actually needs, no wider.

    The prediction is deliberately not fed the exact true rate: a real
    receiver's last tracked rate before hand-off is itself an estimate, with
    its own residual error (this project's own tracking accuracy figures,
    e.g. 17_kalman_vs_expfilter_ramp.png, put steady-state Kalman rate error
    at a small fraction of the true rate, not exactly zero) -- modeled here
    as a +/-5% relative jitter per trial, a conservative placeholder for that
    residual error rather than a number re-derived from a full within-packet
    tracking simulation, which is out of this experiment's scope. The true
    rate itself (637.3 Hz/s, not a round 640) is deliberately not a clean
    multiple of the search step either -- a round true_rate x round gap
    landed the true CFO suspiciously exactly on the blind search's own
    0-centered grid in early testing, artificially inflating blind's
    measured success rate.

    A successful acquisition is one landing within half a bin of the true
    CFO (the threshold that actually matters: close enough for
    fft_correlation_demod to decode correctly from there)."""
    cfg = make_cfg(9, 125e3)  # SF9/125kHz: half_bin=122Hz, a config from the table above
    half_bin = cfg.bandwidth / cfg.M / 2
    span, step, n_bursts = 1500.0, 50.0, 8
    true_rate = 637.3  # Hz/s, close to the table's worst-case 640, deliberately not round
    # KalmanAFCLoop's rate_tracked is Hz *per burst* (its own model: cfo[i+1]=cfo[i]+rate[i],
    # one burst per step), not Hz/s -- convert the real-world rate before use.
    true_rate_per_burst = true_rate * cfg.symbol_duration

    gaps_s = np.arange(0.0, 65.0, 5.0)
    snr_db = -10.0
    n_trials = 40
    rng_jitter = np.random.default_rng(42)

    success_blind = np.empty(len(gaps_s))
    success_predicted = np.empty(len(gaps_s))
    centers = np.empty(len(gaps_s))
    true_cfos = np.empty(len(gaps_s))
    for i, gap_s in enumerate(gaps_s):
        n_equiv_bursts = gap_s / cfg.symbol_duration
        true_new_cfo = true_rate * gap_s
        true_cfos[i] = true_new_cfo

        ok_blind = ok_pred = 0
        for t in range(n_trials):
            rate_jitter = 1.0 + rng_jitter.uniform(-0.05, 0.05)
            center_pred = predicted_center(0.0, true_rate_per_burst * rate_jitter, n_equiv_bursts)
            acq_blind = acquire(cfg, true_new_cfo, center=0.0, span=span, step=step,
                                 n_bursts=n_bursts, snr_db=snr_db, seed=t)
            acq_pred = acquire(cfg, true_new_cfo, center=center_pred, span=span, step=step,
                                n_bursts=n_bursts, snr_db=snr_db, seed=t + 500_000)
            ok_blind += int(abs(acq_blind - true_new_cfo) < half_bin)
            ok_pred += int(abs(acq_pred - true_new_cfo) < half_bin)
        centers[i] = center_pred  # last trial's center, for reference/plotting
        success_blind[i] = ok_blind / n_trials
        success_predicted[i] = ok_pred / n_trials
        print(f"gap={gap_s:6.0f}s  true_new_cfo={true_new_cfo:8.0f}Hz  predicted_center~={center_pred:8.0f}Hz  "
              f"blind={success_blind[i]:.2f}  predicted={success_predicted[i]:.2f}")

    return gaps_s, true_cfos, success_blind, success_predicted, span, half_bin, true_rate


def plot_duty_cycle_findings(rows, gaps_s, true_cfos, success_blind, success_predicted, span, half_bin, true_rate):
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, (ax_table, ax_acq) = plt.subplots(1, 2, figsize=(13, 4.5))

    if rows:
        labels = [f"SF{r['sf']}/{int(r['bw']/1e3)}k/{r['payload_len']}B" for r in rows[::2]]
        drift_lo = [r["drift"] for r in rows[0::2]]
        drift_hi = [r["drift"] for r in rows[1::2]]
        x = np.arange(len(labels))
        ax_table.bar(x, drift_hi, color="tab:red", alpha=0.5, label=f"{LEO_RATE_RANGE[1]:.0f} Hz/s (worst-case)")
        ax_table.bar(x, drift_lo, color="tab:blue", alpha=0.7, label=f"{LEO_RATE_RANGE[0]:.0f} Hz/s")
        half_bins = [r["half_bin"] for r in rows[0::2]]
        ax_table.plot(x, half_bins, "k_", ms=14, mew=2, label="half-bin tolerance (this config)")
        ax_table.axhline(25000.0, color="k", linestyle="--", linewidth=1, label="25kHz acquire_span")
        ax_table.set(xlabel="", ylabel="CFO drift during 1% duty-cycle gap (Hz)", yscale="log",
                     title="Between-packet Doppler drift vs. real thresholds")
        ax_table.set_xticks(x)
        ax_table.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        ax_table.legend(fontsize=7, loc="upper left")

    ax_acq.plot(gaps_s, success_blind, marker="o", ms=4, label="blind search (centered at 0)")
    ax_acq.plot(gaps_s, success_predicted, marker="o", ms=4, label="rate-predicted center (+/-5% rate error)")
    span_edge_gap = span / true_rate
    ax_acq.axvline(span_edge_gap, color="k", linestyle="--", linewidth=1,
                    label=f"gap where true drift = span ({span_edge_gap:.0f}s)")
    ax_acq.set(xlabel="duty-cycle gap since last packet (s)", ylabel="acquisition success rate",
               ylim=(-0.05, 1.05),
               title=f"Same-width (+/-{span:.0f}Hz, alias-safe) acquisition, blind vs. rate-predicted center\n"
                     f"(SF9/125kHz, true rate={true_rate:.1f}Hz/s worst-case LEO, SNR=-10dB)")
    ax_acq.legend(fontsize=8)
    ax2 = ax_acq.twiny()
    ax2.set_xlim(ax_acq.get_xlim())
    tick_gaps = [g for g in [0, 15, 30, 45, 60] if g <= gaps_s.max()]
    ax2.set_xticks(tick_gaps)
    ax2.set_xticklabels([f"{true_rate*g/1e3:.1f}kHz" for g in tick_gaps], fontsize=7)
    ax2.set_xlabel("true CFO drift at that gap", fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "25_duty_cycle_reacquisition.png"), dpi=150)
    plt.close(fig)


def main():
    print("=== Real numbers: between-packet Doppler drift vs. duty-cycle gap ===")
    rows = real_numbers_table()
    print("\n=== Blind vs. rate-predicted acquisition centering ===")
    gaps_s, true_cfos, success_blind, success_predicted, span, half_bin, true_rate = experiment_predicted_vs_blind_acquisition()
    plot_duty_cycle_findings(rows, gaps_s, true_cfos, success_blind, success_predicted, span, half_bin, true_rate)
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
