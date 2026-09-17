"""Does the duty-cycle reacquisition idea (examples/duty_cycle_experiments.py)
survive real LoRaWAN Adaptive Data Rate -- where consecutive uplinks from
the *same* device routinely use *different* spreading factors, not the same
SF the whole time?

Real LoRaWAN devices don't fix their SF: ADR steps the spreading factor
(and so the datarate) up or down as link quality changes, always picking
the fastest (lowest-SF) datarate the current link margin can support --
lower SF means shorter airtime, less duty-cycle budget spent, more capacity
for everyone. A device whose link is degrading (moving away from a
gateway, or a satellite pass receding) gets stepped to a higher, more
robust SF between one uplink and the next.

The problem this creates for cross-packet CFO-rate extrapolation:
KalmanAFCLoop's own rate_tracked is in Hz *per burst*, and a burst's
duration (cfg.symbol_duration = M/bandwidth) depends on SF. Carry a rate
estimate from one packet across an SF change without accounting for that,
and the predicted center is wrong -- not a rounding difference, but wrong
by close to the ratio of the two symbol durations, $2^{|SF_b - SF_a|}$.
Confirmed directly below, not assumed: SF7-to-SF9 at 125kHz (a real,
plausible two-step ADR move) is a 4x error.

duty_cycle_experiments.py's own predicted_center_hz_per_s fixes this by
taking a physical rate (Hz/s) and a real gap (seconds), converting to
whichever config is actually in play internally -- removing the ambiguity
the two-separately-computed-arguments API invited, rather than trusting
every call site to get the conversion right.

Run with: python examples/adr_experiments.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from duty_cycle_experiments import acquire, make_cfg, predicted_center, predicted_center_hz_per_s

try:
    from lora_phy import LoRaTransmitter
    HAVE_LORA_PHY = True
except ImportError:
    HAVE_LORA_PHY = False

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
BANDWIDTH = 125e3

# Standard LoRaWAN (EU868-style) per-SF demodulation floor, dB -- the SNR a
# receiver needs to decode at all at that spreading factor. 2.5dB per SF
# step is the widely-cited, standard figure (e.g. Semtech's own LoRa
# datasheets); consulted as a standard reference value, not independently
# re-derived from this project's own SER curves.
LORAWAN_REQUIRED_SNR = {7: -7.5, 8: -10.0, 9: -12.5, 10: -15.0, 11: -17.5, 12: -20.0}


def adr_pick_sf(measured_snr_db, margin_db=10.0, available_sfs=range(7, 13)):
    """The core of real LoRaWAN ADR: pick the fastest (lowest-SF) datarate
    whose required SNR still leaves at least margin_db of headroom against
    the measured link SNR -- shorter airtime (less duty-cycle budget spent)
    whenever the link can support it, falling back to a more robust
    (higher-SF, more time-on-air) datarate only when it can't. Falls back
    to the most robust available SF if none has enough margin."""
    for sf in sorted(available_sfs):
        if measured_snr_db - LORAWAN_REQUIRED_SNR[sf] >= margin_db:
            return sf
    return max(available_sfs)


def adr_realism_table():
    """A degrading link (e.g. a receding satellite pass, or a device moving
    away from its gateway) forces ADR to step up through SFs -- shown
    directly against the standard required-SNR table, not just asserted."""
    print(f"{'measured SNR (dB)':>18}  {'ADR-picked SF':>13}  {'margin at that SF':>18}")
    for snr_db in [10, 5, 0, -5, -10, -15, -20]:
        sf = adr_pick_sf(snr_db)
        margin = snr_db - LORAWAN_REQUIRED_SNR[sf]
        print(f"{snr_db:18.1f}  {sf:13d}  {margin:18.1f}")


def gap_for_sf(sf, payload_len=12):
    """The 1%-duty-cycle-mandated gap after transmitting one packet at this
    SF/BW -- reuses lora_phy's own time_in_air, same convention as
    duty_cycle_experiments.py's own real_numbers_table."""
    cfg = make_cfg(sf, BANDWIDTH)
    tx = LoRaTransmitter(sf, BANDWIDTH, cfg.sample_rate, has_header=True, coding_rate=4, enable_crc=True)
    on_air_s = tx.time_in_air(payload_len) / 1000.0
    return on_air_s * 99.0  # gap >= 99x on-air time for a 1% duty cycle


def experiment_sf_transition_prediction_error():
    """For a range of (SF_a -> SF_b) ADR transitions, the buggy (mismatched-
    unit) prediction's error vs. the safe wrapper's error -- both against
    the same true drift, same true rate, real duty-cycle gap."""
    true_rate = 637.3  # Hz/s, this project's own grounded (non-round) LEO rate
    transitions = [(7, 7), (7, 8), (7, 9), (7, 10), (7, 12)]

    rows = []
    for sf_a, sf_b in transitions:
        cfg_a = make_cfg(sf_a, BANDWIDTH)
        cfg_b = make_cfg(sf_b, BANDWIDTH)
        gap_s = gap_for_sf(sf_a)
        true_new_cfo = true_rate * gap_s

        rate_a_per_burst = true_rate * cfg_a.symbol_duration
        n_equiv_bursts_b_units = gap_s / cfg_b.symbol_duration
        buggy = predicted_center(0.0, rate_a_per_burst, n_equiv_bursts_b_units)
        fixed = predicted_center_hz_per_s(0.0, true_rate, gap_s, cfg_b)

        rows.append(dict(sf_a=sf_a, sf_b=sf_b, gap_s=gap_s, true_new_cfo=true_new_cfo,
                          buggy_err=buggy - true_new_cfo, fixed_err=fixed - true_new_cfo,
                          buggy=buggy, fixed=fixed))
        print(f"SF{sf_a}->SF{sf_b}  gap={gap_s:6.2f}s  true={true_new_cfo:9.1f}Hz  "
              f"buggy={buggy:9.1f}Hz (err={buggy-true_new_cfo:+9.1f})  "
              f"fixed={fixed:9.1f}Hz (err={fixed-true_new_cfo:+9.1f})")
    return rows


def experiment_reacquisition_across_adr_switch():
    """The practical consequence: does the prediction error above actually
    cost acquisition success at the new SF? Same alias-safe span sizing
    rule duty_cycle_experiments.py established (a fixed span in Hz is not
    SF-independent -- scale to the new config's own bin width), centered on
    blind (0), buggy, and fixed predictions."""
    true_rate = 637.3
    transitions = [(7, 8), (7, 9), (7, 10)]
    snr_db = -10.0
    n_trials = 40

    results = {}
    for sf_a, sf_b in transitions:
        cfg_a = make_cfg(sf_a, BANDWIDTH)
        cfg_b = make_cfg(sf_b, BANDWIDTH)
        bin_hz = cfg_b.bandwidth / cfg_b.M
        half_bin = bin_hz / 2
        span, step = 6 * bin_hz, half_bin / 3
        n_bursts = 8

        gap_s = gap_for_sf(sf_a)
        true_new_cfo = true_rate * gap_s
        rate_a_per_burst = true_rate * cfg_a.symbol_duration
        n_equiv_bursts_b_units = gap_s / cfg_b.symbol_duration
        buggy_center = predicted_center(0.0, rate_a_per_burst, n_equiv_bursts_b_units)
        fixed_center = predicted_center_hz_per_s(0.0, true_rate, gap_s, cfg_b)

        for name, center in [("blind (0)", 0.0), ("buggy (unit-mismatched)", buggy_center),
                              ("fixed (Hz/s-safe)", fixed_center)]:
            ok = 0
            for t in range(n_trials):
                acq = acquire(cfg_b, true_new_cfo, center=center, span=span, step=step,
                               n_bursts=n_bursts, snr_db=snr_db, seed=t)
                ok += int(abs(acq - true_new_cfo) < half_bin)
            results[(sf_a, sf_b, name)] = ok / n_trials
            print(f"SF{sf_a}->SF{sf_b}  {name:24s}  success={ok}/{n_trials}")
    return transitions, results


def plot_adr_findings(rows, transitions, results):
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, (ax_err, ax_acq) = plt.subplots(1, 2, figsize=(13, 4.8))

    labels = [f"SF{r['sf_a']}→SF{r['sf_b']}" for r in rows]
    x = np.arange(len(labels))
    ax_err.bar(x - 0.18, [abs(r["buggy_err"]) for r in rows], width=0.35, label="buggy (unit-mismatched)")
    ax_err.bar(x + 0.18, [abs(r["fixed_err"]) for r in rows], width=0.35, label="fixed (Hz/s-safe)")
    ax_err.set(xlabel="ADR spreading-factor transition", ylabel="prediction error (Hz)", yscale="log",
               title="Cross-SF rate-extrapolation error: buggy vs. fixed")
    ax_err.set_xticks(x)
    ax_err.set_xticklabels(labels)
    ax_err.legend(fontsize=8)

    names = ["blind (0)", "buggy (unit-mismatched)", "fixed (Hz/s-safe)"]
    x2 = np.arange(len(transitions))
    width = 0.25
    for i, name in enumerate(names):
        vals = [results[(sf_a, sf_b, name)] for sf_a, sf_b in transitions]
        ax_acq.bar(x2 + (i - 1) * width, vals, width=width, label=name)
    ax_acq.set(xlabel="ADR spreading-factor transition", ylabel="reacquisition success rate",
               ylim=(-0.05, 1.05), title="Reacquisition at the new SF after an ADR switch\n(SNR=-10dB)")
    ax_acq.set_xticks(x2)
    ax_acq.set_xticklabels([f"SF{a}→SF{b}" for a, b in transitions])
    ax_acq.legend(fontsize=8)

    fig.suptitle("Does cross-packet reacquisition survive a real ADR spreading-factor switch?")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "27_adr_reacquisition.png"), dpi=150)
    plt.close(fig)


def main():
    if not HAVE_LORA_PHY:
        print("lora_phy is not installed (pip install .[refcheck]) -- skipping ADR experiments.")
        return
    print("=== ADR rule vs. link SNR ===")
    adr_realism_table()
    print("\n=== Cross-SF rate-extrapolation error ===")
    rows = experiment_sf_transition_prediction_error()
    print("\n=== Reacquisition success across an ADR switch ===")
    transitions, results = experiment_reacquisition_across_adr_switch()
    plot_adr_findings(rows, transitions, results)
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
