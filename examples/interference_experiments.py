"""Does a real LoRaWAN receiver (lora_phy) still recover a desired packet
when a second, colliding transmission overlaps it on the same channel --
the actual multi-device scenario a LoRaWAN gateway faces, never tested by
this project before (every other test here used a single transmitter and
AWGN only)?

Two real LoRa PHY properties this checks for directly, not assumed from the
literature: the **capture effect** (a receiver locks onto whichever signal
is currently stronger, so a sufficiently-dominant desired packet survives a
same-strength or weaker interferer) and **spreading-factor quasi-
orthogonality** (two LoRa transmissions on the same channel at *different*
spreading factors interfere with each other far less than same-SF
transmissions do, because their chirp rates don't correlate well against
each other's dechirp reference -- a real, well-known, and here directly
measured property, not a byproduct of anything this project's own nonlinear
trajectories touch).

Scope note matching examples/lorawan_experiments.py: this only exercises
`g=linear` (the only shape a real receiver can demodulate at all), and uses
`lora_phy`'s own full receiver pipeline (real preamble detection and sync,
not this project's own instrumented burst slicing) -- the closest this
project comes to simulating what a real gateway actually sees.

Run with: python examples/interference_experiments.py
"""
import os
import time

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

try:
    from lora_phy import LoRaReceiver, LoRaTransmitter
    from lora_phy import common as lp_common
    HAVE_LORA_PHY = True
except ImportError:
    HAVE_LORA_PHY = False

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
BANDWIDTH = 125e3
SAMPLE_RATE = 4 * BANDWIDTH
RF_FREQ = 868e6
PAYLOAD_LEN = 12
PREAMBLE_LEN = 8


def make_packet(sf, seed, payload_len=PAYLOAD_LEN):
    tx = LoRaTransmitter(sf, BANDWIDTH, SAMPLE_RATE, has_header=True, coding_rate=4,
                          enable_crc=True, preamble_len=PREAMBLE_LEN)
    rng = np.random.default_rng(seed)
    payload = rng.integers(0, 256, payload_len, dtype=np.uint8)
    sig = tx.modulate(tx.encode(payload))
    return sig, payload


def try_decode(sig, sf, payload_len, desired_payload):
    """True only if the DESIRED payload specifically is recovered with a
    passing CRC -- not just "some CRC passed" (an interferer is itself a
    validly-encoded packet with its own correct CRC, so a naive
    any-CRC-passed check would silently credit decoding the interferer
    instead of the desired signal)."""
    rx = LoRaReceiver(rf_freq=RF_FREQ, spreading_factor=sf, bandwidth=BANDWIDTH,
                       sample_rate=SAMPLE_RATE, preamble_len=PREAMBLE_LEN)
    try:
        all_syms, _cfos, _netid = rx.demodulate(sig)
    except Exception:
        return False
    for syms in all_syms:
        try:
            data, checksum = rx.decode(syms)
        except Exception:
            continue
        if len(checksum) == 2 and list(data[payload_len:payload_len + 2]) == checksum:
            if np.array_equal(data[:payload_len], desired_payload):
                return True
    return False


def collision_trial(desired_sf, interferer_sf, sir_db, snr_db, offset_frac, seed):
    """One collision: a desired packet plus an interferer at sir_db (desired
    relative to interferer, so positive = desired stronger), the interferer
    starting offset_frac of the way into the desired packet's own duration
    (0 = fully aligned from the start, 1 = interferer starts as the desired
    packet ends -- effectively no overlap). Background noise is referenced
    to the desired signal's own power, so SNR keeps its usual meaning
    regardless of interferer strength."""
    desired, payload = make_packet(desired_sf, seed)
    interferer, _ = make_packet(interferer_sf, seed + 1_000_000)
    n = len(desired)

    offset = int(round(offset_frac * n))
    combined = desired.astype(complex).copy()
    scale = 10 ** (-sir_db / 20)
    end = min(n, offset + len(interferer))
    combined[offset:end] += scale * interferer[:end - offset]

    rng = np.random.default_rng(seed + 2_000_000)
    sig_power = np.mean(np.abs(desired) ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    rx_sig = combined + noise

    return try_decode(rx_sig, desired_sf, PAYLOAD_LEN, payload)


def test_correctness_no_interference():
    """No interferer at all, clean-ish SNR: the desired packet should
    always decode -- the baseline every SIR/orthogonality claim below is
    measured relative to."""
    n_trials = 20
    fails = 0
    for t in range(n_trials):
        ok = collision_trial(7, 7, sir_db=999.0, snr_db=15.0, offset_frac=0.0, seed=t)
        fails += int(not ok)
    print(f"no-interference baseline: {fails}/{n_trials} failures")
    assert fails == 0


def experiment_capture_vs_sf_separation():
    """The main sweep: capture probability vs. SIR, one curve per
    interferer spreading factor (same SF through +3), full overlap
    (offset_frac=0), moderate background SNR so interference -- not thermal
    noise -- is the effect under test."""
    desired_sf = 7
    sir_range = np.arange(-20.0, 16.0, 2.0)
    snr_db = 15.0
    n_trials = 60

    results = {}
    for delta_sf in [0, 1, 2, 3]:
        interferer_sf = desired_sf + delta_sf
        t0 = time.time()
        capture = np.empty(len(sir_range))
        for i, sir_db in enumerate(sir_range):
            ok_count = 0
            for t in range(n_trials):
                if collision_trial(desired_sf, interferer_sf, sir_db, snr_db, offset_frac=0.0, seed=t):
                    ok_count += 1
            capture[i] = ok_count / n_trials
        print(f"delta_SF={delta_sf}  done in {time.time()-t0:.1f}s")
        results[delta_sf] = capture

    return sir_range, results


def experiment_capture_vs_overlap():
    """Secondary check: does how much of the desired packet the interferer
    actually overlaps matter, not just its relative strength?

    Same-SF SIR=-3dB, an interferer clearly strong enough to always win a
    *full*-overlap collision (checked directly: 0/40 capture at
    offset_frac=0..0.9). The interesting range isn't spread evenly across
    [0,1] -- found by direct probing, not assumed -- it's a sharp transition
    concentrated near *full* overlap (0.90-0.95): even a strong interferer
    only needs to collide with roughly the last 5-10% of the packet to break
    the whole thing, not something close to half. That's consistent with
    Hamming FEC/interleaving correcting scattered errors but not a
    contiguous corrupted block landing on the CRC-bearing symbols
    specifically -- a burst error, not a scattered one. The sweep below is
    concentrated in that actual transition region rather than wasted on the
    flat 0% tail."""
    desired_sf = 7
    sir_db = -3.0
    snr_db = 15.0
    n_trials = 60
    offsets = np.concatenate([np.arange(0.0, 0.85, 0.2), np.arange(0.85, 1.001, 0.02)])

    capture = np.empty(len(offsets))
    t0 = time.time()
    for i, offset_frac in enumerate(offsets):
        ok_count = 0
        for t in range(n_trials):
            if collision_trial(desired_sf, desired_sf, sir_db, snr_db, offset_frac, seed=t):
                ok_count += 1
        capture[i] = ok_count / n_trials
    print(f"overlap sweep done in {time.time()-t0:.1f}s")
    return offsets, capture


def plot_interference_findings(sir_range, sf_results, offsets, overlap_capture):
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, (ax_sf, ax_overlap) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    for delta_sf, capture in sf_results.items():
        label = "same SF (0)" if delta_sf == 0 else f"interferer SF+{delta_sf}"
        ax_sf.plot(sir_range, capture, marker="o", ms=4, label=label)
    ax_sf.axhline(0.5, color="k", linewidth=0.5, linestyle=":")
    ax_sf.set(xlabel="SIR (dB, desired relative to interferer)", ylabel="capture probability",
              ylim=(-0.05, 1.05),
              title="Capture effect vs. spreading-factor separation\n(full overlap, SNR=15dB, 60 trials/point)")
    ax_sf.legend(fontsize=8)

    ax_overlap.plot(offsets, overlap_capture, marker="o", ms=4, color="tab:red")
    ax_overlap.set(xlabel="interferer start (fraction of desired packet duration)",
                   ylabel="capture probability", ylim=(-0.05, 1.05),
                   title="Capture probability vs. collision overlap\n(same-SF interferer, SIR=-3dB, SNR=15dB)")

    fig.suptitle("Real LoRaWAN multi-device collisions (lora_phy's own receiver, real preamble detection + Hamming/CRC)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "26_lorawan_interference.png"), dpi=150)
    plt.close(fig)


def plot_sf_spectrogram_overlay(sfs=(7, 8, 9, 10)):
    """The visual behind the left panel of 26_lorawan_interference.png:
    why does spreading-factor separation matter at all? Overlay one
    real up-chirp per SF (lora_phy's own construction, all sharing the same
    bandwidth and so the same frequency axis), each tiled to repeat enough
    times to fill the longest (highest-SF) symbol's own duration -- since a
    lower-SF transmitter really does send several symbols in the time a
    higher-SF one sends one. Same SF would trace the *same* diagonal every
    time, permanently coincident in frequency at every instant (hence the
    near step-function capture behavior in the SIR sweep); different SFs
    sweep the same band at different rates, crossing each other only
    briefly rather than staying aligned -- the direct visual reason a
    different-SF interferer is so much less disruptive than a same-SF one."""
    chirps = {sf: lp_common.chirp(True, sf, BANDWIDTH, SAMPLE_RATE, start_freq_offset=0) for sf in sfs}
    n_total = len(chirps[max(sfs)])  # longest (highest-SF) symbol duration

    combined = np.zeros(n_total, dtype=complex)
    for c in chirps.values():
        tiled = np.tile(c, n_total // len(c))
        combined[:len(tiled)] += tiled

    f, t, Sxx = spectrogram(combined, fs=SAMPLE_RATE, nperseg=256, noverlap=248,
                             return_onesided=False, mode="magnitude")
    f = np.fft.fftshift(f)
    Sxx = np.fft.fftshift(Sxx, axes=0)

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.pcolormesh(t * 1e3, f / 1e3, 20 * np.log10(Sxx + 1e-6), shading="gouraud",
                        cmap="magma", vmin=-50, vmax=-15)
    ax.set(xlabel="time (ms)", ylabel="frequency (kHz)", ylim=(-BANDWIDTH / 2 / 1e3, BANDWIDTH / 2 / 1e3),
           title=f"{len(sfs)} real LoRa spreading factors sharing one channel (SF{min(sfs)}–SF{max(sfs)}, "
                 f"{BANDWIDTH/1e3:.0f}kHz)\nsame band, different chirp rates: briefly aligned, never coincident")
    cbar = fig.colorbar(im)
    cbar.set_label("magnitude (dB)")

    # Label each SF where its own first sweep peaks (just before it wraps into its next
    # repeat) -- these times are, by construction, as different as the SFs' own durations
    # (dur_ms, 2*dur_ms, 4*dur_ms, ...), so the labels land well separated from each other
    # even though they share the same frequency (the top of the band).
    for sf in sfs:
        dur_ms = len(chirps[sf]) / SAMPLE_RATE * 1e3
        label_t = dur_ms * 0.95
        ax.annotate(f"SF{sf}", xy=(label_t, BANDWIDTH / 2 / 1e3), xytext=(label_t - 0.05, 53),
                    color="white", fontsize=10, fontweight="bold", ha="center",
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "28_lorawan_sf_spectrogram.png"), dpi=150)
    plt.close(fig)


def main():
    if not HAVE_LORA_PHY:
        print("lora_phy is not installed (pip install .[refcheck]) -- skipping interference experiments.")
        return
    test_correctness_no_interference()
    sir_range, sf_results = experiment_capture_vs_sf_separation()
    offsets, overlap_capture = experiment_capture_vs_overlap()
    plot_interference_findings(sir_range, sf_results, offsets, overlap_capture)
    plot_sf_spectrogram_overlay()
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
