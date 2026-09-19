"""A real LEO-satellite Doppler pass, computed from actual orbital
mechanics (not a synthetic linear/UAV-reversal-style profile), driving the
same real-LoRaWAN-framed packet pipeline as nonlinear_lorawan_experiments.py.

Ties two threads of this project together with real numbers instead of
illustrative ones:

- The narrowband profiles nonlinear_lorawan_experiments.py already used
  (constant 20kHz, UAV reversal) were chosen for illustration. This file
  asks what an actual LEO satellite pass looks like -- altitude, orbital
  velocity, and Doppler derived from orbital mechanics -- and drives the
  exact same packet pipeline with it.

- It also settles a question the wideband alpha=1.05 demo
  (31_nonlinear_lorawan_wideband_scale.png) leaves open: is that time-scale
  factor realistic for an RF satellite link? No. A Doppler *scale* departs
  from 1 by v/c_signal; even a fast LEO satellite (~7.6km/s) only reaches
  alpha ~ 1.000025 -- five orders of magnitude below 1.05. Reaching
  alpha=1.05 for RF needs v ~ 15,000km/s (meaningless). But c_signal for
  underwater acoustics is only ~1500m/s, so the same alpha=1.05 needs just
  ~75m/s relative velocity -- well within a fast torpedo/submarine's range.
  alpha=1.05 was always implicitly an acoustic/sonar-domain number riding
  on an RF example, consistent with paired_sweep's own DHFM/active-sonar
  heritage (NARRATIVE.md, Kroszczynski 1969) -- not an RF satellite
  scenario. Real LEO Doppler is a narrowband (CFO-domain) story, which is
  exactly what this file demonstrates.

Run with: python examples/leo_satellite_experiments.py
"""
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from lorawan_experiments import N_PREAMBLE
from nonlinear_lorawan_experiments import BANDWIDTH, HAVE_LORA_PHY, SF, make_cfg, run_nonlinear_lorawan_packet

if HAVE_LORA_PHY:
    from lora_phy import LoRaReceiver, LoRaTransmitter

from nlfsc_lora.afc import KalmanAFCLoop, measurement_noise_for

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# Orbital mechanics constants
MU_EARTH_KM3_S2 = 398600.4418
R_EARTH_KM = 6371.0
C_MPS = 299_792_458.0
F_CARRIER_HZ = 868e6  # EU868, matches the rest of this project's LoRaWAN work

ALTITUDE_KM = 550.0  # Starlink/typical LoRaWAN-satellite-IoT altitude band


def leo_orbital_velocity_kms(altitude_km: float) -> float:
    """Circular-orbit velocity from vis-viva at r = R_earth + altitude."""
    r_km = R_EARTH_KM + altitude_km
    return float(np.sqrt(MU_EARTH_KM3_S2 / r_km))


def doppler_profile_hz(t_s: np.ndarray, altitude_km: float, velocity_kms: float, f_carrier_hz: float) -> np.ndarray:
    """Doppler shift vs time through an overhead LEO pass, flat-Earth
    closest-approach model (t=0 at closest approach/zenith): the satellite
    travels along a straight line at height h with speed v, so range
    R(t) = sqrt(h^2 + (v*t)^2) and Doppler f_d(t) = -f_carrier/c * dR/dt.
    The standard first-order approximation for the region around closest
    approach, where satcom Doppler is steepest -- the same approximation
    used throughout the CubeSat/satellite-IoT Doppler-compensation
    literature for pass geometry near zenith."""
    h_m = altitude_km * 1000.0
    v_mps = velocity_kms * 1000.0
    x_m = v_mps * np.asarray(t_s, dtype=float)
    range_rate = v_mps * x_m / np.sqrt(h_m ** 2 + x_m ** 2)
    return -f_carrier_hz / C_MPS * range_rate


def max_doppler_hz(velocity_kms: float, f_carrier_hz: float) -> float:
    return f_carrier_hz / C_MPS * velocity_kms * 1000.0


def max_doppler_rate_hz_per_s(altitude_km: float, velocity_kms: float, f_carrier_hz: float) -> float:
    """Doppler rate at closest approach (t=0), where it peaks."""
    h_m = altitude_km * 1000.0
    v_mps = velocity_kms * 1000.0
    return f_carrier_hz / C_MPS * v_mps ** 2 / h_m


def realistic_wideband_scale_context():
    """How far real LEO Doppler *scale* (alpha) sits from the 1.05 used in
    the wideband demo, and what relative velocity actually reaches
    alpha=1.05 for an acoustic (underwater) signal instead of RF."""
    v_leo_mps = leo_orbital_velocity_kms(ALTITUDE_KM) * 1000.0
    alpha_leo = 1.0 + v_leo_mps / C_MPS
    c_sound_water = 1500.0
    v_needed_underwater = 0.05 * c_sound_water
    return alpha_leo, v_needed_underwater


def _packet_time_grid(t_half_s: float, step_s: float) -> np.ndarray:
    return np.arange(-t_half_s, t_half_s + step_s, step_s)


def simulate_leo_pass(cfg, tx, rx, payload_len_bytes, pass_times_s, altitude_km, velocity_kms, f_carrier_hz,
                       snr_db, n_trials):
    """For each packet transmission time across the pass, build that
    packet's own per-burst true_cfo_sequence from the actual instantaneous
    Doppler at each burst's offset within the packet -- this captures both
    the packet's absolute CFO (large toward the edges of the pass) and any
    in-packet drift (largest near closest approach, where the Doppler
    *rate* peaks) -- then run it n_trials times and report the pass rate."""
    n_total = N_PREAMBLE + tx.encode(np.zeros(payload_len_bytes, dtype=np.uint8)).shape[0]
    burst_offsets_s = np.arange(n_total) * cfg.symbol_duration
    pass_rates = np.empty(len(pass_times_s))
    for i, t_center in enumerate(pass_times_s):
        burst_times = t_center + burst_offsets_s
        cfo_seq = doppler_profile_hz(burst_times, altitude_km, velocity_kms, f_carrier_hz)
        passes = 0
        for trial in range(n_trials):
            loop = KalmanAFCLoop(measurement_noise=measurement_noise_for(cfg))
            ok, _b, _s = run_nonlinear_lorawan_packet(cfg, tx, rx, payload_len_bytes, cfo_seq, snr_db,
                                                        seed=trial, loop=loop)
            passes += int(ok)
        pass_rates[i] = passes / n_trials
    return pass_rates


def experiment_leo_pass_reliability():
    payload_len = 12
    velocity_kms = leo_orbital_velocity_kms(ALTITUDE_KM)
    t_half_s = 300.0
    step_s = 15.0
    pass_times = _packet_time_grid(t_half_s, step_s)
    snr_db = -15.0
    n_trials = 25

    results = {}
    for shape in ["linear", "hyperbolic"]:
        cfg = make_cfg(shape)
        tx = LoRaTransmitter(SF, BANDWIDTH, cfg.sample_rate, has_header=True, coding_rate=4, enable_crc=True)
        rx = LoRaReceiver(rf_freq=F_CARRIER_HZ, spreading_factor=SF, bandwidth=BANDWIDTH, sample_rate=cfg.sample_rate)
        t0 = time.time()
        pass_rates = simulate_leo_pass(cfg, tx, rx, payload_len, pass_times, ALTITUDE_KM, velocity_kms,
                                        F_CARRIER_HZ, snr_db, n_trials)
        print(f"[{shape}] LEO pass sweep done in {time.time()-t0:.1f}s")
        results[shape] = pass_rates

    return pass_times, results, velocity_kms, snr_db


def plot_leo_pass(pass_times, results, velocity_kms, snr_db):
    os.makedirs(OUT_DIR, exist_ok=True)
    doppler_curve = doppler_profile_hz(pass_times, ALTITUDE_KM, velocity_kms, F_CARRIER_HZ)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 7.5), sharex=True)

    ax1.plot(pass_times / 60.0, doppler_curve / 1e3, color="black")
    ax1.axhline(0, color="gray", lw=0.7, ls=":")
    ax1.set(ylabel="Doppler shift (kHz)",
            title=f"LEO satellite pass, {ALTITUDE_KM:.0f}km altitude, {velocity_kms:.2f}km/s, "
                  f"{F_CARRIER_HZ/1e6:.0f}MHz carrier\n(orbital-mechanics-derived, flat-Earth closest-approach model)")

    for shape in ["linear", "hyperbolic"]:
        ax2.plot(pass_times / 60.0, results[shape], marker="o", ms=4, label=f"{shape} PHY")
    ax2.set(xlabel="time from closest approach (min)", ylabel="packet CRC pass rate", ylim=(-0.05, 1.05),
            title=f"Real LoRaWAN-framed packet reliability across the pass (SNR={snr_db:.0f}dB)")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "32_leo_satellite_pass.png"), dpi=150)
    plt.close(fig)


def main():
    if not HAVE_LORA_PHY:
        print("lora_phy is not installed (pip install .[refcheck]) -- skipping LEO satellite experiments.")
        return

    v = leo_orbital_velocity_kms(ALTITUDE_KM)
    f_max = max_doppler_hz(v, F_CARRIER_HZ)
    rate_max = max_doppler_rate_hz_per_s(ALTITUDE_KM, v, F_CARRIER_HZ)
    alpha_leo, v_underwater = realistic_wideband_scale_context()
    print(f"LEO orbital velocity at {ALTITUDE_KM:.0f}km: {v:.3f} km/s")
    print(f"Max Doppler at {F_CARRIER_HZ/1e6:.0f}MHz: +/-{f_max/1e3:.2f}kHz")
    print(f"Max Doppler rate (at closest approach): {rate_max/1e3:.1f} kHz/s")
    print(f"Real LEO Doppler *scale* alpha: {alpha_leo:.8f} (vs. 1.05 used in the wideband demo)")
    print(f"alpha=1.05 needs v={v_underwater:.0f}m/s relative to the speed of sound in water -- realistic "
          f"for a fast underwater vehicle, not RF")

    pass_times, results, velocity_kms, snr_db = experiment_leo_pass_reliability()
    plot_leo_pass(pass_times, results, velocity_kms, snr_db)
    print(f"Wrote figure to {OUT_DIR}")


if __name__ == "__main__":
    main()
