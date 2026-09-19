"""Pins for examples/leo_satellite_experiments.py: a LEO satellite Doppler
pass computed from real orbital mechanics (not a synthetic profile),
driving the same real-LoRaWAN-framed packet pipeline as
nonlinear_lorawan_experiments.py. Skips cleanly if lora_phy isn't
installed.
"""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("lora_phy")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
from leo_satellite_experiments import (
    ALTITUDE_KM,
    F_CARRIER_HZ,
    doppler_profile_hz,
    leo_orbital_velocity_kms,
    max_doppler_hz,
    max_doppler_rate_hz_per_s,
    realistic_wideband_scale_context,
)


def test_leo_orbital_velocity_matches_known_range():
    """~550km LEO orbital velocity should be close to the well-known
    ~7.5-7.6km/s figure for this altitude band (ISS/Starlink-ish)."""
    v = leo_orbital_velocity_kms(ALTITUDE_KM)
    assert 7.4 < v < 7.7


def test_max_doppler_matches_known_satellite_iot_range():
    """Real satellite-IoT/LEO systems in sub-GHz bands see maximum Doppler
    shifts on the order of tens of kHz (e.g. Iridium ~37kHz at 1.6GHz
    scales to ~20kHz at 868MHz) -- this should land in the same
    ballpark."""
    v = leo_orbital_velocity_kms(ALTITUDE_KM)
    f_max = max_doppler_hz(v, F_CARRIER_HZ)
    assert 15e3 < f_max < 30e3


def test_doppler_profile_is_zero_at_closest_approach_and_antisymmetric():
    v = leo_orbital_velocity_kms(ALTITUDE_KM)
    t = np.array([-100.0, 0.0, 100.0])
    f_d = doppler_profile_hz(t, ALTITUDE_KM, v, F_CARRIER_HZ)
    assert abs(f_d[1]) < 1e-6
    assert np.isclose(f_d[0], -f_d[2])
    assert f_d[0] > 0  # approaching -> blueshift


def test_doppler_profile_approaches_max_far_from_closest_approach():
    v = leo_orbital_velocity_kms(ALTITUDE_KM)
    f_max = max_doppler_hz(v, F_CARRIER_HZ)
    f_d_far = doppler_profile_hz(np.array([-1e6]), ALTITUDE_KM, v, F_CARRIER_HZ)[0]
    assert np.isclose(f_d_far, f_max, rtol=1e-3)


def test_doppler_rate_peaks_at_closest_approach():
    v = leo_orbital_velocity_kms(ALTITUDE_KM)
    rate_analytic = max_doppler_rate_hz_per_s(ALTITUDE_KM, v, F_CARRIER_HZ)
    dt = 0.001
    f_before, f_after = doppler_profile_hz(np.array([-dt / 2, dt / 2]), ALTITUDE_KM, v, F_CARRIER_HZ)
    rate_numeric = (f_after - f_before) / dt
    assert np.isclose(rate_analytic, -rate_numeric, rtol=1e-3)


def test_real_leo_doppler_scale_is_far_below_the_wideband_demo_value():
    """The whole point of realistic_wideband_scale_context: alpha=1.05
    used in the wideband demo (31_nonlinear_lorawan_wideband_scale.png) is
    not an RF satellite number -- real LEO alpha departs from 1 by
    roughly 2.5e-5, several orders of magnitude smaller."""
    alpha_leo, v_underwater = realistic_wideband_scale_context()
    assert abs(alpha_leo - 1.0) < 1e-4
    assert abs(alpha_leo - 1.0) > 1e-6
    assert 50.0 < v_underwater < 100.0  # a fast underwater vehicle, not RF
