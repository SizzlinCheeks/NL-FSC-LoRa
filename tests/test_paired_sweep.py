import numpy as np
import pytest

from nlfsc_lora.chirp import ChirpConfig, base_waveform, symbol_waveform
from nlfsc_lora.channel import apply_doppler_scale, awgn
from nlfsc_lora.doppler import peak_response
from nlfsc_lora.receiver import fft_correlation_demod
from nlfsc_lora.trajectories import TRAJECTORIES, hyperbolic_center_freq
from nlfsc_lora.paired_sweep import (
    acquire_doppler_scale,
    acquire_doppler_scale_calibrated,
    build_alpha_calibration,
    combine_paired_lags,
    correct_doppler_scale,
    doppler_scale_from_bias,
    fit_bias_ratio,
    paired_sweep_decode,
    raw_lag_estimate,
    reversed_trajectory,
)

Q = 1.0 / 3.0


def make_cfgs(sf=7, bw=125e3, os=4, q=Q):
    g_up, _ = TRAJECTORIES["hyperbolic"]
    g_down = reversed_trajectory(g_up)
    f_center = hyperbolic_center_freq(bw, q)
    cfg_up = ChirpConfig(sf=sf, bandwidth=bw, sample_rate=os * bw, g=g_up, f_center=f_center)
    cfg_down = ChirpConfig(sf=sf, bandwidth=bw, sample_rate=os * bw, g=g_down, f_center=f_center)
    return cfg_up, cfg_down


def test_reversed_trajectory_matches_boundary_conditions_and_preserves_doppler_tolerance():
    """g_down(0)=g(1)=1, g_down(1)=g(0)=0 -- a genuine down-sweep (starts at the
    top of the band, ends at the bottom), not a no-op. Its matched-filter peak
    should stay high under a Doppler scale, the same as the up-sweep's --
    confirmed directly, since this is what distinguishes the correct
    construction (time-reversal, g(1-u)) from the wrong one (amplitude-flip,
    1-g(u)), which was tried first and found to degrade badly."""
    cfg_up, cfg_down = make_cfgs()
    assert cfg_down.g(0.0) == pytest.approx(1.0)
    assert cfg_down.g(1.0) == pytest.approx(0.0, abs=1e-9)

    for alpha in [0.9, 1.1]:
        peak_up, _ = peak_response(cfg_up, alpha)
        peak_down, _ = peak_response(cfg_down, alpha)
        assert peak_up > 0.8
        assert peak_down > 0.8


def test_delta_down_equals_minus_q_times_delta_up():
    """The module's central closed form, checked directly against simulated
    lags rather than assumed: Delta_down(alpha) = -q * Delta_up(alpha), not a
    naive equal-and-opposite pair."""
    cfg_up, cfg_down = make_cfgs()
    ref_up = base_waveform(cfg_up)
    ref_down = base_waveform(cfg_down)
    for alpha in [0.85, 0.95, 1.05, 1.15]:
        lag_up = raw_lag_estimate(apply_doppler_scale(ref_up, alpha), cfg_up)
        lag_down = raw_lag_estimate(apply_doppler_scale(ref_down, alpha), cfg_down)
        assert lag_down == pytest.approx(-Q * lag_up, abs=1.0)


def test_combine_paired_lags_exact_solve():
    """Direct algebraic check: feeding combine_paired_lags exactly
    shift + Delta and shift - q*Delta should recover both exactly."""
    shift, delta = 37.5, 12.0
    lag_up = shift + delta
    lag_down = shift - Q * delta
    shift_hat, delta_hat = combine_paired_lags(lag_up, lag_down, Q, n_samples=512)
    assert shift_hat == pytest.approx(shift, abs=1e-9)
    assert delta_hat == pytest.approx(delta, abs=1e-9)


def test_combine_paired_lags_handles_wraparound():
    """raw_lag_estimate wraps each lag independently into (-n/2, n/2]; a true
    shift near that branch cut can put lag_up and lag_down on opposite sides
    of it even though they measure the same shift. combine_paired_lags must
    resolve that, not silently corrupt the combination by close to n_samples."""
    n = 512
    shift, delta = n / 2 - 3, 8.0  # near the branch cut
    lag_up_true = shift + delta
    lag_down_true = shift - Q * delta
    # simulate raw_lag_estimate's own (-n/2, n/2] wrap on each independently
    def wrap(x):
        x = x % n
        return x - n if x > n / 2 else x
    lag_up, lag_down = wrap(lag_up_true), wrap(lag_down_true)
    shift_hat, _delta_hat = combine_paired_lags(lag_up, lag_down, Q, n)
    shift_hat_wrapped = shift_hat % n
    assert min(abs(shift_hat_wrapped - shift), abs(shift_hat_wrapped - shift - n)) < 1.0


def test_doppler_scale_from_bias_round_trips_through_combine():
    """acquire_doppler_scale's whole pipeline (raw_lag_estimate -> combine ->
    invert), checked end to end on a noiseless m=0 preamble pair, should
    recover alpha to a small fraction of a percent."""
    cfg_up, cfg_down = make_cfgs()
    for alpha_true in [0.85, 0.95, 1.0, 1.05, 1.15]:
        ref_up = base_waveform(cfg_up)
        ref_down = base_waveform(cfg_down)
        rx_up = apply_doppler_scale(ref_up, alpha_true)
        rx_down = apply_doppler_scale(ref_down, alpha_true)
        alpha_hat = acquire_doppler_scale(rx_up, rx_down, cfg_up, cfg_down, Q)
        assert alpha_hat == pytest.approx(alpha_true, abs=2e-3)


def test_paired_sweep_decode_exact_near_m_zero():
    """paired_sweep_decode's documented exact-case: m at/near 0, no cyclic-shift
    wrap discontinuity inside the burst."""
    cfg_up, cfg_down = make_cfgs()
    for m_true in [0, 1, 2]:
        for alpha in [0.9, 1.1]:
            rx_up = apply_doppler_scale(symbol_waveform(cfg_up, m_true), alpha)
            rx_down = apply_doppler_scale(symbol_waveform(cfg_down, m_true), alpha)
            m_hat, alpha_hat = paired_sweep_decode(rx_up, rx_down, cfg_up, cfg_down, Q)
            assert m_hat == m_true
            assert alpha_hat == pytest.approx(alpha, abs=1e-2)


def test_acquire_and_correct_recovers_full_decode_accuracy_across_full_m_range():
    """The actual validated capability: acquire alpha from a m=0 preamble pair,
    correct an ordinary single-sweep payload burst with it, decode normally --
    across the *entire* symbol alphabet, not just near m=0 (paired_sweep_decode's
    own limitation). An uncorrected receiver fails completely (~100% SER) at
    these Doppler scales (matches 08_lora_ser_vs_doppler_scale.png's own
    half-bin-sensitivity finding); this should recover near-0% SER."""
    cfg_up, cfg_down = make_cfgs()
    rng = np.random.default_rng(0)
    for alpha_true in [0.9, 1.1]:
        alpha_hat = acquire_doppler_scale(
            apply_doppler_scale(base_waveform(cfg_up), alpha_true),
            apply_doppler_scale(base_waveform(cfg_down), alpha_true),
            cfg_up, cfg_down, Q,
        )
        errors_corrected, errors_uncorrected = 0, 0
        n_trials = 100
        for _ in range(n_trials):
            m_true = int(rng.integers(0, cfg_up.M))
            rx = apply_doppler_scale(symbol_waveform(cfg_up, m_true), alpha_true)
            if fft_correlation_demod(rx, cfg_up) != m_true:
                errors_uncorrected += 1
            if fft_correlation_demod(correct_doppler_scale(rx, alpha_hat), cfg_up) != m_true:
                errors_corrected += 1
        assert errors_uncorrected > 0.9 * n_trials
        assert errors_corrected == 0


def test_acquire_doppler_scale_robust_at_negative_snr():
    """The same full-symbol coherent gain that fixed afc.py's negative-SNR
    floor (raw_lag_estimate correlates the whole symbol) should carry over
    here: acquisition should stay accurate well below 0dB SNR."""
    cfg_up, cfg_down = make_cfgs()
    rng = np.random.default_rng(1)
    alpha_true = 1.10
    errs = []
    for _ in range(200):
        rx_up = awgn(apply_doppler_scale(base_waveform(cfg_up), alpha_true), -10.0, rng)
        rx_down = awgn(apply_doppler_scale(base_waveform(cfg_down), alpha_true), -10.0, rng)
        alpha_hat = acquire_doppler_scale(rx_up, rx_down, cfg_up, cfg_down, Q)
        errs.append(alpha_hat - alpha_true)
    assert np.std(errs) < 0.01


def make_shape_cfgs(shape, sf=7, bw=125e3, os=4):
    """Same construction as make_cfgs(), but for any registered trajectory
    shape rather than hardcoding hyperbolic -- used to check whether the
    paired-sweep mechanism generalizes past the one shape with a closed
    form."""
    g_up, _ = TRAJECTORIES[shape]
    g_down = reversed_trajectory(g_up)
    f_center = hyperbolic_center_freq(bw, Q)  # same nonzero band for every shape, apples to apples
    cfg_up = ChirpConfig(sf=sf, bandwidth=bw, sample_rate=os * bw, g=g_up, f_center=f_center)
    cfg_down = ChirpConfig(sf=sf, bandwidth=bw, sample_rate=os * bw, g=g_down, f_center=f_center)
    return cfg_up, cfg_down


@pytest.mark.parametrize("shape", ["quadratic", "exponential"])
def test_calibrated_acquisition_restores_full_accuracy_for_well_behaved_shapes(shape):
    """The generalized (fit_bias_ratio + calibration-curve) path, checked
    directly: for quadratic and exponential -- neither has hyperbolic's exact
    self-similarity theorem behind it -- acquiring alpha from a m=0 preamble
    pair and correcting payload bursts with it should restore the same
    near-0% SER acquire_doppler_scale() gives hyperbolic, across the full
    symbol alphabet, at the same 10dB SNR an uncorrected receiver fails
    completely at."""
    cfg_up, cfg_down = make_shape_cfgs(shape)
    q_eff = fit_bias_ratio(cfg_up, cfg_down)
    calibration = build_alpha_calibration(cfg_up, cfg_down, q_eff)

    rng = np.random.default_rng(0)
    for alpha_true in [0.9, 1.1]:
        rx_pre_up = awgn(apply_doppler_scale(base_waveform(cfg_up), alpha_true), 10.0, rng)
        rx_pre_down = awgn(apply_doppler_scale(base_waveform(cfg_down), alpha_true), 10.0, rng)
        alpha_hat = acquire_doppler_scale_calibrated(rx_pre_up, rx_pre_down, cfg_up, cfg_down, q_eff, calibration)
        assert alpha_hat == pytest.approx(alpha_true, abs=2e-3)

        errors_corrected, errors_uncorrected = 0, 0
        n_trials = 100
        for _ in range(n_trials):
            m_true = int(rng.integers(0, cfg_up.M))
            rx = apply_doppler_scale(symbol_waveform(cfg_up, m_true), alpha_true)
            if fft_correlation_demod(rx, cfg_up) != m_true:
                errors_uncorrected += 1
            if fft_correlation_demod(correct_doppler_scale(rx, alpha_hat), cfg_up) != m_true:
                errors_corrected += 1
        assert errors_uncorrected > 0.9 * n_trials
        assert errors_corrected == 0


def test_calibrated_acquisition_degrades_for_sigmoid_near_the_edge_of_the_alpha_range():
    """The honest negative finding, pinned as a regression test rather than
    just described: sigmoid's own calibration curve fits decently on average,
    but at 10dB SNR with a single preamble burst its alpha estimate is
    measurably noisier than quadratic's at the same extreme alpha -- traced
    to sigmoid's near-flat instantaneous frequency at the band edges
    (trajectories.py::sigmoid's own "lingers near the band edges" docstring)
    carrying less Doppler-scale information there. This isn't a claim that
    sigmoid never works (it's fine near alpha=1, see the generalization
    figure) -- only that, unlike quadratic/exponential, it does not match
    hyperbolic's robustness at the edges of the tested range."""
    alpha_true = 0.90
    n_trials = 60

    def alpha_err_std(shape):
        cfg_up, cfg_down = make_shape_cfgs(shape)
        q_eff = fit_bias_ratio(cfg_up, cfg_down)
        calibration = build_alpha_calibration(cfg_up, cfg_down, q_eff)
        rng = np.random.default_rng(2)
        errs = []
        for _ in range(n_trials):
            rx_up = awgn(apply_doppler_scale(base_waveform(cfg_up), alpha_true), 10.0, rng)
            rx_down = awgn(apply_doppler_scale(base_waveform(cfg_down), alpha_true), 10.0, rng)
            alpha_hat = acquire_doppler_scale_calibrated(rx_up, rx_down, cfg_up, cfg_down, q_eff, calibration)
            errs.append(alpha_hat - alpha_true)
        return float(np.std(errs))

    std_sigmoid = alpha_err_std("sigmoid")
    std_quadratic = alpha_err_std("quadratic")
    assert std_sigmoid > 5 * std_quadratic
