import numpy as np
import pytest

from nlfsc_lora.afc import AFCLoop, KalmanAFCLoop, dual_edge_cfo_estimate, run_afc_sequence
from nlfsc_lora.channel import apply_cfo
from nlfsc_lora.chirp import ChirpConfig, symbol_waveform
from nlfsc_lora.trajectories import TRAJECTORIES, hyperbolic_center_freq


def make_cfg(sf=7, bw=125e3, os=4):
    g, dg = TRAJECTORIES["hyperbolic"]
    cfg = ChirpConfig(sf=sf, bandwidth=bw, sample_rate=os * bw, g=g, f_center=hyperbolic_center_freq(bw))
    return cfg, dg


@pytest.mark.parametrize("m_true,cfo_true", [(0, 0.0), (33, 300.0), (100, -200.0), (127, 450.0)])
def test_dual_edge_estimate_exact_given_correct_symbol_noiseless(m_true, cfo_true):
    """The combined estimate should be accurate regardless of m -- even for m=100,
    where the edge-A window happens to straddle this symbol's wrap glitch (mismatch_a
    ~1.06): quality-weighting should let the clean edge (B) dominate rather than
    averaging in the corrupted one. That's the point of using two edges instead of
    one -- checked directly, not just asserted, since a single window can and does
    land on the glitch for some symbols."""
    cfg, dg = make_cfg()
    rx = apply_cfo(symbol_waveform(cfg, m_true), cfo_true, cfg.sample_rate)
    cfo_est, mismatch_a, mismatch_b = dual_edge_cfo_estimate(rx, cfg, dg, m_true)
    assert cfo_est == pytest.approx(cfo_true, abs=2.0)
    assert min(mismatch_a, mismatch_b) < 0.01  # at least one edge must be clean


def test_dual_edge_estimate_flags_wrong_symbol_hypothesis():
    """A wrong m_hat should show up as a large rate mismatch at both edges,
    not silently produce a plausible-looking but wrong CFO estimate."""
    cfg, dg = make_cfg()
    rx = apply_cfo(symbol_waveform(cfg, 50), 300.0, cfg.sample_rate)
    _cfo_correct, mismatch_a_correct, mismatch_b_correct = dual_edge_cfo_estimate(rx, cfg, dg, 50)
    _cfo_wrong, mismatch_a_wrong, mismatch_b_wrong = dual_edge_cfo_estimate(rx, cfg, dg, 90)
    assert mismatch_a_wrong > mismatch_a_correct
    assert mismatch_b_wrong > mismatch_b_correct


def test_afc_loop_converges_toward_repeated_measurement():
    loop = AFCLoop(gain=0.3, cfo_tracked=0.0)
    for _ in range(50):
        loop.update(500.0)
    assert loop.cfo_tracked == pytest.approx(500.0, abs=1.0)


def test_afc_loop_ignores_untrusted_none_measurement():
    loop = AFCLoop(gain=0.3, cfo_tracked=123.0)
    result = loop.update(None)
    assert result == 123.0


def test_afc_loop_gates_a_wild_outlier_measurement():
    """Discovered by testing over long (thousands-of-bursts) sequences, not
    assumed: dual_edge_cfo_estimate's mismatch check occasionally (~1 in 400,
    measured directly) passes a measurement that is off by 1-4.5 kHz anyway.
    Without a sanity gate, AFCLoop trusts it, jumps far outside
    fft_correlation_demod's own capture range, and every subsequent decode
    (and thus every subsequent measurement) is corrupted with no way back --
    a run long enough to hit this even once effectively never recovers. The
    gate should reject the single wild reading and keep tracking normally
    once good measurements resume."""
    loop = AFCLoop(gain=0.3, cfo_tracked=500.0, max_jump_hz=1200.0)
    loop.update(510.0)  # ordinary small measurement: passes through
    assert loop.cfo_tracked == pytest.approx(503.0, abs=0.5)

    tracked_before_outlier = loop.cfo_tracked
    loop.update(tracked_before_outlier - 5000.0)  # wild outlier: must be rejected
    assert loop.cfo_tracked == pytest.approx(tracked_before_outlier, abs=0.5)

    loop.update(tracked_before_outlier + 20.0)  # loop keeps tracking normally afterward
    assert loop.cfo_tracked != pytest.approx(tracked_before_outlier, abs=0.001)


def test_kalman_loop_converges_toward_repeated_measurement():
    loop = KalmanAFCLoop()
    for _ in range(50):
        loop.update(500.0)
    assert loop.cfo_tracked == pytest.approx(500.0, abs=5.0)


def test_kalman_loop_ignores_untrusted_none_measurement():
    loop = KalmanAFCLoop(cfo_tracked=123.0, rate_tracked=0.0)
    result = loop.update(None)
    # a None measurement is predict-only: with rate_tracked=0 the state shouldn't move
    assert result == pytest.approx(123.0, abs=1e-9)


def test_kalman_loop_gates_a_wild_outlier_innovation():
    """The Kalman-native version of AFCLoop's max_jump_hz check: reject a
    measurement whose innovation is implausible relative to the filter's own
    predicted uncertainty, rather than absorbing it into the state estimate."""
    loop = KalmanAFCLoop(cfo_tracked=500.0, rate_tracked=0.0)
    for _ in range(10):
        loop.update(500.0)  # let the filter converge and its uncertainty shrink
    tracked_before_outlier = loop.cfo_tracked
    loop.update(tracked_before_outlier + 50000.0)  # wild outlier
    assert loop.cfo_tracked == pytest.approx(tracked_before_outlier, abs=5.0)


def test_tracking_survives_a_drift_that_exceeds_the_static_capture_range():
    """The core claim: a receiver using the dual-edge loop should keep decoding
    correctly as CFO drifts far past the ~half-bin static tolerance, while a
    receiver that only corrects once (no tracking) should fail once the drift
    moves away from wherever it was initially acquired.

    Drift rate matters, not just total drift: a plain proportional loop has a
    real, textbook steady-state lag tracking a ramp (~drift_rate/gain -- a
    type-1 control system's velocity error), so the drift here is chosen slow
    enough relative to the bin size and loop gain that the lag stays well
    inside the tolerance (verified: max residual ~484Hz vs a ~977Hz half-bin
    at this SF). A much faster ramp against the same gain can and does fail
    this way -- that tradeoff is real, not a bug, and is worth documenting
    rather than tuned away silently.
    """
    cfg, dg = make_cfg(sf=6)
    bin_hz = cfg.bandwidth / cfg.M
    rng = np.random.default_rng(3)
    n_bursts = 100
    true_symbols = rng.integers(0, cfg.M, n_bursts)
    true_cfo = np.linspace(0, 4 * bin_hz, n_bursts)  # drifts to well past the half-bin tolerance

    decoded_tracked, _tr, _res = run_afc_sequence(cfg, dg, true_symbols, true_cfo, snr_db=10, gain=0.3, seed=0)
    decoded_static, _tr2, _res2 = run_afc_sequence(cfg, dg, true_symbols, true_cfo, snr_db=10, gain=0.0, seed=0)

    acc_tracked = np.mean(decoded_tracked == true_symbols)
    acc_static = np.mean(decoded_static == true_symbols)
    assert acc_tracked > 0.9
    assert acc_static < acc_tracked


def test_cold_start_beyond_capture_range_needs_acquisition():
    """Documents the discovered pull-in-range limitation: without the acquisition
    step, a cold start against an offset beyond fft_correlation_demod's own
    half-bin tolerance decodes the very first burst wrong, corrupting the loop."""
    cfg, dg = make_cfg(sf=6)
    bin_hz = cfg.bandwidth / cfg.M
    rng = np.random.default_rng(4)
    n_bursts = 20
    true_symbols = rng.integers(0, cfg.M, n_bursts)
    true_cfo = np.full(n_bursts, 4 * bin_hz)  # comfortably beyond the half-bin tolerance

    decoded_with_acq, _t1, _r1 = run_afc_sequence(
        cfg, dg, true_symbols, true_cfo, snr_db=15, seed=0, acquire=True, acquire_span=2 * true_cfo[0]
    )
    decoded_without_acq, _t2, _r2 = run_afc_sequence(cfg, dg, true_symbols, true_cfo, snr_db=15, seed=0, acquire=False)

    assert np.mean(decoded_with_acq == true_symbols) > 0.9
    assert np.mean(decoded_without_acq == true_symbols) < 0.5


def test_multi_burst_acquisition_beats_single_burst_at_low_snr():
    """Discovered while building an end-to-end packet test (preamble + random
    payload): joint_cfo_symbol_search's wide candidate search is itself less
    SNR-robust than a plain single-hypothesis decode, since testing many CFO
    candidates against one noisy burst gives more chances for a noise-induced
    false peak (measured: 27% single-burst acquisition symbol error at -15dB
    where plain decoding had 0%). acquire_bursts > 1 -- what a multi-symbol
    preamble is actually for -- majority-votes across several independent
    acquisition bursts instead of trusting one, closing most of that gap."""
    cfg, dg = make_cfg()
    true_cfo = 300.0  # small enough to stay within acquire_span throughout
    n_bursts = 20
    snr_db = -15.0
    n_trials = 60

    errs_single, errs_multi = 0, 0
    for seed in range(n_trials):
        rng = np.random.default_rng(seed)
        true_symbols = rng.integers(0, cfg.M, n_bursts)
        seq = np.full(n_bursts, true_cfo)
        dec_single, _t, _r = run_afc_sequence(cfg, dg, true_symbols, seq, snr_db, seed=seed, acquire_bursts=1)
        dec_multi, _t, _r = run_afc_sequence(cfg, dg, true_symbols, seq, snr_db, seed=seed, acquire_bursts=8)
        errs_single += int(np.sum(dec_single != true_symbols))
        errs_multi += int(np.sum(dec_multi != true_symbols))

    assert errs_multi < errs_single


def _flyover_cfo(max_cfo: float, t0: float, half: int, n_bursts: int) -> np.ndarray:
    """Constant-velocity closest-point-of-approach Doppler curve: an S-curve
    saturating to +/-max_cfo far from t=0, crossing zero at t=0 -- a UAV or
    drone approaching, passing near the receiver, then receding, unlike a
    satellite pass's roughly-monotonic ramp. t0 (in burst units) sets how many
    bursts the sign reversal takes; max_cfo/t0 is the instantaneous rate at
    the steepest point, t=0."""
    t = np.linspace(-half, half, n_bursts)
    return -max_cfo * t / np.sqrt(t ** 2 + t0 ** 2)


def test_afc_tracks_through_a_doppler_sign_reversal():
    """The UAV case: unlike a satellite pass, Doppler here swings from positive
    to negative within the same run. Measured directly (see
    examples/output/19_afc_rate_of_change_limit.png), both trackers hold
    decode accuracy near 1.0 once the instantaneous rate stays below roughly
    55 Hz/burst -- itself many orders of magnitude past realistic UAV or
    satellite Doppler acceleration. t0=90 here (max rate ~33 Hz/burst) is
    comfortably inside that working region."""
    cfg, dg = make_cfg()
    max_cfo, t0, half = 3000.0, 90.0, 360
    n_bursts = 2 * half
    true_cfo = _flyover_cfo(max_cfo, t0, half, n_bursts)
    rng = np.random.default_rng(7)
    true_symbols = rng.integers(0, cfg.M, n_bursts)

    dec_afc, _t, _r = run_afc_sequence(cfg, dg, true_symbols, true_cfo, snr_db=10, seed=0, loop=AFCLoop(gain=0.3))
    dec_kal, _t, _r = run_afc_sequence(cfg, dg, true_symbols, true_cfo, snr_db=10, seed=0, loop=KalmanAFCLoop())

    assert np.mean(dec_afc == true_symbols) > 0.95
    assert np.mean(dec_kal == true_symbols) > 0.95


def test_afc_fails_for_an_unrealistically_fast_doppler_reversal():
    """Documents the real, discovered rate-of-change limit rather than implying
    dual-edge tracking works at any speed: a reversal fast enough that the
    per-burst CFO change alone (~200 Hz/burst here) exceeds what the tracker
    can keep up with fails outright, for both trackers -- this is a genuine
    physical limit, not the outlier-measurement bug max_jump_hz/innovation_gate
    fix elsewhere in this file. See examples/output/19_afc_rate_of_change_limit.png
    for where the cliff actually sits (~55 Hz/burst) relative to this."""
    cfg, dg = make_cfg()
    max_cfo, t0, half = 3000.0, 15.0, 150
    n_bursts = 2 * half
    true_cfo = _flyover_cfo(max_cfo, t0, half, n_bursts)
    rng = np.random.default_rng(7)
    true_symbols = rng.integers(0, cfg.M, n_bursts)

    dec_afc, _t, _r = run_afc_sequence(cfg, dg, true_symbols, true_cfo, snr_db=10, seed=0, loop=AFCLoop(gain=0.3))
    assert np.mean(dec_afc == true_symbols) < 0.7
