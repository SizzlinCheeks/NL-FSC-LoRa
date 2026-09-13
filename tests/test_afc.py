import numpy as np
import pytest

from nlfsc_lora.afc import AFCLoop, dual_edge_cfo_estimate, run_afc_sequence
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
