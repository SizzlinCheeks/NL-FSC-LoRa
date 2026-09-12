import numpy as np
import pytest

from nlfsc_lora.trajectories import TRAJECTORIES, numerical_derivative


@pytest.mark.parametrize("name", TRAJECTORIES.keys())
def test_endpoints(name):
    g, _ = TRAJECTORIES[name]
    u = np.array([0.0, 1.0])
    g_vals = g(u)
    assert g_vals[0] == pytest.approx(0.0, abs=1e-9)
    assert g_vals[1] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("name", ["linear", "quadratic", "cubic", "sigmoid", "sinusoidal"])
def test_monotonic_increasing(name):
    g, _ = TRAJECTORIES[name]
    u = np.linspace(0, 1, 2001)
    g_vals = g(u)
    assert np.all(np.diff(g_vals) >= -1e-9)


@pytest.mark.parametrize("name,dg", [(n, TRAJECTORIES[n][1]) for n in TRAJECTORIES if TRAJECTORIES[n][1]])
def test_closed_form_derivative_matches_numerical(name, dg):
    g, _ = TRAJECTORIES[name]
    u = np.linspace(0.02, 0.98, 50)
    assert np.allclose(dg(u), numerical_derivative(g, u), atol=1e-3)
