"""Numerical tests of the integrators: order of convergence, time reversibility, circular orbit."""

import numpy as np

from nbody import NBodySystem, scenarios, simulate
from nbody.forces import make_acceleration_function
from nbody.integrators import INTEGRATORS, velocity_verlet_step


def _measured_order(name, t_final=1.5):
    """Order p estimated from the final error with h and h/2: p = log2(e(h) / e(h/2))."""
    sys = scenarios.figure_eight()
    ref = simulate(sys, t_final, t_final / 20_000, "yoshida4", save_every=20_000).positions[-1]
    errs = []
    for n in (200, 400):
        x = simulate(sys, t_final, t_final / n, name, save_every=n).positions[-1]
        errs.append(np.linalg.norm(x - ref))
    return np.log2(errs[0] / errs[1])


def test_convergence_orders():
    for name, info in INTEGRATORS.items():
        p = _measured_order(name)
        assert abs(p - info.order) < 0.35, f"{name}: measured order {p:.2f}, expected {info.order}"


def test_verlet_is_time_reversible():
    """n steps forward, reverse v, n steps forward again: back to the initial state."""
    sys = scenarios.figure_eight()
    accel = make_acceleration_function(sys.masses, sys.G)
    x, v = sys.positions.copy(), sys.velocities.copy()
    a = accel(x)
    for _ in range(2000):
        x, v, a = velocity_verlet_step(x, v, a, 1e-3, accel)
    v = -v
    for _ in range(2000):
        x, v, a = velocity_verlet_step(x, v, a, 1e-3, accel)
    assert np.allclose(x, sys.positions, atol=1e-10)
    assert np.allclose(-v, sys.velocities, atol=1e-10)


def test_circular_orbit_returns_after_one_period():
    """Test mass on a circular orbit: T = 2 pi sqrt(r^3 / G M)."""
    M, r, G = 1.0, 1.0, 1.0
    sys = NBodySystem([M, 1e-12], [[0, 0], [r, 0]], [[0, 0], [0, np.sqrt(G * M / r)]], G=G)
    period = 2 * np.pi * np.sqrt(r ** 3 / (G * M))
    traj = simulate(sys, period, period / 4000, "yoshida4", save_every=4000)
    assert np.linalg.norm(traj.positions[-1, 1] - traj.positions[0, 1]) < 1e-9
