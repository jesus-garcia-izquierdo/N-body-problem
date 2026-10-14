"""Conservation laws, virial theorem, scenarios and utilities."""

import os
import tempfile

import numpy as np

from nbody import NBodySystem, Trajectory, chaos, diagnostics, scenarios, simulate


def _random_system(n=5, seed=1):
    rng = np.random.default_rng(seed)
    return NBodySystem(rng.uniform(0.5, 1.5, n), rng.normal(size=(n, 3)), 0.3 * rng.normal(size=(n, 3)),
                       G=1.0, softening=0.05)


def test_momentum_and_angular_momentum_conserved_to_roundoff():
    """Verlet conserves P and L exactly (up to round-off): central, pairwise internal forces."""
    traj = simulate(_random_system(), 5.0, 1e-3, "verlet", save_every=50)
    P, L = traj.linear_momentum(), traj.angular_momentum()
    assert diagnostics.relative_error(P, diagnostics.momentum_scale(traj.masses, traj.velocities)).max() < 1e-12
    assert diagnostics.relative_error(
        L, diagnostics.angular_momentum_scale(traj.masses, traj.positions, traj.velocities)).max() < 1e-12


def test_symplectic_energy_bounded_vs_euler_drift():
    sys = scenarios.figure_eight()
    T = scenarios.FIGURE_EIGHT_PERIOD
    errs = {}
    for name in ("euler", "verlet"):
        traj = simulate(sys, 20 * T, T / 500, name, save_every=50)
        errs[name] = diagnostics.relative_error(traj.total_energy())
    # Verlet: small error with no trend (the 2nd half is not worse than the 1st)
    v = errs["verlet"]
    assert v.max() < 1e-3
    assert v[len(v) // 2:].max() < 2 * v[: len(v) // 2].max()
    # Euler: the error grows much more
    assert errs["euler"][-1] > 100 * v.max()


def test_virial_theorem_figure_eight():
    T = scenarios.FIGURE_EIGHT_PERIOD
    traj = simulate(scenarios.figure_eight(), 5 * T, T / 2000, save_every=10)
    assert abs(diagnostics.virial_ratio(traj.kinetic_energy(), traj.potential_energy()) - 1) < 1e-3


def test_center_of_mass_frame():
    sys = scenarios.sun_earth_moon(center_of_mass_frame=False).to_center_of_mass_frame()
    assert np.allclose(sys.center_of_mass(), 0, atol=1e-3)                  # metres, out of 1e11
    assert np.allclose(sys.masses @ sys.velocities, 0, atol=1e-6 * sys.total_mass)


def test_planar_input_is_padded_to_3d():
    sys = scenarios.figure_eight()
    assert sys.positions.shape == (3, 3) and np.all(sys.positions[:, 2] == 0)


def test_perturbed_returns_copy():
    base = scenarios.six_body_report()
    p = base.perturbed(0, "position", component=0, relative=1e-3)
    assert p.positions[0, 0] == base.positions[0, 0] * (1 + 1e-3)
    assert base.positions[0, 0] == 1e5                                          # the original is unchanged


def test_routh_criterion():
    assert not scenarios.routh_stable((1, 1, 1))
    assert scenarios.routh_stable((1, 1e-3, 1e-6))


def test_plummer_starts_virialized():
    sys = scenarios.plummer_cluster(400, seed=3)
    traj = simulate(sys, 1e-3, 1e-3)
    Q = 2 * traj.kinetic_energy()[0] / -traj.potential_energy()[0]
    assert 0.8 < Q < 1.2


def test_divergence_of_identical_runs_is_zero():
    traj = simulate(scenarios.figure_eight(), 1.0, 1e-2)
    assert np.all(chaos.position_divergence(traj, traj) == 0)


def test_trajectory_save_load_roundtrip():
    traj = simulate(scenarios.figure_eight(), 0.5, 1e-2)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.npz")
        traj.save(path)
        back = Trajectory.load(path)
    assert np.array_equal(back.positions, traj.positions) and back.names == traj.names


def test_lagrange_growth_rate_theory():
    """Equal masses: sigma = omega / sqrt(2); Sun-Jupiter-Trojan: linearly stable (sigma = 0)."""
    assert abs(scenarios.lagrange_growth_rate((1, 1, 1)) - np.sqrt(3) / np.sqrt(2)) < 1e-9
    assert scenarios.lagrange_growth_rate((1, 1e-3, 1e-6)) == 0.0
