"""Physics of the forces: Newton's third law, an analytical case, F = -grad U."""

import numpy as np

from nbody.forces import accelerations, potential_energy

RNG = np.random.default_rng(42)


def random_config(n=6):
    return RNG.uniform(0.5, 2.0, n), RNG.normal(size=(n, 3))


def test_third_law_total_force_is_zero():
    """Action-reaction: sum_i m_i a_i = sum F_i = 0 (internal forces cancel in pairs)."""
    m, x = random_config()
    a = accelerations(x, m, G=1.0)
    assert np.allclose(m @ a, 0.0, atol=1e-12)


def test_two_body_matches_newton():
    """Two bodies a distance d apart: |a_1| = G m_2 / d^2, pointing towards body 2."""
    m = np.array([3.0, 5.0])
    x = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    a = accelerations(x, m, G=1.5)
    assert np.allclose(a[0], [1.5 * 5.0 / 4.0, 0, 0])
    assert np.allclose(a[1], [-1.5 * 3.0 / 4.0, 0, 0])


def test_no_self_interaction_single_body():
    a = accelerations(np.zeros((1, 3)), np.array([1.0]), G=1.0)
    assert np.all(a == 0) and np.all(np.isfinite(a))


def test_potential_matches_pair_sum():
    m, x = random_config(4)
    U = sum(-m[i] * m[j] / np.linalg.norm(x[i] - x[j]) for i in range(4) for j in range(i + 1, 4))
    assert np.isclose(potential_energy(x, m, G=1.0), U)


def _check_force_is_minus_gradient(softening):
    """m_i a_i = -dU/dr_i, checked with centred finite differences."""
    m, x = random_config(5)
    a = accelerations(x, m, G=1.0, softening=softening)
    eps = 1e-6
    grad = np.zeros_like(x)
    for i in range(x.shape[0]):
        for k in range(3):
            xp, xm = x.copy(), x.copy()
            xp[i, k] += eps
            xm[i, k] -= eps
            grad[i, k] = (potential_energy(xp, m, 1.0, softening) - potential_energy(xm, m, 1.0, softening)) / (2 * eps)
    assert np.allclose(m[:, None] * a, -grad, rtol=1e-6, atol=1e-8)


def test_force_is_minus_gradient_of_potential():
    _check_force_is_minus_gradient(0.0)


def test_softened_force_is_consistent_with_softened_potential():
    _check_force_is_minus_gradient(0.3)
