r"""
Chaos: sensitivity to initial conditions and the Lyapunov exponent.

Motivation
----------
The original report showed chaos "by eye": changing an initial condition by
0.1 % produced visibly different trajectories. To make it *quantitative* we
measure how the separation delta(t) between two solutions that start a tiny
distance delta_0 apart grows in time:

    regular (integrable) motion :  delta(t) ~ delta_0 * t          (linear growth)
    chaotic motion              :  delta(t) ~ delta_0 * e^{lambda t} (exponential growth)

lambda > 0 is the **maximal Lyapunov exponent**. Its inverse t_L = 1/lambda is
the *Lyapunov time*: every t_L the initial error is multiplied by e ~ 2.7.
Practical consequence: to predict up to time t with tolerance Delta we must
know the initial conditions to a precision delta_0 ~ Delta e^{-lambda t}; each
gain in prediction horizon requires exponentially better measurements.

Benettin's method (1980)
------------------------
If delta(t) is left to grow, it saturates at the size of the system and stops
measuring anything. Benettin's recipe: every tau time units
    1. measure d = |delta|,
    2. accumulate ln(d / delta_0),
    3. rescale the perturbation back to length delta_0 (same direction).
Then lambda ~ (1 / t) sum ln(d_k / delta_0). The direction of delta aligns by
itself with the direction of maximal stretching, so we get the *largest*
exponent.

Distance in phase space
-----------------------
Positions and velocities have different units; we make them dimensionless
with a length L (rms radius about the CM) and a time T = L / v_rms:

    d^2 = sum_i |dr_i|^2 / L^2  +  sum_i |dv_i|^2 T^2 / L^2

Data layout
-----------
    x, v, xs, vs : (N, 3)   reference state and perturbed "shadow" state
    t_samples    : (K,)     renormalization times
    lyap         : (K,)     running estimate lambda(t) (converges with t)
"""

from __future__ import annotations

import numpy as np

from .forces import make_acceleration_function
from .integrators import get_integrator
from .system import NBodySystem


def phase_space_scales(system: NBodySystem) -> tuple[float, float]:
    """Characteristic length L and time T of the system (to make phase space dimensionless)."""
    cm = system.to_center_of_mass_frame()
    L = float(np.sqrt(np.mean(np.sum(cm.positions ** 2, axis=1))))
    v_rms = float(np.sqrt(np.mean(np.sum(cm.velocities ** 2, axis=1))))
    T = L / v_rms if v_rms > 0 else 1.0
    return L, T


def position_divergence(traj_a, traj_b) -> np.ndarray:
    """delta_r(t) = sqrt(sum_i |r_i^a(t) - r_i^b(t)|^2), shape (T,). Both runs must share their times."""
    if traj_a.t.shape != traj_b.t.shape:
        raise ValueError("Both trajectories must have the same stored instants")
    d = traj_a.positions - traj_b.positions
    return np.sqrt(np.einsum("tnk,tnk->t", d, d))


def lyapunov_benettin(system: NBodySystem, t_final: float, h: float, delta0: float = 1e-8,
                      renorm_every: int = 10, integrator: str = "verlet",
                      seed: int | None = 0) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the maximal Lyapunov exponent lambda (in 1/[time unit]).

    Parameters
    ----------
    delta0       : dimensionless size of the perturbation (small, but far above
                   round-off ~1e-16)
    renorm_every : steps between renormalizations (tau = renorm_every * h)

    Returns (t_samples, lyap): lambda estimated as a function of time. If the
    curve levels off at a value > 0 the system is chaotic; if it decays to 0
    like ln(t)/t, the motion is regular.
    """
    step = get_integrator(integrator).step
    accel = make_acceleration_function(system.masses, system.G, system.softening)
    L, T = phase_space_scales(system)

    # Initial perturbation along a random direction of phase space (6N dimensions)
    rng = np.random.default_rng(seed)
    dx, dv = rng.normal(size=system.positions.shape), rng.normal(size=system.velocities.shape)

    def scaled_norm(dx, dv):
        return float(np.sqrt(np.sum(dx ** 2) / L ** 2 + np.sum(dv ** 2) * T ** 2 / L ** 2))

    norm = scaled_norm(dx, dv)
    dx, dv = dx * delta0 / norm, dv * delta0 / norm

    x, v = system.positions.copy(), system.velocities.copy()
    xs, vs = x + dx, v + dv
    a, a_s = accel(x), accel(xs)

    n_steps = int(np.ceil(t_final / h - 1e-9))
    n_samples = n_steps // renorm_every
    t_samples, lyap = np.empty(n_samples), np.empty(n_samples)
    log_sum = 0.0
    for k in range(n_samples):
        for _ in range(renorm_every):
            x, v, a = step(x, v, a, h, accel)
            xs, vs, a_s = step(xs, vs, a_s, h, accel)
        dx, dv = xs - x, vs - v
        d = scaled_norm(dx, dv)
        log_sum += np.log(d / delta0)
        # Rescale the shadow back to distance delta0 along the same direction
        xs, vs = x + dx * (delta0 / d), v + dv * (delta0 / d)
        a_s = accel(xs)
        t_samples[k] = (k + 1) * renorm_every * h
        lyap[k] = log_sum / t_samples[k]
    return t_samples, lyap
