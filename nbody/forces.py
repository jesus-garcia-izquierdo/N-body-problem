r"""
Gravitational accelerations and potential energy, **fully vectorized**.

Physics
-------
Newton's law of universal gravitation. The acceleration of body i is the
superposition of the pulls of all the other bodies:

    a_i = G sum_{j != i} m_j (r_j - r_i) / |r_j - r_i|^3

With *softening* eps the denominator becomes (|r_ij|^2 + eps^2)^{3/2}.
Motivation: when two bodies come very close, 1/r^3 blows up and a fixed-step
integrator gives an unphysical "kick" (the body is ejected). eps removes the
singularity: each body behaves like a fuzzy cloud of size ~eps. Planetary
systems use eps = 0 (exact); star clusters use a small eps > 0.

The potential energy of the system counts every **pair** once:

    U = - G sum_{i<j} m_i m_j / sqrt(|r_ij|^2 + eps^2)

(Same result as the original report, which summed over all i, j and divided
by 2 because every pair appeared twice.)

Numerical strategy (why there are no loops)
-------------------------------------------
The MATLAB code looped over bodies with two nested ``for`` loops and grew the
vectors with ``delta_x(end+1)=...``. Here the whole matrix of separations is
built at once with *broadcasting*:

    positions[np.newaxis, :, :]   shape (1, N, 3)   -> a "row" with all r_j
    positions[:, np.newaxis, :]   shape (N, 1, 3)   -> a "column" with all r_i
    subtract them                  shape (N, N, 3)   -> dr[i, j] = r_j - r_i

NumPy virtually repeats each array along its length-1 axis (no copy). The
cost is still O(N^2), unavoidable with direct summation, but the loop now
runs in compiled C inside NumPy instead of in the Python interpreter.

Data layout
-----------
    positions : (N, 3)    float64
    masses    : (N,)      float64
    dr        : (N, N, 3) float64   separations r_j - r_i
    r2        : (N, N)    float64   squared distances (+ eps^2)
    inv_r3    : (N, N)    float64   1/r^3 (0 on the diagonal: no self-interaction)
    returns   : (N, 3)    float64   accelerations
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# Type of the acceleration function used by the integrators:
# takes positions (N, 3) and returns accelerations (N, 3).
AccelerationFunction = Callable[[np.ndarray], np.ndarray]


def pairwise_separations(positions: np.ndarray) -> np.ndarray:
    """dr[i, j] = r_j - r_i, shape (N, N, 3). Antisymmetric: dr[j, i] = -dr[i, j]."""
    return positions[np.newaxis, :, :] - positions[:, np.newaxis, :]


def accelerations(positions: np.ndarray, masses: np.ndarray, G: float,
                  softening: float = 0.0) -> np.ndarray:
    """Gravitational acceleration of every body, shape (N, 3)."""
    dr = pairwise_separations(positions)                       # (N, N, 3)

    # |dr|^2 for every pair. einsum("ijk,ijk->ij") sums dr*dr over k without
    # creating the intermediate array dr**2 (saves memory).
    r2 = np.einsum("ijk,ijk->ij", dr, dr) + softening * softening

    # Self-interaction: r_ii = 0 would give 0/0 = NaN. Setting r^2 = inf on the
    # diagonal makes 1/r^3 = 0 naturally. (The MATLAB code fixed it afterwards
    # by searching for NaN with find(isnan(...)); this is cleaner and faster,
    # and a genuine collision between two different bodies is not hidden.)
    np.fill_diagonal(r2, np.inf)
    inv_r3 = r2 ** -1.5                                        # (N, N)

    # a_i = G sum_j m_j dr_ij / r_ij^3. The weight w_ij = m_j / r_ij^3 is an
    # (N, N) matrix; einsum("ij,ijk->ik") contracts index j (the sum over bodies).
    weights = inv_r3 * masses[np.newaxis, :]
    return G * np.einsum("ij,ijk->ik", weights, dr)


def potential_energy(positions: np.ndarray, masses: np.ndarray, G: float,
                     softening: float = 0.0) -> float:
    """Total potential energy U (a scalar) at one instant."""
    i, j = np.triu_indices(masses.size, k=1)                   # pairs i<j, each counted once
    dr = positions[j] - positions[i]                           # (n_pairs, 3)
    r = np.sqrt(np.einsum("pk,pk->p", dr, dr) + softening * softening)
    return float(-G * np.sum(masses[i] * masses[j] / r))


def make_acceleration_function(masses: np.ndarray, G: float,
                               softening: float = 0.0) -> AccelerationFunction:
    """Return a(r) with masses, G and eps "frozen" inside (a closure).

    Why: integrators only need "give me the acceleration at these positions".
    Hiding the rest keeps the integrators generic: they would work for any
    position-dependent force, not only gravity.
    """
    masses = np.asarray(masses, dtype=np.float64)

    def accel(positions: np.ndarray) -> np.ndarray:
        return accelerations(positions, masses, G, softening)

    return accel
