r"""
Conserved quantities and physical checks along a trajectory.

Why they matter: the N-body problem has no analytical solution for N > 2, so
there is no "exact answer" to compare with. What we do know for certain is
what must be conserved (Noether's theorem):

    symmetry of the isolated system       ->  conserved quantity
    ---------------------------------------------------------------
    translations in space                 ->  linear momentum  P = sum m_i v_i
    rotations                             ->  angular momentum L = sum m_i r_i x v_i
    translations in time                  ->  energy           E = K + U

How far P, L and E drift from their initial values measures the numerical
quality of the run. In addition, for bound orbits the **virial theorem**
relates the time averages: 2<K> = -<U>.

Data layout (T instants, N bodies)
----------------------------------
    input   positions, velocities : (T, N, 3)   masses : (N,)
    output  energies              : (T,)
            P, L, R_cm            : (T, 3)  (vectors; take np.linalg.norm for the modulus)

Difference with the original report: there |P| and |L| were plotted. Here the
full vectors are returned (the moduli can be conserved while the components
are not, or vice versa), and errors are measured **relative to the initial
value**, which is what is physically meaningful (subtracting the mean, as the
report did, depends on the whole series and hides slow drifts).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Energies
# ---------------------------------------------------------------------------
def kinetic_energy(masses: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    """K(t) = 1/2 sum_i m_i |v_i|^2.   velocities (T, N, 3) -> (T,)

    einsum("tnk,tnk->tn") computes |v_i(t)|^2 for every t and i at once.
    """
    v2 = np.einsum("tnk,tnk->tn", velocities, velocities)     # (T, N)
    return 0.5 * v2 @ masses                                   # (T, N)(N,) = (T,)


def potential_energy(masses: np.ndarray, positions: np.ndarray, G: float,
                     softening: float = 0.0) -> np.ndarray:
    """U(t) = -G sum_{i<j} m_i m_j / r_ij(t).   positions (T, N, 3) -> (T,)

    Loop over bodies, vectorized in time and over partners. Memory is O(T N)
    instead of O(T N^2) (with T = 300 000 and N = 6 the full pair matrix would
    take ~260 MB; this way, a few MB).
    """
    n = masses.size
    U = np.zeros(positions.shape[0])
    for i in range(n - 1):
        dr = positions[:, i + 1:, :] - positions[:, i:i + 1, :]          # (T, n-i-1, 3)
        r = np.sqrt(np.einsum("tjk,tjk->tj", dr, dr) + softening * softening)
        U -= G * masses[i] * (masses[i + 1:] / r).sum(axis=1)
    return U


def total_energy(masses, positions, velocities, G, softening=0.0) -> np.ndarray:
    return kinetic_energy(masses, velocities) + potential_energy(masses, positions, G, softening)


# ---------------------------------------------------------------------------
# Momenta and centre of mass
# ---------------------------------------------------------------------------
def linear_momentum(masses: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    """P(t) = sum_i m_i v_i  ->  (T, 3).   Equation (7) of the report, for the 3 components."""
    return np.einsum("n,tnk->tk", masses, velocities)


def angular_momentum(masses: np.ndarray, positions: np.ndarray, velocities: np.ndarray,
                     origin: np.ndarray | None = None) -> np.ndarray:
    """L(t) = sum_i r_i x (m_i v_i)  ->  (T, 3), about ``origin`` (default 0).

    np.cross works on the last axis (the components), so it computes all T*N
    cross products at once; then we sum over bodies. Without external forces
    L is conserved about ANY fixed point.
    """
    r = positions if origin is None else positions - origin
    p = masses[np.newaxis, :, np.newaxis] * velocities                    # (T, N, 3)
    return np.cross(r, p).sum(axis=1)


def center_of_mass(masses: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """R_cm(t) = sum m_i r_i / M  ->  (T, 3). Must move in a straight line at constant speed."""
    return np.einsum("n,tnk->tk", masses, positions) / masses.sum()


# ---------------------------------------------------------------------------
# Conservation errors
# ---------------------------------------------------------------------------
def relative_error(series: np.ndarray, scale: float | None = None) -> np.ndarray:
    """|Q(t) - Q(0)| / scale.

    Default scale = |Q(0)|. For vectors (T, 3) the norm of the difference is
    used. If Q(0) ~ 0 (e.g. P in the CM frame) a relative error is meaningless:
    pass a physical scale instead (see momentum_scale).
    """
    s = np.asarray(series)
    diff = s - s[0]
    if s.ndim == 2:
        diff_norm = np.linalg.norm(diff, axis=1)
        ref = np.linalg.norm(s[0])
    else:
        diff_norm = np.abs(diff)
        ref = abs(s[0])
    if scale is None:
        scale = ref
    if scale == 0:
        return diff_norm
    return diff_norm / scale


def momentum_scale(masses: np.ndarray, velocities: np.ndarray) -> float:
    """Natural momentum scale: max_t sum_i m_i |v_i(t)|.

    Useful when P_total = 0 (CM frame). The maximum over time is used because
    it may be 0 at t = 0 (e.g. a cold collapse starts at rest).
    """
    return float((np.linalg.norm(velocities, axis=2) @ masses).max())


def angular_momentum_scale(masses: np.ndarray, positions: np.ndarray, velocities: np.ndarray) -> float:
    """Natural angular-momentum scale: max_t sum_i m_i |r_i(t)| |v_i(t)|."""
    rv = np.linalg.norm(positions, axis=2) * np.linalg.norm(velocities, axis=2)    # (T, N)
    return float((rv @ masses).max())


# ---------------------------------------------------------------------------
# Virial theorem
# ---------------------------------------------------------------------------
def virial_ratio(K: np.ndarray, U: np.ndarray) -> float:
    """Q = 2<K> / (-<U>). The virial theorem predicts Q -> 1 for bound orbits.

    Conditions (important!):
      1. Bound (E < 0) and bounded system: no particle escapes.
      2. K measured in the centre-of-mass frame (otherwise 1/2 M V_cm^2 adds to
         <K> and the ratio exceeds 1).
      3. Average over a long time (many periods) or an integer number of periods.
    """
    return float(2.0 * np.mean(K) / (-np.mean(U)))


def running_virial_ratio(K: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Q(t) using cumulative averages from 0 to t: shows how it converges to 1."""
    n = np.arange(1, K.size + 1)
    return 2.0 * (np.cumsum(K) / n) / (-(np.cumsum(U) / n))


def conservation_summary(traj) -> dict:
    """Maximum conservation errors and virial ratio of a trajectory (as numbers)."""
    K, U = traj.kinetic_energy(), traj.potential_energy()
    P, L = traj.linear_momentum(), traj.angular_momentum()
    return {
        "energy": float(relative_error(K + U).max()),
        "momentum": float(relative_error(P, momentum_scale(traj.masses, traj.velocities)).max()),
        "angular_momentum": float(relative_error(
            L, angular_momentum_scale(traj.masses, traj.positions, traj.velocities)).max()),
        "virial": virial_ratio(K, U),
    }


def conservation_report(traj, label: str | None = None) -> str:
    """Plain-text summary of the quality of a run."""
    s = conservation_summary(traj)
    lines = [
        f"── {label or traj.integrator} ──  N={traj.n_bodies}, stored steps={traj.t.size}, "
        f"wall time={traj.wall_time:.2f} s",
        f"  max |ΔE/E0|                = {s['energy']:.3e}",
        f"  max |ΔP| / max Σ m|v|      = {s['momentum']:.3e}",
        f"  max |ΔL| / max Σ m|r||v|   = {s['angular_momentum']:.3e}",
        f"  virial ratio 2<K>/(-<U>)   = {s['virial']:.5f}   (-> 1 if bound and in the CM frame)",
    ]
    return "\n".join(lines)
