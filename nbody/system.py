"""
State of an N-body system at one instant: masses, positions and velocities.

Data layout
-----------
    N = number of bodies

    masses      : np.ndarray, shape (N,),   float64   [kg]   m_i
    positions   : np.ndarray, shape (N, 3), float64   [m]    r_i = (x_i, y_i, z_i)
    velocities  : np.ndarray, shape (N, 3), float64   [m/s]  v_i
    G           : float                                       gravitational constant
    softening   : float                                [m]    softening length eps (see forces.py)
    names       : list[str]                                   labels used in plots and videos

Convention: **row = body, column = Cartesian component**. (The original MATLAB
code used rows = bodies and columns = time steps, with three separate matrices
x, y, z. Keeping x, y, z together in the last axis lets us operate on 3D
vectors in one go.)

Why a ``dataclass``: it bundles the data that always travels together and
generates ``__init__`` and ``__repr__`` for free. Methods that "modify" the
system return a NEW object (immutable style). That way a set of initial
conditions reused by several runs can never be altered by accident, which is
essential when studying chaos (we compare almost identical systems).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import G_SI


def _as_vectors3d(arr, n_bodies: int, name: str) -> np.ndarray:
    """Convert the input to a float64 array of shape (N, 3).

    Planar (N, 2) input is also accepted and padded with z = 0, so 2D
    scenarios (figure-eight, Lagrange triangle, ...) can be written naturally.
    """
    a = np.array(arr, dtype=np.float64)          # np.array copies: no shared memory with the caller
    if a.ndim != 2 or a.shape[0] != n_bodies or a.shape[1] not in (2, 3):
        raise ValueError(f"{name} must have shape (N, 3) or (N, 2) with N={n_bodies}; got {a.shape}")
    if a.shape[1] == 2:
        a = np.column_stack([a, np.zeros(n_bodies)])
    return a


@dataclass
class NBodySystem:
    """(Initial) conditions of a gravitating N-body system."""

    masses: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    G: float = G_SI
    softening: float = 0.0
    names: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Validation on construction
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        self.masses = np.array(self.masses, dtype=np.float64).reshape(-1)
        n = self.masses.size
        if n < 1:
            raise ValueError("At least one body is required")
        if np.any(self.masses <= 0):
            raise ValueError("All masses must be positive")
        self.positions = _as_vectors3d(self.positions, n, "positions")
        self.velocities = _as_vectors3d(self.velocities, n, "velocities")
        if not self.names:
            self.names = [f"Body {i + 1}" for i in range(n)]
        if len(self.names) != n:
            raise ValueError("names must contain one entry per body")
        self.G = float(self.G)
        self.softening = float(self.softening)

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------
    @property
    def n_bodies(self) -> int:
        return self.masses.size

    @property
    def total_mass(self) -> float:
        return float(self.masses.sum())

    def center_of_mass(self) -> np.ndarray:
        """R_cm = sum_i m_i r_i / M  ->  shape (3,).

        ``masses @ positions`` is the product (1 x N)(N x 3) = (3,): exactly
        the weighted sum, with no loop (the MATLAB equivalent is ``m'*x``).
        """
        return self.masses @ self.positions / self.total_mass

    def center_of_mass_velocity(self) -> np.ndarray:
        """V_cm = sum_i m_i v_i / M = P_total / M  ->  shape (3,)."""
        return self.masses @ self.velocities / self.total_mass

    # ------------------------------------------------------------------
    # Transformations (all return a new system)
    # ------------------------------------------------------------------
    def copy(self) -> "NBodySystem":
        return NBodySystem(self.masses.copy(), self.positions.copy(), self.velocities.copy(),
                           self.G, self.softening, list(self.names))

    def to_center_of_mass_frame(self) -> "NBodySystem":
        """Move to the centre-of-mass (CM) frame.

        Physics: without external forces the CM moves with constant velocity
        (dP/dt = 0). Subtracting R_cm and V_cm is a Galilean transformation to
        another inertial frame, so the relative dynamics is unchanged.
        Benefits:
          * P_total = 0 exactly -> the system does not drift out of the plot.
          * The kinetic energy no longer contains the trivial term 1/2 M V_cm^2,
            which is required to test the virial theorem.
        """
        new = self.copy()
        new.positions -= self.center_of_mass()        # broadcasting: (N,3) - (3,)
        new.velocities -= self.center_of_mass_velocity()
        return new

    def perturbed(self, body: int, quantity: str, component: int | None = None,
                  relative: float = 1e-3) -> "NBodySystem":
        """Copy with a small relative perturbation (used to study chaos).

        Parameters
        ----------
        body      : body index (0-based, as everywhere in Python)
        quantity  : "position", "velocity" or "mass"
        component : 0, 1, 2 (x, y, z); ignored for "mass"
        relative  : relative change, e.g. 1e-3 = 0.1 %
        """
        new = self.copy()
        if quantity == "mass":
            new.masses[body] *= 1.0 + relative
        elif quantity in ("position", "velocity"):
            if component is None:
                raise ValueError("A component (0, 1 or 2) is required")
            arr = new.positions if quantity == "position" else new.velocities
            arr[body, component] *= 1.0 + relative
        else:
            raise ValueError("quantity must be 'position', 'velocity' or 'mass'")
        return new

    def __repr__(self) -> str:  # readable console summary
        return (f"NBodySystem(N={self.n_bodies}, M_total={self.total_mass:.4g}, "
                f"G={self.G:.4g}, softening={self.softening:.3g})")
