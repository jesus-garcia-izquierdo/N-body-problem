"""
nbody — Simulation of the gravitational N-body problem with symplectic integrators.

Minimal usage
-------------
>>> from nbody import scenarios, simulate
>>> system = scenarios.figure_eight()
>>> traj = simulate(system, t_final=6.33, h=1e-3, integrator="verlet")
>>> traj.total_energy()[[0, -1]]          # conserved
array([-1.28..., -1.28...])

Package layout (bottom-up)
--------------------------
    constants    G, AU, year, masses...
    system       NBodySystem: masses, positions, velocities (initial state)
    forces       accelerations and potential energy (vectorized, O(N^2))
    integrators  Euler, symplectic Euler, Verlet, Yoshida-4, RK4
    simulation   simulate() -> Trajectory
    diagnostics  E, P, L, centre of mass, virial ratio
    chaos        trajectory divergence, Lyapunov exponent
    scenarios    initial conditions (original report + classic problems)
    plotting     static figures (orbits, conservation laws)
    video        3D MP4 videos
"""

from . import chaos, constants, diagnostics, forces, integrators, scenarios
from .integrators import INTEGRATORS
from .simulation import Trajectory, simulate
from .system import NBodySystem

__version__ = "1.0.0"

__all__ = [
    "NBodySystem", "Trajectory", "simulate", "INTEGRATORS",
    "chaos", "constants", "diagnostics", "forces", "integrators", "scenarios",
]
