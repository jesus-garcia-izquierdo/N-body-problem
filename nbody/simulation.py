"""
Simulation loop and result container (trajectory).

Program logic
-------------
    simulate(system, t_final, h, integrator)
      1. a_0 = a(r_0)                          (the only "extra" force evaluation)
      2. allocate memory for the instants that will be stored
      3. for n = 1 .. n_steps:
             (r, v, a) <- step(r, v, a, h)      (integrators.py)
             if n % save_every == 0: store r, v
      4. return a Trajectory

Data layout
-----------
    T = number of stored instants,  N = number of bodies

    Trajectory.t          : (T,)       float64   times [s]
    Trajectory.positions  : (T, N, 3)  float64   r_i(t)
    Trajectory.velocities : (T, N, 3)  float64   v_i(t)
    Trajectory.masses     : (N,)       float64

Axis 0 = time, axis 1 = body, axis 2 = component. So ``positions[k]`` is a
"snapshot" of the system at instant k and ``positions[:, i]`` is the orbit of
body i.

Why memory is allocated up front (``np.empty``): the MATLAB code did
``x = [x x1]`` at every step, which copies the whole matrix each time -> total
cost O(T^2). With 300 000 steps that is ~10^11 copied numbers. Allocating once,
each step writes into its own slot: O(T).

``save_every``: a small h is needed for accuracy, not for plotting. Storing
1 out of k steps divides the memory by k without changing the integration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import diagnostics as diag
from .forces import make_acceleration_function
from .integrators import get_integrator
from .system import NBodySystem


@dataclass
class Trajectory:
    """Result of a simulation, plus shortcuts to the conserved quantities."""

    t: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    masses: np.ndarray
    G: float
    softening: float = 0.0
    names: list[str] = field(default_factory=list)
    integrator: str = "verlet"
    h: float = float("nan")
    wall_time: float = 0.0

    @property
    def n_bodies(self) -> int:
        return self.masses.size

    # -- physical quantities (each returns a time series) -------------------
    def kinetic_energy(self) -> np.ndarray:
        return diag.kinetic_energy(self.masses, self.velocities)

    def potential_energy(self) -> np.ndarray:
        return diag.potential_energy(self.masses, self.positions, self.G, self.softening)

    def total_energy(self) -> np.ndarray:
        return self.kinetic_energy() + self.potential_energy()

    def linear_momentum(self) -> np.ndarray:
        return diag.linear_momentum(self.masses, self.velocities)

    def angular_momentum(self) -> np.ndarray:
        return diag.angular_momentum(self.masses, self.positions, self.velocities)

    def center_of_mass(self) -> np.ndarray:
        return diag.center_of_mass(self.masses, self.positions)

    def final_state(self) -> NBodySystem:
        """Last instant as an NBodySystem (to chain simulations)."""
        return NBodySystem(self.masses, self.positions[-1], self.velocities[-1],
                           self.G, self.softening, list(self.names))

    # -- save / load (NumPy .npz: compressed and portable) ------------------
    def save(self, path: str) -> None:
        np.savez_compressed(path, t=self.t, positions=self.positions, velocities=self.velocities,
                            masses=self.masses, G=self.G, softening=self.softening,
                            names=np.array(self.names), integrator=self.integrator, h=self.h)

    @classmethod
    def load(cls, path: str) -> "Trajectory":
        d = np.load(path)
        return cls(d["t"], d["positions"], d["velocities"], d["masses"], float(d["G"]),
                   float(d["softening"]), [str(s) for s in d["names"]], str(d["integrator"]), float(d["h"]))


def simulate(system: NBodySystem, t_final: float, h: float, integrator: str = "verlet",
             save_every: int = 1, verbose: bool = False) -> Trajectory:
    """Integrate the system from t = 0 to t_final with fixed step h.

    Parameters
    ----------
    system     : initial conditions (left unchanged)
    t_final    : total time [same units as h]
    h          : time step. Rule of thumb: h <~ 1/100 of the shortest orbital
                 period in the system (Moon: 27 days -> h <~ 2e4 s).
    integrator : "verlet" (default), "yoshida4", "rk4", "symplectic_euler", "euler"
    save_every : store 1 out of every ``save_every`` steps
    verbose    : print progress
    """
    if h <= 0 or t_final <= 0:
        raise ValueError("h and t_final must be positive")
    step = get_integrator(integrator).step
    accel = make_acceleration_function(system.masses, system.G, system.softening)

    # Number of steps: round up so that t_final is reached.
    n_steps = int(np.ceil(t_final / h - 1e-9))
    n_saved = n_steps // save_every + 1

    t = np.empty(n_saved)
    pos = np.empty((n_saved, system.n_bodies, 3))
    vel = np.empty((n_saved, system.n_bodies, 3))

    x, v = system.positions.copy(), system.velocities.copy()
    a = accel(x)
    t[0], pos[0], vel[0] = 0.0, x, v

    k = 1                                   # index of the next free slot
    report_every = max(n_steps // 10, 1)
    t0 = time.perf_counter()
    for n in range(1, n_steps + 1):
        x, v, a = step(x, v, a, h, accel)
        if n % save_every == 0:
            t[k], pos[k], vel[k] = n * h, x, v   # n*h rather than t += h: no accumulated round-off
            k += 1
        if verbose and n % report_every == 0:
            print(f"  {100 * n / n_steps:5.1f} %  ({time.perf_counter() - t0:.1f} s)")

    return Trajectory(t[:k], pos[:k], vel[:k], system.masses.copy(), system.G, system.softening,
                      list(system.names), integrator, h, time.perf_counter() - t0)
