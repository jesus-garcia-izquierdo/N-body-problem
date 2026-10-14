"""
Example 1 — The Sun-Earth-Moon system (Section 4.1 of the original report)
=========================================================================

What it does
------------
1. Real initial conditions: Earth at aphelion, Moon at apogee, all in the
   ecliptic plane (the Moon's 5° inclination is ignored).
2. Integrates 2 years with velocity Verlet and h = 1000 s.
3. Checks the conservation laws and the virial theorem.
4. Repeats WITHOUT moving to the centre-of-mass frame: here the effect is
   tiny (the Sun dominates the total mass), but in example 3 it is not.

Why h = 1000 s
--------------
The shortest period is the Moon's (27.3 days ~ 2.4e6 s): h = 1000 s gives
~2400 steps per lunar orbit, plenty for Verlet (error ~ h^2).

What you should see
-------------------
* The Moon's orbit is invisible at Solar-System scale (the Earth-Moon distance
  is 1/375 of the Sun-Earth distance), so it is also drawn relative to Earth.
  Its ellipse precesses: the Sun perturbs it.
* |dE/E| ~ 1e-9, oscillating and bounded (Verlet is symplectic); P and L
  conserved down to machine round-off (~1e-15).
* 2<K>/(-<U>) ~ 1 in the CM frame.
"""

from _common import save  # noqa: F401  (also puts the repository root on sys.path)

import numpy as np

from nbody import constants as c
from nbody import diagnostics, plotting, scenarios, simulate

T_FINAL = 2 * c.YEAR
H = 1000.0

# --- Main run (centre-of-mass frame) ----------------------------------------
system = scenarios.sun_earth_moon(center_of_mass_frame=True)
print(system)
traj = simulate(system, T_FINAL, H, integrator="verlet", save_every=20)
print(diagnostics.conservation_report(traj, "Sun-Earth-Moon (CM frame)"))

# --- Figure 1: heliocentric orbits + Moon relative to Earth ------------------
fig, (ax1, ax2) = plotting.new_figure(11, 5.0, ncols=2)
plotting.plot_orbits(traj, ax=ax1, projection="xy", length_unit=c.AU, unit_label="AU",
                     title="Orbits around the centre of mass (2 years)", legend="inside")
rel = (traj.positions[:, 2] - traj.positions[:, 1]) / 1e6          # r_Moon - r_Earth, in 1000 km
ax2.plot(rel[:, 0], rel[:, 1], color=plotting.PALETTE[2], lw=0.6)
ax2.plot(0, 0, "o", color=plotting.PALETTE[1], ms=8, label="Earth")
ax2.set_aspect("equal", adjustable="datalim")
ax2.set_xlabel("x (10³ km)"); ax2.set_ylabel("y (10³ km)")
ax2.set_title("Moon seen from Earth (the ellipse precesses)")
ax2.legend(loc="upper right")
save(fig, "01_sun_earth_moon_orbits.png")

# --- Figure 2: conservation laws --------------------------------------------
fig = plotting.plot_conservation(traj, time_unit=c.YEAR, time_label="years",
                                 title="Sun-Earth-Moon — velocity Verlet, h = 1000 s")
save(fig, "01_sun_earth_moon_conservation.png")

# --- Virial theorem: convergence of the time average ------------------------
K, U = traj.kinetic_energy(), traj.potential_energy()
q = diagnostics.running_virial_ratio(K, U)
print(f"  <K> = {K.mean():.5e} J,  -1/2 <U> = {-0.5 * U.mean():.5e} J")

fig, ax = plotting.new_figure(7.5, 3.6)
ax.plot(traj.t / c.YEAR, q, color=plotting.PALETTE[0])
ax.axhline(1, color=plotting.TEXT_2, lw=0.8, ls="--")
ax.set_xlabel("t (years)"); ax.set_ylabel("2⟨K⟩ / (−⟨U⟩)")
ax.set_title("Virial theorem: the running average tends to 1")
ax.set_ylim(0.99, 1.01)
save(fig, "01_sun_earth_moon_virial.png")

# --- Comparison: without moving to the CM frame -------------------------------
lab = simulate(scenarios.sun_earth_moon(center_of_mass_frame=False), T_FINAL, H, save_every=20)
K_lab, U_lab = lab.kinetic_energy(), lab.potential_energy()
v_cm = np.linalg.norm(lab.linear_momentum()[0]) / lab.masses.sum()
print(f"\nWithout the CM frame the centre of mass moves at {v_cm:.2f} m/s "
      f"({v_cm * T_FINAL / 1e3:.0f} km in 2 years)")
print(f"  virial ratio = {diagnostics.virial_ratio(K_lab, U_lab):.7f}  "
      f"(CM frame: {diagnostics.virial_ratio(K, U):.7f})")
print(f"  1/2 M V_cm^2 / <K> = {0.5 * lab.masses.sum() * v_cm**2 / K.mean():.1e}  -> negligible here, "
      "because V_cm = m_E v_E / M ~ 9 cm/s")
