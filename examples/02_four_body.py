"""
Example 2 — Four bodies around a massive central body (Section 4.2 of the report)
================================================================================

Configuration (Table 2 of the report, reproduced as is)
-------------------------------------------------------
* Body 3: mass 2e40 kg (~10^10 solar masses, like a galactic nucleus) at rest
  at (7.48e13, 0, 0) m.
* Bodies 1 and 2: ~1 solar mass each, in the XY plane, on either side of it.
* Body 4: light, in the XZ plane (perpendicular orbit).

Physics illustrated
-------------------
* An almost pure Kepler problem: the central body dominates, so each body
  traces a nearly independent ellipse. The report's speeds use the distance
  1.496e14 m (twice the real one) -> v^2 = GM/(2r), and the vis-viva equation
  v^2 = GM(2/r - 1/a) gives a = 2r/3: ellipses with e = 0.5 whose apocentre
  is the starting point.
* Conservation of E, P and L over 15 years (~250 orbits of body 1).
* Virial theorem with eccentric orbits.

Run time: ~15 s for the simulation, ~1 min for the video.
"""

from _common import MAKE_VIDEOS, save, video_path

import numpy as np

from nbody import constants as c
from nbody import diagnostics, plotting, scenarios, simulate

T_FINAL = 15 * c.YEAR
H = 1000.0

system = scenarios.four_body_report()
traj = simulate(system, T_FINAL, H, integrator="verlet", save_every=50, verbose=True)
print(diagnostics.conservation_report(traj, "Four bodies"))

# Measured eccentricity of body 1 around the central body
r1 = np.linalg.norm(traj.positions[:, 0] - traj.positions[:, 2], axis=1)
e_meas = (r1.max() - r1.min()) / (r1.max() + r1.min())
print(f"  eccentricity of body 1: measured {e_meas:.4f}  (vis-viva prediction: 0.5000)")

ax = plotting.plot_orbits(traj, projection="3d", length_unit=1e13, unit_label="10¹³ m",
                          title="Four-body system (15 years)")
save(ax.figure, "02_four_body_orbits.png")

fig = plotting.plot_conservation(traj, time_unit=c.YEAR, time_label="years",
                                 title="Four bodies — velocity Verlet, h = 1000 s")
save(fig, "02_four_body_conservation.png")

if MAKE_VIDEOS:
    from nbody.video import render_video
    # The first half year (~8 orbits of bodies 1 and 2): finer sampling for a smooth video
    short = simulate(system, 0.5 * c.YEAR, H, save_every=5)
    mp4, png = video_path("four_body")
    render_video(short, mp4, poster=png, title="Four-body system",
                 subtitle="A 10¹⁰ M☉ central body with two stars and a light body on a polar orbit",
                 length_unit=1e13, unit_label="10¹³ m", time_unit=c.DAY, time_label="days",
                 time_decimals=0, duration=20, trail=30 * c.DAY, sizes=[70, 60, 190, 35],
                 elev=20, rotation=70, zoom=1.15)
