"""
Example 5 — Lagrange triangle: linear stability and Routh's criterion
===================================================================

Physics
-------
Lagrange (1772) found that three bodies at the vertices of an equilateral
triangle, rotating rigidly with omega^2 = G M / d^3, are an exact solution for
ANY masses. But exact does not mean observable: it is stable only if one mass
dominates (Routh's criterion, 1875):

        27 (m1 m2 + m2 m3 + m3 m1) < (m1 + m2 + m3)^2

* Sun (1) - Jupiter (1e-3) - Trojan asteroid (~0): 27e-3 < 1 -> stable.
  That is why Trojan asteroids exist at Jupiter's L4 and L5 points.
* Three equal masses: 27*3 = 81 > 9 -> unstable.

Numerical experiment
--------------------
Verlet's truncation error (local error ~h^3 per step) acts as a tiny
perturbation. If the solution is unstable it grows as e^{sigma t} until the
triangle breaks up; if it is stable it stays bounded. The slope of log(delta)
versus t measures the growth rate sigma, compared with linear theory
(sigma = omega / sqrt(2) ~ 1.22 for equal masses).

The initial oscillation of amplitude ~1e-6 is NOT an instability: Verlet
solves exactly a slightly different problem (the "shadow Hamiltonian", see
docs/NUMERICS.md), whose circular orbit has a slightly different radius.
"""

from _common import MAKE_VIDEOS, save, video_path

import numpy as np

from nbody import plotting, scenarios, simulate

H, T_FINAL = 1e-3, 40.0
cases = {"Equal masses (1, 1, 1)": (1.0, 1.0, 1.0),
         "Sun-Jupiter-Trojan (1, 10⁻³, 10⁻⁶)": (1.0, 1e-3, 1e-6)}


def side_deviation(traj):
    """max_k |d_k(t) - d_0| / d_0: how much the triangle is deformed."""
    p = traj.positions
    sides = np.stack([np.linalg.norm(p[:, i] - p[:, j], axis=1) for i, j in [(0, 1), (1, 2), (0, 2)]], axis=1)
    return np.abs(sides / sides[0] - 1).max(axis=1)


fig, ax = plotting.new_figure(9.5, 4.6)
for k, (label, m) in enumerate(cases.items()):
    traj = simulate(scenarios.lagrange_triangle(m), T_FINAL, H, save_every=20)
    dev = side_deviation(traj)
    stable = scenarios.routh_stable(m)
    print(f"{label}: Routh says {'STABLE' if stable else 'UNSTABLE'};  final deformation = {dev[-1]:.2e}")
    ax.semilogy(traj.t, np.where(dev > 0, dev, np.nan), color=plotting.PALETTE[k], lw=1.4, label=label)
    if not stable:
        # Linear fit of log(delta) during the exponential phase (1e-4 < delta < 1e-1)
        sel = (dev > 1e-4) & (dev < 1e-1)
        sigma, _ = np.polyfit(traj.t[sel], np.log(dev[sel]), 1)
        sigma_th = scenarios.lagrange_growth_rate(m)
        print(f"  growth rate: measured sigma = {sigma:.3f},  linear theory sigma = {sigma_th:.3f}")
        ax.annotate(f"slope: σ = {sigma:.3f}\nlinear theory: σ = {sigma_th:.3f}", (20.0, 3e-2),
                    ha="right", color=plotting.TEXT_2, fontsize=9)
ax.set_xlabel("t (N-body units)"); ax.set_ylabel("deformation of the triangle")
ax.set_title("Routh's criterion in action: exponential growth vs. stability")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
save(fig, "05_lagrange_stability.png")

if MAKE_VIDEOS:
    from nbody.video import render_video
    traj = simulate(scenarios.lagrange_triangle((1, 1, 1)), 34.0, H, save_every=2)
    mp4, png = video_path("lagrange_unstable")
    render_video(traj, mp4, poster=png, poster_time=0.93, title="Lagrange's triangle breaks up",
                 subtitle="Equal masses violate Routh's criterion: a 10⁻⁷ numerical error grows as exp(1.22 t)",
                 time_label="time units", time_decimals=1, duration=20, trail=1.2, t_range=(14.0, 34.0),
                 limits=(np.zeros(3), np.array([3.0, 3.0, 1.2])), sizes=[70] * 3,
                 elev=35, rotation=60, zoom=1.25, names=["m₁", "m₂", "m₃"])
