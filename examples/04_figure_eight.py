"""
Example 4 — The figure-eight choreography (Chenciner-Montgomery)
===============================================================

Physics
-------
In 1993 Cris Moore found numerically, and in 2000 Chenciner and Montgomery
proved, that three equal masses can chase each other forever along a single
figure-eight curve. It is a *choreography*: the three bodies run along the
same curve, T/3 apart. Properties:
  * total angular momentum L = 0 (and yet they rotate: the two lobes are
    traversed in opposite senses and their contributions cancel),
  * centre of mass fixed at the origin,
  * stable: a small perturbation does NOT grow exponentially (the orbit is
    surrounded by KAM tori), unlike example 3.

N-body units: G = 1, m = 1, period T ~ 6.3259.

What it does
------------
1. Integrates 10 periods with Verlet (h = T/6000 ~ 1e-3).
2. Checks that after each period the bodies return to their starting points.
3. Perturbs one velocity by 0.1 % and shows that the separation grows only
   slowly (no exponential) -> stable orbit.
4. Renders the 3D video shown at the top of the README.
"""

from _common import MAKE_VIDEOS, save, video_path

import numpy as np

from nbody import chaos, diagnostics, plotting, scenarios, simulate

T = scenarios.FIGURE_EIGHT_PERIOD
H = T / 6000                # one period = exactly 6000 steps -> r(kT) can be compared with r(0)

system = scenarios.figure_eight()
L0 = diagnostics.angular_momentum(system.masses, system.positions[None], system.velocities[None])[0]
print(f"initial L = {L0}")
traj = simulate(system, 10 * T, H, save_every=6)        # 1000 stored instants per period
print(diagnostics.conservation_report(traj, "Figure-eight"))

# Return after each period: distance to the initial state at t = k T
for k in (1, 5, 10):
    err = np.linalg.norm(traj.positions[1000 * k] - traj.positions[0])
    print(f"  after {k:2d} periods: |r(kT) - r(0)| = {err:.2e}")

# --- Figure: the orbit and the energy error ---------------------------------------
fig, (ax1, ax2) = plotting.new_figure(11, 4.2, ncols=2, gridspec_kw={"width_ratios": [1.35, 1]})
plotting.plot_orbits(traj, ax=ax1, projection="xy", unit_label="length units",
                     title="Three bodies, one curve (10 periods)", legend="inside")
ax2.semilogy(traj.t / T, diagnostics.relative_error(traj.total_energy()), color=plotting.PALETTE[6], lw=0.8)
ax2.set_xlabel("t / T"); ax2.set_ylabel("|ΔE / E₀|"); ax2.set_title("Energy error (bounded)")
save(fig, "04_figure_eight.png")

# --- Stability against a perturbation ----------------------------------------------
pert = simulate(system.perturbed(0, "velocity", component=0, relative=1e-3), 10 * T, H, save_every=6)
d = chaos.position_divergence(traj, pert)
fig = plotting.plot_divergence(traj.t, {"v_x,A + 0.1 %": d}, time_unit=T, time_label="periods",
                               title="Perturbed figure-eight: the separation does NOT grow exponentially",
                               ylabel="δr(t) (length units)")
save(fig, "04_figure_eight_stability.png")

# --- Video ------------------------------------------------------------------------
if MAKE_VIDEOS:
    from nbody.video import render_video
    run = simulate(system, 3 * T, H, save_every=2)
    mp4, png = video_path("figure_eight")
    render_video(run, mp4, poster=png, poster_time=0.4, title="Figure-eight choreography",
                 subtitle="Three equal masses chasing each other along a single curve (Chenciner & Montgomery, 2000)",
                 time_unit=T, time_label="periods", duration=20, trail=T / 3, sizes=[80] * 3,
                 elev=30, rotation=70, zoom=1.35, unit_label="")
