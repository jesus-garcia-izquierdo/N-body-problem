"""
Example 8 — Star clusters: cold collapse and the virial theorem
==============================================================

Physics
-------
Virial theorem (for 1/r^2 forces, time-averaging a bound system):
2<K> + <U> = 0, i.e. Q = 2K / |U| -> 1.

Two experiments with N = 400 "stars" (N-body units, G = M = 1):

1. **Plummer sphere**: born in equilibrium (Q ~ 1) and stays there; it only
   fluctuates because N is finite (~1/sqrt(N)).
2. **Cold collapse**: a homogeneous sphere with every star at rest (Q = 0).
   Gravity makes it collapse in one free-fall time

        t_ff = (pi/2) sqrt(R^3 / (2 G M)) ~ 1.11,

   it bounces, oscillates and within a few t_ff settles into a
   quasi-stationary state with Q ~ 1. This process, *violent relaxation*
   (Lynden-Bell, 1967), is how elliptical galaxies are thought to have formed:
   fast, and without star-star collisions.

Diagnostics
-----------
* Q(t) = 2K / |U|.
* Lagrangian radii: radii of the spheres (centred on the CM) that contain
  10 %, 50 % and 90 % of the mass. The 90 % radius grows at the end: some stars
  gain energy during the bounce and form an extended halo (a few escape).

Softening eps = 0.05: prevents close two-star encounters from forcing a tiny
time step. Reasonable here because we are interested in collective dynamics.

Run time: ~1 minute for the simulations, ~1 minute for the video.
"""

from _common import MAKE_VIDEOS, save, video_path

import numpy as np

from nbody import diagnostics, plotting, scenarios, simulate

N, T_FINAL, H = 400, 8.0, 2e-3
T_FF = np.pi / 2 * np.sqrt(1 / 2)


def lagrangian_radii(traj, fractions=(0.1, 0.5, 0.9)):
    """Radii that enclose each mass fraction (equal masses -> sorting is enough)."""
    cm = traj.center_of_mass()                                        # (T, 3)
    r = np.linalg.norm(traj.positions - cm[:, None, :], axis=2)       # (T, N)
    r.sort(axis=1)
    return {f: r[:, int(f * traj.n_bodies) - 1] for f in fractions}


cold_ic = scenarios.uniform_sphere(N, virial_ratio=0.0)
cold = simulate(cold_ic, T_FINAL, H, save_every=5)
plum = simulate(scenarios.plummer_cluster(N), T_FINAL, H, save_every=5)
for traj, name in ((cold, "Cold collapse"), (plum, "Plummer")):
    print(diagnostics.conservation_report(traj, name))

# --- Figure 1: Q(t) and Lagrangian radii ------------------------------------------
fig, (ax1, ax2) = plotting.new_figure(12, 4.4, ncols=2)
for k, (traj, name) in enumerate(((cold, "Cold collapse (Q₀ = 0)"), (plum, "Plummer sphere (Q₀ ≈ 1)"))):
    Q = 2 * traj.kinetic_energy() / -traj.potential_energy()
    ax1.plot(traj.t / T_FF, Q, color=plotting.PALETTE[k], label=name)
ax1.axhline(1, color=plotting.TEXT_2, ls="--", lw=0.8)
ax1.set_xlabel("t / t_ff"); ax1.set_ylabel("Q = 2K / |U|")
ax1.set_title("Virial theorem: Q → 1"); ax1.legend(loc="lower right")

styles = {0.1: ":", 0.5: "-", 0.9: "--"}
for f, r in lagrangian_radii(cold).items():
    ax2.semilogy(cold.t / T_FF, r, color=plotting.PALETTE[0], ls=styles[f], label=f"{int(f * 100)} % of the mass")
ax2.set_xlabel("t / t_ff"); ax2.set_ylabel("Lagrangian radius (length units)")
ax2.set_title("Cold collapse: Lagrangian radii"); ax2.legend(loc="lower right")
save(fig, "08_cluster_virial.png")

# --- Figure 2: snapshots of the collapse ----------------------------------------------
times = [0.0, 0.85 * T_FF, 1.1 * T_FF, T_FINAL]
fig, axs = plotting.new_figure(13, 3.9, ncols=4)
for ax, tt in zip(axs, times):
    i = np.argmin(np.abs(cold.t - tt))
    p = cold.positions[i] - cold.center_of_mass()[i]
    ax.plot(p[:, 0], p[:, 1], "o", ms=1.8, color=plotting.PALETTE[0], alpha=0.65)
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2); ax.set_aspect("equal")
    ax.set_title(f"t = {cold.t[i] / T_FF:.2f} t_ff"); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Cold collapse of a homogeneous sphere (xy projection)", fontweight="bold",
             color=plotting.TEXT, fontsize=12.5)
save(fig, "08_cold_collapse.png")

# --- Video: stars colored by their initial radius ------------------------------------
if MAKE_VIDEOS:
    import matplotlib as mpl
    from nbody.video import render_video
    r0 = np.linalg.norm(cold_ic.positions, axis=1)
    star_colors = mpl.colormaps["plasma"](0.4 + 0.58 * (1 - r0 / r0.max()))   # inner stars = yellow
    mp4, png = video_path("cold_collapse")
    render_video(cold, mp4, poster=png, poster_time=0.6, title="Cold collapse of a star cluster",
                 subtitle=f"{N} stars released from rest: collapse, bounce and virialization (color = initial radius)",
                 time_unit=T_FF, time_label="free-fall times", time_decimals=2, duration=22,
                 colors=star_colors, sizes=np.full(N, 12.0), legend=False,
                 limits=(np.zeros(3), np.full(3, 3.0)), elev=22, rotation=90, zoom=1.15)
