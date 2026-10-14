"""
Example 7 — The Solar System and Kepler's third law
==================================================

What it does
------------
1. Sun + 8 planets with real masses, semi-major axes, eccentricities and
   inclinations (illustrative initial phases, see ``scenarios.solar_system``).
2. Integrates one full orbit of Neptune (165 years) with h = 0.5 days. The
   step is set by Mercury: 88 days / 0.5 days = 176 steps per orbit.
3. MEASURES the period of every planet in the simulation (counting turns of
   its polar angle around the Sun) and checks Kepler's third law:

        T^2 = (4 pi^2 / G M_sun) a^3    ->   in years and AU:  T = a^{3/2}

   On a log-log scale this is a straight line of slope 3/2.
4. Two 3D videos: the inner planets (2 years) and the outer planets (165 years).

Run time: ~10 s for the simulation, ~2 minutes for the videos.
"""

from _common import MAKE_VIDEOS, save, video_path

import numpy as np

from nbody import constants as c
from nbody import diagnostics, plotting, scenarios, simulate

system = scenarios.solar_system()
traj = simulate(system, 165 * c.YEAR, 0.5 * c.DAY, save_every=4, verbose=True)
print(diagnostics.conservation_report(traj, "Solar System"))

# --- Measured periods ---------------------------------------------------------
print(f"\n{'planet':10s} {'a (AU)':>8s} {'measured T (yr)':>16s} {'a^1.5':>8s}")
a_list, T_list = [], []
for i, (name, _, a_au, _, _) in enumerate(scenarios.PLANETS, start=1):
    rel = traj.positions[:, i] - traj.positions[:, 0]                 # position relative to the Sun
    theta = np.unwrap(np.arctan2(rel[:, 1], rel[:, 0]))                # cumulative angle (no 2 pi jumps)
    revolutions = (theta[-1] - theta[0]) / (2 * np.pi)
    T_meas = traj.t[-1] / revolutions / c.YEAR
    a_list.append(a_au); T_list.append(T_meas)
    print(f"{name:10s} {a_au:8.3f} {T_meas:16.4f} {a_au ** 1.5:8.4f}")
slope = np.polyfit(np.log(a_list), np.log(T_list), 1)[0]
print(f"Slope of log T vs log a = {slope:.4f}  (Kepler: 1.5)")

# --- Figure 1: inner and outer orbits -------------------------------------------
colors = [plotting.TEXT_2] + [plotting.body_color(i, 8) for i in range(1, 9)]   # grey Sun
inner = traj.t <= 2 * c.YEAR
fig, (ax1, ax2) = plotting.new_figure(12, 5.6, ncols=2)
for i, name in enumerate(traj.names):
    p = traj.positions[:, i] / c.AU
    if i <= 4:
        ax1.plot(p[inner, 0], p[inner, 1], color=colors[i], lw=1.2, label=name)
    if i == 0 or i >= 5:
        ax2.plot(p[:, 0], p[:, 1], color=colors[i], lw=1.0, label=name)
    else:
        ax2.plot(p[:, 0], p[:, 1], color=colors[i], lw=0.6)
for ax, ttl, lim in ((ax1, "Inner planets (2 years)", 1.75), (ax2, "Whole system (165 years)", 32)):
    ax.set_aspect("equal"); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("x (AU)"); ax.set_ylabel("y (AU)"); ax.set_title(ttl)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
save(fig, "07_solar_system_orbits.png")

# --- Figure 2: Kepler's third law -----------------------------------------------
fig, ax = plotting.new_figure(7.5, 5.2)
aa = np.logspace(-0.5, 1.6, 50)
ax.loglog(aa, aa ** 1.5, color=plotting.TEXT_2, lw=1, ls="--", label=r"Kepler: $T = a^{3/2}$", zorder=1)
ax.loglog(a_list, T_list, "o", color=plotting.PALETTE[0], ms=8, label="measured period", zorder=2)
for a_au, T_m, (name, *_) in zip(a_list, T_list, scenarios.PLANETS):
    ax.annotate(name, (a_au, T_m), textcoords="offset points", xytext=(8, -12), fontsize=8.5,
                color=plotting.TEXT_2)
ax.set_xlabel("semi-major axis a (AU)"); ax.set_ylabel("period T (years)")
ax.set_title(f"Kepler's third law from the simulation (slope {slope:.3f})")
ax.legend(loc="upper left")
save(fig, "07_kepler_third_law.png")

fig = plotting.plot_conservation(traj, time_unit=c.YEAR, time_label="years",
                                 title="Solar System — velocity Verlet, h = 0.5 days")
save(fig, "07_solar_system_conservation.png")

# --- Videos ---------------------------------------------------------------------
if MAKE_VIDEOS:
    from nbody.video import render_video
    sun = "#ffd166"
    v_cols = [sun, "#b8b8b8", "#ffc98b", "#5aa2ff", "#ff6b4a"]
    inner_run = simulate(scenarios.solar_system(n_planets=4), 2 * c.YEAR, 0.25 * c.DAY, save_every=2)
    mp4, png = video_path("solar_system_inner")
    render_video(inner_run, mp4, poster=png, title="The inner Solar System",
                 subtitle="Real masses, eccentricities and inclinations (z exaggerated ×4 for visibility)",
                 length_unit=c.AU, unit_label="AU", time_unit=c.DAY, time_label="days", time_decimals=0,
                 duration=20, trail=60 * c.DAY, colors=v_cols, sizes=[260, 22, 40, 42, 30],
                 limits=(np.zeros(3), np.array([3.4, 3.4, 1.0])), elev=28, rotation=45, zoom=1.2,
                 z_scale=4.0)

    outer = traj  # 165-year run of the whole system
    v_cols8 = [sun, "#b8b8b8", "#ffc98b", "#5aa2ff", "#ff6b4a", "#e8b37a", "#f2d49b", "#8fe3f0", "#6c8cff"]
    mp4, png = video_path("solar_system_outer")
    render_video(outer, mp4, poster=png, title="The Solar System over one orbit of Neptune",
                 subtitle="165 years in 25 seconds (z exaggerated ×4 for visibility)",
                 length_unit=c.AU, unit_label="AU", time_unit=c.YEAR, time_label="years", time_decimals=1,
                 duration=25, trail=40 * c.YEAR, colors=v_cols8,
                 sizes=[300, 6, 10, 10, 8, 120, 100, 60, 60],
                 limits=(np.zeros(3), np.array([64.0, 64.0, 18.0])), elev=34, rotation=60, zoom=1.45,
                 z_scale=4.0)
