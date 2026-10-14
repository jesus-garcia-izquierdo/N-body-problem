"""
Example 6 — Why Verlet? Comparing integrators
=============================================

Two experiments with the figure-eight orbit (G = 1, period T ~ 6.33):

A) Long term: 200 periods with a "coarse" step h = T/300.
   * Explicit Euler: the energy drifts without control (the system "gains"
     energy at every step) -> useless result.
   * RK4: very accurate at first, but its error grows linearly and
     systematically (it is not symplectic).
   * Symplectic Euler, Verlet, Yoshida-4: the energy error oscillates but does
     NOT grow -> reliable for very long integrations.

B) Order of convergence: position error after 1 period versus the number of
   force evaluations (the real cost). On a log-log scale the slope is -p, with
   p the order of the method: error ~ h^p ~ (number of evaluations)^{-p}.

Moral: high order wins in the short term; symplectic structure wins in the
long term. Yoshida-4 has both.

The video shows the same orbit integrated with explicit Euler and with Verlet,
using the same step.
"""

from _common import MAKE_VIDEOS, save, video_path

import numpy as np

from nbody import INTEGRATORS, diagnostics, plotting, scenarios, simulate

T = scenarios.FIGURE_EIGHT_PERIOD
system = scenarios.figure_eight()
names = ["euler", "symplectic_euler", "verlet", "yoshida4", "rk4"]
colors = {n: plotting.PALETTE[i] for i, n in enumerate(names)}

# ---------------------------------------------------------------------------
# A) Long-term energy error
# ---------------------------------------------------------------------------
# The error oscillates many times per period; to show the TREND we plot its
# envelope: the maximum within each period.
SAMPLES_PER_PERIOD = 10                                       # 300 steps / save_every=30
fig, ax = plotting.new_figure(10, 4.6)
for name in names:
    info = INTEGRATORS[name]
    traj = simulate(system, 200 * T, T / 300, integrator=name, save_every=30)
    err = diagnostics.relative_error(traj.total_energy())
    err = np.where(np.isfinite(err), err, np.inf)[1:]          # drop t = 0 (zero error)
    n_per = err.size // SAMPLES_PER_PERIOD
    envelope = err[:n_per * SAMPLES_PER_PERIOD].reshape(n_per, SAMPLES_PER_PERIOD).max(axis=1)
    envelope = np.where(envelope < 10, envelope, np.nan)       # Euler "explodes": stop the curve
    ax.semilogy(np.arange(1, n_per + 1), envelope, color=colors[name], lw=1.8,
                label=f"{info.label}\n(order {info.order}{', symplectic' if info.symplectic else ''})")
    print(f"{info.label:18s} final energy error: {err[-1]:.2e}   ({traj.wall_time:.1f} s)")
ax.set_ylim(1e-8, 10)
ax.set_xlabel("t / T  (periods)"); ax.set_ylabel("max |E − E₀| / |E₀| per period")
ax.set_title("Long-term energy error (h = T/300)")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0, labelspacing=0.9)
save(fig, "06_integrators_energy.png")

# ---------------------------------------------------------------------------
# B) Order of convergence
# ---------------------------------------------------------------------------
# Reference solution: Yoshida-4 with a very small step
ref = simulate(system, T, T / 40_000, integrator="yoshida4", save_every=40_000).positions[-1]

steps_per_period = np.array([100, 200, 400, 800, 1600, 3200])
fig, ax = plotting.new_figure(9.5, 5.0)
for name in names:
    info = INTEGRATORS[name]
    errs, evals = [], []
    for n in steps_per_period:
        traj = simulate(system, T, T / n, integrator=name, save_every=int(n))
        errs.append(np.linalg.norm(traj.positions[-1] - ref))
        evals.append(n * info.force_evals)
    errs, evals = np.array(errs), np.array(evals)
    # Slope from the three smallest steps (asymptotic regime, h -> 0)
    slope = np.polyfit(np.log(evals[-3:]), np.log(errs[-3:]), 1)[0]
    ax.loglog(evals, errs, "o-", color=colors[name], ms=5, lw=1.5, label=f"{info.label}: slope {slope:.2f}")
    print(f"{info.label:18s} measured order ~ {-slope:.2f} (theory {info.order})")
ax.set_xlabel("force evaluations per period  (cost)")
ax.set_ylabel("position error after one period")
ax.set_title("Order of convergence: error ∝ cost^(−p)")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
save(fig, "06_integrators_convergence.png")

# ---------------------------------------------------------------------------
# Video: explicit Euler vs. Verlet with the same step
# ---------------------------------------------------------------------------
if MAKE_VIDEOS:
    from nbody.video import render_video
    h = T / 300
    runs = [simulate(system, 6 * T, h, integrator=k, save_every=1) for k in ("euler", "verlet")]
    mp4, png = video_path("euler_vs_verlet")
    render_video(runs, mp4, poster=png, poster_time=0.75, titles=["Explicit Euler", "Velocity Verlet"],
                 title="Same orbit, same time step, two integrators",
                 subtitle="Euler pumps energy into the system at every step; Verlet (symplectic) keeps it bounded",
                 time_unit=T, time_label="periods", duration=20, trail=T / 2, sizes=[60] * 3,
                 limits=(np.zeros(3), np.array([3.6, 3.6, 1.2])), elev=40, rotation=30, zoom=0.95)
