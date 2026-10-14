"""
Example 3 — Chaos in a six-body system (Section 4.3 of the report), now quantified
=================================================================================

What it does
------------
1. The four runs of the report: the reference and three runs with a 0.1 %
   relative change in (a) the x coordinate of body 1, (b) the velocity v_y of
   body 2 and (c) the mass of body 3. The orbits are plotted.
2. Measures the separation delta_r(t) between each perturbed run and the
   reference. A straight line on a log scale means exponential growth -> CHAOS.
3. Extra experiment: the SAME initial condition integrated with h and with h/2.
   It also diverges exponentially! The numerical error (~h^2) acts as one more
   perturbation: after a while neither run is "the true" trajectory. Yet E, P
   and L are still conserved, so statistical properties remain reliable (the
   *shadowing* lemma).
4. Estimates the maximal Lyapunov exponent with Benettin's method.
5. Renders a side-by-side video: reference vs. perturbed run.

Run time: ~1 minute for the simulations, ~2 minutes for the video.
"""

from _common import MAKE_VIDEOS, save, video_path

import numpy as np

from nbody import chaos, diagnostics, plotting, scenarios, simulate
from nbody import constants as c

T_FINAL = 10 * c.YEAR
H = 1000.0
SAVE = 50

base = scenarios.six_body_report()
cases = {
    "Reference": base,
    "x₁ + 0.1 %": base.perturbed(body=0, quantity="position", component=0, relative=1e-3),
    "v_y,2 + 0.1 %": base.perturbed(body=1, quantity="velocity", component=1, relative=1e-3),
    "m₃ + 0.1 %": base.perturbed(body=2, quantity="mass", relative=1e-3),
}

trajs = {}
for name, sys in cases.items():
    trajs[name] = simulate(sys, T_FINAL, H, save_every=SAVE)
    print(diagnostics.conservation_report(trajs[name], name))

# Same initial condition with step h/2 (stored at the same instants)
trajs["Reference, h/2"] = simulate(base, T_FINAL, H / 2, save_every=2 * SAVE)

# --- Figure 1: orbits (as Figure 9 of the report) ---------------------------
fig = plotting.plt.figure(figsize=(10.5, 9.4), layout="constrained")
for k, name in enumerate(cases):
    ax = fig.add_subplot(2, 2, k + 1, projection="3d")
    plotting.plot_orbits(trajs[name], ax=ax, length_unit=1e5, unit_label="10⁵ m", title=name,
                         legend=None, markers=False, linewidth=0.9, zoom=1.05)
handles = [plotting.plt.Line2D([], [], color=plotting.PALETTE[i], lw=2, label=f"Body {i + 1}")
           for i in range(6)]
fig.legend(handles=handles, loc="outside lower center", ncol=6)
fig.suptitle("Six bodies: a 0.1 % change in a single initial condition",
             fontweight="bold", color=plotting.TEXT, fontsize=13)
save(fig, "03_chaos_orbits.png")

# --- Figure 2: separation between trajectories --------------------------------
ref = trajs["Reference"]
deltas = {name: chaos.position_divergence(ref, tr) for name, tr in trajs.items() if name != "Reference"}
fig = plotting.plot_divergence(ref.t, deltas, time_unit=c.YEAR, time_label="years",
                               title="Exponential growth of the differences",
                               reference_level=(1e5, "size of the system"))
save(fig, "03_chaos_divergence.png")

# --- Lyapunov exponent (Benettin) ---------------------------------------------
t_l, lam = chaos.lyapunov_benettin(base, T_FINAL, H, delta0=1e-8, renorm_every=20)
lam_final = lam[-len(lam) // 5:].mean()                     # average of the last 20 %
print(f"\nLyapunov exponent lambda ~ {lam_final:.3e} 1/s  ->  Lyapunov time 1/lambda ~ "
      f"{1 / lam_final / c.DAY:.0f} days")
print(f"In 10 years an initial error is amplified ~e^{T_FINAL * lam_final:.0f} ~ "
      f"10^{T_FINAL * lam_final / np.log(10):.0f} times: not even double precision (1e-16) helps.")

fig, ax = plotting.new_figure(8.5, 4.0)
ax.plot(t_l / c.YEAR, lam * c.YEAR, color=plotting.PALETTE[6])
ax.axhline(lam_final * c.YEAR, color=plotting.TEXT_2, ls="--", lw=0.8)
ax.text(0.98, 0.06, f"λ ≈ {lam_final * c.YEAR:.1f} yr⁻¹   (1/λ ≈ {1 / lam_final / c.DAY:.0f} days)",
        transform=ax.transAxes, ha="right", color=plotting.TEXT_2, fontsize=9.5)
ax.set_xlabel("t (years)"); ax.set_ylabel("λ (1/year)")
ax.set_title("Maximal Lyapunov exponent (Benettin's method)")
save(fig, "03_chaos_lyapunov.png")

# --- Video: reference vs. perturbed, side by side ---------------------------------
if MAKE_VIDEOS:
    from nbody.video import render_video
    mp4, png = video_path("six_body_chaos")
    render_video([trajs["Reference"], trajs["x₁ + 0.1 %"]], mp4, poster=png, poster_time=0.55,
                 titles=["Reference", "x₁ changed by 0.1 %"],
                 title="Chaos in a six-body system",
                 subtitle="Same equations, almost the same start, completely different futures",
                 length_unit=1e5, unit_label="10⁵ m", time_unit=c.YEAR, time_label="years",
                 duration=30, trail=2.0 * c.YEAR, sizes=[55] * 6, elev=20, rotation=40, zoom=1.0)
