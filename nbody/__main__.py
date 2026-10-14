"""
Command-line interface:  python -m nbody <scenario> [options]

Examples
--------
    python -m nbody figure-eight --t 3 --video eight.mp4    # --t in the scenario's unit
    python -m nbody sun-earth-moon --t 2 --plot              # years
    python -m nbody plummer --n 200 --t 10 --h 0.005 --video cluster.mp4
    python -m nbody six-body --integrator yoshida4 --t 5
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from . import constants as c
from . import diagnostics, plotting, scenarios
from .integrators import INTEGRATORS
from .simulation import simulate

# scenario -> (builder, time unit, unit name, default t_final, default h, length unit, length label)
SCENARIOS = {
    "sun-earth-moon": (scenarios.sun_earth_moon, c.YEAR, "yr", 2.0, 1000.0, c.AU, "AU"),
    "four-body": (scenarios.four_body_report, c.YEAR, "yr", 1.0, 1000.0, 1e13, "1e13 m"),
    "six-body": (scenarios.six_body_report, c.YEAR, "yr", 2.0, 1000.0, 1e5, "1e5 m"),
    "solar-system": (scenarios.solar_system, c.YEAR, "yr", 165.0, 0.5 * c.DAY, c.AU, "AU"),
    "figure-eight": (scenarios.figure_eight, scenarios.FIGURE_EIGHT_PERIOD, "periods", 3.0, 1e-3, 1.0, ""),
    "lagrange": (scenarios.lagrange_triangle, 1.0, "", 30.0, 1e-3, 1.0, ""),
    "plummer": (None, 1.0, "", 10.0, 5e-3, 1.0, ""),
    "cold-collapse": (None, 1.0, "", 10.0, 5e-3, 1.0, ""),
}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m nbody", description="Gravitational N-body simulator")
    p.add_argument("scenario", choices=SCENARIOS)
    p.add_argument("--t", type=float, help="final time in the scenario's unit (years, periods...)")
    p.add_argument("--h", type=float, help="time step in internal units (s for SI scenarios)")
    p.add_argument("--integrator", default="verlet", choices=INTEGRATORS)
    p.add_argument("--n", type=int, default=100, help="number of stars (plummer, cold-collapse)")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--plot", action="store_true", help="show orbits and conservation laws")
    p.add_argument("--video", help="save a 3D MP4 video to this file (requires ffmpeg)")
    args = p.parse_args(argv)

    builder, t_unit, t_label, t_default, h_default, l_unit, l_label = SCENARIOS[args.scenario]
    if args.scenario == "plummer":
        system = scenarios.plummer_cluster(args.n)
    elif args.scenario == "cold-collapse":
        system = scenarios.uniform_sphere(args.n, virial_ratio=0.0)
    else:
        system = builder()
    t_final = (args.t if args.t is not None else t_default) * t_unit
    h = args.h if args.h is not None else h_default

    print(f"{args.scenario}: N={system.n_bodies}, t_final={t_final / t_unit:g} {t_label}, "
          f"h={h:g}, integrator={args.integrator}")
    traj = simulate(system, t_final, h, args.integrator, save_every=args.save_every, verbose=True)
    print(diagnostics.conservation_report(traj))

    if args.video:
        from .video import render_video
        render_video(traj, args.video, title=args.scenario, length_unit=l_unit, unit_label=l_label,
                     time_unit=t_unit, time_label=t_label, duration=12)
    if args.plot:
        plotting.plot_orbits(traj, length_unit=l_unit, unit_label=l_label or "length units",
                             title=args.scenario)
        plotting.plot_conservation(traj, time_unit=t_unit, time_label=t_label or "time units")
        plt.show()


if __name__ == "__main__":
    main()
