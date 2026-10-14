"""
Static figures with Matplotlib (videos live in video.py).

Functions
---------
    plot_orbits(traj, ...)          orbits in 3D or as a 2D projection
    plot_conservation(traj, ...)    energies + relative errors of E, P and L
    plot_divergence(t, deltas, ...) separation between trajectories (log scale)
    new_figure(w, h)                figure with constrained layout (no overlaps)
    set_language("en" | "es")       language of the built-in labels

Design decisions
----------------
* ``constrained_layout``: Matplotlib computes the space needed by titles,
  labels and legends, so nothing overlaps. Legends with several entries go
  OUTSIDE the axes, to the right, where they never cover data.
* Fixed categorical palette (8 colors validated for color-vision deficiency);
  body i always has the same color in every figure and video.
* Errors on a logarithmic scale: an error of 1e-12 and one of 1e-6 look the
  same on a linear axis but differ by six orders of magnitude.
* Dense curves (hundreds of thousands of points) are rasterized, so exported
  PDF figures stay light.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from . import diagnostics as diag

# Categorical palette (fixed order). More than 8 bodies -> a single color.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"

# ---------------------------------------------------------------------------
# Built-in labels (English by default; Spanish is used by the report builder)
# ---------------------------------------------------------------------------
_TEXTS = {
    "en": {
        "energies": "Energies", "kinetic": "Kinetic K", "potential": "Potential U",
        "total": "Total E = K + U", "energy": "Energy", "energy_error": "Relative energy error",
        "momenta": "Momentum conservation", "linear": "Linear momentum P",
        "angular": "Angular momentum L", "rel_error": "relative error",
        "conservation": "Conservation laws", "separation": "Separation between trajectories",
    },
    "es": {
        "energies": "Energías", "kinetic": "Cinética K", "potential": "Potencial U",
        "total": "Total E = K + U", "energy": "Energía", "energy_error": "Error relativo de la energía",
        "momenta": "Conservación de los momentos", "linear": "Momento lineal P",
        "angular": "Momento angular L", "rel_error": "error relativo",
        "conservation": "Leyes de conservación", "separation": "Separación entre trayectorias",
    },
}
_LANG = "en"


def set_language(lang: str) -> None:
    """Choose the language of the built-in labels: "en" (default) or "es"."""
    global _LANG
    if lang not in _TEXTS:
        raise ValueError(f"Unsupported language {lang!r}")
    _LANG = lang


def tr(key: str) -> str:
    return _TEXTS[_LANG][key]


def body_color(i: int, n: int) -> str:
    return PALETTE[i % len(PALETTE)] if n <= len(PALETTE) else PALETTE[0]


def apply_style() -> None:
    """Clean, readable style shared by every figure of the package."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": "#cfcdc6", "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
        "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.labelsize": 10,
        "axes.titlepad": 8, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 1.4,
        "font.family": "DejaVu Sans", "figure.dpi": 110, "savefig.dpi": 200,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.06, "figure.constrained_layout.w_pad": 0.06,
        "axes.formatter.use_mathtext": True, "axes.formatter.limits": (-3, 4),
    })


apply_style()


def new_figure(width: float = 8.0, height: float = 5.0, **subplot_kw):
    """Figure + axes with constrained layout (``plt.subplots`` arguments allowed)."""
    return plt.subplots(figsize=(width, height), layout="constrained", **subplot_kw)


# ---------------------------------------------------------------------------
# Orbits
# ---------------------------------------------------------------------------
def _equal_3d(ax, pos: np.ndarray, min_fraction=(0.45, 0.45, 0.25), pad: float = 0.06,
              zoom: float = 0.9) -> None:
    """Equal scale on the three axes, with a minimum z extent for flat systems.

    Matplotlib 3D axes stretch each axis independently, which deforms circles
    into ellipses. We give every axis the same length unit per data unit
    (``set_box_aspect`` with the real ranges); flat systems get a minimum z
    extent so the box does not collapse into a sheet. ``zoom`` < 1 leaves room
    for the axis labels.
    """
    lo, hi = pos.reshape(-1, 3).min(axis=0), pos.reshape(-1, 3).max(axis=0)
    center, span = (lo + hi) / 2, (hi - lo) * (1 + 2 * pad)
    span = np.maximum(span, span.max() * np.asarray(min_fraction))
    for setter, cc, s in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center, span):
        setter(cc - s / 2, cc + s / 2)
    ax.set_box_aspect(span / span.max(), zoom=zoom)


def plot_orbits(traj, ax=None, projection: str = "3d", length_unit: float = 1.0,
                unit_label: str = "m", title: str | None = None, legend: str | None = "outside",
                markers: bool = True, linewidth: float = 1.3, colors=None, names=None,
                elev: float = 24, azim: float = -58, zoom: float = 0.9):
    """Draw the orbits of all bodies.

    projection  : "3d", "xy", "xz" or "yz"
    length_unit : positions are divided by it (e.g. AU for the Solar System)
    legend      : "outside" (right of the axes), "inside", or None
    zoom        : size of the 3D box inside its axes (< 1 leaves room for labels)
    """
    pos = traj.positions / length_unit
    n = traj.n_bodies
    names = names if names is not None else traj.names
    colors = colors if colors is not None else [body_color(i, n) for i in range(n)]
    many = n > len(PALETTE)
    is3d = projection == "3d"
    if ax is None:
        fig = plt.figure(figsize=(7.2, 5.6) if is3d else (6.6, 5.4), layout="constrained")
        ax = fig.add_subplot(projection="3d") if is3d else fig.add_subplot()
    a, b = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}.get(projection, (0, 1))
    axis_names = "xyz"

    for i in range(n):
        kw = dict(color=colors[i], lw=0.4 if many else linewidth, alpha=0.5 if many else 1.0,
                  label=None if many else names[i], rasterized=pos.shape[0] > 20_000)
        if is3d:
            ax.plot(pos[:, i, 0], pos[:, i, 1], pos[:, i, 2], **kw)
            if markers:
                ax.scatter(*pos[-1, i], color=colors[i], s=22, depthshade=False,
                           edgecolors=SURFACE, linewidths=0.8, zorder=5)
        else:
            ax.plot(pos[:, i, a], pos[:, i, b], **kw)
            if markers:
                ax.plot(pos[-1, i, a], pos[-1, i, b], "o", color=colors[i], ms=5.5,
                        markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=5)

    if is3d:
        _equal_3d(ax, pos, zoom=zoom)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel(f"x ({unit_label})", labelpad=6)
        ax.set_ylabel(f"y ({unit_label})", labelpad=6)
        ax.set_zlabel(f"z ({unit_label})", labelpad=6)
        ax.tick_params(labelsize=8, pad=1)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_facecolor(SURFACE); pane.set_edgecolor(GRID)
    else:
        ax.set_xlabel(f"{axis_names[a]} ({unit_label})"); ax.set_ylabel(f"{axis_names[b]} ({unit_label})")
        ax.set_aspect("equal", adjustable="datalim")
    if title:
        ax.set_title(title)
    if legend and not many:
        if legend == "outside":
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
        else:
            ax.legend(loc="best")
    return ax


# ---------------------------------------------------------------------------
# Conservation laws
# ---------------------------------------------------------------------------
def plot_conservation(traj, time_unit: float = 1.0, time_label: str = "s",
                      energy_unit: float = 1.0, energy_label: str = "J", title: str | None = None,
                      width: float = 9.0, height: float = 7.8):
    """Three stacked panels sharing the time axis:

        (a) K, U and E                         (linear scale)
        (b) |E(t) - E0| / |E0|                 (log scale)
        (c) relative errors of P and L         (log scale)

    Legends sit to the right of each panel, so they never cover the curves.
    """
    t = traj.t / time_unit
    K, U = traj.kinetic_energy(), traj.potential_energy()
    E = K + U
    P, L = traj.linear_momentum(), traj.angular_momentum()
    p_scale = diag.momentum_scale(traj.masses, traj.velocities)
    l_scale = diag.angular_momentum_scale(traj.masses, traj.positions, traj.velocities)
    dense = t.size > 20_000

    fig, axs = plt.subplots(3, 1, figsize=(width, height), sharex=True, layout="constrained",
                            gridspec_kw={"height_ratios": [1.15, 1, 1]})
    ax = axs[0]
    lw = 0.7 if dense else 1.3
    ax.plot(t, K / energy_unit, color=PALETTE[0], lw=lw, label=tr("kinetic"), rasterized=dense)
    ax.plot(t, U / energy_unit, color=PALETTE[1], lw=lw, label=tr("potential"), rasterized=dense)
    ax.plot(t, E / energy_unit, color=TEXT, lw=1.6, label=tr("total"))
    ax.set_ylabel(f"{tr('energy')} ({energy_label})")
    ax.set_title(f"(a) {tr('energies')}", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)

    def clean(y):
        return np.where(y > 0, y, np.nan)                       # log(0) is undefined: hide it

    ax = axs[1]
    ax.semilogy(t, clean(diag.relative_error(E)), color=PALETTE[6], lw=0.8, rasterized=dense,
                label="|E(t) − E₀| / |E₀|")
    ax.set_ylabel(tr("rel_error"))
    ax.set_title(f"(b) {tr('energy_error')}", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)

    ax = axs[2]
    ax.semilogy(t, clean(diag.relative_error(P, p_scale)), color=PALETTE[2], lw=0.9,
                rasterized=dense, label=f"{tr('linear')}\n|ΔP| / max Σ mᵢ|vᵢ|")
    ax.semilogy(t, clean(diag.relative_error(L, l_scale)), color=PALETTE[3], lw=0.9,
                rasterized=dense, label=f"{tr('angular')}\n|ΔL| / max Σ mᵢ|rᵢ||vᵢ|")
    ax.set_ylabel(tr("rel_error"))
    ax.set_title(f"(c) {tr('momenta')}", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0, labelspacing=1.0)
    ax.set_xlabel(f"t ({time_label})")

    if title:
        fig.suptitle(title, color=TEXT, fontweight="bold", fontsize=12.5)
    return fig


def plot_divergence(t, deltas: dict, time_unit: float = 1.0, time_label: str = "s",
                    title: str | None = None, ylabel: str = "δr(t) (m)", ax=None,
                    reference_level: tuple[float, str] | None = None):
    """delta(t) on a log scale: a straight line means exponential growth (chaos)."""
    if ax is None:
        fig, ax = new_figure(8.5, 4.4)
    for k, (label, d) in enumerate(deltas.items()):
        ax.semilogy(t / time_unit, np.where(d > 0, d, np.nan), color=PALETTE[k % len(PALETTE)],
                    lw=1.5, label=label)
    if reference_level:
        level, text = reference_level
        ax.axhline(level, color=TEXT_2, ls="--", lw=0.9)
        ax.annotate(text, (0.01, level), xycoords=("axes fraction", "data"), xytext=(0, 4),
                    textcoords="offset points", color=TEXT_2, fontsize=8.5, va="bottom")
    ax.set_xlabel(f"t ({time_label})"); ax.set_ylabel(ylabel)
    ax.set_title(title or tr("separation"))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    return ax.figure
