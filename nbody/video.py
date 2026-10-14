"""
High-quality 3D videos (MP4, H.264) of N-body trajectories.

What a video shows
------------------
* Bodies as glowing spheres (a bright core plus a faint halo); size ~ mass^(1/3)
  by default, like real spheres of equal density.
* Fading trails: the recent past of each orbit, transparent at the tail and
  opaque at the head, so the direction of motion is obvious.
* A slowly rotating camera: the eye reconstructs depth from motion parallax,
  which is what makes a 3D plot readable on a flat screen.
* Overlays: title, subtitle, elapsed time and legend.
* One or several panels side by side with the SAME camera (e.g. a reference
  run next to a perturbed one, to show chaos).

Data flow
---------
    render_video(trajs, path, ...)
        frame k  ->  stored sample idx[k]       (frames are evenly spaced in time)
        trail    ->  samples idx[k]-n_trail .. idx[k]  (all stored samples, smooth curve)
        writer   ->  ffmpeg, libx264, yuv420p, CRF 18 (visually lossless), +faststart

Requirements: ffmpeg must be installed (``conda install ffmpeg``,
``brew install ffmpeg`` or ``apt install ffmpeg``).

Why Matplotlib and not a game engine: it needs no extra dependency, it is
deterministic (the same script always produces the same video) and the
figure code is the same one used for the static plots.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# Dark "space" theme. Colors are the same hues as the static palette, lifted
# so that they read well on a near-black background.
BACKGROUND = "#06080d"
PANE = "#0b0f17"
GRID = "#1d2430"
TICKS = "#7d8696"
TEXT = "#eef1f5"
TEXT_2 = "#a7afbd"
MANY_BODIES = 12                     # above this, bodies are drawn as dots without trails
VIDEO_PALETTE = ["#5aa2ff", "#ff8a50", "#3ddc97", "#ffc233", "#ff7eb6", "#58c75a", "#a89cff", "#ff6b6b"]


@dataclass
class _Panel:
    """Everything needed to redraw one 3D panel."""
    ax: object
    pos: np.ndarray            # (T, N, 3) positions in display units
    trails: list               # Line3DCollection per body (empty for many-body runs)
    halo: object               # scatter (glow)
    core: object               # scatter (body)
    trail_rgba: np.ndarray     # (N, 4) base colors
    lo: np.ndarray             # (3,) lower corner of the displayed box
    hi: np.ndarray             # (3,) upper corner of the displayed box

    def inside(self, x: np.ndarray) -> np.ndarray:
        """Boolean mask of the bodies that lie inside the displayed box."""
        return np.all((x >= self.lo) & (x <= self.hi), axis=1)


def _cube_limits(pos_list, pad=0.08, min_fraction=(0.6, 0.6, 0.35)):
    """Common limits for all panels with the SAME scale on x, y and z.

    Each axis spans at least ``min_fraction`` of the largest span, so a thin
    or flat system sits in a well-proportioned box instead of a sliver (the
    limits grow; the data are never stretched).
    """
    allpos = np.concatenate([p.reshape(-1, 3) for p in pos_list])
    lo, hi = np.percentile(allpos, 0.2, axis=0), np.percentile(allpos, 99.8, axis=0)
    center, span = (lo + hi) / 2, (hi - lo) * (1 + 2 * pad)
    span = np.maximum(span, span.max() * np.asarray(min_fraction))
    return center, span


def _style_axes(ax, center, span, unit_label, show_axes, zoom):
    ax.set_facecolor(BACKGROUND)
    for setter, cc, s in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center, span):
        setter(cc - s / 2, cc + s / 2)
    ax.set_box_aspect(span / span.max(), zoom=zoom)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(PANE)
        axis.pane.set_edgecolor(GRID)
        axis._axinfo["grid"].update(color=GRID, linewidth=0.6)
        axis.label.set_color(TICKS)
    ax.tick_params(colors=TICKS, labelsize=8, pad=0)
    if unit_label:
        ax.set_xlabel(f"x ({unit_label})", labelpad=2, fontsize=9)
        ax.set_ylabel(f"y ({unit_label})", labelpad=2, fontsize=9)
        ax.set_zlabel(f"z ({unit_label})", labelpad=2, fontsize=9)
    if not show_axes:
        ax.set_axis_off()


def _make_frame_function(fig, panels, idx, t_display, n_trail, alpha_ramp, elev, azim0,
                         rotation, time_text, time_fmt):
    n_frames = len(idx)

    def update(k):
        i = idx[k]
        i0 = max(0, i - n_trail)
        for p in panels:
            x = p.pos[i]                                             # (N, 3) current positions
            if not p.trails:                                         # many-body mode: hide escapers
                x = np.where(p.inside(x)[:, None], x, np.nan)        # outside the box (NaN = not drawn)
            p.core._offsets3d = (x[:, 0], x[:, 1], x[:, 2])
            p.halo._offsets3d = (x[:, 0], x[:, 1], x[:, 2])
            if p.trails:
                seg_pos = p.pos[i0:i + 1]
                stride = max(1, len(seg_pos) // 350)                 # <= ~350 segments per trail
                seg_pos = seg_pos[::-1][::stride][::-1]              # keep the newest point exactly
                if len(seg_pos) >= 2:
                    ramp = alpha_ramp(len(seg_pos) - 1)
                    for b, coll in enumerate(p.trails):
                        pts = seg_pos[:, b, :]
                        segs = np.stack([pts[:-1], pts[1:]], axis=1)    # (S, 2, 3)
                        col = np.tile(p.trail_rgba[b], (len(segs), 1))
                        col[:, 3] = ramp
                        coll.set_segments(segs)
                        coll.set_color(col)
            p.ax.view_init(elev=elev, azim=azim0 + rotation * k / max(n_frames - 1, 1))
        if time_text is not None:
            time_text.set_text(time_fmt(t_display[i]))
        return []

    return update


def render_video(trajs, path: str, *, titles=None, title: str = "", subtitle: str = "",
                 duration: float = 20.0, fps: int = 30, size=(1280, 720),
                 length_unit: float = 1.0, unit_label: str = "", time_unit: float = 1.0,
                 time_label: str = "", time_decimals: int = 2, trail: float | None = None,
                 colors=None, sizes=None, names=None, legend: bool = True,
                 elev: float = 22.0, azim: float = -60.0, rotation: float = 50.0,
                 limits=None, show_axes: bool = True, zoom: float = 1.2, z_scale: float = 1.0,
                 t_range=None, crf: int = 18,
                 poster: str | None = None, poster_time: float = 0.6, verbose: bool = True) -> str:
    """Render one or several trajectories (side-by-side panels) to an MP4 file.

    Parameters
    ----------
    trajs        : a Trajectory or a list of Trajectories (same stored times)
    titles       : one title per panel (optional)
    duration     : length of the video in seconds
    size         : (width, height) in pixels; both must be even (H.264 requirement)
    length_unit  : positions are divided by it; ``unit_label`` names it
    time_unit    : times are divided by it for the clock; ``time_label`` names it
    trail        : trail length in the same units as traj.t (default: 1/12 of the run)
    colors       : one color per body (default: video palette; a single color for N > 12)
    sizes        : marker area per body (default: ~ mass^(1/3), between 22 and 132)
    rotation     : total camera rotation in degrees over the video
    limits       : None (auto), or (center (3,), span (3,)) in display units
    zoom         : camera zoom of the 3D box (1 = Matplotlib default)
    z_scale      : multiply z by this factor for display (say so in the subtitle!)
    t_range      : (t_start, t_end) to render only part of the run
    crf          : H.264 quality (lower = better; 18 is visually lossless)
    poster       : if given, also save a PNG still at ``poster_time`` (fraction of the video)
    """
    if not isinstance(trajs, (list, tuple)):
        trajs = [trajs]
    n_panels = len(trajs)
    ref = trajs[0]
    n_bodies = ref.n_bodies
    many = n_bodies > MANY_BODIES        # many-body mode: no trails, small dots
    names = names if names is not None else ref.names
    if colors is None:
        colors = [VIDEO_PALETTE[i % len(VIDEO_PALETTE)] for i in range(n_bodies)] if not many \
            else [VIDEO_PALETTE[0]] * n_bodies
    if sizes is None:
        r = np.cbrt(ref.masses / ref.masses.max())
        sizes = (22 + 110 * r) if not many else np.full(n_bodies, 7.0)
    sizes = np.asarray(sizes, dtype=float)

    # ---- frames: evenly spaced in time within t_range -------------------
    t = ref.t
    i_start, i_end = 0, t.size - 1
    if t_range is not None:
        i_start = int(np.searchsorted(t, t_range[0]))
        i_end = int(np.searchsorted(t, t_range[1], side="right") - 1)
    n_frames = int(round(duration * fps))
    idx = np.linspace(i_start, i_end, n_frames).round().astype(int)
    dt = t[1] - t[0] if t.size > 1 else 1.0
    trail = trail if trail is not None else (t[i_end] - t[i_start]) / 12
    n_trail = 0 if many else max(2, int(round(trail / dt)))

    pos_list = [tr.positions / length_unit for tr in trajs]
    if z_scale != 1.0:                   # optional vertical exaggeration (e.g. nearly flat planetary systems)
        pos_list = [p * np.array([1.0, 1.0, z_scale]) for p in pos_list]
    center, span = (np.asarray(limits[0]), np.asarray(limits[1])) if limits is not None \
        else _cube_limits(pos_list)

    # ---- figure ----------------------------------------------------------
    dpi = 100
    fig = plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi, facecolor=BACKGROUND, layout="none")
    top = 0.86 if (title or subtitle) else 0.95
    panel_w = 0.98 / n_panels
    panels = []
    for p_i, (pos, tr_) in enumerate(zip(pos_list, trajs)):
        ax = fig.add_axes([0.01 + p_i * panel_w, 0.06, panel_w, top - 0.06], projection="3d",
                          computed_zorder=False)
        _style_axes(ax, center, span, unit_label, show_axes, zoom)
        if titles:
            ax.set_title(titles[p_i], color=TEXT, fontsize=12, fontweight="bold", pad=0, y=0.98)
        rgba = np.array([to_rgba(cc) for cc in colors])
        trails = []
        if not many:
            for b in range(n_bodies):
                x_start = pos[idx[0], b]
                coll = Line3DCollection([np.stack([x_start, x_start])], linewidths=1.6,
                                        capstyle="round", colors=[(0, 0, 0, 0)], zorder=3,
                                        clip_on=False)
                ax.add_collection3d(coll, autolim=False)
                trails.append(coll)
        x0 = pos[idx[0]]
        halo = ax.scatter(x0[:, 0], x0[:, 1], x0[:, 2], s=sizes * (5.0 if not many else 3.0),
                          c=rgba, alpha=0.16, depthshade=False, linewidths=0, zorder=4, clip_on=False)
        core = ax.scatter(x0[:, 0], x0[:, 1], x0[:, 2], s=sizes, c=rgba, depthshade=False,
                          edgecolors="white" if not many else "none", linewidths=0.4,
                          alpha=1.0 if not many else 0.9, zorder=5, clip_on=False)
        panels.append(_Panel(ax, pos, trails, halo, core, rgba, center - span / 2, center + span / 2))

    # ---- overlays ----------------------------------------------------------
    if title:
        fig.text(0.025, 0.955, title, color=TEXT, fontsize=17, fontweight="bold", va="top")
    if subtitle:
        fig.text(0.025, 0.905, subtitle, color=TEXT_2, fontsize=11.5, va="top")
    time_text = fig.text(0.025, 0.035, "", color=TEXT, fontsize=13, family="DejaVu Sans Mono")
    if legend and not many and n_bodies <= 10:
        handles = [Line2D([], [], marker="o", ls="-", lw=1.6, color=colors[b], markersize=7,
                          markeredgecolor="white", markeredgewidth=0.4, label=names[b])
                   for b in range(n_bodies)]
        leg = fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.97),
                         frameon=False, fontsize=10, labelcolor=TEXT, ncol=1 if n_bodies <= 5 else 2)
        leg.set_zorder(10)

    t_display = t / time_unit

    def time_fmt(value):
        return f"t = {value:,.{time_decimals}f} {time_label}".rstrip()

    def alpha_ramp(n_seg):
        return np.linspace(0.0, 0.95, n_seg) ** 1.6                  # fades towards the tail

    update = _make_frame_function(fig, panels, idx, t_display, n_trail, alpha_ramp, elev, azim,
                                  rotation, time_text, time_fmt)

    # ---- poster frame (PNG) -------------------------------------------------
    if poster:
        update(int(poster_time * (n_frames - 1)))
        fig.savefig(poster, dpi=dpi, facecolor=BACKGROUND)

    # ---- encode -------------------------------------------------------------
    writer = animation.FFMpegWriter(
        fps=fps, codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-crf", str(crf), "-preset", "slow",
                    "-movflags", "+faststart"])
    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    t0 = time.perf_counter()
    with warnings.catch_warnings():                 # harmless layout-engine warning from Matplotlib
        warnings.simplefilter("ignore", UserWarning)
        anim.save(path, writer=writer, dpi=dpi, savefig_kwargs={"facecolor": BACKGROUND})
    plt.close(fig)
    if verbose:
        print(f"  video -> {path}  ({n_frames} frames, {time.perf_counter() - t0:.0f} s)")
    return path
