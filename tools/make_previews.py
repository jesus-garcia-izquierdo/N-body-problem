"""
Build the animated previews shown in the README from the MP4 videos.

    python tools/make_previews.py

Why previews: GitHub does not play MP4 files stored in a repository inside a
README (it only plays videos uploaded through its web editor). Animated GIFs,
however, autoplay everywhere. Each preview is the same 3D render as its MP4,
sped up and downscaled, with a palette optimized for that clip.

Requires ffmpeg. Output: previews/<name>.gif (480 px wide) and
previews/figure_eight_hero.gif (720 px wide, used at the top of the README).
"""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIDEOS, PREVIEWS = ROOT / "videos", ROOT / "previews"


def make_gif(src: Path, dst: Path, width: int, fps: int, speed: float, colors: int) -> None:
    """Two-pass GIF encoding: an optimized palette (palettegen) and ordered dithering (paletteuse)."""
    vf = (f"setpts={1 / speed}*PTS,fps={fps},scale={width}:-1:flags=lanczos,split[a][b];"
          f"[a]palettegen=max_colors={colors}:stats_mode=full[p];"
          f"[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", vf, str(dst)], check=True)
    print(f"  {dst.relative_to(ROOT)}  ({dst.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    PREVIEWS.mkdir(exist_ok=True)
    for mp4 in sorted(VIDEOS.glob("*.mp4")):
        if mp4.stem == "figure_eight":
            make_gif(mp4, PREVIEWS / "figure_eight_hero.gif", width=720, fps=15, speed=1 / 0.6, colors=112)
        else:
            make_gif(mp4, PREVIEWS / f"{mp4.stem}.gif", width=480, fps=12, speed=2.0, colors=160)


if __name__ == "__main__":
    main()
