"""Helpers shared by the examples: output folders and figure saving."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
VIDEOS = ROOT / "videos"
FIGURES.mkdir(exist_ok=True)
VIDEOS.mkdir(exist_ok=True)

# Lets the examples run without installing the package (python examples/xx.py)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Set MAKE_VIDEOS = False (or run with --no-video) to skip the slower video rendering.
MAKE_VIDEOS = "--no-video" not in sys.argv


def save(fig, name: str) -> Path:
    """Save a figure as figures/<name> and return its path."""
    path = FIGURES / name
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.12)   # nothing is ever cut
    print(f"  figure -> {path.relative_to(ROOT)}")
    return path


def video_path(name: str) -> tuple[str, str]:
    """Paths of an MP4 video and its PNG poster frame."""
    return str(VIDEOS / f"{name}.mp4"), str(VIDEOS / f"{name}.png")
