#!/usr/bin/env python3
"""Write BGR frames with Wan/VACE cache_video.

This helper lets the dynamic clip executor restitch clips from a lightweight
Python environment while still saving the final video through the Wan encoder
used by VACE clip outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Save a .npz stack of BGR frames with wan.utils.cache_video.")
    parser.add_argument("--frames_npz", required=True, help=".npz containing frames as uint8 BGR array [T,H,W,3].")
    parser.add_argument("--output", required=True, help="Output mp4 path.")
    parser.add_argument("--fps", type=float, required=True, help="Video FPS.")
    args = parser.parse_args()

    data = np.load(args.frames_npz)
    frames = data["frames"].astype(np.uint8)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out), fps=args.fps, codec="libx264", quality=8)
    try:
        for frame in frames:
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        writer.close()


if __name__ == "__main__":
    main()
