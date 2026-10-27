"""Stitch the per-clip outputs of one manifest into a single video.

Clips are planned with a rolling first-frame anchor: every clip after the first
is generated with the previous clip's final frame as its opening anchor, so
consecutive clips share exactly one frame. That duplicate is dropped here, so a
401-frame plan stitches back to exactly 401 frames. Frames are streamed one at
a time, so stitching costs no real memory.

    python -m fluidgen.tools.stitch MANIFEST OUT.mp4 [FPS]
"""
from __future__ import annotations

import json
import sys

import cv2


def stitch(manifest_path: str, out_path: str, fps: float = 16.0) -> int:
    m = json.load(open(manifest_path))
    # The planner writes `vace_clip_jobs` or `tora_clip_jobs` depending on the
    # backend; the clip geometry is the same either way.
    jobs = m.get("vace_clip_jobs") or m.get("tora_clip_jobs") or []
    if not jobs:
        raise ValueError(f"no clip jobs in {manifest_path}")
    jobs = sorted(jobs, key=lambda j: j["clip_id"])
    writer, written, per_clip = None, 0, []
    for i, j in enumerate(jobs):
        path = j.get("final_save_file") or j.get("save_file")
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(path)
        # Tora pads every clip to its model's frame_num (49); the real length is
        # `actual_frame_num`. Reading to the end of the file would append the padding.
        keep = int(j.get("actual_frame_num") or j.get("frame_num") or 0) or None
        n = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            n += 1
            if keep and n > keep:
                break
            if i > 0 and n == 1:
                continue                      # drop the shared anchor frame
            if writer is None:
                h, w = fr.shape[:2]
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(fr)
            written += 1
        per_clip.append((j["clip_id"], n, j.get("actual_frame_num"), j.get("frame_num")))
        cap.release()
    if writer is not None:
        writer.release()
    print("PERCLIP", per_clip)
    print("STITCHED", written, "frames ->", out_path)
    return written


if __name__ == "__main__":
    stitch(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 16.0)
