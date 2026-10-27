"""Video/image sampling helpers for OpenAI VLM calls."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


@dataclass
class FrameSample:
    source_path: str
    source_label: str
    frame_index: int
    data_url: str


def _encode_jpeg_data_url(frame, quality: int = 82) -> str:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for video frame sampling.")
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Failed to JPEG-encode sampled frame.")
    encoded = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def image_to_data_url(path: str | Path, quality: int = 88) -> str:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for image encoding.")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return _encode_jpeg_data_url(image, quality=quality)


def sample_video_frames(
    video_path: str | Path | None,
    source_label: str,
    frame_interval: int = 5,
    max_frames: int = 0,
    quality: int = 82,
) -> list[FrameSample]:
    """Sample one frame every `frame_interval` frames.

    Args:
        max_frames: 0 means no cap. A positive value caps samples after stride sampling.
    """
    if video_path is None:
        return []
    path = Path(video_path)
    if not path.exists():
        return []
    if cv2 is None:
        raise RuntimeError("OpenCV is required for video frame sampling.")
    interval = max(1, int(frame_interval))
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    samples: list[FrameSample] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % interval == 0:
            samples.append(
                FrameSample(
                    source_path=str(path),
                    source_label=source_label,
                    frame_index=frame_idx,
                    data_url=_encode_jpeg_data_url(frame, quality=quality),
                )
            )
            if max_frames and len(samples) >= max_frames:
                break
        frame_idx += 1
    cap.release()
    return samples


def sample_bundle_videos(bundle, frame_interval: int = 5, max_frames_per_video: int = 0) -> list[FrameSample]:
    paths = [
        (getattr(bundle, "flow_video_path", None), "optical_flow_video"),
        (getattr(bundle, "sim_video_path", None), "simulation_preview_video"),
        (getattr(bundle, "segmap_video_path", None), "segmap_video"),
        (getattr(bundle, "fluid_mask_video_path", None), "fluid_mask_video"),
        (getattr(bundle, "fg_mask_video_path", None), "foreground_mask_video"),
    ]
    out: list[FrameSample] = []
    for path, label in paths:
        out.extend(sample_video_frames(path, label, frame_interval, max_frames=max_frames_per_video))
    return out
