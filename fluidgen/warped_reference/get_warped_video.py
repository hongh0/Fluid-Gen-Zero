#!/usr/bin/env python3
"""
get_warped_video.py

Generate a warped video by rigidly translating an object in the first frame
according to centroid-based displacement derived from a per-frame mask video
(or from optical flow sampled at the object centroid).

Pipeline
--------
1. Load first frame + resize to --size (default 720×1280).
2. Load mask video (per-frame binary masks); resize each frame to --size.
3. Compute centroid and bbox of mask_video[0]  →  use as SAM2 prompts on first frame.
4. Run SAM2 image predictor  →  first-frame object mask (precise boundary).
5. Inpaint background using the SAM2 mask.
6. For each frame t: displacement = centroid(mask_video[t]) − centroid(mask_video[0]).
   Translate the object BGRA by that displacement (rigid affine shift).
7. Alpha-composite translated object over the static inpainted background.

Outputs (saved next to --output):
  <stem>.mp4          warped composite video
  <stem>_frame0.png   resized first frame (debug)
  <stem>_mask0.png    SAM2-generated first-frame object mask (debug)
  <stem>_disp_vis.mp4 per-frame displacement colour-wheel visualisation

Usage
-----
python get_warped_video.py \\
    --image       frame0.png \\
    --mask-video  masks.mp4 \\
    --sam2-checkpoint  /path/to/sam2.1_hiera_large.pt \\
    --sam2-config      /path/to/sam2/configs/sam2.1/sam2.1_hiera_l.yaml \\
    --size   720x1280 \\
    --output warped.mp4
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
import cv2
import numpy as np
import imageio
from typing import List, Optional, Tuple
from flux_controlnet_inpainting.inpainter import FluxInpainter


def segment_with_sam2(
    image_bgr: np.ndarray,
    point_xy: Tuple[float, float],
    # box_xyxy: Optional[Tuple[float, float, float, float]],
    checkpoint: str,
    config: str,
    device: str = "cuda",
    box_scale: float = 1.5,
) -> np.ndarray:
    """
    Run SAM2 image predictor with foreground point and optional loose bbox prompts.

    Parameters
    ----------
    image_bgr  : (H, W, 3) uint8 BGR image — already resized to target resolution.
    point_xy   : (x, y) pixel coordinate of the foreground point prompt.
    box_xyxy   : optional (x0, y0, x1, y1) box prompt in pixel coordinates.
    checkpoint : path to SAM2 .pt checkpoint file.
    config     : Hydra config name (e.g. 'configs/sam2.1/sam2.1_hiera_l.yaml')
                 OR absolute/relative path to the .yaml file.
    device     : 'cuda' or 'cpu'.
    box_scale  : scale factor for loosening the bbox around the foreground point.

    Returns
    -------
    (H, W) bool mask — union of SAM2 candidate masks.
    """
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    # build_sam2 uses Hydra compose(config_name=...).
    # If an absolute/relative path is given, extract the relative portion starting
    # from 'configs/' so Hydra can find it; copy the file locally if needed.
    cfg_file = config
    if os.path.exists(config):
        norm = config.replace("\\", "/")
        try:
            ci = next(i for i, p in enumerate(norm.split("/")) if p == "configs")
            cfg_file = "/".join(norm.split("/")[ci:])
        except StopIteration:
            parts = norm.split("/")
            cfg_file = "/".join(parts[-3:]) if len(parts) >= 3 else norm
        if not os.path.exists(cfg_file):
            os.makedirs(os.path.dirname(cfg_file), exist_ok=True)
            shutil.copy(config, cfg_file)

    dev = device if (device == "cpu" or not __import__("torch").cuda.is_available()) \
          else device
    print(f"  Building SAM2 model  config={cfg_file}  device={dev}")
    sam2_model = build_sam2(cfg_file, checkpoint, device=dev)
    predictor  = SAM2ImagePredictor(sam2_model)

    image_rgb    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    point_coords = np.array([[point_xy[0], point_xy[1]]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)   # 1 = foreground
    box = None
    with torch.inference_mode():
        predictor.set_image(image_rgb)
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=True,     # get candidates, then union them
        )
    # return masks[0]
    union_mask = np.any(masks.astype(bool), axis=0)
    score_str = ", ".join(f"{float(s):.3f}" for s in scores)
    print(f"  SAM2 done — union of {len(masks)} masks, scores=[{score_str}]")
    return union_mask


# ---------------------------------------------------------------------------
# Helpers: image resize
# ---------------------------------------------------------------------------

def _load_image_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _resize_to(img: np.ndarray, hw: Tuple[int, int],
               interp=cv2.INTER_AREA) -> np.ndarray:
    H, W = hw
    if img.shape[:2] == (H, W):
        return img
    return cv2.resize(img, (W, H), interpolation=interp)


# ---------------------------------------------------------------------------
# Mask video loading
# ---------------------------------------------------------------------------

def load_mask_video(path: str, resize_hw: Optional[Tuple[int, int]] = None) -> List[np.ndarray]:
    """
    Load a binary mask video (or image sequence).
    Returns a list of (H, W) bool arrays, one per frame.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open mask video: {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if resize_hw is not None:
            gray = _resize_to(gray, resize_hw, interp=cv2.INTER_NEAREST)
        frames.append(gray > 127)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from mask video: {path}")
    return frames


# ---------------------------------------------------------------------------
# Centroid
# ---------------------------------------------------------------------------

def mask_centroid(mask_bool: np.ndarray) -> Tuple[float, float]:
    """
    Return (cx, cy) centroid of the True region in a bool mask.
    Falls back to image centre if the mask is empty.
    """
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        H, W = mask_bool.shape
        return W / 2.0, H / 2.0
    return float(xs.mean()), float(ys.mean())


def mask_bbox_xyxy(mask_bool: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    """
    Return (x0, y0, x1, y1) bbox of the True region in a bool mask.
    Returns None if the mask is empty.
    """
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())



# ---------------------------------------------------------------------------
# Flow visualisation  (Baker et al. ICCV 2007 color wheel)
# ---------------------------------------------------------------------------

def _make_colorwheel() -> np.ndarray:
    """
    Generates the standard optical-flow color wheel (Baker et al., ICCV 2007).
    Returns ndarray of shape (55, 3) uint8.
    """
    RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
    ncols = RY + YG + GC + CB + BM + MR
    cw = np.zeros((ncols, 3))
    col = 0
    cw[col:col+RY, 0] = 255
    cw[col:col+RY, 1] = np.floor(255 * np.arange(RY) / RY);          col += RY
    cw[col:col+YG, 0] = 255 - np.floor(255 * np.arange(YG) / YG)
    cw[col:col+YG, 1] = 255;                                           col += YG
    cw[col:col+GC, 1] = 255
    cw[col:col+GC, 2] = np.floor(255 * np.arange(GC) / GC);          col += GC
    cw[col:col+CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    cw[col:col+CB, 2] = 255;                                           col += CB
    cw[col:col+BM, 2] = 255
    cw[col:col+BM, 0] = np.floor(255 * np.arange(BM) / BM);          col += BM
    cw[col:col+MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    cw[col:col+MR, 0] = 255
    return cw


def _flow_uv_to_colors(u: np.ndarray, v: np.ndarray,
                        convert_to_bgr: bool = False) -> np.ndarray:
    """
    Apply the color wheel to normalised flow components u, v ∈ [−1, 1].
    Returns (H, W, 3) uint8 RGB (or BGR if convert_to_bgr=True).
    """
    colorwheel = _make_colorwheel()
    ncols = colorwheel.shape[0]
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.uint8)

    rad = np.sqrt(np.square(u) + np.square(v))
    a   = np.arctan2(-v, -u) / np.pi
    fk  = (a + 1) / 2 * (ncols - 1)
    k0  = np.floor(fk).astype(np.int32)
    k1  = k0 + 1
    k1[k1 == ncols] = 0
    f   = fk - k0

    for i in range(3):
        tmp  = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col  = (1 - f) * col0 + f * col1
        idx  = rad <= 1
        col[idx]  = 1 - rad[idx] * (1 - col[idx])   # increase saturation near centre
        col[~idx] = col[~idx] * 0.75                 # out-of-range: slightly darker
        ch = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch] = np.floor(255 * col)

    return flow_image


def flow_to_color_bgr(flow_xy: np.ndarray,
                      max_mag: Optional[float] = None) -> np.ndarray:
    """
    Visualise a (H, W, 2) float32 flow field using the Baker et al. color wheel.
    Magnitude is normalised globally by max_mag (or the per-frame max if None).
    Returns (H, W, 3) uint8 BGR.
    """
    u = flow_xy[:, :, 0]
    v = flow_xy[:, :, 1]
    rad = np.sqrt(np.square(u) + np.square(v))
    if max_mag is None or max_mag < 1e-6:
        max_mag = float(rad.max()) if rad.max() > 1e-6 else 1.0
    eps = 1e-5
    u_n = u / (max_mag + eps)
    v_n = v / (max_mag + eps)
    return _flow_uv_to_colors(u_n, v_n, convert_to_bgr=True)


def displacements_to_flow_vis(displacements: np.ndarray, H: int, W: int,
                               fps: float, path: str):
    """
    Convert (T, 2) rigid displacement vectors to a color-wheel video.
    Each frame is a uniform color field whose hue/saturation encodes the
    direction and relative magnitude of that frame's translation.
    An arrow is drawn at the canvas centre for clarity.
    """
    mags = np.sqrt((displacements ** 2).sum(axis=1))
    max_mag = float(mags.max()) if mags.max() > 1e-6 else 1.0

    vis_frames = []
    for dx, dy in displacements:
        # Uniform flow field = same vector everywhere → solid color wheel cell
        flow = np.zeros((H, W, 2), dtype=np.float32)
        flow[:, :, 0] = dx
        flow[:, :, 1] = dy
        frame = flow_to_color_bgr(flow, max_mag=max_mag)
        # Arrow overlay
        cx, cy = W // 2, H // 2
        scale  = min(H, W) * 0.35 / (max_mag + 1e-6)
        ex = int(cx + dx * scale)
        ey = int(cy + dy * scale)
        cv2.arrowedLine(frame, (cx, cy), (ex, ey),
                        (255, 255, 255), 2, tipLength=0.25)
        vis_frames.append(frame)

    print(f"  Saving displacement vis → {path}  (max={max_mag:.1f} px)")
    save_video_mp4(vis_frames, path, fps=fps)


# ---------------------------------------------------------------------------
# Background inpainting
# ---------------------------------------------------------------------------

def inpaint_background_flux(
    image_bgr: np.ndarray,
    mask_bool: np.ndarray,
    inpainter: "FluxInpainter",
    prompt: str = "",
    num_inference_steps: int = 24,
    seed: int = 42,
) -> np.ndarray:
    """
    Inpaint the object region using FluxInpainter.

    Parameters
    ----------
    image_bgr  : (H, W, 3) uint8 BGR — original first frame.
    mask_bool  : (H, W) bool — True = object region to erase.
    inpainter  : loaded FluxInpainter instance.
    prompt     : text prompt describing the background to synthesise.
    num_inference_steps : diffusion steps (more = higher quality, slower).
    seed       : RNG seed for reproducibility.

    Returns
    -------
    (H, W, 3) uint8 BGR — inpainted background.
    """
    import torch

    H, W = image_bgr.shape[:2]

    # BGR → RGB → float [0,1] → (C, H, W) torch tensor
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_t   = torch.from_numpy(image_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)

    # bool mask → float (H, W) torch tensor  (1 = inpaint region)
    mask_t = torch.from_numpy(mask_bool.astype(np.float32))

    # FluxInpainter expects size=(W, H)
    result_pil = inpainter(
        image_t, mask_t,
        prompt=prompt,
        size=(W, H),
        num_inference_steps=num_inference_steps,
        seed=seed,
    )

    # PIL RGB → numpy BGR, resize to original resolution if needed
    result_bgr = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
    if result_bgr.shape[:2] != (H, W):
        result_bgr = cv2.resize(result_bgr, (W, H), interpolation=cv2.INTER_CUBIC)

    return result_bgr


# ---------------------------------------------------------------------------
# Rigid translation of object BGRA
# ---------------------------------------------------------------------------

def translate_object(object_bgra: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Rigidly translate `object_bgra` by (dx, dy) pixels using an affine shift.
    Pixels that shift outside the canvas become transparent (alpha=0).
    """
    H, W = object_bgra.shape[:2]
    M = np.array([[1, 0, dx],
                  [0, 1, dy]], dtype=np.float32)
    return cv2.warpAffine(object_bgra, M, (W, H),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0, 0))


def alpha_composite(fg_bgra: np.ndarray, bg_bgr: np.ndarray) -> np.ndarray:
    """Alpha-composite BGRA `fg_bgra` over BGR `bg_bgr`. Returns BGR uint8."""
    alpha = fg_bgra[:, :, 3:4].astype(np.float32) / 255.0
    fg    = fg_bgra[:, :, :3].astype(np.float32)
    bg    = bg_bgr.astype(np.float32)
    out   = fg * alpha + bg * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Video writing
# ---------------------------------------------------------------------------

def save_video_mp4(frames_bgr: List[np.ndarray], path: str, fps: float = 24.0):
    """Write BGR/BGRA frames to an H.264 MP4 via imageio/FFmpeg."""
    if not frames_bgr:
        raise ValueError("No frames to write")
    h, w = frames_bgr[0].shape[:2]
    hh, ww = h - h % 2, w - w % 2

    rgb_frames = []
    for f in frames_bgr:
        if f.ndim == 2:
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        elif f.shape[2] == 4:
            f = cv2.cvtColor(f, cv2.COLOR_BGRA2BGR)
        if f.dtype != np.uint8:
            f = np.clip(f, 0, 255).astype(np.uint8)
        rgb_frames.append(cv2.cvtColor(f[:hh, :ww], cv2.COLOR_BGR2RGB))

    ffmpeg_params = ["-movflags", "+faststart",
                     "-colorspace", "bt709",
                     "-color_primaries", "bt709",
                     "-color_trc", "bt709"]
    try:
        writer = imageio.get_writer(path, format="ffmpeg", fps=float(fps),
                                    codec="libx264", pixelformat="yuv420p",
                                    ffmpeg_params=ffmpeg_params)
    except Exception:
        writer = imageio.get_writer(path, format="ffmpeg", fps=float(fps),
                                    codec="mpeg4", pixelformat="yuv420p",
                                    ffmpeg_params=["-movflags", "+faststart"])
    try:
        for frm in rgb_frames:
            writer.append_data(frm)
    finally:
        writer.close()

    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise RuntimeError("FFmpeg produced an empty file.")
    return path


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def generate_warped_video(
    image_path: str,
    output_dir: str,
    mask_video_path: str,
    sam2_checkpoint: str,
    sam2_config: str,
    fps: float = 24.0,
    size_hw: Optional[Tuple[int, int]] = None,
    sam2_device: str = "cuda",
    inpaint_prompt: str = "",
    inpaint_steps: int = 24,
    inpaint_seed: int = 42,
    inpaint_device: str = "cuda",
    object_mask_path: Optional[str] = None,
):
    """
    Generate a warped video via mask-centroid rigid translation.

    Steps
    -----
    1. Resize first frame and every mask-video frame to size_hw.
    2. Compute centroid and bbox of mask_video[0]  →  SAM2 prompts.
    3. Run SAM2 on the resized first frame  →  precise first-frame object mask.
    4. Inpaint the background using FluxInpainter.
    5. For each frame t: translate the object so its centroid coincides with
       centroid(mask_video[t]).  Displacement = centroid_t − centroid_0.
    6. Alpha-composite translated object over the static inpainted background.

    All outputs are written to output_dir:
      first_frame.png      resized first frame
      mask0.png            SAM2-generated object mask
      background.png       Flux-inpainted background
      motion_signal.mp4    warped composite video
      disp_vis.mp4         displacement colour-wheel visualisation

    Parameters
    ----------
    image_path      : first frame image.
    output_dir      : directory where all outputs are saved (created if needed).
    mask_video_path : per-frame binary mask video (white = object).
    sam2_checkpoint : path to SAM2 .pt checkpoint.
    sam2_config     : SAM2 Hydra config name or path to .yaml.
    fps             : output frame rate.
    size_hw         : (H, W) target resolution; everything is resized to this.
    sam2_device     : torch device for SAM2 ('cuda' or 'cpu').
    inpaint_prompt  : text prompt for FluxInpainter background synthesis.
    inpaint_steps   : diffusion inference steps for FluxInpainter.
    inpaint_seed    : RNG seed for FluxInpainter.
    inpaint_device  : torch device for FluxInpainter.
    """

    os.makedirs(output_dir, exist_ok=True)

    def out(filename: str) -> str:
        return os.path.join(output_dir, filename)

    # ---- 1. Load + resize first frame ----
    image = _load_image_bgr(image_path)
    if size_hw is not None:
        image = _resize_to(image, size_hw, interp=cv2.INTER_AREA)
    H, W = image.shape[:2]
    print(f"First frame: {image_path}  →  {H}×{W}")
    cv2.imwrite(out("first_frame.png"), image)

    # ---- 2. Load mask video + resize ----
    print(f"Loading mask video: {mask_video_path}")
    mask_frames = load_mask_video(mask_video_path, resize_hw=(H, W))
    T = len(mask_frames)
    print(f"  {T} mask frames at {H}×{W}")

    cx0, cy0 = mask_centroid(mask_frames[0])
    # ---- 3. First-frame object mask ----
    mask0 = None
    if object_mask_path and os.path.exists(object_mask_path):
        m = cv2.imread(object_mask_path, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            if m.shape[:2] != (H, W):
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            mask0 = m > 127
            print(f"  using the reconstruction's object mask ({mask0.mean() * 100:.1f}% "
                  f"of the frame) → {object_mask_path}")
        else:
            print(f"  could not read {object_mask_path}; falling back to SAM2")
    if mask0 is None or not mask0.any():
        print("Running SAM2 on first frame …")
        mask0 = segment_with_sam2(
            image_bgr=image,
            point_xy=(cx0, cy0),
            # box_xyxy=bbox0,
            checkpoint=sam2_checkpoint,
            config=sam2_config,
            device=sam2_device,
        )
    cv2.imwrite(out("mask0.png"), mask0.astype(np.uint8) * 255)
    print(f"  SAM2 mask saved → {out('mask0.png')}")
    # ---- 4. Inpaint background with FluxInpainter ----
    print("  Loading FluxInpainter …")
    inpainter = FluxInpainter(device=inpaint_device)
    print(f"  Inpainting background  prompt={inpaint_prompt!r}  steps={inpaint_steps} …")
    background = inpaint_background_flux(
        image, mask0, inpainter,
        prompt=inpaint_prompt,
        num_inference_steps=inpaint_steps,
        seed=inpaint_seed,
    )
    cv2.imwrite(out("background.png"), background)
    print(f"  Inpainted background saved → {out('background.png')}")

    # ---- 5. Per-frame displacements: centroid_t − centroid_0 ----
    print("  Computing per-frame centroid displacements …")
    displacements = np.array(
        [(mask_centroid(mf)[0] - cx0, mask_centroid(mf)[1] - cy0)
         for mf in mask_frames],
        dtype=np.float32,
    )   # (T, 2)

    displacements_to_flow_vis(displacements, H, W, fps, out("disp_vis.mp4"))

    # ---- 6. Pack object BGRA, translate, composite ----
    object_bgra = np.zeros((H, W, 4), dtype=np.uint8)
    object_bgra[:, :, :3] = image
    object_bgra[:, :, 3]  = mask0.astype(np.uint8) * 255

    # mask BGRA for translation (alpha channel carries the mask shape)
    mask_bgra = np.zeros((H, W, 4), dtype=np.uint8)
    mask_bgra[:, :, :3] = 255          # white object colour
    mask_bgra[:, :, 3]  = mask0.astype(np.uint8) * 255

    print(f"  Generating {T} frames …")
    frames_out = []
    mask_frames_out = []
    for i, (dx, dy) in enumerate(displacements):
        translated        = translate_object(object_bgra, dx, dy)
        composite         = alpha_composite(translated, background)
        frames_out.append(composite)

        # Translated binary mask: white where the object is, black elsewhere
        translated_mask   = translate_object(mask_bgra, dx, dy)
        mask_gray         = translated_mask[:, :, 3]   # alpha = mask shape
        mask_frames_out.append(mask_gray)

        if (i + 1) % 20 == 0 or (i + 1) == T:
            print(f"    {i + 1}/{T}  dx={dx:.1f} dy={dy:.1f}")

    # ---- Save ----
    video_path = out("motion_signal.mp4")
    print(f"Saving warped video → {video_path}")
    save_video_mp4(frames_out, video_path, fps=fps)

    mask_video_out_path = out("mask.mp4")
    print(f"Saving mask video   → {mask_video_out_path}")
    save_video_mp4(
        [cv2.cvtColor(m, cv2.COLOR_GRAY2BGR) for m in mask_frames_out],
        mask_video_out_path, fps=fps,
    )

    print(f"Done. All outputs in: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Warp a first frame by rigidly translating the object "
                    "according to centroid displacement derived from a mask video. "
                    "The first-frame object mask is generated by SAM2 using the "
                    "centroid and bbox of mask_video[0] as prompts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # --- required ---
    parser.add_argument("--image", required=True,
                        help="First frame image (PNG/JPG/…)")
    parser.add_argument("--mask-video", required=True,
                        help="Per-frame binary object mask video. "
                             "Frame 0 centroid + bbox → SAM2 prompts; "
                             "per-frame centroids → displacement trajectory.")
    parser.add_argument("--sam2-checkpoint",
                        help="Path to SAM2 .pt checkpoint file "
                             "(e.g. /path/to/sam2.1_hiera_large.pt)", default=None)
    parser.add_argument("--sam2-config",
                        help="SAM2 Hydra config name or path to .yaml "
                             "(e.g. /path/to/sam2/configs/sam2.1/sam2.1_hiera_l.yaml)",default="/configs/sam2.1/sam2.1_hiera_l.yaml")
    # --- optional ---
    parser.add_argument("--object-mask", default="",
                        help="First-frame object mask from reconstruction; used instead of "
                             "prompting SAM2 with a single point.")
    parser.add_argument("--output-dir", default="warped_output",
                        help="Directory where all outputs are saved (default: warped_output)")
    parser.add_argument("--fps", type=float, default=24.0,
                        help="Output FPS (default: 24)")
    parser.add_argument("--size", default="720x1280", metavar="HxW",
                        help="Resize image + masks to this resolution before all "
                             "processing (default: 720x1280)")
    parser.add_argument("--sam2-device", default="cuda",
                        help="Torch device for SAM2 (default: cuda)")
    parser.add_argument("--inpaint-prompt", default="background, nothing",
                        help="Text prompt for FluxInpainter background synthesis "
                             "(default: empty string)")
    parser.add_argument("--inpaint-steps", type=int, default=24,
                        help="Diffusion inference steps for FluxInpainter (default: 24)")
    parser.add_argument("--inpaint-seed", type=int, default=42,
                        help="RNG seed for FluxInpainter (default: 42)")
    parser.add_argument("--inpaint-device", default="cuda",
                        help="Torch device for FluxInpainter (default: cuda)")
    args = parser.parse_args()
    if not args.sam2_checkpoint:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from fluidgen import paths
        args.sam2_checkpoint = paths.SAM2_CKPT

    size_hw = None
    if args.size:
        try:
            h, w = map(int, args.size.split("x"))
            size_hw = (h, w)
        except ValueError:
            parser.error("--size must be HxW, e.g. 720x1280")

    generate_warped_video(
        image_path=args.image,
        output_dir=args.output_dir,
        mask_video_path=args.mask_video,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
        fps=args.fps,
        size_hw=size_hw,
        sam2_device=args.sam2_device,
        inpaint_prompt=args.inpaint_prompt,
        inpaint_steps=args.inpaint_steps,
        inpaint_seed=args.inpaint_seed,
        inpaint_device=args.inpaint_device,
        object_mask_path=args.object_mask,
    )


if __name__ == "__main__":
    main()
