"""Stage 1 -- decompose the image into object / fluid / background.

The liquid is segmented the same way the object is: one text prompt through
LangSAM (Grounding DINO grounds the phrase, SAM 2 cuts the mask), with the
liquid's material word (from the intent parser) instead of the object's noun.
The fluid prompts are unioned and the object is subtracted; what is left is an
explicit background class (walls, tabletops, backdrops).

When grounding returns no plausible object, a detector box (Grounding DINO
SwinT / OWLv2) prompts SAM 2, then the VLM, then SAM 2's automatic masks.

CLIPSeg is kept as the fallback for when grounding finds nothing. Its absolute
probabilities are not comparable across images ("purple milk" scores 0.05, an
ordinary pond 0.95), so it is only ever used *contrastively*: a region is fluid
when it looks more like the liquid than like the background vocabulary.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from contracts import SegmentationArtifact

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fluidgen import paths  # noqa: E402

SAM2_CKPT = paths.SAM2_CKPT
SAM2_CFG = paths.SAM2_CFG
DINO_CFG = paths.GDINO_SWINT_CFG
DINO_CKPT = paths.GDINO_SWINT_CKPT
CLIPSEG_CKPT = paths.CLIPSEG_CKPT
OWLV2_CKPT = paths.OWLV2_CKPT
GDINO_CKPT = paths.GDINO_CKPT
GENERIC_FLUID_WORDS = ["fluid", "water", "liquid"]
BACKGROUND_WORDS = ["wooden wall", "table surface", "floor", "wall",
                    "container rim", "background", "sky"]


# --------------------------------------------------------------------------
MAX_OBJECT_FRACTION = 0.5
MIN_OBJECT_FRACTION = 0.005


def material_word(scene_prompt: str, default: str = "water") -> str:
    """Liquid material name from the intent parser's table, or `default`.

    The parser lives in `fluidgen.agent` and uses relative imports, so it has
    to be reached through the package root rather than by file path.
    """
    if not scene_prompt:
        return default
    try:
        from fluidgen.agent.intent_parser import _detect_material
        return _detect_material(scene_prompt, default=default)
    except Exception:
        return default


_LANGSAM = {}


def _langsam(device: str):
    """LangSAM = GroundingDINO (transformers build) + SAM2, loaded once."""
    if device not in _LANGSAM:
        from lang_sam import LangSAM
        _LANGSAM[device] = LangSAM(gdino_model_ckpt_path=GDINO_CKPT,
                                   gdino_processor_ckpt_path=GDINO_CKPT,
                                   device=device)
    return _LANGSAM[device]


def _grounded_mask(image_rgb: np.ndarray, prompt: str, device: str,
                   box_threshold: float = 0.30) -> Optional[np.ndarray]:
    """Union of LangSAM masks for one text prompt; None if it could not run.
    """
    from PIL import Image

    h, w = image_rgb.shape[:2]
    try:
        model = _langsam(device)
        res = model.predict([Image.fromarray(image_rgb)], [prompt],
                            box_threshold=box_threshold)[0]
        out = np.zeros((h, w), dtype=bool)
        for m in res.get("masks", []):
            out |= np.asarray(m).astype(bool)
        return out
    except Exception:
        return None
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _load_sam2(device: str):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(SAM2_CFG, SAM2_CKPT, device=device)
    return SAM2ImagePredictor(model)


def _clipseg_probs(image_rgb: np.ndarray, prompts: List[str], device: str) -> np.ndarray:
    """Per-prompt probability maps at full image resolution, (P, H, W)."""
    from PIL import Image
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

    proc = CLIPSegProcessor.from_pretrained(CLIPSEG_CKPT)
    model = CLIPSegForImageSegmentation.from_pretrained(CLIPSEG_CKPT).to(device).eval()
    pil = Image.fromarray(image_rgb)
    inputs = proc(text=prompts, images=[pil] * len(prompts),
                  padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    logits = logits.reshape(len(prompts), *logits.shape[-2:])
    probs = torch.sigmoid(logits).float().cpu().numpy()
    h, w = image_rgb.shape[:2]
    return np.stack([cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR) for p in probs])


def _fluid_vs_background(image_rgb: np.ndarray, fluid_word: str,
                         device: str) -> Tuple[np.ndarray, np.ndarray]:
    """Dense (fluid_score, background_score); compare them, never threshold one."""
    fluid_words = [f"a pool of {fluid_word}", fluid_word, *GENERIC_FLUID_WORDS]
    probs = _clipseg_probs(image_rgb, fluid_words + BACKGROUND_WORDS, device)
    return probs[:len(fluid_words)].max(0), probs[len(fluid_words):].max(0)


def _owlv2_box(image_rgb: np.ndarray, prompt: str, device: str) -> Optional[np.ndarray]:
    """Text-conditioned box via OWLv2; the stand-in for GroundingDINO here."""
    try:
        from PIL import Image
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        proc = Owlv2Processor.from_pretrained(OWLV2_CKPT)
        model = Owlv2ForObjectDetection.from_pretrained(OWLV2_CKPT).to(device).eval()
        pil = Image.fromarray(image_rgb)
        inputs = proc(text=[[f"a photo of a {prompt}", prompt]], images=pil,
                      return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        h, w = image_rgb.shape[:2]
        res = proc.post_process_grounded_object_detection(
            outputs, target_sizes=torch.tensor([[h, w]]), threshold=0.15)[0]
        if len(res["boxes"]) == 0:
            return None
        best = int(torch.argmax(res["scores"]))
        return res["boxes"][best].cpu().numpy()
    except Exception:
        return None


def _dino_box(image_rgb: np.ndarray, prompt: str, device: str) -> Optional[np.ndarray]:
    """Text-conditioned box via GroundingDINO; None if unavailable/not found."""
    try:
        from groundingdino.util.inference import Model as DinoModel
    except Exception:
        return None
    try:
        dino = DinoModel(model_config_path=DINO_CFG, model_checkpoint_path=DINO_CKPT, device=device)
        det = dino.predict_with_classes(
            image=cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
            classes=[prompt], box_threshold=0.3, text_threshold=0.25)
        if det.xyxy is None or len(det.xyxy) == 0:
            return None
        # Prefer the highest-scoring box of plausible size rather than the
        # largest: detectors often also return a whole-frame box.
        h_img, w_img = image_rgb.shape[:2]
        areas = (det.xyxy[:, 2] - det.xyxy[:, 0]) * (det.xyxy[:, 3] - det.xyxy[:, 1])
        frac = areas / float(h_img * w_img)
        ok = np.nonzero((frac > 1e-3) & (frac < MAX_OBJECT_FRACTION))[0]
        if len(ok) == 0:
            return None
        conf = getattr(det, "confidence", None)
        pick = ok[int(np.argmax(conf[ok]))] if conf is not None else ok[int(np.argmax(areas[ok]))]
        return det.xyxy[int(pick)]
    except Exception:
        return None


def _ask_vlm_json(text: str, image_bgr: np.ndarray) -> dict:
    """One image + one instruction -> a JSON object, from the pipeline's VLM.

    Uses the same model as intent parsing, risk annotation and clip prompts
    (FGZ_VLM_MODEL, see fluidgen/agent/openai_adapter.py) through the same
    Responses API, so a run talks to exactly one VLM.
    """
    import base64
    import json as _json

    from openai import OpenAI
    from fluidgen.agent.openai_adapter import DEFAULT_OPENAI_MODEL

    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise ValueError("could not encode image")
    b64 = base64.b64encode(buf.tobytes()).decode()
    rsp = OpenAI(api_key=os.environ["OPENAI_API_KEY"]).responses.create(
        model=DEFAULT_OPENAI_MODEL,
        input=[{"role": "user", "content": [
            {"type": "input_text", "text": text},
            {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"}]}],
        text={"format": {"type": "json_object"}})
    return _json.loads(rsp.output_text)


def _vlm_locate_once(image_rgb: np.ndarray, prompt: str,
                     notes: List[str]) -> Optional[np.ndarray]:
    """A coarse box for the object from the VLM, in pixel xyxy, or None.

    Returns None whenever the VLM is unavailable; this is a fallback.
    """
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key or not prompt:
        return None
    try:
        d = _ask_vlm_json(
            f"Locate the {prompt} in this image. Reply JSON only: "
            '{"x0":0.0,"y0":0.0,"x1":0.0,"y1":0.0} -- the bounding box of '
            "the object, normalised to [0,1], origin top-left. If the "
            "object is cut off by the frame edge, box the visible part.",
            cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        h, w = image_rgb.shape[:2]
        box = np.array([float(d["x0"]) * w, float(d["y0"]) * h,
                        float(d["x1"]) * w, float(d["y1"]) * h], np.float32)
        box[0::2] = np.clip(box[0::2], 0, w - 1)
        box[1::2] = np.clip(box[1::2], 0, h - 1)
        if box[2] - box[0] < 4 or box[3] - box[1] < 4:
            return None
        notes.append(f"VLM box {np.round(box).astype(int).tolist()}")
        return box
    except Exception as exc:
        notes.append(f"VLM localisation unavailable: {type(exc).__name__}")
        return None


def _vlm_locate(image_rgb: np.ndarray, prompt: str, notes: List[str],
                attempts: int = 3) -> Optional[np.ndarray]:
    """`_vlm_locate_once`, retried, because a single call is not reliable."""
    for k in range(attempts):
        box = _vlm_locate_once(image_rgb, prompt, notes)
        if box is not None:
            return box
    notes.append(f"VLM gave no usable box in {attempts} attempts")
    return None


def _box_agreement(mask: np.ndarray, box: np.ndarray) -> float:
    """IoU of a candidate mask with the VLM's coarse box.

    IoU rather than "does the mask cover the box": a mask spanning the whole
    frame covers any box completely, and the union term is what rules it out.
    """
    x0, y0, x1, y1 = np.round(box).astype(int)
    b = np.zeros(mask.shape, bool)
    b[y0:y1 + 1, x0:x1 + 1] = True
    return float((mask & b).sum()) / float(max((mask | b).sum(), 1))


def _vlm_guided_mask(image_rgb: np.ndarray, prompt: str, device: str,
                     notes: List[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """SAM 2 proposals around the VLM's box, picked by agreement with it."""
    box = _vlm_locate(image_rgb, prompt, notes)
    if box is None:
        return None, None
    predictor = _load_sam2(device)
    predictor.set_image(image_rgb)
    cx, cy = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
    cands = []
    for kwargs in ({"box": box[None, :]},
                   {"point_coords": np.array([[cx, cy]], np.float32),
                    "point_labels": np.array([1])}):
        try:
            masks, _, _ = predictor.predict(multimask_output=True, **kwargs)
        except Exception:
            continue
        cands += [np.asarray(m).astype(bool) for m in masks]
    cands = [m for m in cands if _plausible_object(m)]
    if not cands:
        notes.append("no SAM 2 proposal at the VLM box was a plausible object")
        return None, box
    best = max(cands, key=lambda m: _box_agreement(m, box))
    notes.append(f"object from VLM box + SAM 2, {100.0 * best.mean():.1f}% of the image "
                 f"(box agreement {_box_agreement(best, box):.2f}, "
                 f"{len(cands)} proposals)")
    return best, box


def _vlm_choose(image_rgb: np.ndarray, prompt: str, masks: List[np.ndarray],
                notes: List[str], max_shown: int = 12) -> Optional[int]:
    """Let the VLM pick the object out of a numbered overlay of proposals.
    """
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key or not prompt or not masks:
        return None
    order = sorted(range(len(masks)), key=lambda i: -masks[i].sum())[:max_shown]
    vis = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()
    rng = np.random.default_rng(0)
    for n, i in enumerate(order):
        col = tuple(int(c) for c in rng.integers(60, 255, 3))
        cnts, _ = cv2.findContours(masks[i].astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, col, 3)
        ys, xs = np.nonzero(masks[i])
        org = (int(xs.mean()) - 10, int(ys.mean()) + 10)
        cv2.putText(vis, str(n), org, cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 7)
        cv2.putText(vis, str(n), org, cv2.FONT_HERSHEY_SIMPLEX, 1.3, col, 3)
    try:
        k = int(_ask_vlm_json(
            f"Each numbered outline is a candidate region. Which single "
            f"number outlines the {prompt} in its entirety? Prefer the "
            f"outline covering the whole object over one covering part of "
            f'it. Reply JSON only: {{"choice": <number>}}, or '
            '{"choice": -1} if none of them is that object.', vis)["choice"])
    except Exception as exc:
        notes.append(f"VLM choice unavailable: {type(exc).__name__}")
        return None
    if not (0 <= k < len(order)):
        notes.append(f"VLM chose {k}, which is not one of the {len(order)} regions")
        return None
    notes.append(f"VLM chose region {k} of {len(order)}")
    return order[k]


def _promote_to_whole(i: int, pool: List[int], masks: List[np.ndarray],
                      notes: List[str], contain: float = 0.9) -> int:
    """Trade a chosen part for the whole it sits in, if one was proposed.

    SAM 2 proposes an object together with its parts; take the largest proposal
    that contains almost all of whatever was chosen.
    """
    best, area = i, int(masks[i].sum())
    for j in pool:
        if j == i:
            continue
        mj = masks[j]
        if int(mj.sum()) > int(masks[best].sum()) and \
                (masks[i] & mj).sum() >= contain * area:
            best = j
    if best != i:
        notes.append(f"promoted the choice from {area} px to {int(masks[best].sum())} px: "
                     "a larger proposal contains it")
    return best


def _auto_masks(image_rgb: np.ndarray, device: str):
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    model = build_sam2(SAM2_CFG, SAM2_CKPT, device=device, apply_postprocessing=False)
    gen = SAM2AutomaticMaskGenerator(model, points_per_side=24,
                                     pred_iou_thresh=0.7, stability_score_thresh=0.9)
    return gen.generate(image_rgb)


def _plausible_object(mask: np.ndarray) -> bool:
    """Is this mask usable as the object at all?

    Weaker than `_score_object`, which ranks unlabelled proposals by centrality:
    a text-grounded mask only has to be neither a speck nor the whole frame.
    """
    frac = float(mask.sum()) / float(mask.size)
    return MIN_OBJECT_FRACTION <= frac <= MAX_OBJECT_FRACTION


def _score_object(mask: np.ndarray) -> float:
    """Prefer compact, reasonably sized regions near the image centre.

    Only reached when both detectors fail. Overhead fluid shots put the object
    in the middle of a large uniform liquid field, so centrality plus
    compactness separates the two.
    """
    h, w = mask.shape
    area = float(mask.sum())
    if area < MIN_OBJECT_FRACTION * h * w or area > MAX_OBJECT_FRACTION * h * w:
        return -1.0
    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean() / h, xs.mean() / w
    centrality = 1.0 - 2.0 * max(abs(cy - 0.5), abs(cx - 0.5))
    extent = area / max(1.0, (ys.ptp() + 1) * (xs.ptp() + 1))
    touches_border = (ys.min() == 0 or xs.min() == 0 or ys.max() == h - 1 or xs.max() == w - 1)
    return centrality + 0.5 * extent - (0.6 if touches_border else 0.0)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if n <= 2:
        return mask
    sizes = [(labels == i).sum() for i in range(1, n)]
    return labels == (int(np.argmax(sizes)) + 1)


def _close(mask: np.ndarray, k: int = 5) -> np.ndarray:
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                            np.ones((k, k), np.uint8)).astype(bool)


# --------------------------------------------------------------------------
def segment(image_path: str, out_dir: str, object_prompt: str = "",
            fluid_prompt: str = "", scene_prompt: str = "",
            device: str = "cuda") -> SegmentationArtifact:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(image_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    notes: List[str] = []

    # ---- object: one text prompt through LangSAM ------------------------
    obj = None
    cands = None
    vlm_box = None
    if object_prompt:
        obj = _grounded_mask(rgb, object_prompt, device, box_threshold=0.20)
        if obj is not None and _plausible_object(obj):
            notes.append(f"object from LangSAM for {object_prompt!r}")
        else:
            if obj is not None and obj.sum() > 0:
                notes.append(f"rejected LangSAM object mask for {object_prompt!r}: "
                             f"{100.0 * obj.sum() / obj.size:.1f}% of the image, not an object")
            obj = None
        if obj is None:
            box = _dino_box(rgb, object_prompt, device) or _owlv2_box(rgb, object_prompt, device)
            if box is not None:
                predictor = _load_sam2(device)
                predictor.set_image(rgb)
                masks, _, _ = predictor.predict(box=box[None, :], multimask_output=False)
                cand = np.asarray(masks[0]).astype(bool)
                if _plausible_object(cand):
                    obj = cand
                    notes.append(f"object from detector box for {object_prompt!r}")
                else:
                    notes.append(f"rejected detector-box object mask for {object_prompt!r}: "
                                 f"{100.0 * cand.sum() / cand.size:.1f}% of the image")
        if obj is None:
            obj, vlm_box = _vlm_guided_mask(rgb, object_prompt, device, notes)
    if obj is None or obj.sum() == 0:
        cands = _auto_masks(rgb, device)
        segs = [np.asarray(c["segmentation"]).astype(bool) for c in cands]
        scored = [(_score_object(m), i) for i, m in enumerate(segs)]
        scored = [t for t in scored if t[0] > 0]
        how = "SAM2 automatic masks (centred-compact heuristic)"
        if not scored:
            pool = [i for i, m in enumerate(segs) if _plausible_object(m)]
            pick = _vlm_choose(rgb, object_prompt, [segs[i] for i in pool], notes)
            if pick is not None:
                scored = [(1.0, _promote_to_whole(pool[pick], pool, segs, notes))]
                how = "SAM2 automatic masks chosen by the VLM"
            elif vlm_box is not None:
                scored = [(_box_agreement(segs[i], vlm_box), i) for i in pool]
                how = "SAM2 automatic masks ranked by VLM box agreement"
        if not scored:
            raise RuntimeError(
                "no plausible object mask found; notes=" + " | ".join(notes[-6:]))
        best_score, best_i = max(scored)
        obj = segs[best_i]
        notes.append(f"object from {how} (score {best_score:.3f})")

    obj = _close(_largest_component(obj))

    # ---- fluid: its own semantic prompt, exactly like the object --------
    fluid_word = fluid_prompt or material_word(scene_prompt)
    fluid_words = [fluid_word] + [g for g in GENERIC_FLUID_WORDS if g != fluid_word]

    fluid = np.zeros((h, w), dtype=bool)
    grounded: List[str] = []
    for word in fluid_words:
        got = _grounded_mask(rgb, word, device)
        if got is not None and got.any():
            fluid |= got
            grounded.append(f"{word}:{100.0 * got.mean():.0f}%")
    if grounded:
        notes.append("fluid from LangSAM, union of " + ", ".join(grounded))

    if not fluid.any():
        # Grounding found no liquid at all. CLIPSeg still has an opinion, but
        # only a relative one, so read it out over SAM2's regions: its own
        # 352px output is far too coarse to cut a liquid line against a wall.
        p_fluid, p_bg = _fluid_vs_background(rgb, fluid_word, device)
        if cands is None:
            cands = _auto_masks(rgb, device)
        covered = np.zeros((h, w), dtype=bool)
        for c in cands:
            seg = np.asarray(c["segmentation"]).astype(bool) & ~obj
            if seg.sum() < 0.0005 * h * w:
                continue
            covered |= seg
            if p_fluid[seg].mean() > p_bg[seg].mean():
                fluid |= seg
        fluid |= ~covered & ~obj & (p_fluid > p_bg)
        notes.append(f"grounding found no liquid; CLIPSeg contrast for {fluid_word!r}")

    fluid = _close(fluid & ~obj)
    fluid &= ~obj
    if fluid.sum() == 0:
        raise RuntimeError(f"no fluid found for {fluid_word!r}")
    else:
        keep = _largest_component(fluid)
        # A pool split by the object is still one pool; only drop specks.
        if keep.sum() >= 0.5 * fluid.sum():
            fluid = keep

    background = ~(obj | fluid)
    if background.sum() > 0:
        notes.append(f"background is {100.0 * background.mean():.1f}% of the image")

    for name, m in [("object_mask", obj), ("fluid_mask", fluid), ("background_mask", background)]:
        cv2.imwrite(str(out / f"{name}.png"), (m.astype(np.uint8) * 255))

    art = SegmentationArtifact(
        image_path=str(image_path),
        object_mask_path=str(out / "object_mask.png"),
        fluid_mask_path=str(out / "fluid_mask.png"),
        background_mask_path=str(out / "background_mask.png"),
        width=w, height=h,
        object_pixels=int(obj.sum()), fluid_pixels=int(fluid.sum()),
        background_pixels=int(background.sum()),
        overlap_pixels=int((obj & fluid).sum()),
        uncovered_pixels=int((~(obj | fluid | background)).sum()),
        object_prompt=object_prompt, fluid_prompt=fluid_word, notes=notes)
    art.save(out / "segmentation.json")
    return art


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--object_prompt", default="")
    p.add_argument("--fluid_prompt", default="",
                   help="liquid material word; derived from --scene_prompt when omitted")
    p.add_argument("--scene_prompt", default="",
                   help="full scene text, parsed for the liquid material")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    art = segment(a.image, a.out_dir, a.object_prompt, a.fluid_prompt,
                  a.scene_prompt, a.device)
    print(json.dumps({"object_px": art.object_pixels, "fluid_px": art.fluid_pixels,
                      "background_px": art.background_pixels,
                      "fluid_prompt": art.fluid_prompt,
                      "overlap": art.overlap_pixels, "uncovered": art.uncovered_pixels,
                      "notes": art.notes}, indent=2))


if __name__ == "__main__":
    main()
