"""Hybrid simulation complexity metrics for fluid-object VACE generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .intent_parser import DEFAULT_ALIGNMENT_SET
from .clip_planner import ClipPlannerSettings, ComplexityAwareClipPlanner
from .schema import (
    ComplexityGroupScore,
    ComplexityPlan,
    FrameComplexity,
    IntentParseResult,
    SemanticEvent,
    SimulationBundle,
    VLMComplexitySuggestion,
)

try:  # Keep the planner importable in lightweight environments.
    import cv2  # type: ignore
except Exception:  # pragma: no cover - only used when OpenCV is unavailable.
    cv2 = None


def _load_xyz(path: Optional[str]) -> Optional[np.ndarray]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    data = np.load(p)
    if "xyz" not in data:
        raise ValueError(f"Expected key 'xyz' in {p}")
    xyz = np.asarray(data["xyz"], dtype=np.float32)
    if xyz.ndim == 2 and xyz.shape[-1] == 3:
        xyz = xyz[None, ...]
    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"Expected xyz shape (T, N, 3), got {xyz.shape} from {p}")
    return xyz


def _robust_norm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.nanpercentile(values, 5))
    hi = float(np.nanpercentile(values, 95))
    if hi <= lo + 1e-8:
        return np.zeros_like(values)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def _pad_or_trim(values: np.ndarray, length: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == length:
        return values
    if values.size > length:
        return values[:length]
    if values.size == 0:
        return np.zeros(length, dtype=np.float32)
    pad = np.full(length - values.size, float(values[-1]), dtype=np.float32)
    return np.concatenate([values, pad], axis=0)


def _safe_mean_stack(values: Sequence[np.ndarray], length: int) -> np.ndarray:
    arrays = [_pad_or_trim(v, length) for v in values if v.size]
    if not arrays:
        return np.zeros(length, dtype=np.float32)
    return np.mean(np.stack(arrays, axis=0), axis=0).astype(np.float32)


def _sample_points(points: np.ndarray, max_points: int = 384) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    idx = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
    return points[idx]


def _nearest_distance_stats(a: np.ndarray, b: np.ndarray, threshold: float) -> tuple[float, float]:
    if a.size == 0 or b.size == 0:
        return 0.0, 0.0
    min_dist = np.full(a.shape[0], np.inf, dtype=np.float32)
    chunk = 128
    for i in range(0, a.shape[0], chunk):
        diff = a[i : i + chunk, None, :] - b[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        min_dist[i : i + chunk] = np.minimum(min_dist[i : i + chunk], dist.min(axis=1))
    contact_ratio = float(np.mean(min_dist <= threshold))
    closeness = float(np.mean(np.exp(-min_dist / max(threshold, 1e-6))))
    return contact_ratio, closeness


def _read_mask_video(path: Optional[str], max_frames: int) -> Optional[np.ndarray]:
    if not path or cv2 is None or not Path(path).exists():
        return None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    frames: list[np.ndarray] = []
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        frames.append((gray > 20).astype(np.uint8))
    cap.release()
    if not frames:
        return None
    return np.stack(frames, axis=0)


@dataclass
class ComplexityConfig:
    min_clip_frames: int = 13
    default_clip_frames: int = 41
    max_clip_frames: int = 81
    high_threshold: float = 0.72
    medium_threshold: float = 0.45
    alpha_physical: float = 0.40
    beta_visual: float = 0.40
    gamma_semantic: float = 0.20
    contact_sample_points: int = 384
    event_window: int = 6

    def __post_init__(self) -> None:
        for field_name in ["min_clip_frames", "default_clip_frames", "max_clip_frames"]:
            value = getattr(self, field_name)
            if value < 5 or (value - 1) % 4 != 0:
                raise ValueError(f"{field_name} must be 4n+1 and >=5, got {value}")
        if self.min_clip_frames > self.default_clip_frames or self.default_clip_frames > self.max_clip_frames:
            raise ValueError("Expected min_clip_frames <= default_clip_frames <= max_clip_frames")


class SimulationComplexityAnalyzer:
    """Computes C(t)=alpha*Cphys+beta*Cvis+gamma*Csemantic."""

    def __init__(self, config: Optional[ComplexityConfig] = None):
        self.config = config or ComplexityConfig()

    def analyze(
        self,
        bundle: SimulationBundle,
        intent: IntentParseResult,
        base_prompt: str,
        vlm_suggestions: Optional[Sequence[VLMComplexitySuggestion]] = None,
    ) -> ComplexityPlan:
        vlm_suggestions = list(vlm_suggestions or [])
        fg = _load_xyz(bundle.fg_traj_path)
        fluid = _load_xyz(bundle.fluid_traj_path)
        total_frames = self._infer_total_frames(fg, fluid, bundle.flow_npy_path, bundle.fluid_mask_video_path, bundle.fg_mask_video_path)
        if total_frames < 2:
            raise ValueError("Need at least two simulation frames to analyze complexity.")

        transitions = total_frames - 1
        physical_components = self._physical_components(fg, fluid, transitions)
        visual_components = self._visual_components(bundle, transitions, total_frames)
        aligned_events = self._align_events(intent.key_events, physical_components, visual_components, transitions)
        semantic_components = self._semantic_components(aligned_events, vlm_suggestions, physical_components, visual_components, transitions)

        priority_weights = intent.constraint_priority.normalized_complexity_weights()
        base_total = max(
            1e-6,
            self.config.alpha_physical + self.config.beta_visual + self.config.gamma_semantic,
        )
        base_weights = {
            "physical": self.config.alpha_physical / base_total,
            "visual": self.config.beta_visual / base_total,
            "semantic": self.config.gamma_semantic / base_total,
        }
        weights = {
            key: 0.5 * base_weights[key] + 0.5 * priority_weights[key]
            for key in base_weights
        }
        if not physical_components:
            weights["visual"] += weights["physical"] * 0.7
            weights["semantic"] += weights["physical"] * 0.3
            weights["physical"] = 0.0
        if not visual_components:
            weights["physical"] += weights["visual"] * 0.7
            weights["semantic"] += weights["visual"] * 0.3
            weights["visual"] = 0.0
        weight_total = max(1e-6, sum(weights.values()))
        weights = {k: float(v / weight_total) for k, v in weights.items()}

        physical_score = _safe_mean_stack(list(physical_components.values()), transitions)
        visual_score = _safe_mean_stack(list(visual_components.values()), transitions)
        semantic_score = _safe_mean_stack(list(semantic_components.values()), transitions)
        hybrid = (
            weights["physical"] * physical_score
            + weights["visual"] * visual_score
            + weights["semantic"] * semantic_score
        ).astype(np.float32)

        transition_scores: list[FrameComplexity] = []
        all_components = {**physical_components, **visual_components, **semantic_components}
        event_names_by_t = self._event_names_by_transition(aligned_events, transitions)
        for i in range(transitions):
            components = {key: float(value[i]) for key, value in all_components.items()}
            transition_scores.append(
                FrameComplexity(
                    frame_index=i,
                    score=float(hybrid[i]),
                    group_scores=ComplexityGroupScore(
                        physical=float(physical_score[i]),
                        visual=float(visual_score[i]),
                        semantic=float(semantic_score[i]),
                        hybrid=float(hybrid[i]),
                    ),
                    components=components,
                    semantic_events=event_names_by_t[i],
                )
            )

        planner = ComplexityAwareClipPlanner(
            ClipPlannerSettings(
                min_clip_frames=self.config.min_clip_frames,
                default_clip_frames=self.config.default_clip_frames,
                max_clip_frames=self.config.max_clip_frames,
                high_threshold=self.config.high_threshold,
                medium_threshold=self.config.medium_threshold,
            )
        )
        clips = planner.plan(
            transition_scores=transition_scores,
            total_frames=total_frames,
            base_prompt=base_prompt,
            events=aligned_events,
            vlm_suggestions=vlm_suggestions,
        )

        return ComplexityPlan(
            total_frames=total_frames,
            transition_scores=transition_scores,
            clips=clips,
            thresholds={"medium": self.config.medium_threshold, "high": self.config.high_threshold},
            weights=weights,
            event_timeline=aligned_events,
            vlm_suggestions=list(vlm_suggestions),
            summary={
                "mean_score": float(np.mean(hybrid)),
                "max_score": float(np.max(hybrid)),
                "mean_physical": float(np.mean(physical_score)),
                "mean_visual": float(np.mean(visual_score)),
                "mean_semantic": float(np.mean(semantic_score)),
                "num_clips": len(clips),
                "uses_flow": bundle.flow_npy_path is not None and Path(bundle.flow_npy_path).exists(),
                "uses_pointcloud": fg is not None or fluid is not None,
                "uses_masks": bool(bundle.fluid_mask_video_path or bundle.fg_mask_video_path or bundle.segmap_video_path),
                "components": sorted(all_components.keys()),
            },
        )

    def _infer_total_frames(
        self,
        fg: Optional[np.ndarray],
        fluid: Optional[np.ndarray],
        flow_path: Optional[str],
        fluid_mask_path: Optional[str],
        fg_mask_path: Optional[str],
    ) -> int:
        candidates: List[int] = []
        for arr in [fg, fluid]:
            if arr is not None:
                candidates.append(int(arr.shape[0]))
        if flow_path and Path(flow_path).exists():
            flow = np.load(flow_path, mmap_mode="r")
            if flow.ndim != 4:
                raise ValueError(f"Expected flow shape (T, 2, H, W), got {flow.shape}")
            candidates.append(int(flow.shape[0]) + 1)
        for path in [fluid_mask_path, fg_mask_path]:
            if path and cv2 is not None and Path(path).exists():
                cap = cv2.VideoCapture(str(path))
                if cap.isOpened():
                    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if count > 0:
                        candidates.append(count)
                cap.release()
        if not candidates:
            raise FileNotFoundError("No fg/fluid trajectory, flow npy, or mask video found for complexity analysis.")
        return max(2, min(candidates))

    def _physical_components(self, fg: Optional[np.ndarray], fluid: Optional[np.ndarray], transitions: int) -> Dict[str, np.ndarray]:
        comps: Dict[str, np.ndarray] = {}
        for name, xyz in [("fg", fg), ("fluid", fluid)]:
            if xyz is None or xyz.shape[0] < 2:
                continue
            xyz = xyz[: transitions + 1]
            vel = np.diff(xyz, axis=0)
            speed = np.linalg.norm(vel, axis=2)
            centroid = xyz.mean(axis=1)
            centroid_speed = np.linalg.norm(np.diff(centroid, axis=0), axis=1)
            particle_speed_mean = np.mean(speed, axis=1)
            particle_speed_var = np.var(speed, axis=1)
            p95_disp = np.percentile(speed, 95, axis=1)
            if vel.shape[0] >= 2:
                accel = np.linalg.norm(np.diff(vel, axis=0), axis=2)
                accel_mag = np.mean(accel, axis=1)
            else:
                accel_mag = np.zeros(transitions, dtype=np.float32)
            spread = np.mean(np.linalg.norm(xyz - centroid[:, None, :], axis=2), axis=1)
            spread_change = np.abs(np.diff(spread, axis=0))
            entropy = self._particle_entropy_curve(xyz)
            entropy_change = np.abs(np.diff(entropy, axis=0))

            comps[f"{name}_centroid_speed"] = _robust_norm(_pad_or_trim(centroid_speed, transitions))
            comps[f"{name}_velocity_mean"] = _robust_norm(_pad_or_trim(particle_speed_mean, transitions))
            comps[f"{name}_velocity_var"] = _robust_norm(_pad_or_trim(particle_speed_var, transitions))
            comps[f"{name}_p95_disp"] = _robust_norm(_pad_or_trim(p95_disp, transitions))
            comps[f"{name}_acceleration"] = _robust_norm(_pad_or_trim(accel_mag, transitions))
            comps[f"{name}_spread_change"] = _robust_norm(_pad_or_trim(spread_change, transitions))
            comps[f"{name}_entropy_change"] = _robust_norm(_pad_or_trim(entropy_change, transitions))

        if fg is not None and fluid is not None and fg.shape[0] >= 2 and fluid.shape[0] >= 2:
            contact, closeness = self._contact_curves(fg[: transitions + 1], fluid[: transitions + 1], transitions)
            comps["contact_intensity"] = _robust_norm(contact)
            comps["contact_closeness"] = _robust_norm(closeness)
            comps["contact_change"] = _robust_norm(_pad_or_trim(np.abs(np.diff(contact)), transitions))
        return comps

    def _particle_entropy_curve(self, xyz: np.ndarray) -> np.ndarray:
        out = []
        for frame in xyz:
            pts = _sample_points(frame, 2048)
            if pts.shape[0] < 4:
                out.append(0.0)
                continue
            mins = pts.min(axis=0)
            maxs = pts.max(axis=0)
            span = np.maximum(maxs - mins, 1e-6)
            idx = np.floor((pts - mins) / span * 8).clip(0, 7).astype(np.int32)
            flat = idx[:, 0] * 64 + idx[:, 1] * 8 + idx[:, 2]
            counts = np.bincount(flat, minlength=512).astype(np.float32)
            prob = counts[counts > 0] / max(1.0, counts.sum())
            entropy = -np.sum(prob * np.log(prob + 1e-8)) / np.log(512)
            out.append(float(entropy))
        return np.asarray(out, dtype=np.float32)

    def _contact_curves(self, fg: np.ndarray, fluid: np.ndarray, transitions: int) -> tuple[np.ndarray, np.ndarray]:
        contact = []
        closeness = []
        init = np.concatenate([_sample_points(fg[0], 512), _sample_points(fluid[0], 512)], axis=0)
        bbox_diag = float(np.linalg.norm(init.max(axis=0) - init.min(axis=0))) if init.size else 1.0
        threshold = max(1e-5, bbox_diag * 0.03)
        for t in range(min(fg.shape[0], fluid.shape[0], transitions + 1)):
            fgs = _sample_points(fg[t], self.config.contact_sample_points)
            fls = _sample_points(fluid[t], self.config.contact_sample_points)
            ratio, close = _nearest_distance_stats(fls, fgs, threshold)
            contact.append(ratio)
            closeness.append(close)
        return _pad_or_trim(np.asarray(contact, dtype=np.float32), transitions), _pad_or_trim(np.asarray(closeness, dtype=np.float32), transitions)

    def _visual_components(self, bundle: SimulationBundle, transitions: int, total_frames: int) -> Dict[str, np.ndarray]:
        comps: Dict[str, np.ndarray] = {}
        if bundle.flow_npy_path and Path(bundle.flow_npy_path).exists():
            flow = np.load(bundle.flow_npy_path, mmap_mode="r")[:transitions].astype(np.float32)
            mag = np.sqrt(np.square(flow[:, 0]) + np.square(flow[:, 1]))
            comps["flow_mean_mag"] = _robust_norm(_pad_or_trim(np.mean(mag, axis=(1, 2)), transitions))
            comps["flow_var_mag"] = _robust_norm(_pad_or_trim(np.var(mag, axis=(1, 2)), transitions))
            comps["flow_p95_mag"] = _robust_norm(_pad_or_trim(np.percentile(mag, 95, axis=(1, 2)), transitions))
            comps["flow_active_ratio"] = _robust_norm(_pad_or_trim(np.mean(mag > 0.5, axis=(1, 2)), transitions))
            comps["motion_saliency_concentration"] = _robust_norm(_pad_or_trim(self._saliency_concentration(mag), transitions))
            discontinuity, curl_abs, div_abs = self._flow_gradient_stats(flow)
            comps["flow_discontinuity"] = _robust_norm(_pad_or_trim(discontinuity, transitions))
            comps["flow_curl_abs"] = _robust_norm(_pad_or_trim(curl_abs, transitions))
            comps["flow_divergence_abs"] = _robust_norm(_pad_or_trim(div_abs, transitions))

        masks = self._load_masks(bundle, total_frames)
        if masks is not None:
            comps.update(self._mask_components(masks, transitions))
        return comps

    def _saliency_concentration(self, mag: np.ndarray) -> np.ndarray:
        out = []
        for frame in mag:
            flat = frame.reshape(-1)
            total = float(np.sum(flat))
            if total <= 1e-6:
                out.append(0.0)
                continue
            k = max(1, int(flat.size * 0.1))
            top = np.partition(flat, -k)[-k:]
            out.append(float(np.sum(top) / total))
        return np.asarray(out, dtype=np.float32)

    def _flow_gradient_stats(self, flow: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        discontinuity = []
        curl_abs = []
        div_abs = []
        for frame in flow:
            u, v = frame[0], frame[1]
            du_dy, du_dx = np.gradient(u)
            dv_dy, dv_dx = np.gradient(v)
            grad_mag = np.sqrt(du_dx**2 + du_dy**2 + dv_dx**2 + dv_dy**2)
            discontinuity.append(float(np.mean(grad_mag)))
            curl_abs.append(float(np.mean(np.abs(dv_dx - du_dy))))
            div_abs.append(float(np.mean(np.abs(du_dx + dv_dy))))
        return (
            np.asarray(discontinuity, dtype=np.float32),
            np.asarray(curl_abs, dtype=np.float32),
            np.asarray(div_abs, dtype=np.float32),
        )

    def _load_masks(self, bundle: SimulationBundle, total_frames: int) -> Optional[np.ndarray]:
        masks = []
        for path in [bundle.fluid_mask_video_path, bundle.fg_mask_video_path, bundle.segmap_video_path]:
            arr = _read_mask_video(path, total_frames)
            if arr is not None:
                masks.append(arr[:total_frames])
        if not masks:
            return None
        min_t = min(x.shape[0] for x in masks)
        combined = np.zeros_like(masks[0][:min_t], dtype=np.uint8)
        for arr in masks:
            combined = np.maximum(combined, arr[:min_t].astype(np.uint8))
        return combined

    def _mask_components(self, masks: np.ndarray, transitions: int) -> Dict[str, np.ndarray]:
        if cv2 is None:
            return {}
        masks = masks.astype(np.uint8)
        area = masks.reshape(masks.shape[0], -1).mean(axis=1)
        boundary_density = []
        fragmentation = []
        for mask in masks:
            kernel = np.ones((3, 3), dtype=np.uint8)
            eroded = cv2.erode(mask, kernel, iterations=1)
            boundary = np.logical_and(mask > 0, eroded == 0)
            boundary_density.append(float(boundary.mean()))
            num_labels, _ = cv2.connectedComponents(mask, connectivity=8)
            fragmentation.append(float(max(0, num_labels - 1)))
        silhouette_change = []
        for a, b in zip(masks[:-1], masks[1:]):
            union = np.logical_or(a > 0, b > 0).sum()
            xor = np.logical_xor(a > 0, b > 0).sum()
            silhouette_change.append(float(xor / max(1, union)))
        return {
            "mask_area_change": _robust_norm(_pad_or_trim(np.abs(np.diff(area)), transitions)),
            "mask_boundary_density": _robust_norm(_pad_or_trim(np.asarray(boundary_density[:-1]), transitions)),
            "mask_fragmentation": _robust_norm(_pad_or_trim(np.asarray(fragmentation[:-1]), transitions)),
            "mask_fragmentation_change": _robust_norm(_pad_or_trim(np.abs(np.diff(fragmentation)), transitions)),
            "mask_silhouette_change": _robust_norm(_pad_or_trim(np.asarray(silhouette_change), transitions)),
        }

    def _align_events(
        self,
        events: Sequence[SemanticEvent],
        physical_components: Dict[str, np.ndarray],
        visual_components: Dict[str, np.ndarray],
        transitions: int,
    ) -> List[SemanticEvent]:
        components = {**physical_components, **visual_components}
        aligned: list[SemanticEvent] = []
        for event in events:
            signals = [components[key] for key in event.preferred_signals if key in components]
            if not signals:
                # The type's alignment set is unavailable for this rollout (e.g. no
                # masks): use the default row of the table, then every signal.
                signals = [components[key] for key in DEFAULT_ALIGNMENT_SET if key in components]
            if not signals:
                signals = list(components.values())
            curve = _safe_mean_stack(signals, transitions)
            if curve.size == 0 or float(np.max(curve)) <= 1e-6:
                start, end, conf = 0, min(transitions, self.config.event_window), 0.2
            else:
                peak = int(np.argmax(curve))
                radius = max(2, self.config.event_window)
                start = max(0, peak - radius)
                end = min(transitions, peak + radius)
                conf = float(max(event.confidence, curve[peak]))
            aligned.append(
                SemanticEvent(
                    name=event.name,
                    description=event.description,
                    importance=event.importance,
                    keywords=event.keywords,
                    preferred_signals=event.preferred_signals,
                    frame_start=start,
                    frame_end=end,
                    confidence=max(0.0, min(1.0, conf)),
                )
            )
        return aligned

    def _semantic_components(
        self,
        events: Sequence[SemanticEvent],
        vlm_suggestions: Sequence[VLMComplexitySuggestion],
        physical_components: Dict[str, np.ndarray],
        visual_components: Dict[str, np.ndarray],
        transitions: int,
    ) -> Dict[str, np.ndarray]:
        comps: Dict[str, np.ndarray] = {}
        for event in events:
            curve = np.zeros(transitions, dtype=np.float32)
            if event.frame_start is not None and event.frame_end is not None:
                start = max(0, event.frame_start)
                end = min(transitions, event.frame_end)
                if end > start:
                    x = np.linspace(-1.0, 1.0, end - start, dtype=np.float32)
                    bump = np.exp(-3.0 * x**2) * event.importance * max(0.25, event.confidence)
                    curve[start:end] = np.maximum(curve[start:end], bump)
            comps[f"semantic_event_{event.name}"] = curve
        vlm_curve = np.zeros(transitions, dtype=np.float32)
        for suggestion in vlm_suggestions:
            start = max(0, suggestion.frame_start)
            end = min(transitions, suggestion.frame_end)
            if end <= start:
                continue
            risk_weight = {"low": 0.3, "medium": 0.6, "high": 0.9}.get(suggestion.risk.lower(), 0.6)
            vlm_curve[start:end] = np.maximum(vlm_curve[start:end], risk_weight * suggestion.confidence)
        if np.any(vlm_curve > 0):
            comps["semantic_vlm_risk"] = vlm_curve
        if not comps:
            comps["semantic_general"] = np.zeros(transitions, dtype=np.float32)
        return comps

    def _event_names_by_transition(self, events: Sequence[SemanticEvent], transitions: int) -> list[list[str]]:
        names = [[] for _ in range(transitions)]
        for event in events:
            if event.frame_start is None or event.frame_end is None:
                continue
            for t in range(max(0, event.frame_start), min(transitions, event.frame_end)):
                names[t].append(event.name)
        return names
