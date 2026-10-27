"""Complexity-aware clip segmentation for VACE 4n+1 generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .schema import ClipPlan, FrameComplexity, SemanticEvent, VLMComplexitySuggestion


@dataclass
class ClipPlannerSettings:
    min_clip_frames: int = 13
    default_clip_frames: int = 41
    max_clip_frames: int = 81
    high_threshold: float = 0.72
    medium_threshold: float = 0.45
    boundary_search_radius: int = 12
    high_event_min_importance: float = 0.75


class ComplexityAwareClipPlanner:
    """Plan clips so high-risk motion is short but semantic events are not cut."""

    def __init__(self, settings: ClipPlannerSettings):
        self.settings = settings

    def plan(
        self,
        transition_scores: Sequence[FrameComplexity],
        total_frames: int,
        base_prompt: str,
        events: Sequence[SemanticEvent] | None = None,
        vlm_suggestions: Sequence[VLMComplexitySuggestion] | None = None,
    ) -> List[ClipPlan]:
        events = list(events or [])
        vlm_suggestions = list(vlm_suggestions or [])
        scores = np.array([x.score for x in transition_scores], dtype=np.float32)
        clips: List[ClipPlan] = []
        start = 0
        clip_id = 0
        while start < total_frames - 1:
            remaining = total_frames - start
            end, frame_num, label, cut_reason = self._choose_end(
                start=start,
                remaining_frames=remaining,
                scores=scores,
                events=events,
                vlm_suggestions=vlm_suggestions,
            )
            actual_frame_num = end - start + 1
            clip_score = self._clip_score(scores, start, end)
            risk_tags = self._risk_tags(transition_scores, start, end, events, vlm_suggestions)
            notes = ["frame_num satisfies VACE 4n+1"]
            if frame_num != actual_frame_num:
                notes.append(f"pad {frame_num - actual_frame_num} trailing frame(s) for VACE 4n+1 input")
            if any(self._is_no_cut_inside(start, end, e) for e in events):
                notes.append("contains a semantic event interval; do not split internally")

            clips.append(
                ClipPlan(
                    clip_id=clip_id,
                    start_frame=start,
                    end_frame=end,
                    actual_frame_num=actual_frame_num,
                    frame_num=frame_num,
                    complexity_score=clip_score,
                    complexity_label=label,
                    prompt=base_prompt,
                    risk_tags=risk_tags,
                    anchor_strategy="rolling_first_frame_anchor",
                    overlap_left=0 if clip_id == 0 else 1,
                    overlap_right=1 if end < total_frames - 1 else 0,
                    cut_reason=cut_reason,
                    notes=notes,
                )
            )
            start = end
            clip_id += 1
        return clips

    def _choose_end(
        self,
        start: int,
        remaining_frames: int,
        scores: np.ndarray,
        events: Sequence[SemanticEvent],
        vlm_suggestions: Sequence[VLMComplexitySuggestion],
    ) -> tuple[int, int, str, str]:
        max_len = min(self.settings.max_clip_frames, remaining_frames)
        valid_max_len = self._floor_4n1(max_len)
        valid_max_len = max(5, valid_max_len)
        if remaining_frames <= self.settings.max_clip_frames:
            padded = self._ceil_4n1(remaining_frames)
            return start + remaining_frames - 1, padded, self._label(scores, start, start + remaining_frames - 1), "final_segment"

        local = scores[start : min(len(scores), start + valid_max_len - 1)]
        peak = float(np.max(local)) if local.size else 0.0
        mean = float(np.mean(local)) if local.size else 0.0
        if peak >= self.settings.high_threshold:
            target = self.settings.min_clip_frames
            label = "high"
        elif mean >= self.settings.medium_threshold:
            target = self.settings.default_clip_frames
            label = "medium"
        else:
            target = self.settings.max_clip_frames
            label = "low"

        candidates = self._candidate_lengths(start, target, valid_max_len)
        best: tuple[float, int, str] | None = None
        for frame_num in candidates:
            end = start + frame_num - 1
            cut_frame = end
            stability = self._boundary_instability(scores, cut_frame)
            length_penalty = abs(frame_num - target) / max(1, self.settings.max_clip_frames)
            forbidden = self._forbidden_cut(cut_frame, events, vlm_suggestions, scores)
            event_bonus = self._event_boundary_bonus(cut_frame, events, vlm_suggestions)
            cost = stability + 0.35 * length_penalty - event_bonus
            reason = "stable_boundary"
            if forbidden:
                cost += 10.0
                reason = "fallback_boundary_forbidden_penalty"
            elif event_bonus > 0:
                reason = "event_boundary"
            if best is None or cost < best[0]:
                best = (cost, end, reason)

        if best is None:
            frame_num = max(5, self._floor_4n1(min(target, valid_max_len)))
            return start + frame_num - 1, frame_num, label, "fallback_length"

        end = best[1]
        frame_num = end - start + 1
        return end, frame_num, label, best[2]

    def _candidate_lengths(self, start: int, target: int, valid_max_len: int) -> list[int]:
        lo = max(5, self.settings.min_clip_frames)
        hi = max(lo, valid_max_len)
        lengths = set(range(self._ceil_4n1(lo), self._floor_4n1(hi) + 1, 4))
        radius = self.settings.boundary_search_radius
        for raw in range(max(lo, target - radius), min(hi, target + radius) + 1):
            lengths.add(self._floor_4n1(raw))
            lengths.add(self._ceil_4n1(raw))
        return sorted(x for x in lengths if lo <= x <= hi and (x - 1) % 4 == 0)

    @staticmethod
    def _ceil_4n1(length: int) -> int:
        return ((max(1, length) - 1 + 3) // 4) * 4 + 1

    @staticmethod
    def _floor_4n1(length: int) -> int:
        return ((max(1, length) - 1) // 4) * 4 + 1

    def _clip_score(self, scores: np.ndarray, start: int, end: int) -> float:
        local = scores[start:end]
        if local.size == 0:
            return 0.0
        return float(max(np.mean(local), np.percentile(local, 90)))

    def _label(self, scores: np.ndarray, start: int, end: int) -> str:
        score = self._clip_score(scores, start, end)
        if score >= self.settings.high_threshold:
            return "high"
        if score >= self.settings.medium_threshold:
            return "medium"
        return "low"

    def _boundary_instability(self, scores: np.ndarray, cut_frame: int) -> float:
        lo = max(0, cut_frame - 2)
        hi = min(len(scores), cut_frame + 2)
        if hi <= lo:
            return 0.0
        return float(np.mean(scores[lo:hi]))

    def _forbidden_cut(
        self,
        cut_frame: int,
        events: Sequence[SemanticEvent],
        vlm_suggestions: Sequence[VLMComplexitySuggestion],
        scores: np.ndarray,
    ) -> bool:
        if 0 <= cut_frame - 1 < len(scores) and scores[cut_frame - 1] >= self.settings.high_threshold:
            return True
        for event in events:
            if event.importance < self.settings.high_event_min_importance:
                continue
            if event.frame_start is None or event.frame_end is None:
                continue
            if event.frame_start < cut_frame < event.frame_end:
                return True
        for suggestion in vlm_suggestions:
            if suggestion.no_cut and suggestion.frame_start < cut_frame < suggestion.frame_end:
                return True
        return False

    def _event_boundary_bonus(
        self,
        cut_frame: int,
        events: Sequence[SemanticEvent],
        vlm_suggestions: Sequence[VLMComplexitySuggestion],
    ) -> float:
        bonus = 0.0
        for event in events:
            for boundary in [event.frame_start, event.frame_end]:
                if boundary is not None and abs(cut_frame - boundary) <= 2:
                    bonus = max(bonus, 0.25 * event.importance)
        for suggestion in vlm_suggestions:
            if any(abs(cut_frame - x) <= 2 for x in suggestion.suggested_cut_frames):
                bonus = max(bonus, 0.25 * suggestion.confidence)
        return bonus

    @staticmethod
    def _is_no_cut_inside(start: int, end: int, event: SemanticEvent) -> bool:
        if event.frame_start is None or event.frame_end is None:
            return False
        return start <= event.frame_start and event.frame_end <= end

    def _risk_tags(
        self,
        transition_scores: Sequence[FrameComplexity],
        start: int,
        end: int,
        events: Sequence[SemanticEvent],
        vlm_suggestions: Sequence[VLMComplexitySuggestion],
    ) -> list[str]:
        local = transition_scores[start:end]
        tags: set[str] = set()
        if not local:
            return []
        max_phys = max(x.group_scores.physical for x in local)
        max_vis = max(x.group_scores.visual for x in local)
        max_sem = max(x.group_scores.semantic for x in local)
        if max_phys >= self.settings.high_threshold:
            tags.add("physical_dynamics_high")
        if max_vis >= self.settings.high_threshold:
            tags.add("visual_motion_high")
        if max_sem >= self.settings.high_threshold:
            tags.add("semantic_event_high")
        component_max: dict[str, float] = {}
        for item in local:
            for key, value in item.components.items():
                component_max[key] = max(component_max.get(key, 0.0), value)
        for key, tag in [
            ("contact_intensity", "object_fluid_contact"),
            ("contact_change", "contact_change"),
            ("mask_fragmentation", "topology_fragmentation"),
            ("flow_discontinuity", "flow_discontinuity"),
            ("flow_curl_abs", "swirling_motion"),
            ("mask_silhouette_change", "rapid_occlusion_or_silhouette_change"),
        ]:
            if component_max.get(key, 0.0) >= self.settings.medium_threshold:
                tags.add(tag)
        for event in events:
            if event.frame_start is not None and event.frame_end is not None and not (event.frame_end < start or event.frame_start > end):
                tags.add(f"event:{event.name}")
        for suggestion in vlm_suggestions:
            if not (suggestion.frame_end < start or suggestion.frame_start > end):
                tags.add(f"vlm:{suggestion.risk}")
        return sorted(tags)
