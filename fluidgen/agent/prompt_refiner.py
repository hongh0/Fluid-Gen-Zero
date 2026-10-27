"""Simulation-aware prompt refinement for VACE clips."""

from __future__ import annotations

from typing import Any, List, Sequence

from .schema import ClipPlan, ComplexityPlan, IntentParseResult, SemanticEvent


class PromptRefiner:
    """Build one unified caption prompt per clip, not merely polished prose."""

    def build_unified_caption(
        self,
        intent: IntentParseResult,
        events: Sequence[SemanticEvent] | None = None,
    ) -> str:
        events = list(events or intent.key_events)
        d = intent.dynamics
        a = intent.appearance
        event_text = "; ".join(
            f"{event.name}: {event.description}" for event in events[:5]
        ) or "general fluid-object interaction"
        hard = " ".join(intent.constraint_priority.hard_constraints)
        return (
            f"{intent.raw_prompt}. Unified simulation caption: a {a.style} scene with {a.material} fluid, "
            f"{a.lighting}, {a.background}. The motion follows the provided simulation masks and optical flow: "
            f"{d.speed} flow, {d.fluid_volume} volume, gravity {d.gravity_mag:g} m/s^2, "
            f"viscosity prior {d.viscosity:g}. Key event timeline: {event_text}. "
            f"Visual requirements: preserve material appearance, coherent liquid surface, stable silhouettes, "
            f"realistic reflections/refractions, and no object identity drift. Hard constraints: {hard}"
        )

    def refine_global(self, intent: IntentParseResult, complexity_plan: ComplexityPlan | None = None) -> str:
        events = complexity_plan.event_timeline if complexity_plan is not None else intent.key_events
        return (
            f"{self.build_unified_caption(intent, events)} Use the simulation conditions as hard motion guidance. "
            "Keep temporal continuity across clip boundaries using the previous generated last frame as the next anchor."
        )

    def refine_for_clip(self, intent: IntentParseResult, clip: ClipPlan, complexity_plan: ComplexityPlan | None = None) -> str:
        events = []
        if complexity_plan is not None:
            events = [
                event for event in complexity_plan.event_timeline
                if event.frame_start is not None
                and event.frame_end is not None
                and not (event.frame_end < clip.start_frame or event.frame_start > clip.end_frame)
            ]
        event_text = "; ".join(event.description for event in (events or intent.key_events)[:3])
        local_motion = self._local_motion_hint(clip)
        d = intent.dynamics
        a = intent.appearance
        raw_prompt = intent.raw_prompt.strip().rstrip(".!?")
        return (
            f"{raw_prompt}. A {a.style} video scene with {a.material} fluid, {a.lighting}, "
            f"{a.background}. Keep the same subject, material, lighting, background, and framing consistent "
            f"across the sequence. The motion should feel guided by the supplied control references: "
            f"{d.speed} flow, {d.fluid_volume} fluid volume, natural gravity, and physically coherent liquid behavior. "
            f"{local_motion} "
            f"Key visual action: {event_text or 'the fluid and object continue their planned interaction with smooth temporal continuity'}. "
            f"Preserve object identity, stable geometry, coherent liquid surfaces, clean silhouettes, realistic reflections and refractions, "
            f"accurate object-fluid contact, and natural continuation from the previous moment. Do not hallucinate new objects or change the user's intent."
        )

    @staticmethod
    def _word_count(text: str) -> int:
        return len([token for token in text.replace("\n", " ").split(" ") if token.strip()])

    @staticmethod
    def _local_motion_hint(clip: ClipPlan) -> str:
        tags = " ".join(clip.risk_tags).lower()
        hints: list[str] = []
        if any(token in tags for token in ["splash", "spray", "droplet", "impact"]):
            hints.append("Emphasize the splash, spray, and small droplets with crisp local detail while keeping the main object clearly identifiable.")
        if any(token in tags for token in ["topology", "split", "merge", "fragmentation"]):
            hints.append("Keep separating and merging fluid shapes coherent, with believable surface tension and no unstable blob flicker.")
        if any(token in tags for token in ["occlusion", "silhouette", "contact"]):
            hints.append("Maintain clean object-fluid contact boundaries and stable silhouettes when the liquid crosses or partially covers the object.")
        if any(token in tags for token in ["swirl", "vortex", "flow_discontinuity", "visual_motion_high"]):
            hints.append("Let the fast local motion read as continuous fluid movement rather than noisy texture changes.")
        if hints:
            return " ".join(hints[:2])
        if clip.complexity_label == "high":
            return "Give this moment extra physical clarity, especially around contact, deformation, and thin fluid details."
        if clip.complexity_label == "medium":
            return "Keep the motion direction, surface deformation, and contact timing smooth and readable."
        return "Let the motion remain smooth and stable, with faithful material appearance and gentle temporal continuity."

    @staticmethod
    def _forbidden_prompt_terms() -> list[str]:
        return [
            "clip frames",
            "vace frame_num",
            "frame_num",
            "complexity=",
            "complexity:",
            "risk cues",
            "risk tags",
            "clip-specific focus",
            "low-risk interval",
            "moderate-risk interval",
            "high-risk interval",
            "anchor strategy",
            "optical flow",
            "simulation masks",
            "point cloud",
        ]

    def _clean_refined_focus(self, prompt: str) -> str:
        cleaned = prompt.strip()
        replacements = {
            "simulation masks and optical flow": "motion conditioning",
            "simulation masks": "motion conditioning",
            "optical flow": "motion guidance",
        }
        for src, dst in replacements.items():
            cleaned = cleaned.replace(src, dst).replace(src.title(), dst)

        sentences = [part.strip() for part in cleaned.replace("\n", " ").split(".") if part.strip()]
        kept = [
            sentence
            for sentence in sentences
            if not any(term in sentence.lower() for term in self._forbidden_prompt_terms())
        ]
        return ". ".join(kept).strip()

    def _needs_rich_prompt_guard(self, prompt: str, global_style_prompt: str = "") -> bool:
        lower = prompt.lower()
        forbidden = self._forbidden_prompt_terms()
        has_forbidden_label = any(term in lower for term in forbidden)
        has_motion_guidance = any(term in lower for term in ["motion", "flow", "fluid", "splash", "continuity", "contact"])
        has_visual_constraints = any(term in lower for term in ["identity", "silhouette", "reflection", "geometry", "coherent", "surface"])
        has_global_stem = True
        if global_style_prompt.strip():
            stem = " ".join(global_style_prompt.lower().split()[:8])
            has_global_stem = bool(stem) and prompt.lower().startswith(stem)
        return self._word_count(prompt) < 85 or not (has_motion_guidance and has_visual_constraints) or not has_global_stem or has_forbidden_label

    def _compose_rich_refined_prompt(
        self,
        intent: IntentParseResult,
        clip: ClipPlan,
        refined_prompt: str,
        global_style_prompt: str = "",
    ) -> str:
        d = intent.dynamics
        a = intent.appearance
        global_stem = (
            global_style_prompt.strip()
            or (
                f"{intent.raw_prompt}. A {a.style} video scene with {a.material} fluid, "
                f"{a.lighting}, {a.background}."
            )
        ).rstrip(".")
        refined_focus = self._clean_refined_focus(refined_prompt).rstrip(".")
        if refined_focus:
            local_focus = refined_focus.rstrip(".")
        else:
            local_focus = "Preserve the planned local fluid-object motion with smooth temporal continuity"
        local_motion = self._local_motion_hint(clip)

        return (
            f"{global_stem}. Keep this same subject, material, lighting, background, and framing consistent across the entire sequence. "
            f"Use the supplied motion conditioning as a strong guide for timing, direction, and liquid deformation, "
            f"with {d.speed} {a.material} flow, {d.fluid_volume} fluid volume, natural gravity, and physically coherent viscosity. "
            f"{local_focus}. {local_motion} "
            f"Preserve object identity, stable geometry, coherent liquid surfaces, clean silhouettes, realistic reflections and refractions, "
            f"accurate object-fluid contact, and natural continuation from the previous moment. Do not hallucinate new objects or change the user's intent."
        )

    def apply(self, intent: IntentParseResult, clips: List[ClipPlan], complexity_plan: ComplexityPlan | None = None) -> List[ClipPlan]:
        for clip in clips:
            clip.prompt = self.refine_for_clip(intent, clip, complexity_plan)
        return clips

    def apply_with_openai(
        self,
        intent: IntentParseResult,
        clips: List[ClipPlan],
        complexity_plan: ComplexityPlan | None = None,
        openai_adapter: Any = None,
        image_path: str | None = None,
    ) -> dict | None:
        draft_prompts: dict[int, str] = {}
        for clip in clips:
            draft = self.refine_for_clip(intent, clip, complexity_plan)
            clip.prompt = draft
            draft_prompts[clip.clip_id] = draft

        if openai_adapter is None:
            for clip in clips:
                clip.notes.append("OpenAI prompt refinement skipped: no adapter configured.")
            return {"_api_status": "adapter_missing"}

        response = openai_adapter.refine_clip_prompts(
            intent=intent,
            clips=clips,
            complexity_plan=complexity_plan,
            image_path=image_path,
            draft_prompts=draft_prompts,
        )
        if not response or response.get("_api_status"):
            status = response.get("_api_status", "empty_response") if isinstance(response, dict) else "empty_response"
            for clip in clips:
                clip.notes.append(f"OpenAI prompt refinement not applied; fallback prompt kept ({status}).")
            return response

        by_id = {clip.clip_id: clip for clip in clips}
        applied = 0
        for item in response.get("clip_prompts", []):
            try:
                clip_id = int(item.get("clip_id"))
            except Exception:
                continue
            prompt = str(item.get("prompt", "")).strip()
            if not prompt or clip_id not in by_id:
                continue
            clip = by_id[clip_id]
            global_style_prompt = str(response.get("global_style_prompt", ""))
            enriched = False
            if self._needs_rich_prompt_guard(prompt, global_style_prompt=global_style_prompt):
                prompt = self._compose_rich_refined_prompt(
                    intent,
                    clip,
                    prompt,
                    global_style_prompt=global_style_prompt,
                )
                enriched = True
                clip.notes.append("OpenAI prompt refinement was enriched with simulation-control context.")
            clip.prompt = prompt
            item["prompt"] = prompt
            if enriched:
                item["enriched_by_agent"] = True
            clip.notes.append("OpenAI prompt refinement applied for VACE generation prompt.")
            applied += 1
        for clip in clips:
            if "OpenAI prompt refinement applied for VACE generation prompt." not in clip.notes:
                clip.notes.append("OpenAI prompt refinement did not return this clip; fallback prompt kept.")
        response["_applied_clip_prompts"] = applied
        return response
