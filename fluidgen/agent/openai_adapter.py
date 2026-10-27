"""OpenAI API adapter for fluid-object agentic planning.

The API key is intentionally not hard-coded. Pass it explicitly, set
OPENAI_API_KEY, or leave it empty to write/return request payloads without
calling the API.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .physics_mapping import OBJECT_PHYSICS_ENUMS, OBJECT_PHYSICS_PROMPT_GUIDE
from .schema import ClipPlan, ComplexityPlan, IntentParseResult, SimulationBundle, to_dict
from .video_sampler import image_to_data_url, sample_bundle_videos

# The VLM used for all three calls (intent parsing, risk annotation, clip-level
# prompts). Override with FGZ_VLM_MODEL; OPENAI_BASE_URL is honoured by the SDK.
DEFAULT_OPENAI_MODEL = os.environ.get("FGZ_VLM_MODEL", "").strip() or "gpt-5.4-mini-2026-03-17"
DEFAULT_FRAME_INTERVAL = 5

INTENT_PHYSICS_SYSTEM_PROMPT = """You are an expert parser for a fluid-object video generation agent.
Your job is to convert an input image and user prompt into structured controls for a simulation-conditioned video pipeline.

You must output only JSON matching the requested schema. Do not include markdown.

Interpret four categories:
1. Appearance: material, lighting, background, style, visual prompt details.
2. Dynamics: semantic fluid behavior plus gravity, fluid volume, speed, push force, force direction, dt/substeps, and sim_pipeline_overrides.
3. Object Physics Semantics: qualitative labels describing the foreground object's material and contact behavior.
4. Constraint Priority: relative priority for physical, visual, semantic, and temporal constraints.
5. Object Grounding: the noun phrase a text-grounded segmenter should use for the solid object, and its real-world size.

Key domain assumptions:
- object_grounding.object_size_m is the object's LARGEST real-world dimension in metres, and it is the ONLY metric anchor the reconstruction has: monocular depth is defined up to scale, so this number sets the size of everything else in the scene and therefore every gravity timescale. Judge it from the IMAGE, not from the words: how much of the container or frame the object spans, the apparent depth of field, and whether it is a real specimen or a small prop. These are mostly table-top close-ups, so 0.02-0.30 m is the usual range; only go larger when the picture genuinely shows a large scene. Put what set the scale in size_evidence and be honest in confidence.
- Do not estimate new numeric object material constants such as Young's modulus or Poisson ratio directly. Use object_physics_semantics labels only; local code will map them to elastic_E and elastic_nu.
- Treat viscosity and fluid_volume as semantic priors. The local parser may derive liquid_rho and liquid_viscous from the parsed fluid/liquid material.
- Use appearance.material for the visible liquid identity when it is clear, for example water, milk, cola, honey, oil, or lava.
- Extract semantic events that increase generation risk: impact, splash, split_merge, swirl, occlusion, material_detail, or other user-critical interactions.
- Do not invent new user intent. If uncertain, preserve ambiguity in notes.
- Prefer physically plausible parameters over exaggerated values unless the prompt explicitly asks for stylization.
"""

VLM_COMPLEXITY_SYSTEM_PROMPT = """You are a VLM critic for simulation-conditioned fluid-object video generation.
Your role is not to make the final segmentation alone. You provide high-level semantic and visual risk hints for a metric-based clip planner.

You will receive:
- the original user prompt,
- sampled frames from optical-flow, simulation-preview, mask, or segmap videos,
- a compact complexity curve from quantitative metrics.

Judge which time intervals are risky for the video generator: chaotic fluid motion, topology split/merge, splash, occlusion, thin droplets, high local detail, difficult object-fluid contact, or important semantic events.

Return only JSON. For each interval, provide frame_start, frame_end, risk, reason, no_cut, suggested_cut_frames, and confidence.
Use no_cut=true for event interiors that should not be split, such as active splash, split/merge, or rapid occlusion.
"""

PROMPT_REFINEMENT_SYSTEM_PROMPT = """You are a senior prompt writer for a simulation-conditioned fluid-object video generation model.

Your task is to refine structured agent controls into rich, direct video-generation prompts. These prompts are not short captions, but they must read like natural positive prompts rather than engineering logs.

Important rules:
- Preserve the user's intent, object identity, material, scene style, and fluid-object interaction.
- Keep a consistent subject and scene description across all clips. Each clip prompt should begin with a shared scene/style stem so the generated clips do not drift.
- Use the metadata and draft_agent_prompt to understand timing, difficulty, local events, and continuity, but do not copy engineering labels into the final prompt.
- Do not mention or imitate labels such as Clip frames, VACE frame_num, complexity, risk cues, Clip-specific focus, Low-risk interval, Moderate-risk interval, High-risk interval, Anchor strategy, masks, optical flow, point clouds, or simulator.
- Convert that control information into fluent visual language: consistent subject, guided motion, coherent fluid surface, stable silhouettes, object-fluid contact, reflections/refractions, temporal continuity, and no hallucinated objects.
- Do not introduce new objects, camera moves, materials, events, or user intent that were not implied by the input.
- Each clip prompt should be detailed enough for generation, typically 90-160 words. Avoid very short two-sentence captions.
- Clip prompts should differ mainly in the local event focus; the core subject, appearance, constraints, and conditioning language should remain similar.

Return only JSON matching the requested schema.
"""

def strictify(node: Any) -> Any:
    """Make a JSON Schema satisfy OpenAI structured output's `strict` mode.
    """
    if isinstance(node, list):
        return [strictify(x) for x in node]
    if not isinstance(node, dict):
        return node
    out = {k: strictify(v) for k, v in node.items()}
    if out.get("type") == "object":
        out.setdefault("properties", {})
        out["additionalProperties"] = False
        out["required"] = list(out["properties"].keys())
    return out


INTENT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "appearance": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "material": {"type": "string"},
                "lighting": {"type": "string"},
                "background": {"type": "string"},
                "style": {"type": "string"},
                "extra": {"type": "object"},
            },
            "required": ["material", "lighting", "background", "style", "extra"],
        },
        "dynamics": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "viscosity": {"type": "number"},
                "gravity_mag": {"type": "number"},
                "fluid_volume": {"type": "string"},
                "speed": {"type": "string"},
                "push_force": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                "force_direction": {"anyOf": [{"type": "array", "items": {"type": "number"}}, {"type": "null"}]},
                "extra": {"type": "object"},
            },
            "required": [
                "viscosity",
                "gravity_mag",
                "fluid_volume",
                "speed",
                "push_force",
                "force_direction",
                "extra",
            ],
        },
        "object_grounding": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "object_prompt": {"type": "string"},
                "object_size_m": {"type": "number"},
                "object_kind": {"type": "string",
                                "enum": ["rigid", "elastic", "elastoplastic",
                                         "sand", "snow", "liquid"]},
                "size_evidence": {"type": "string"},
                "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            },
            "required": ["object_prompt", "object_size_m", "object_kind",
                         "size_evidence", "confidence"],
        },
        "object_physics_semantics": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "material_class": {"type": "string", "enum": OBJECT_PHYSICS_ENUMS["material_class"]},
                "solidity": {"type": "string", "enum": OBJECT_PHYSICS_ENUMS["solidity"]},
                "hardness_level": {"type": "string", "enum": OBJECT_PHYSICS_ENUMS["hardness_level"]},
                "bounce_category": {"type": "string", "enum": OBJECT_PHYSICS_ENUMS["bounce_category"]},
                "surface_roughness_level": {"type": "integer", "minimum": 1, "maximum": 5},
                "friction_tendency": {"type": "string", "enum": OBJECT_PHYSICS_ENUMS["friction_tendency"]},
                "size_thickness_hint": {"type": "string", "enum": OBJECT_PHYSICS_ENUMS["size_thickness_hint"]},
                "justification": {"type": "string"},
            },
            "required": [
                "material_class",
                "solidity",
                "hardness_level",
                "bounce_category",
                "surface_roughness_level",
                "friction_tendency",
                "size_thickness_hint",
                "justification",
            ],
        },
        "constraint_priority": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "physical": {"type": "number"},
                "visual": {"type": "number"},
                "semantic": {"type": "number"},
                "temporal": {"type": "number"},
                "hard_constraints": {"type": "array", "items": {"type": "string"}},
                "soft_constraints": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["physical", "visual", "semantic", "temporal", "hard_constraints", "soft_constraints"],
        },
        "key_events": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "importance": {"type": "number"},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                    "preferred_signals": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number"},
                },
                "required": ["name", "description", "importance", "keywords", "preferred_signals", "confidence"],
            },
        },
        "visual_prompt": {"type": "string"},
        "overall_prompt": {"type": "string"},
        "negative_prompt": {"type": "string"},
        "notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "appearance",
        "dynamics",
        "object_physics_semantics",
        "constraint_priority",
        "key_events",
        "visual_prompt",
        "overall_prompt",
        "negative_prompt",
        "notes",
        "object_grounding",
    ],
}

VLM_COMPLEXITY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "frame_start": {"type": "integer"},
                    "frame_end": {"type": "integer"},
                    "risk": {"type": "string"},
                    "reason": {"type": "string"},
                    "no_cut": {"type": "boolean"},
                    "suggested_cut_frames": {"type": "array", "items": {"type": "integer"}},
                    "confidence": {"type": "number"},
                },
                "required": ["frame_start", "frame_end", "risk", "reason", "no_cut", "suggested_cut_frames", "confidence"],
            },
        }
    },
    "required": ["suggestions"],
}

PROMPT_REFINEMENT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "global_style_prompt": {"type": "string"},
        "clip_prompts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "clip_id": {"type": "integer"},
                    "prompt": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["clip_id", "prompt", "reason"],
            },
        },
        "notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["global_style_prompt", "clip_prompts", "notes"],
}


class OpenAIAPIAdapter:
    def __init__(
        self,
        api_key: str = "",
        model: str = DEFAULT_OPENAI_MODEL,
        frame_interval: int = DEFAULT_FRAME_INTERVAL,
        max_frames_per_video: int = 0,
        request_dir: str | Path | None = None,
    ):
        self.api_key = api_key if api_key != "" else os.getenv("OPENAI_API_KEY", "")
        self.model = model
        self.frame_interval = max(1, int(frame_interval))
        self.max_frames_per_video = max(0, int(max_frames_per_video))
        self.request_dir = Path(request_dir) if request_dir else None

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _write_request(self, name: str, payload: Dict[str, Any]) -> Optional[str]:
        if self.request_dir is None:
            return None
        self.request_dir.mkdir(parents=True, exist_ok=True)
        path = self.request_dir / f"{name}.json"
        safe = dict(payload)
        safe["input"] = self._strip_data_urls(safe.get("input"))
        path.write_text(json.dumps(safe, indent=2, ensure_ascii=False) + "\n")
        return str(path)

    def _strip_data_urls(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            out = {}
            for key, value in obj.items():
                if key == "image_url" and isinstance(value, str) and value.startswith("data:image"):
                    out[key] = "<base64 image omitted from saved request>"
                else:
                    out[key] = self._strip_data_urls(value)
            return out
        if isinstance(obj, list):
            return [self._strip_data_urls(x) for x in obj]
        return obj

    def _responses_json(self, name: str, system_prompt: str, user_content: list[dict], schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        payload = {
            "model": self.model,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                {"role": "user", "content": user_content},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": name,
                    "schema": strictify(schema),
                }
            },
        }
        request_path = self._write_request(name, payload)
        if not self.enabled:
            return {"_api_status": "missing_api_key", "_request_path": request_path}
        try:
            from openai import OpenAI
        except Exception as exc:
            return {"_api_status": "openai_import_error", "_error": str(exc), "_request_path": request_path}

        client = OpenAI(api_key=self.api_key)
        try:
            response = client.responses.create(**payload)
        except Exception as exc:
            return {"_api_status": "api_call_error", "_error": str(exc), "_request_path": request_path}
        text = getattr(response, "output_text", None)
        if text is None:
            # Fallback for SDK response shapes that do not expose output_text.
            text = response.model_dump_json() if hasattr(response, "model_dump_json") else str(response)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"_api_status": "json_parse_error", "_raw_text": text, "_request_path": request_path}

    def parse_intent(self, prompt: str, image_path: str | None = None) -> Optional[Dict[str, Any]]:
        content: list[dict] = [
            {
                "type": "input_text",
                "text": (
                    "Parse this image+prompt into simulation-conditioned video controls.\n"
                    f"User prompt: {prompt}\n"
                    f"{OBJECT_PHYSICS_PROMPT_GUIDE}\n"
                    "Do not estimate numeric elastic_E or elastic_nu directly. "
                    "Return object_physics_semantics instead.\n"
                    "If the liquid identity is visually clear, set appearance.material to a concise canonical name "
                    "such as water, milk, cola, honey, oil, or lava.\n"
                    "Return only the requested JSON."
                ),
            }
        ]
        if image_path:
            try:
                content.append({"type": "input_image", "image_url": image_to_data_url(image_path)})
            except Exception as exc:
                content[0]["text"] += f"\nImage could not be encoded locally: {exc}"
        return self._responses_json("intent_physics_parse", INTENT_PHYSICS_SYSTEM_PROMPT, content, INTENT_SCHEMA)

    def judge_complexity(
        self,
        prompt: str,
        bundle: SimulationBundle,
        complexity_plan: ComplexityPlan | None = None,
    ) -> Optional[Dict[str, Any]]:
        samples = sample_bundle_videos(bundle, frame_interval=self.frame_interval, max_frames_per_video=self.max_frames_per_video)
        curve = []
        if complexity_plan is not None:
            for item in complexity_plan.transition_scores:
                if item.frame_index % self.frame_interval == 0:
                    curve.append(
                        {
                            "frame": item.frame_index,
                            "hybrid": item.group_scores.hybrid,
                            "physical": item.group_scores.physical,
                            "visual": item.group_scores.visual,
                            "semantic": item.group_scores.semantic,
                            "events": item.semantic_events,
                        }
                    )
        user_text = {
            "type": "input_text",
            "text": json.dumps(
                {
                    "user_prompt": prompt,
                    "frame_sampling": f"Every {self.frame_interval} frames from available simulation/flow/mask videos.",
                    "source_paths": {s.source_label: s.source_path for s in samples},
                    "sampled_frames": [
                        {"source": s.source_label, "frame_index": s.frame_index}
                        for s in samples
                    ],
                    "complexity_curve_sampled_every_n_frames": self.frame_interval,
                    "complexity_curve": curve,
                    "output_instruction": "Return risk/no-cut suggestions only; the metric planner makes final cuts.",
                },
                ensure_ascii=False,
            ),
        }
        content = [user_text]
        for sample in samples:
            content.append({"type": "input_text", "text": f"{sample.source_label}, frame {sample.frame_index}"})
            content.append({"type": "input_image", "image_url": sample.data_url})
        return self._responses_json("vlm_complexity_judge", VLM_COMPLEXITY_SYSTEM_PROMPT, content, VLM_COMPLEXITY_SCHEMA)

    def refine_clip_prompts(
        self,
        intent: IntentParseResult,
        clips: Sequence[ClipPlan],
        complexity_plan: ComplexityPlan | None = None,
        image_path: str | None = None,
        draft_prompts: Optional[Dict[int, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        events = []
        if complexity_plan is not None:
            for event in complexity_plan.event_timeline:
                events.append(
                    {
                        "name": event.name,
                        "description": event.description,
                        "importance": event.importance,
                        "frame_start": event.frame_start,
                        "frame_end": event.frame_end,
                    }
                )
        else:
            events = [
                {
                    "name": event.name,
                    "description": event.description,
                    "importance": event.importance,
                    "frame_start": event.frame_start,
                    "frame_end": event.frame_end,
                }
                for event in intent.key_events
            ]

        payload = {
            "task": "Refine clip prompts for direct use in a video generation model.",
            "user_prompt": intent.raw_prompt,
            "visual_prompt": intent.visual_prompt,
            "overall_prompt": intent.overall_prompt,
            "negative_prompt": intent.negative_prompt,
            "appearance": to_dict(intent.appearance),
            "dynamics_summary": {
                "speed": intent.dynamics.speed,
                "fluid_volume": intent.dynamics.fluid_volume,
                "gravity_mag": intent.dynamics.gravity_mag,
                "viscosity": intent.dynamics.viscosity,
                "material": intent.appearance.material,
            },
            "semantic_events": events,
            "clip_controls": [
                {
                    "clip_id": clip.clip_id,
                    "complexity_label": clip.complexity_label,
                    "risk_tags": clip.risk_tags,
                    "start_frame": clip.start_frame,
                    "end_frame": clip.end_frame,
                    "actual_frame_num": clip.actual_frame_num,
                    "vace_frame_num": clip.frame_num,
                    "anchor_strategy": clip.anchor_strategy,
                    "overlap_left": clip.overlap_left,
                    "overlap_right": clip.overlap_right,
                    "draft_agent_prompt": (draft_prompts or {}).get(clip.clip_id, clip.prompt),
                }
                for clip in clips
            ],
            "style_instruction": (
                "Return rich positive prompts suitable for video generation. Use the structured controls as hidden guidance, "
                "but do not write engineering labels such as frame numbers, VACE frame_num, complexity, risk cues, or anchor strategy. "
                "Make every clip share a consistent scene/style stem while varying only the clip-local action."
            ),
        }
        content: list[dict] = [{"type": "input_text", "text": json.dumps(payload, ensure_ascii=False)}]
        if image_path:
            try:
                content.append({"type": "input_image", "image_url": image_to_data_url(image_path)})
            except Exception as exc:
                payload["image_encoding_error"] = str(exc)
                content[0]["text"] = json.dumps(payload, ensure_ascii=False)
        return self._responses_json("prompt_refinement", PROMPT_REFINEMENT_SYSTEM_PROMPT, content, PROMPT_REFINEMENT_SCHEMA)
