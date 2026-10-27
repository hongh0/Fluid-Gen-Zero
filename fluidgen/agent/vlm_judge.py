"""VLM/LLM adapter for high-level simulation complexity judgement.

This module intentionally does not call a remote API. It creates a structured
request that can be sent to a VLM, and parses the VLM's JSON response back into
planner suggestions. Metrics remain the source of truth; VLM suggestions are
used as semantic no-cut/risk hints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .schema import ComplexityPlan, SimulationBundle, VLMComplexitySuggestion, to_dict


class VLMComplexityJudge:
    @staticmethod
    def build_request(
        prompt: str,
        bundle: SimulationBundle,
        complexity_plan: ComplexityPlan | None = None,
        sampled_frames: Sequence[int] | None = None,
    ) -> Dict[str, Any]:
        curve_preview = []
        if complexity_plan is not None:
            for item in complexity_plan.transition_scores:
                if item.frame_index % max(1, len(complexity_plan.transition_scores) // 24 or 1) == 0:
                    curve_preview.append(
                        {
                            "frame": item.frame_index,
                            "hybrid": item.group_scores.hybrid,
                            "physical": item.group_scores.physical,
                            "visual": item.group_scores.visual,
                            "semantic": item.group_scores.semantic,
                            "events": item.semantic_events,
                        }
                    )
        return {
            "task": "Judge which simulation intervals are high-risk for a video generator.",
            "user_prompt": prompt,
            "simulation_assets": to_dict(bundle),
            "sampled_frames": list(sampled_frames or []),
            "complexity_curve_preview": curve_preview,
            "instructions": [
                "Use preview frames/videos plus the complexity curve.",
                "Identify intervals with chaotic fluid motion, topology split/merge, splash, occlusion, or difficult object-fluid contact.",
                "Do not make the final segmentation alone; provide hints for the metric-based planner.",
                "Prefer no_cut=true for event interiors such as active splash or split/merge.",
            ],
            "expected_json_schema": {
                "suggestions": [
                    {
                        "frame_start": 12,
                        "frame_end": 28,
                        "risk": "high",
                        "reason": "major splash and thin droplets",
                        "no_cut": True,
                        "suggested_cut_frames": [8, 32],
                        "confidence": 0.8,
                    }
                ]
            },
        }

    @staticmethod
    def write_request(path: str | Path, request: Dict[str, Any]) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(request, indent=2, ensure_ascii=False) + "\n")

    @staticmethod
    def load_suggestions(path: str | Path | None) -> List[VLMComplexitySuggestion]:
        if not path:
            return []
        data = json.loads(Path(path).read_text())
        raw_items = data.get("suggestions", data if isinstance(data, list) else [])
        suggestions: list[VLMComplexitySuggestion] = []
        for item in raw_items:
            suggestions.append(
                VLMComplexitySuggestion(
                    frame_start=int(item["frame_start"]),
                    frame_end=int(item["frame_end"]),
                    risk=str(item.get("risk", "medium")),
                    reason=str(item.get("reason", "VLM suggested high complexity interval")),
                    no_cut=bool(item.get("no_cut", False)),
                    suggested_cut_frames=[int(x) for x in item.get("suggested_cut_frames", [])],
                    confidence=float(item.get("confidence", 0.5)),
                )
            )
        return suggestions

    @staticmethod
    def suggestions_from_payload(data: Dict[str, Any] | None) -> List[VLMComplexitySuggestion]:
        if not data:
            return []
        if data.get("_api_status"):
            return []
        suggestions: list[VLMComplexitySuggestion] = []
        for item in data.get("suggestions", []):
            suggestions.append(
                VLMComplexitySuggestion(
                    frame_start=int(item["frame_start"]),
                    frame_end=int(item["frame_end"]),
                    risk=str(item.get("risk", "medium")),
                    reason=str(item.get("reason", "OpenAI VLM suggested high complexity interval")),
                    no_cut=bool(item.get("no_cut", False)),
                    suggested_cut_frames=[int(x) for x in item.get("suggested_cut_frames", [])],
                    confidence=float(item.get("confidence", 0.5)),
                )
            )
        return suggestions

    @staticmethod
    def judge_with_openai(
        prompt: str,
        bundle: SimulationBundle,
        complexity_plan: ComplexityPlan,
        openai_adapter,
    ) -> tuple[List[VLMComplexitySuggestion], Dict[str, Any] | None]:
        if openai_adapter is None:
            return [], None
        data = openai_adapter.judge_complexity(prompt, bundle, complexity_plan)
        return VLMComplexityJudge.suggestions_from_payload(data), data
