"""Generation-time planning (Sec. 3.2 of the paper).

    intent_parser    prompt + image -> intent record I = {appearance, dynamics, events}
    complexity       simulation rollout -> temporal complexity C_t = w_phys C^phys + w_vis C^vis + w_sem C^sem
    vlm_judge        VLM risk intervals / no-cut flags / suggested cut frames (semantic term)
    clip_planner     boundary cost J(b) and greedy left-to-right boundary selection
    prompt_refiner   clip-level prompts p_i
    routing          clip plan -> per-backbone generation jobs (VACE / Tora)
    clip_executor, tora_clip_executor
                     run the jobs with the patched backbones (see backbones/)
    cli              `scripts/run_agent.py` entry point
"""

from .clip_planner import ClipPlannerSettings, ComplexityAwareClipPlanner
from .complexity import ComplexityConfig, SimulationComplexityAnalyzer
from .intent_parser import IntentPhysicsParser
from .openai_adapter import OpenAIAPIAdapter
from .prompt_refiner import PromptRefiner
from .routing import build_tora_clip_jobs, build_vace_clip_jobs
from .schema import (
    AppearanceSpec,
    ClipPlan,
    ComplexityPlan,
    ConstraintPriority,
    DynamicsSpec,
    IntentParseResult,
    SemanticEvent,
    SimulationBundle,
    StageCommand,
    VLMComplexitySuggestion,
)
from .vlm_judge import VLMComplexityJudge

__all__ = [
    "AppearanceSpec",
    "ClipPlan",
    "ClipPlannerSettings",
    "ComplexityAwareClipPlanner",
    "ComplexityConfig",
    "ComplexityPlan",
    "ConstraintPriority",
    "DynamicsSpec",
    "IntentParseResult",
    "IntentPhysicsParser",
    "OpenAIAPIAdapter",
    "PromptRefiner",
    "SemanticEvent",
    "SimulationBundle",
    "SimulationComplexityAnalyzer",
    "StageCommand",
    "VLMComplexityJudge",
    "VLMComplexitySuggestion",
    "build_vace_clip_jobs",
    "build_tora_clip_jobs",
]
