"""Shared data structures for the fluid-object agentic framework."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class AppearanceSpec:
    material: str = "water"
    lighting: str = "natural studio lighting"
    background: str = "clean realistic background"
    style: str = "photorealistic"
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DynamicsSpec:
    viscosity: float = 1.0
    gravity_mag: float = 9.8
    fluid_volume: str = "medium"
    speed: str = "medium"
    elastic_E: float = 5e7
    elastic_nu: float = 0.27
    liquid_rho: float = 1000.0
    liquid_viscous: bool = False
    push_force: Optional[float] = None
    force_direction: Optional[List[float]] = None
    force_frame: str = "scene"
    dt: float = 1e-3
    substeps: int = 30
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_sim_pipeline_overrides(self) -> Dict[str, Any]:
        """Return the simulation-facing fields of the dynamics spec."""
        out: Dict[str, Any] = {
            "gravity_mag": self.gravity_mag,
            "dt": self.dt,
            "substeps": self.substeps,
            "elastic_E": self.elastic_E,
            "elastic_nu": self.elastic_nu,
            "liquid_rho": self.liquid_rho,
            "liquid_viscous": self.liquid_viscous,
        }
        if self.push_force is not None:
            out["push_force"] = self.push_force
        if self.force_direction is not None:
            out["force_direction"] = self.force_direction
            out["force_frame"] = self.force_frame
        out.update(self.extra.get("sim_pipeline_overrides", {}))
        return out


@dataclass
class ConstraintPriority:
    physical: float = 0.35
    visual: float = 0.35
    semantic: float = 0.20
    temporal: float = 0.10
    hard_constraints: List[str] = field(default_factory=list)
    soft_constraints: List[str] = field(default_factory=list)

    def normalized_complexity_weights(self) -> Dict[str, float]:
        total = max(1e-6, self.physical + self.visual + self.semantic)
        return {
            "physical": self.physical / total,
            "visual": self.visual / total,
            "semantic": self.semantic / total,
        }


@dataclass
class SemanticEvent:
    name: str
    description: str
    importance: float = 0.5
    keywords: List[str] = field(default_factory=list)
    preferred_signals: List[str] = field(default_factory=list)
    frame_start: Optional[int] = None
    frame_end: Optional[int] = None
    confidence: float = 0.0


@dataclass
class ObjectGrounding:
    """What the reconstruction needs to know about the solid object.
    """

    object_prompt: str = ""          # noun phrase for the text-grounded segmenter
    object_size_m: Optional[float] = None
    object_kind: str = "rigid"       # rigid, or one of the MPM solids
    size_evidence: str = ""
    confidence: str = "low"


@dataclass
class IntentParseResult:
    raw_prompt: str
    image_path: Optional[str]
    appearance: AppearanceSpec
    dynamics: DynamicsSpec
    constraint_priority: ConstraintPriority
    key_events: List[SemanticEvent]
    visual_prompt: str
    overall_prompt: str
    negative_prompt: str = "deformed objects, disappearing objects, inconsistent fluid, flicker, hallucinated objects"
    object_grounding: "ObjectGrounding" = field(default_factory=lambda: ObjectGrounding())
    notes: List[str] = field(default_factory=list)


@dataclass
class StageCommand:
    name: str
    command: List[str]
    cwd: str
    outputs: Dict[str, str] = field(default_factory=dict)
    env: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    returncode: Optional[int] = None


@dataclass
class SimulationBundle:
    root_dir: str
    fg_traj_path: Optional[str] = None
    fluid_traj_path: Optional[str] = None
    flow_npy_path: Optional[str] = None
    flow_video_path: Optional[str] = None
    sim_video_path: Optional[str] = None
    segmap_dir: Optional[str] = None
    fg_mask_video_path: Optional[str] = None
    fluid_mask_video_path: Optional[str] = None
    segmap_video_path: Optional[str] = None
    warped_reference_dir: Optional[str] = None
    warped_reference_video_path: Optional[str] = None
    warped_reference_mask_path: Optional[str] = None
    vace_condition_dir: Optional[str] = None
    vace_src_video_path: Optional[str] = None
    vace_src_mask_path: Optional[str] = None

    @classmethod
    def from_dir(cls, root_dir: str) -> "SimulationBundle":
        root = Path(root_dir)
        segmap = root / "segmap"
        warped = root / "warped_reference"
        vace_condition = root / "vace_condition"
        return cls(
            root_dir=str(root),
            fg_traj_path=str(root / "fg_traj.npz") if (root / "fg_traj.npz").exists() else None,
            fluid_traj_path=str(root / "fluid_traj.npz") if (root / "fluid_traj.npz").exists() else None,
            flow_npy_path=str(root / "optical_flow_flow.npy") if (root / "optical_flow_flow.npy").exists() else None,
            flow_video_path=str(root / "optical_flow.mp4") if (root / "optical_flow.mp4").exists() else None,
            sim_video_path=str(root / "simulation.mp4") if (root / "simulation.mp4").exists() else None,
            segmap_dir=str(segmap) if segmap.exists() else None,
            fg_mask_video_path=str(segmap / "fg_mask.mp4") if (segmap / "fg_mask.mp4").exists() else None,
            fluid_mask_video_path=str(segmap / "fluid_mask.mp4") if (segmap / "fluid_mask.mp4").exists() else None,
            segmap_video_path=str(segmap / "segmap.mp4") if (segmap / "segmap.mp4").exists() else None,
            warped_reference_dir=str(warped) if warped.exists() else None,
            warped_reference_video_path=str(warped / "motion_signal.mp4") if (warped / "motion_signal.mp4").exists() else None,
            warped_reference_mask_path=str(warped / "mask.mp4") if (warped / "mask.mp4").exists() else None,
            vace_condition_dir=str(vace_condition) if vace_condition.exists() else None,
            vace_src_video_path=str(vace_condition / "src_video-firstframe_ctrl.mp4") if (vace_condition / "src_video-firstframe_ctrl.mp4").exists() else None,
            vace_src_mask_path=str(vace_condition / "src_mask-firstframe_ctrl.mp4") if (vace_condition / "src_mask-firstframe_ctrl.mp4").exists() else None,
        )


@dataclass
class ComplexityGroupScore:
    physical: float = 0.0
    visual: float = 0.0
    semantic: float = 0.0
    hybrid: float = 0.0


@dataclass
class FrameComplexity:
    frame_index: int
    score: float
    group_scores: ComplexityGroupScore = field(default_factory=ComplexityGroupScore)
    components: Dict[str, float] = field(default_factory=dict)
    semantic_events: List[str] = field(default_factory=list)


@dataclass
class VLMComplexitySuggestion:
    frame_start: int
    frame_end: int
    risk: str
    reason: str
    no_cut: bool = False
    suggested_cut_frames: List[int] = field(default_factory=list)
    confidence: float = 0.5


@dataclass
class ClipPlan:
    clip_id: int
    start_frame: int
    end_frame: int
    actual_frame_num: int
    frame_num: int
    complexity_score: float
    complexity_label: str
    prompt: str
    risk_tags: List[str] = field(default_factory=list)
    anchor_strategy: str = "first_frame_anchor"
    overlap_left: int = 0
    overlap_right: int = 0
    cut_reason: str = "complexity-aware"
    notes: List[str] = field(default_factory=list)


@dataclass
class ComplexityPlan:
    total_frames: int
    transition_scores: List[FrameComplexity]
    clips: List[ClipPlan]
    thresholds: Dict[str, float]
    weights: Dict[str, float]
    event_timeline: List[SemanticEvent] = field(default_factory=list)
    vlm_suggestions: List[VLMComplexitySuggestion] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


def to_dict(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, list):
        return [to_dict(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(to_dict(x) for x in obj)
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    return obj
