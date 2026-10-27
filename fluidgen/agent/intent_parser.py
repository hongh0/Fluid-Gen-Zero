"""Intent and physics parser for image+prompt fluid-object generation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

from .physics_mapping import (
    DEFAULT_OBJECT_PHYSICS_SEMANTICS,
    map_object_semantics_to_elastic_params,
    normalize_object_physics_semantics,
)
from .schema import (
    AppearanceSpec,
    ConstraintPriority,
    DynamicsSpec,
    IntentParseResult,
    SemanticEvent,
    to_dict,
)


# Material *names* identify the liquid; *hints* are appearance descriptors that
# only apply when no name is present.
_MATERIALS = {
    "water": {"names": ["water", "清水"], "hints": ["transparent", "clear", "透明"], "viscosity": 1.0, "rho": 1000.0, "viscous": False},
    "lava": {"names": ["lava", "magma", "熔岩", "岩浆"], "hints": [], "viscosity": 10.0, "rho": 2600.0, "viscous": True},
    "oil": {"names": ["oil", "机油"], "hints": ["viscous", "油"], "viscosity": 5.0, "rho": 900.0, "viscous": True},
    "cola": {"names": ["cola", "coke", "可乐"], "hints": ["brown liquid", "褐色"], "viscosity": 1.4, "rho": 1040.0, "viscous": False},
    "milk": {"names": ["milk", "牛奶"], "hints": ["milky", "乳白"], "viscosity": 1.8, "rho": 1030.0, "viscous": False},
    "honey": {"names": ["honey", "syrup", "蜂蜜", "糖浆"], "hints": [], "viscosity": 15.0, "rho": 1400.0, "viscous": True},
    "paint": {"names": ["paint", "颜料", "油漆"], "hints": [], "viscosity": 4.0, "rho": 1300.0, "viscous": True},
    "juice": {"names": ["juice", "cider", "果汁"], "hints": [], "viscosity": 1.2, "rho": 1050.0, "viscous": False},
    "ink": {"names": ["ink", "dye", "墨水", "染料"], "hints": [], "viscosity": 1.1, "rho": 1050.0, "viscous": False},
    "slime": {"names": ["slime", "gel", "史莱姆", "凝胶"], "hints": [], "viscosity": 12.0, "rho": 1200.0, "viscous": True},
    "mud": {"names": ["mud", "sludge", "泥浆", "淤泥"], "hints": [], "viscosity": 8.0, "rho": 1600.0, "viscous": True},
    "mercury": {"names": ["mercury", "水银", "汞"], "hints": [], "viscosity": 1.5, "rho": 13534.0, "viscous": False},
}
for _cfg in _MATERIALS.values():  # kept so _resolve_material_name still works
    _cfg["aliases"] = [*_cfg["names"], *_cfg["hints"]]


def _alias_present(alias: str, lower_prompt: str) -> bool:
    """Word-boundary match for ASCII aliases, substring for CJK."""
    token = alias.lower()
    if re.fullmatch(r"[a-z0-9 .+-]+", token):
        return re.search(rf"(?<![a-z]){re.escape(token)}(?![a-z])", lower_prompt) is not None
    return token in lower_prompt


def _detect_material(prompt: str, default: str = "water") -> str:
    """Longest explicit material name wins; appearance hints are a fallback."""
    lower = prompt.lower()
    for tier in ("names", "hints"):
        best, best_len = None, 0
        for name, cfg in _MATERIALS.items():
            for alias in cfg[tier]:
                if _alias_present(alias, lower) and len(alias) > best_len:
                    best, best_len = name, len(alias)
        if best:
            return best
    return default


_LIGHTING = {
    "dramatic": ["dramatic", "cinematic", "强光", "戏剧", "电影"],
    "soft": ["soft", "diffuse", "柔和", "散射"],
    "backlit": ["backlit", "rim light", "逆光", "轮廓光"],
    "natural": ["natural", "daylight", "自然光", "日光"],
}

_BACKGROUND = {
    "tabletop studio": ["table", "desk", "桌", "台面"],
    "outdoor natural scene": ["outdoor", "grass", "beach", "户外", "草地", "海滩"],
    "minimal clean studio": ["studio", "white background", "棚拍", "白底", "干净背景"],
    "kitchen scene": ["kitchen", "厨房"],
}

_EVENT_RULES = [
    {
        "name": "impact",
        "description": "fluid first contacts or impacts the object/container",
        "keywords": ["impact", "hit", "collide", "contact", "接触", "撞", "冲击", "碰到", "倒入", "倒水"],
        "importance": 0.85,
        "signals": ["contact_intensity", "flow_p95_mag", "mask_silhouette_change"],
    },
    {
        "name": "splash",
        "description": "splash or droplets form with high local motion detail",
        "keywords": ["splash", "spray", "droplet", "飞溅", "溅起", "水花", "喷溅"],
        "importance": 1.0,
        "signals": ["fluid_p95_disp", "flow_discontinuity", "mask_fragmentation"],
    },
    {
        "name": "split_merge",
        "description": "fluid blob splits, merges, or wraps around an obstacle",
        "keywords": ["split", "merge", "separate", "around", "绕过", "分裂", "合并", "包围", "绕流"],
        "importance": 0.9,
        "signals": ["mask_fragmentation", "fluid_spread_change", "flow_discontinuity"],
    },
    {
        "name": "swirl",
        "description": "swirling or turbulent fluid motion is visually important",
        "keywords": ["swirl", "vortex", "turbulent", "whirlpool", "旋涡", "涡流", "湍流", "翻滚"],
        "importance": 0.85,
        "signals": ["flow_curl_abs", "flow_var_mag", "fluid_velocity_var"],
    },
    {
        "name": "occlusion",
        "description": "rapid occlusion or boundary crossing between object and fluid",
        "keywords": ["occlusion", "cover", "behind", "遮挡", "覆盖", "穿过", "前方"],
        "importance": 0.75,
        "signals": ["mask_boundary_density", "mask_silhouette_change", "contact_change"],
    },
    {
        "name": "material_detail",
        "description": "material and appearance fidelity are both strongly constrained",
        "keywords": ["transparent", "clear", "viscous", "glossy", "reflect", "refract", "透明", "粘稠", "反射", "折射", "褐色", "熔岩"],
        "importance": 0.7,
        "signals": ["flow_mean_mag", "mask_boundary_density"],
    },
]


#: Alignment sets Q^evt_j: the simulation signals that localise an event of
#: each type on the rollout timeline. "default" is the fallback row for events
#: whose type is not recognised.
DEFAULT_ALIGNMENT_SET = ["contact_intensity", "flow_mean_mag", "fluid_spread_change"]
EVENT_ALIGNMENT_SETS = {rule["name"]: list(rule["signals"]) for rule in _EVENT_RULES}
EVENT_ALIGNMENT_SETS["default"] = DEFAULT_ALIGNMENT_SET

# Word stems that identify an event type from its name, checked in order. The
# VLM names events freely ("outward_splash", "swirling_eddies"); anything that
# names none of the types (ripples, waves, drift, ...) uses the default row.
_EVENT_TYPE_STEMS = [
    # occlusion before split_merge: "submerge" contains "merge".
    ("occlusion", ["occlu", "cover", "submer", "burial", "buried", "hidden", "behind", "fog", "遮挡", "覆盖", "淹没"]),
    ("split_merge", ["split", "merge", "separat", "wrap", "分裂", "合并", "绕"]),
    ("splash", ["splash", "spray", "droplet", "erupt", "spatter", "飞溅", "溅", "水花"]),
    ("swirl", ["swirl", "vortex", "vortic", "eddy", "eddies", "whirl", "turbul", "spiral", "旋涡", "涡", "湍流"]),
    ("impact", ["impact", "contact", "collid", "collision", "hit", "strike", "pour", "接触", "撞", "冲击", "倒入"]),
    ("material_detail", ["material", "detail", "texture", "coating", "gloss", "reflect", "refract", "transparen",
                         "viscous", "透明", "粘稠", "反射", "折射"]),
]


def event_type(name: str, description: str = "", keywords: list[str] | None = None) -> str:
    """Canonical type of an event (a key of EVENT_ALIGNMENT_SETS), from its name.

    Only the name is used: descriptions mention other phenomena in passing
    ("ripples spreading around the object"), which would mislabel the event.
    Negated names such as "no_splash_behavior" fall back to the default row.
    """
    norm = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "_", (name or "").lower()).strip("_")
    tokens = [t for t in norm.split("_") if t]
    if not tokens or "no" in tokens or "without" in tokens:
        return "default"
    if norm in EVENT_ALIGNMENT_SETS:
        return norm
    for kind, stems in _EVENT_TYPE_STEMS:
        if any(stem in token for token in tokens for stem in stems):
            return kind
    return "default"


def with_alignment_set(event: SemanticEvent) -> SemanticEvent:
    """Give an event the alignment set of its type (replaces any free-form signals)."""
    event.preferred_signals = list(
        EVENT_ALIGNMENT_SETS[event_type(event.name, event.description, event.keywords)])
    return event


def _contains(prompt: str, aliases: list[str]) -> bool:
    lower = prompt.lower()
    return any(alias.lower() in lower for alias in aliases)


def _resolve_material_name(value: Any, default: str = "water") -> str:
    if not isinstance(value, str):
        return default
    lower = value.lower()
    best_name = default
    best_score = -1
    for name, cfg in _MATERIALS.items():
        for alias in [name, *cfg["aliases"]]:
            alias_lower = alias.lower()
            if alias_lower == lower:
                return name
            if alias_lower in lower or lower in alias_lower:
                if len(alias_lower) > best_score:
                    best_name = name
                    best_score = len(alias_lower)
    return best_name


def _pick(mapping: Dict[str, list[str]], prompt: str, default: str) -> str:
    for key, aliases in mapping.items():
        if _contains(prompt, aliases):
            return key
    return default


def _parse_force_direction(prompt: str) -> Optional[list[float]]:
    """Return (right, forward, down) in the *scene* basis, not world axes.
    """
    lower = prompt.lower()
    if any(x in lower for x in [
        "falls into", "fall into", "falling", "drops into", "drop into", "dropped",
        "sinks", "sinking", "plunge", "poured", "pour ", "落入", "掉入", "坠", "下沉", "倒入",
    ]):
        return [0.0, 0.0, 1.0]
    if any(x in lower for x in ["rises", "rising", "floats up", "surfaces", "上浮", "浮起", "升起"]):
        return [0.0, 0.0, -1.0]
    if any(x in lower for x in ["left", "向左", "往左"]):
        return [-1.0, 0.0, 0.0]
    if any(x in lower for x in ["right", "向右", "往右"]):
        return [1.0, 0.0, 0.0]
    if any(x in lower for x in ["forward", "away", "向前", "往前"]):
        return [0.0, 1.0, 0.0]
    if any(x in lower for x in ["backward", "towards camera", "向后", "靠近镜头"]):
        return [0.0, -1.0, 0.0]
    return None


def _parse_speed(prompt: str) -> tuple[str, Optional[float]]:
    lower = prompt.lower()
    if any(x in lower for x in ["slow", "gentle", "缓慢", "慢慢", "轻柔"]):
        return "slow", 0.6
    if any(x in lower for x in ["fast", "rapid", "splash", "快速", "飞溅", "猛烈"]):
        return "fast", 2.0
    return "medium", 1.0


_OBJECT_MATERIAL_KEYWORDS = {
    "soft_plastic_or_rubber": ["rubber", "squishy", "rubber duck", "橡胶", "软胶"],
    "glass_or_ceramic": ["glass", "ceramic", "porcelain", "玻璃", "陶瓷", "瓷"],
    "metal": ["metal", "steel", "iron", "brass", "coin", "金属", "钢", "铁", "硬币"],
    "wood": ["wood", "wooden", "木"],
    "paper_or_cardboard": ["paper", "cardboard", "lantern", "纸", "灯笼"],
    "foam": ["foam", "sponge", "泡沫", "海绵"],
    "stone_or_concrete": ["stone", "concrete", "rock", "石", "混凝土"],
    "hard_plastic": ["plastic", "塑料"],
}

_OBJECT_HARDNESS_KEYWORDS = {
    "very_soft": ["very soft", "极软"],
    "soft": ["soft", "squishy", "软"],
    "hard": ["hard", "rigid", "刚性"],
    "very_hard": ["very hard", "极硬"],
}


def _longest_keyword_match(prompt: str, table: Dict[str, list[str]]) -> Optional[str]:
    """Longest matching keyword wins, independent of table order."""
    lower = prompt.lower()
    best, best_len = None, 0
    for label, keywords in table.items():
        for keyword in keywords:
            token = keyword.lower()
            if token in lower and len(token) > best_len:
                best, best_len = label, len(token)
    return best


def _parse_object_semantics(prompt: str) -> Dict[str, Any]:
    """Rule-path equivalent of the VLM's object_physics_semantics block.

    Both paths feed the same physics_mapping table, so turning the LLM on
    changes the *evidence*, not the parameter scale.
    """
    semantics = dict(DEFAULT_OBJECT_PHYSICS_SEMANTICS)
    material_class = _longest_keyword_match(prompt, _OBJECT_MATERIAL_KEYWORDS)
    if material_class:
        semantics["material_class"] = material_class
    hardness = _longest_keyword_match(prompt, _OBJECT_HARDNESS_KEYWORDS)
    if hardness:
        semantics["hardness_level"] = hardness
    elif material_class in {"soft_plastic_or_rubber", "foam"}:
        semantics["hardness_level"] = "soft"
    semantics["justification"] = "Rule-based keyword parse of the prompt."
    return normalize_object_physics_semantics(semantics)


def _parse_object_elastic(prompt: str) -> tuple[float, float]:
    """Elastic parameters for the object material named in the prompt."""
    return map_object_semantics_to_elastic_params(_parse_object_semantics(prompt))


def _extract_key_events(prompt: str) -> list[SemanticEvent]:
    events: list[SemanticEvent] = []
    for rule in _EVENT_RULES:
        matched = [kw for kw in rule["keywords"] if kw.lower() in prompt.lower()]
        if not matched:
            continue
        events.append(
            SemanticEvent(
                name=rule["name"],
                description=rule["description"],
                importance=float(rule["importance"]),
                keywords=matched,
                preferred_signals=list(rule["signals"]),
                confidence=0.5,
            )
        )
    if not events:
        events.append(
            SemanticEvent(
                name="general_fluid_object_interaction",
                description="fluid-object motion should remain physically plausible and temporally stable",
                importance=0.45,
                keywords=[],
                preferred_signals=list(DEFAULT_ALIGNMENT_SET),
                confidence=0.25,
            )
        )
    return events


def _constraint_priority(prompt: str, appearance: AppearanceSpec, dynamics: DynamicsSpec, events: list[SemanticEvent]) -> ConstraintPriority:
    physical = 0.35
    visual = 0.35
    semantic = 0.20
    temporal = 0.10
    hard = [
        "Respect simulation optical flow and masks as primary motion constraints.",
        "Preserve object identity and object-fluid contact geometry.",
        "Do not hallucinate new objects unless the prompt explicitly asks for them.",
    ]
    soft = [
        f"Maintain {appearance.material} material appearance.",
        f"Use {appearance.lighting} and {appearance.background} consistently.",
    ]

    high_event = any(e.importance >= 0.85 for e in events)
    if dynamics.speed == "fast" or high_event:
        physical += 0.10
        temporal += 0.05
        visual -= 0.05
    if appearance.material in {"water", "cola", "lava", "honey", "oil"}:
        visual += 0.05
    if any(e.name in {"impact", "splash", "split_merge"} for e in events):
        semantic += 0.07
    if _contains(prompt, ["highly detailed", "精细", "细节", "photorealistic", "真实"]):
        visual += 0.08
    if _contains(prompt, ["dramatic", "关键", "important", "must", "必须"]):
        semantic += 0.08

    return ConstraintPriority(
        physical=max(0.05, physical),
        visual=max(0.05, visual),
        semantic=max(0.05, semantic),
        temporal=max(0.05, temporal),
        hard_constraints=hard,
        soft_constraints=soft,
    )


class LLMUnavailableError(RuntimeError):
    """Raised when LLM parsing is required but produced no usable output."""


class IntentPhysicsParser:
    """Rule-based parser with optional JSON override hooks for future VLM/LLM calls."""

    def parse(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        llm_override_json: Optional[str] = None,
    ) -> IntentParseResult:
        material = _detect_material(prompt)

        lighting = _pick(_LIGHTING, prompt, "natural")
        background = _pick(_BACKGROUND, prompt, "clean realistic background")
        style = "photorealistic"
        if _contains(prompt, ["anime", "cartoon", "卡通", "动漫"]):
            style = "stylized animation"
        elif _contains(prompt, ["macro", "close-up", "微距", "特写"]):
            style = "macro photorealistic"

        speed, push_force = _parse_speed(prompt)
        viscosity = float(_MATERIALS[material]["viscosity"])
        elastic_E, elastic_nu = _parse_object_elastic(prompt)
        gravity_mag = 1.6 if _contains(prompt, ["moon", "low gravity", "低重力", "月球"]) else 9.8
        if _contains(prompt, ["heavy gravity", "高重力"]):
            gravity_mag = 14.0

        volume = "medium"
        if _contains(prompt, ["tiny", "small amount", "少量", "一小股"]):
            volume = "small"
        elif _contains(prompt, ["large amount", "flood", "大量", "洪水"]):
            volume = "large"

        appearance = AppearanceSpec(
            material=material,
            lighting=f"{lighting} lighting",
            background=background,
            style=style,
        )
        dynamics = DynamicsSpec(
            viscosity=viscosity,
            gravity_mag=gravity_mag,
            fluid_volume=volume,
            speed=speed,
            elastic_E=elastic_E,
            elastic_nu=elastic_nu,
            liquid_rho=float(_MATERIALS[material]["rho"]),
            liquid_viscous=bool(_MATERIALS[material]["viscous"]),
            push_force=push_force,
            force_direction=_parse_force_direction(prompt),
            force_frame="scene",
        )
        key_events = _extract_key_events(prompt)
        constraint_priority = _constraint_priority(prompt, appearance, dynamics, key_events)

        result = IntentParseResult(
            raw_prompt=prompt,
            image_path=image_path,
            appearance=appearance,
            dynamics=dynamics,
            constraint_priority=constraint_priority,
            key_events=key_events,
            visual_prompt=self.build_visual_prompt(prompt, appearance),
            overall_prompt="",
            notes=[
                "Rule-based parse; pass --llm_override_json to inject VLM/LLM output later.",
                "elastic_E/elastic_nu control the foreground object; liquid_rho/liquid_viscous control the fluid.",
            ],
        )
        result.overall_prompt = self.build_overall_prompt(result)

        if llm_override_json:
            result = self.apply_override(result, llm_override_json)
        return result

    def parse_with_openai(
        self,
        prompt: str,
        image_path: Optional[str],
        openai_adapter,
        llm_override_json: Optional[str] = None,
        require: bool = True,
    ) -> IntentParseResult:
        """Parse with the VLM on top of the rule-based parse.

        With `require=True` a missing key or failed call raises; otherwise the
        rule-based parse is returned with a note. The adapter writes a sanitized
        request JSON either way.
        """
        result = self.parse(prompt, image_path=image_path, llm_override_json=llm_override_json)
        if openai_adapter is None:
            if require:
                raise LLMUnavailableError("No OpenAI adapter provided but LLM parsing is required.")
            return result
        data = openai_adapter.parse_intent(prompt, image_path=image_path)
        if not data:
            if require:
                raise LLMUnavailableError("OpenAI parser returned no data.")
            result.notes.append("OpenAI parser returned no data; using rule-based parse.")
            return result
        status = data.get("_api_status")
        if status and require:
            raise LLMUnavailableError(
                f"OpenAI intent parser failed ({status}): {data.get('_error', '')} "
                f"request={data.get('_request_path')}.")
        if status:
            result.notes.append(
                f"OpenAI parser not applied ({status}); request_path={data.get('_request_path')}. "
                "Using rule-based parse fallback."
            )
            return result
        prepared = self._prepare_openai_payload(result, data)
        return self.apply_payload(result, prepared, note="Applied OpenAI intent parser output with local semantic physics mapping.")

    @staticmethod
    def build_visual_prompt(prompt: str, appearance: AppearanceSpec) -> str:
        return (
            f"{prompt}. Fluid appearance: {appearance.material}, {appearance.style}, "
            f"{appearance.lighting}, {appearance.background}. Preserve object identity, "
            "stable geometry, coherent liquid surface, realistic reflections/refractions, and consistent boundaries."
        )

    @staticmethod
    def build_overall_prompt(result: IntentParseResult) -> str:
        d = result.dynamics
        event_text = "; ".join(e.description for e in result.key_events[:4])
        return (
            f"{result.visual_prompt} Motion should follow a {d.speed} {result.appearance.material} flow "
            f"with gravity {d.gravity_mag:g} m/s^2 and {d.fluid_volume} fluid volume. "
            f"Important events: {event_text}."
        )

    def apply_override(self, result: IntentParseResult, override_path: str) -> IntentParseResult:
        data = json.loads(Path(override_path).read_text())
        return self.apply_payload(result, data, note=f"Applied parser override from {override_path}.")

    def _prepare_openai_payload(self, result: IntentParseResult, data: Dict[str, Any]) -> Dict[str, Any]:
        payload: Dict[str, Any] = dict(data)
        appearance = dict(payload.get("appearance", {}))
        dynamics = dict(payload.get("dynamics", {}))
        extra = dynamics.get("extra")
        if not isinstance(extra, dict):
            extra = {}

        semantics = data.get("object_physics_semantics")
        if isinstance(semantics, dict):
            normalized_semantics = normalize_object_physics_semantics(semantics)
            elastic_E, elastic_nu = map_object_semantics_to_elastic_params(normalized_semantics)
            dynamics["elastic_E"] = elastic_E
            dynamics["elastic_nu"] = elastic_nu
            extra["object_physics_semantics"] = normalized_semantics

        raw_material = appearance.get("material")
        parsed_material = _resolve_material_name(raw_material, default="")
        sim_material = parsed_material or result.appearance.material
        if parsed_material:
            appearance["material"] = parsed_material
        elif not isinstance(raw_material, str) or not raw_material.strip():
            appearance["material"] = result.appearance.material

        if sim_material in _MATERIALS:
            dynamics["liquid_rho"] = float(_MATERIALS[sim_material]["rho"])
            dynamics["liquid_viscous"] = bool(_MATERIALS[sim_material]["viscous"])
            viscosity = dynamics.get("viscosity")
            if not isinstance(viscosity, (int, float)) or viscosity <= 0:
                dynamics["viscosity"] = float(_MATERIALS[sim_material]["viscosity"])

        if extra:
            dynamics["extra"] = extra
        payload["appearance"] = appearance
        payload["dynamics"] = dynamics

        notes = payload.get("notes")
        if isinstance(notes, list):
            payload["notes"] = list(notes)
            if isinstance(semantics, dict):
                payload["notes"].append(
                    "Mapped object_physics_semantics to elastic_E/elastic_nu locally instead of trusting direct VLM estimates."
                )
            if sim_material in _MATERIALS:
                payload["notes"].append(
                    f"Derived liquid_rho/liquid_viscous from fluid material prior '{sim_material}'."
                )
        return payload

    def apply_payload(self, result: IntentParseResult, data: Dict[str, Any], note: str) -> IntentParseResult:
        grounding = data.get("object_grounding")
        if isinstance(grounding, dict):
            g = result.object_grounding
            for key in ("object_prompt", "object_kind", "size_evidence", "confidence"):
                if isinstance(grounding.get(key), str) and grounding[key].strip():
                    setattr(g, key, grounding[key].strip())
            size = grounding.get("object_size_m")
            # a nonsense size is worse than none: it rescales the whole scene
            if isinstance(size, (int, float)) and 1e-3 < float(size) < 1e3:
                g.object_size_m = float(size)
            else:
                result.notes.append(f"ignored implausible object_size_m {size!r} from the parser")

        for section, target in [("appearance", result.appearance), ("dynamics", result.dynamics)]:
            for key, value in data.get(section, {}).items():
                if hasattr(target, key):
                    setattr(target, key, value)
                else:
                    target.extra[key] = value
        if "constraint_priority" in data:
            for key, value in data["constraint_priority"].items():
                if hasattr(result.constraint_priority, key):
                    setattr(result.constraint_priority, key, value)
        if "key_events" in data:
            result.key_events = [with_alignment_set(SemanticEvent(**event)) for event in data["key_events"]]
        if "visual_prompt" in data:
            result.visual_prompt = data["visual_prompt"]
        else:
            result.visual_prompt = self.build_visual_prompt(result.raw_prompt, result.appearance)
        result.overall_prompt = data.get("overall_prompt", self.build_overall_prompt(result))
        if "negative_prompt" in data:
            result.negative_prompt = data["negative_prompt"]
        if "notes" in data:
            result.notes.extend(str(x) for x in data["notes"])
        result.notes.append(note)
        return result

    @staticmethod
    def export_sim_config(result: IntentParseResult) -> Dict[str, Any]:
        return {
            "appearance": to_dict(result.appearance),
            "dynamics": to_dict(result.dynamics),
            "constraint_priority": to_dict(result.constraint_priority),
            "key_events": to_dict(result.key_events),
            "sim_pipeline_overrides": result.dynamics.to_sim_pipeline_overrides(),
            "visual_prompt": result.visual_prompt,
            "overall_prompt": result.overall_prompt,
        }
