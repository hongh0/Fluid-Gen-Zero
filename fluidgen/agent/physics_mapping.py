"""Map qualitative object physics semantics to simulation-friendly elastic params."""

from __future__ import annotations

from typing import Any, Dict


OBJECT_PHYSICS_ENUMS: Dict[str, list[Any]] = {
    "material_class": [
        "metal",
        "wood",
        "hard_plastic",
        "soft_plastic_or_rubber",
        "glass_or_ceramic",
        "paper_or_cardboard",
        "foam",
        "stone_or_concrete",
        "other",
    ],
    "solidity": [
        "solid",
        "mostly_solid",
        "hollow_thin_shell",
        "layered_or_composite",
    ],
    "hardness_level": [
        "very_soft",
        "soft",
        "medium",
        "hard",
        "very_hard",
    ],
    "bounce_category": [
        "almost_no_bounce",
        "low_bounce",
        "medium_bounce",
        "high_bounce",
        "very_high_bounce",
    ],
    "friction_tendency": [
        "very_slippery",
        "slippery",
        "medium",
        "grippy",
        "very_grippy",
    ],
    "size_thickness_hint": [
        "thin_and_small",
        "thin_and_large",
        "thick_and_small",
        "thick_and_large",
    ],
}

DEFAULT_OBJECT_PHYSICS_SEMANTICS: Dict[str, Any] = {
    "material_class": "hard_plastic",
    "solidity": "solid",
    "hardness_level": "medium",
    "bounce_category": "low_bounce",
    "surface_roughness_level": 3,
    "friction_tendency": "medium",
    "size_thickness_hint": "thick_and_small",
    "justification": "Default conservative foreground-object prior.",
}
BASE_ELASTIC_E = {
    "soft_plastic_or_rubber": {
        "very_soft": 3e5,
        "soft": 8e5,
        "medium": 1.5e6,
        "hard": 5e6,
        "very_hard": 1e7,
    },
    "foam": {
        "very_soft": 1e5,
        "soft": 2e5,
        "medium": 5e5,
        "hard": 1e6,
        "very_hard": 2e6,
    },
    "paper_or_cardboard": {
        "very_soft": 5e6,
        "soft": 1e7,
        "medium": 2e7,
        "hard": 3.5e7,
        "very_hard": 5e7,
    },
    "wood": {
        "very_soft": 1e7,
        "soft": 2e7,
        "medium": 4e7,
        "hard": 6e7,
        "very_hard": 8e7,
    },
    "hard_plastic": {
        "very_soft": 8e6,
        "soft": 1.5e7,
        "medium": 5e7,
        "hard": 7.5e7,
        "very_hard": 1e8,
    },
    "glass_or_ceramic": {
        "very_soft": 2e7,
        "soft": 3.5e7,
        "medium": 5e7,
        "hard": 8e7,
        "very_hard": 1e8,
    },
    "stone_or_concrete": {
        "very_soft": 2e7,
        "soft": 3.5e7,
        "medium": 5e7,
        "hard": 8e7,
        "very_hard": 1e8,
    },
    "metal": {
        "very_soft": 3e7,
        "soft": 4e7,
        "medium": 5e7,
        "hard": 9e7,
        "very_hard": 1.2e8,
    },
    "other": {
        "very_soft": 3e5,
        "soft": 1e6,
        "medium": 1e7,
        "hard": 5e7,
        "very_hard": 8e7,
    },
}

SIZE_THICKNESS_MODIFIER = {
    "thin_and_small": 0.85,
    "thin_and_large": 0.70,
    "thick_and_small": 1.0,
    "thick_and_large": 1.10,
}

BASE_POISSON_RATIO = {
    "metal": 0.27,
    "wood": 0.26,
    "hard_plastic": 0.27,
    "soft_plastic_or_rubber": 0.43,
    "glass_or_ceramic": 0.27,
    "paper_or_cardboard": 0.28,
    "foam": 0.32,
    "stone_or_concrete": 0.26,
    "other": 0.27,
}

HARDNESS_NU_OFFSET = {
    "very_soft": 0.03,
    "soft": 0.02,
    "medium": 0.0,
    "hard": -0.01,
    "very_hard": -0.02,
}

MIN_ELASTIC_E = 8e4
MAX_ELASTIC_E = 1.2e8
MIN_ELASTIC_NU = 0.18
MAX_ELASTIC_NU = 0.47

OBJECT_PHYSICS_PROMPT_GUIDE = """
For object_physics_semantics, do not guess numeric Young's modulus or Poisson ratio.
Choose only the following qualitative labels, then local code will map them to simulation parameters.

material_class:
- "metal"
- "wood"
- "hard_plastic"
- "soft_plastic_or_rubber"
- "glass_or_ceramic"
- "paper_or_cardboard"
- "foam"
- "stone_or_concrete"
- "other"

solidity:
- "solid"
- "mostly_solid"
- "hollow_thin_shell"
- "layered_or_composite"

hardness_level:
- "very_soft"
- "soft"
- "medium"
- "hard"
- "very_hard"

bounce_category:
- "almost_no_bounce"
- "low_bounce"
- "medium_bounce"
- "high_bounce"
- "very_high_bounce"

surface_roughness_level:
- integer 1 to 5

friction_tendency:
- "very_slippery"
- "slippery"
- "medium"
- "grippy"
- "very_grippy"

size_thickness_hint:
- "thin_and_small"
- "thin_and_large"
- "thick_and_small"
- "thick_and_large"
""".strip()


LEGACY_OBJECT_PHYSICS_ALIASES: Dict[str, Dict[str, str]] = {
    "material_class": {
        "fabric_or_textile": "other",
    },
}


def _safe_choice(value: Any, key: str) -> Any:
    options = OBJECT_PHYSICS_ENUMS[key]
    if isinstance(value, str):
        value = LEGACY_OBJECT_PHYSICS_ALIASES.get(key, {}).get(value, value)
        if value in options:
            return value
    return DEFAULT_OBJECT_PHYSICS_SEMANTICS[key]


def normalize_object_physics_semantics(semantic_desc: Dict[str, Any] | None = None) -> Dict[str, Any]:
    data = dict(DEFAULT_OBJECT_PHYSICS_SEMANTICS)
    if isinstance(semantic_desc, dict):
        data.update(semantic_desc)

    normalized = {
        "material_class": _safe_choice(data.get("material_class"), "material_class"),
        "solidity": _safe_choice(data.get("solidity"), "solidity"),
        "hardness_level": _safe_choice(data.get("hardness_level"), "hardness_level"),
        "bounce_category": _safe_choice(data.get("bounce_category"), "bounce_category"),
        "friction_tendency": _safe_choice(data.get("friction_tendency"), "friction_tendency"),
        "size_thickness_hint": _safe_choice(data.get("size_thickness_hint"), "size_thickness_hint"),
        "justification": str(data.get("justification", DEFAULT_OBJECT_PHYSICS_SEMANTICS["justification"])),
    }

    roughness = data.get("surface_roughness_level", DEFAULT_OBJECT_PHYSICS_SEMANTICS["surface_roughness_level"])
    if isinstance(roughness, str):
        try:
            roughness = int(roughness)
        except ValueError:
            roughness = DEFAULT_OBJECT_PHYSICS_SEMANTICS["surface_roughness_level"]
    if not isinstance(roughness, int):
        roughness = DEFAULT_OBJECT_PHYSICS_SEMANTICS["surface_roughness_level"]
    normalized["surface_roughness_level"] = max(1, min(5, roughness))
    return normalized


def map_object_semantics_to_elastic_params(semantic_desc: Dict[str, Any] | None = None) -> tuple[float, float]:
    semantics = normalize_object_physics_semantics(semantic_desc)
    material_class = semantics["material_class"]
    hardness_level = semantics["hardness_level"]
    size_thickness_hint = semantics["size_thickness_hint"]

    base_E = BASE_ELASTIC_E.get(material_class, BASE_ELASTIC_E["other"]).get(
        hardness_level,
        BASE_ELASTIC_E["other"]["medium"],
    )
    elastic_E = base_E * SIZE_THICKNESS_MODIFIER.get(size_thickness_hint, 1.0)
    elastic_E = max(MIN_ELASTIC_E, min(MAX_ELASTIC_E, elastic_E))

    elastic_nu = BASE_POISSON_RATIO.get(material_class, BASE_POISSON_RATIO["other"]) + HARDNESS_NU_OFFSET.get(
        hardness_level,
        0.0,
    )
    elastic_nu = max(MIN_ELASTIC_NU, min(MAX_ELASTIC_NU, elastic_nu))
    return float(elastic_E), float(elastic_nu)
