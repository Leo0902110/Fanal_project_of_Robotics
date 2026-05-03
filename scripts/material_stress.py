from __future__ import annotations

from src.perception import build_pseudo_blur_config


MATERIAL_STRESS_PROFILE_CHOICES = ("auto", "mild", "transparent", "dark", "reflective", "low_texture")
PSEUDO_BLUR_PROFILE_CHOICES = ("mild", "transparent", "dark", "reflective", "low_texture")
OBJECT_PROFILE_CHOICES = ("default", "transparent", "dark", "reflective", "low_texture")


def resolve_material_stress_profile(object_profile: str, requested_profile: str = "auto") -> str:
    object_profile = str(object_profile or "default").strip().lower()
    requested_profile = str(requested_profile or "auto").strip().lower()
    if requested_profile != "auto":
        return requested_profile
    return object_profile if object_profile != "default" else "mild"


def build_scene_blur_config(
    *,
    scene: str,
    seed: int,
    object_profile: str = "default",
    pseudo_blur_profile: str = "mild",
    pseudo_blur_severity: float = 1.0,
    material_visual_stress: bool = False,
    material_stress_profile: str = "auto",
):
    scene = str(scene).strip().lower()
    enabled = scene == "pseudo_blur" or bool(material_visual_stress)
    profile = pseudo_blur_profile
    if material_visual_stress:
        profile = resolve_material_stress_profile(object_profile, material_stress_profile)
    return build_pseudo_blur_config(
        enabled=enabled,
        seed=seed,
        profile=profile,
        severity=pseudo_blur_severity,
    )
