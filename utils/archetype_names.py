"""Human-readable names for pitch-level archetypes (cluster metadata)."""
from __future__ import annotations

PITCH_ARCHETYPE_NAMES: dict[int, str] = {
    1: "Running Fastball",
    2: "Power Curve",
    3: "Rising Fastball",
    4: "Sweeping Slider",
    5: "Fading Changeup",
    6: "Heavy Sinker",
    7: "Riding Four-Seam",
    8: "Gyro Slider",
}


def get_pitch_archetype_name(archetype_id: int) -> str:
    return PITCH_ARCHETYPE_NAMES.get(archetype_id, f"Cluster {archetype_id}")
