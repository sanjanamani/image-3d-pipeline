"""Route detected segments to reconstruction methods based on ADE20K labels."""

from __future__ import annotations

from enum import Enum, auto


class ReconstructionMethod(Enum):
    TRELLIS = auto()       # Full Trellis 3D reconstruction
    GROUND_PLANE = auto()  # Flat subdivided ground mesh
    WATER_PLANE = auto()   # Flat PBR water plane
    BILLBOARD = auto()     # Camera-facing quad with texture
    SKY = auto()           # Extract sky color only (no mesh)
    SKIP = auto()          # Too small / unrecognised — skip entirely


# ---------------------------------------------------------------------------
# Label → method lookup tables (ADE20K label names, lower-cased, underscored)
# ---------------------------------------------------------------------------

_TRELLIS_LABELS: frozenset[str] = frozenset({
    "dog", "cat", "person", "animal", "tree", "plant",
    "chair", "bench", "bicycle", "car", "bird", "horse",
    "motorcycle", "boat", "motorcycle", "bus", "truck",
    "cow", "sheep", "elephant", "bear", "zebra", "giraffe",
    "potted_plant", "vase", "teddy_bear", "backpack", "umbrella",
    "handbag", "suitcase", "bottle", "cup", "bowl", "laptop",
    "keyboard", "cell_phone", "book", "clock", "flower_pot",
    "statue", "sculpture", "signboard", "streetlight",
})

_GROUND_LABELS: frozenset[str] = frozenset({
    "grass", "ground", "floor", "earth", "field", "flower",
    "dirt", "sand", "path", "road", "pavement", "soil", "land",
    "rug", "carpet", "sidewalk", "runway", "playingfield",
})

_WATER_LABELS: frozenset[str] = frozenset({
    "water", "lake", "river", "sea", "ocean", "pool", "pond",
    "waterfall", "fountain",
})

_SKY_LABELS: frozenset[str] = frozenset({
    "sky",
})

_BILLBOARD_LABELS: frozenset[str] = frozenset({
    "vegetation", "bush", "shrub", "fence", "wall", "building",
    "bridge", "mountain", "rock", "stone", "hill", "cliff",
    "windowpane", "door", "column", "pillar", "curtain",
    "forest", "tree_(background)",
})


def _normalise(label: str) -> str:
    """Normalise a label for lookup."""
    return label.lower().strip().replace("-", "_").replace(" ", "_")


def route_segment(
    label: str,
    is_thing: bool,
    area_ratio: float,
) -> ReconstructionMethod:
    """Map a segment to a reconstruction method.

    Parameters
    ----------
    label:
        ADE20K label name (will be normalised internally).
    is_thing:
        Whether OneFormer classified this as a "thing" (countable object).
    area_ratio:
        Segment area / total image area (0–1).

    Returns
    -------
    ReconstructionMethod
    """
    norm = _normalise(label)

    # Exact-match lookup first
    if norm in _TRELLIS_LABELS:
        return ReconstructionMethod.TRELLIS
    if norm in _GROUND_LABELS:
        return ReconstructionMethod.GROUND_PLANE
    if norm in _WATER_LABELS:
        return ReconstructionMethod.WATER_PLANE
    if norm in _SKY_LABELS:
        return ReconstructionMethod.SKY
    if norm in _BILLBOARD_LABELS:
        return ReconstructionMethod.BILLBOARD

    # Partial-match fallback (handle compound labels like "wooden_floor")
    for token in norm.split("_"):
        if token in _TRELLIS_LABELS:
            return ReconstructionMethod.TRELLIS
        if token in _GROUND_LABELS:
            return ReconstructionMethod.GROUND_PLANE
        if token in _WATER_LABELS:
            return ReconstructionMethod.WATER_PLANE
        if token in _SKY_LABELS:
            return ReconstructionMethod.SKY
        if token in _BILLBOARD_LABELS:
            return ReconstructionMethod.BILLBOARD

    # Generic fallback rules based on semantic metadata
    if is_thing and area_ratio > 0.03:
        return ReconstructionMethod.TRELLIS
    if is_thing and area_ratio <= 0.03:
        return ReconstructionMethod.BILLBOARD
    # Stuff (non-thing) that doesn't match — treat as ground unless tiny
    if area_ratio > 0.05:
        return ReconstructionMethod.GROUND_PLANE

    return ReconstructionMethod.SKIP
