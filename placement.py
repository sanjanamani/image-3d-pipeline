"""Place a generated asset into the world: real-world scale, upright, snapped.

Given a raw asset mesh (e.g. a unit-normalised Trellis output) and the segment
it came from, compute a 4×4 transform that:

  1. positions the asset at its true world location (depth-unprojected
     ground-contact point),
  2. scales it to its real-world size (from angular extent × depth),
  3. keeps it upright, and
  4. snaps its base onto the ground plane so it rests on the surface.

This replaces the old ``world_x = (cx/w - 0.5)*4.0`` magic-constant placement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from camera import PinholeCamera
from ground_fit import GroundPlane


@dataclass
class SegmentInfo:
    """The 2D evidence for one detected asset."""

    label: str
    mask: np.ndarray                       # bool H×W
    bbox: tuple[int, int, int, int]        # x0, y0, x1, y1
    depth_z: float                         # median metric depth (m)


@dataclass
class PlacedAsset:
    """An asset plus the transform that drops it into the scene."""

    label: str
    transform: np.ndarray                  # 4×4
    world_position: np.ndarray             # (3,) foot position
    real_height_m: float


def _ground_contact_pixel(mask: np.ndarray,
                          bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    """Pixel where the object meets the ground: bottom row of the mask, centred."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        x0, y0, x1, y1 = bbox
        return (x0 + x1) / 2.0, float(y1)
    y_bottom = ys.max()
    near_bottom = ys >= (y_bottom - 2)
    x_center = float(np.median(xs[near_bottom]))
    return x_center, float(y_bottom)


def estimate_real_height(camera: PinholeCamera,
                         bbox: tuple[int, int, int, int],
                         depth_z: float) -> float:
    """Real-world height (m) of an object from its pixel height and depth."""
    _, y0, _, y1 = bbox
    px_height = max(float(y1 - y0), 1.0)
    # Similar triangles: real = (pixels / focal) * depth.
    return (px_height / camera.fy) * abs(depth_z)


def place_asset(
    mesh,
    seg: SegmentInfo,
    camera: PinholeCamera,
    ground: GroundPlane | None,
    upright_axis: int = 1,
) -> PlacedAsset:
    """Compute the placement transform for ``mesh`` given its segment.

    ``mesh`` only needs ``.bounds`` (a 2×3 array) — works for any trimesh.
    The mesh is assumed to be roughly upright (Trellis emits Y-up).

    Returns a :class:`PlacedAsset`; apply ``transform`` to the mesh to seat it.
    """
    # --- 1. World foot position from the ground-contact pixel ---------------
    fx_px, fy_px = _ground_contact_pixel(seg.mask, seg.bbox)
    foot = camera.unproject(fx_px, fy_px, abs(seg.depth_z)).reshape(3)

    # Snap the foot exactly onto the ground plane if we have one.
    if ground is not None:
        foot = ground.project_onto(foot[None, :]).reshape(3)

    # --- 2. Real-world scale -------------------------------------------------
    real_h = estimate_real_height(camera, seg.bbox, seg.depth_z)
    bounds = np.asarray(mesh.bounds, dtype=np.float64)  # (2, 3)
    mesh_h = float(bounds[1, upright_axis] - bounds[0, upright_axis])
    mesh_h = mesh_h if mesh_h > 1e-6 else 1.0
    scale = real_h / mesh_h

    # --- 3. Build transform: scale, then seat the base at the foot point -----
    transform = np.eye(4, dtype=np.float64)
    transform[0, 0] = transform[1, 1] = transform[2, 2] = scale

    # After scaling, the mesh's lowest point (along up axis) must land on foot.
    base_local = bounds[0, upright_axis] * scale          # scaled min-Y
    center_local = ((bounds[0] + bounds[1]) / 2.0) * scale  # scaled centre

    # Translate so the horizontal centre sits over the foot and the base rests
    # at the foot's height.
    transform[0, 3] = foot[0] - center_local[0]
    transform[1, 3] = foot[1] - base_local
    transform[2, 3] = foot[2] - center_local[2]

    return PlacedAsset(
        label=seg.label,
        transform=transform,
        world_position=foot,
        real_height_m=real_h,
    )
