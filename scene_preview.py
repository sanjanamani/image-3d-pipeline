"""Stage-A 'layout preview': real scene composition WITHOUT Trellis.

Runs on a plain CUDA install (no compiled Trellis ops), so it works on native
Windows + an NVIDIA GPU.  It detects objects, estimates depth, places each one
correctly in 3D with the new scene engine, and represents every asset as a
camera-facing **billboard** standing on the fitted floor.  The result is a
real, orbit-able ``scene.glb`` that shows the scene LAYOUT from a single photo.

Stage B later swaps the billboards for full Trellis meshes.

The geometry path (``compose_preview_scene``) depends only on numpy / trimesh /
PIL + the scene engine, so it is unit-testable without any ML models.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import trimesh

from camera import PinholeCamera, estimate_vfov
from ground_fit import fit_ground_plane, fallback_ground_plane, GroundPlane
from placement import SegmentInfo, estimate_real_height
from scene_compose import build_scene, export_glb, scene_report
from object_router import route_segment, ReconstructionMethod


# ---------------------------------------------------------------------------
# Geometry builders (no ML dependencies — testable on CPU)
# ---------------------------------------------------------------------------

def _billboard_world(
    masked_image: Image.Image,
    seg: SegmentInfo,
    camera: PinholeCamera,
    ground: GroundPlane,
) -> trimesh.Trimesh:
    """A camera-facing quad, real-world sized, standing on the ground."""
    x0, y0, x1, y1 = seg.bbox
    fx_px = (x0 + x1) / 2.0
    foot = camera.unproject(fx_px, float(y1), abs(seg.depth_z)).reshape(3)
    foot = ground.project_onto(foot[None, :]).reshape(3)

    real_h = estimate_real_height(camera, seg.bbox, seg.depth_z)
    real_w = (max(x1 - x0, 1) / camera.fx) * abs(seg.depth_z)
    hw = real_w / 2.0

    bx, by, bz = float(foot[0]), float(foot[1]), float(foot[2])
    vertices = np.array([
        [bx - hw, by,          bz],
        [bx + hw, by,          bz],
        [bx + hw, by + real_h, bz],
        [bx - hw, by + real_h, bz],
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    uvs = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float64)

    img = masked_image.convert("RGBA")
    material = trimesh.visual.texture.SimpleMaterial(image=img)
    visual = trimesh.visual.TextureVisuals(uv=uvs, image=img, material=material)
    return trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)


def _floor_world(
    ground: GroundPlane,
    extent_x: tuple[float, float],
    extent_z: tuple[float, float],
    color: tuple[int, int, int],
) -> trimesh.Trimesh:
    """A flat floor quad spanning the scene at the fitted ground height."""
    y = float(ground.point[1])
    x_lo, x_hi = extent_x
    z_lo, z_hi = extent_z
    vertices = np.array([
        [x_lo, y, z_lo], [x_hi, y, z_lo],
        [x_hi, y, z_hi], [x_lo, y, z_hi],
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual.face_colors = np.array([*color, 255], dtype=np.uint8)
    return mesh


# ---------------------------------------------------------------------------
# Composition (pure geometry — unit-testable without ML models)
# ---------------------------------------------------------------------------

def compose_preview_scene(segments, depth_map: np.ndarray, image: Image.Image):
    """Compose a billboard layout scene from detected segments + depth.

    ``segments`` is any iterable of objects exposing ``.label``, ``.mask``
    (bool H×W), ``.bbox`` (x0,y0,x1,y1), ``.is_thing``, ``.area_ratio`` and
    ``.masked_image`` (PIL RGBA) — i.e. ``scene_detector.DetectedSegment``.
    """
    img_w, img_h = image.size
    camera = PinholeCamera(img_w, img_h, vfov_deg=estimate_vfov(img_w, img_h))
    arr = np.asarray(image.convert("RGB"))

    # --- Fit the ground from ground/water segments (fallback if none) -------
    world_grid = camera.unproject_depth_map(depth_map)
    ground_mask = np.zeros((img_h, img_w), dtype=bool)
    floor_color = (120, 120, 120)
    for seg in segments:
        method = route_segment(seg.label, seg.is_thing, seg.area_ratio)
        if method in (ReconstructionMethod.GROUND_PLANE, ReconstructionMethod.WATER_PLANE):
            ground_mask |= seg.mask
            floor_color = tuple(int(c) for c in arr[seg.mask].mean(axis=0))

    if ground_mask.any():
        ground = fit_ground_plane(world_grid[ground_mask]) \
            or fallback_ground_plane(world_grid.reshape(-1, 3))
    else:
        ground = fallback_ground_plane(world_grid.reshape(-1, 3))

    # --- Place every non-floor/sky segment as a billboard -------------------
    placed = []
    foot_pts = []
    for seg in segments:
        method = route_segment(seg.label, seg.is_thing, seg.area_ratio)
        if method in (ReconstructionMethod.GROUND_PLANE,
                      ReconstructionMethod.WATER_PLANE,
                      ReconstructionMethod.SKY,
                      ReconstructionMethod.SKIP):
            continue
        seg_info = SegmentInfo(seg.label, seg.mask, seg.bbox,
                               float(np.median(depth_map[seg.mask]) if seg.mask.any()
                                     else np.median(depth_map)))
        mesh = _billboard_world(seg.masked_image, seg_info, camera, ground)
        from placement import PlacedAsset
        x0, y0, x1, y1 = seg.bbox
        foot = camera.unproject((x0 + x1) / 2.0, float(y1),
                                abs(seg_info.depth_z)).reshape(3)
        foot = ground.project_onto(foot[None, :]).reshape(3)
        foot_pts.append(foot)
        placed.append((mesh, PlacedAsset(seg.label, np.eye(4), foot,
                                         estimate_real_height(camera, seg.bbox,
                                                              seg_info.depth_z))))

    # --- Floor sized to span the placed assets ------------------------------
    if foot_pts:
        fp = np.array(foot_pts)
        pad = 1.5
        ex = (float(fp[:, 0].min()) - pad, float(fp[:, 0].max()) + pad)
        ez = (float(fp[:, 2].min()) - pad, min(0.0, float(fp[:, 2].max()) + pad))
    else:
        ex, ez = (-5.0, 5.0), (-10.0, 0.0)
    floor = _floor_world(ground, ex, ez, floor_color)

    scene = build_scene(placed, room_shell=floor)
    return scene


# ---------------------------------------------------------------------------
# Full preview entrypoint (loads ML models — runs on GPU box)
# ---------------------------------------------------------------------------

def run_preview(image_path: str, output_dir: str = "outputs") -> str:
    """Detect + depth + billboard-layout a real photo into ``scene_preview.glb``."""
    from scene_detector import detect_segments
    from depth_estimator import estimate_depth

    image = Image.open(image_path).convert("RGB")

    print("[scene_preview] Detecting objects …")
    segments = detect_segments(image_path)
    if not segments:
        raise RuntimeError("No segments detected.")

    print("[scene_preview] Estimating depth …")
    depth_map = estimate_depth(image_path)

    print("[scene_preview] Composing layout scene …")
    scene = compose_preview_scene(segments, depth_map, image)

    out = Path(output_dir) / "scene_preview.glb"
    path = export_glb(scene, str(out))
    print(scene_report(scene))
    print(f"[scene_preview] Layout scene → {path}")
    return path
