"""Stage-B scene build: real 3D assets (Trellis) placed by the scene engine.

For each detected object we generate an actual 3D mesh with Trellis, then seat
it in the world with the same camera-correct placement used by the preview
(real-world scale + base-snap to the fitted ground).  Background regions become
a floor; anything Trellis can't handle falls back to a billboard so the scene
always completes.

Needs a CUDA GPU with Trellis installed — run on Colab or WSL2, not native
Windows.  The geometry/placement/compose code is the same tested code path as
the Stage-A preview; only the per-object Trellis call is new.
"""

from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh

from camera import PinholeCamera, estimate_vfov
from placement import SegmentInfo, PlacedAsset, place_asset, estimate_real_height
from object_router import route_segment, ReconstructionMethod
from scene_compose import build_scene, export_glb, scene_report
from scene_preview import (
    fit_ground_from_segments,
    _billboard_world,
    _floor_world,
    _normalize_scene,
)

# ---------------------------------------------------------------------------
# Trellis (lazy, cached)
# ---------------------------------------------------------------------------
_trellis_pipe = None


def _get_trellis():
    global _trellis_pipe
    if _trellis_pipe is not None:
        return _trellis_pipe
    import torch
    from trellis.pipelines import TrellisImageTo3DPipeline

    print("[scene_build] Loading Trellis (JeffreyXiang/TRELLIS-image-large) …")
    pipe = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    if torch.cuda.is_available():
        pipe.cuda()
    _trellis_pipe = pipe
    print("[scene_build] Trellis loaded.")
    return _trellis_pipe


def _trellis_mesh(masked_image: Image.Image, seed: int, simplify: float,
                  texture_size: int) -> trimesh.Trimesh | None:
    """Generate a single textured 3D mesh from a masked object image."""
    from trellis.utils import postprocessing_utils

    pipe = _get_trellis()
    outputs = pipe.run(
        masked_image,
        seed=seed,
        sparse_structure_sampler_params={"steps": 12},
        slat_sampler_params={"steps": 12},
    )
    glb = postprocessing_utils.to_glb(
        outputs["gaussian"][0], outputs["mesh"][0],
        simplify=simplify, texture_size=texture_size,
    )
    mesh = glb if isinstance(glb, trimesh.Trimesh) else None
    if isinstance(glb, trimesh.Scene):
        geoms = [g for g in glb.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(geoms) if geoms else None
    return mesh


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def run_build(image_path: str, output_dir: str = "outputs", seed: int = 42,
              simplify: float = 0.95, texture_size: int = 1024) -> str:
    """Detect → depth → Trellis assets → placed scene graph → scene.glb."""
    from scene_detector import detect_segments
    from depth_estimator import estimate_depth

    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size
    camera = PinholeCamera(img_w, img_h, vfov_deg=estimate_vfov(img_w, img_h))

    print("[scene_build] Detecting objects …")
    segments = detect_segments(image_path)
    if not segments:
        raise RuntimeError("No segments detected.")

    print("[scene_build] Estimating depth …")
    depth_map = estimate_depth(image_path)

    ground, floor_color = fit_ground_from_segments(segments, depth_map, camera, image)

    placed: list[tuple[trimesh.Trimesh, PlacedAsset]] = []
    foot_pts: list[np.ndarray] = []

    for seg in segments:
        method = route_segment(seg.label, seg.is_thing, seg.area_ratio)
        if method in (ReconstructionMethod.SKY, ReconstructionMethod.SKIP,
                      ReconstructionMethod.GROUND_PLANE,
                      ReconstructionMethod.WATER_PLANE):
            continue

        depth_z = float(np.median(depth_map[seg.mask]) if seg.mask.any()
                        else np.median(depth_map))
        seg_info = SegmentInfo(seg.label, seg.mask, seg.bbox, depth_z)

        mesh = None
        if method == ReconstructionMethod.TRELLIS:
            print(f"[scene_build]   {seg.label}: Trellis reconstruction …")
            try:
                mesh = _trellis_mesh(seg.masked_image, seed, simplify, texture_size)
            except Exception as exc:  # noqa: BLE001
                print(f"[scene_build]   Trellis failed for {seg.label} ({exc}); "
                      "using billboard.")
                traceback.print_exc()

        if mesh is not None and len(mesh.faces) > 0:
            # Real 3D asset: place with camera-correct scale + base-snap.
            asset = place_asset(mesh, seg_info, camera, ground)
            placed.append((mesh, asset))
            foot_pts.append(asset.world_position)
        else:
            # Fallback (billboard) for BILLBOARD-routed or failed Trellis.
            bb = _billboard_world(seg.masked_image, seg_info, camera, ground)
            x0, _, x1, y1 = seg.bbox
            foot = camera.unproject((x0 + x1) / 2.0, float(y1), abs(depth_z)).reshape(3)
            foot = ground.project_onto(foot[None, :]).reshape(3)
            foot_pts.append(foot)
            placed.append((bb, PlacedAsset(seg.label, np.eye(4), foot,
                                           estimate_real_height(camera, seg.bbox, depth_z))))

    # Floor sized to span placed assets.
    if foot_pts:
        fp = np.array(foot_pts)
        pad = 1.5
        ex = (float(fp[:, 0].min()) - pad, float(fp[:, 0].max()) + pad)
        ez = (float(fp[:, 2].min()) - pad, min(0.0, float(fp[:, 2].max()) + pad))
    else:
        ex, ez = (-5.0, 5.0), (-10.0, 0.0)
    floor = _floor_world(ground, ex, ez, floor_color)

    scene = build_scene(placed, room_shell=floor)
    _normalize_scene(scene, target_width=8.0)

    out = Path(output_dir) / "scene.glb"
    path = export_glb(scene, str(out))
    print(scene_report(scene))
    print(f"[scene_build] Scene → {path}")
    return path
