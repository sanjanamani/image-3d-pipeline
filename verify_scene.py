"""End-to-end proof of the new scene-composition engine (CPU, no GPU models).

Fabricates a synthetic "lab" — a floor plus three objects at different depths —
using placeholder primitive meshes in place of Trellis assets, then runs the
real pipeline:  camera unproject → ground fit → per-asset placement/scale/snap
→ named scene graph → GLB.  Verifies numerically that assets rest on the floor
at sensible real-world sizes and writes a preview render.

Run:  python3 verify_scene.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from camera import PinholeCamera, estimate_vfov
from ground_fit import fit_ground_plane
from placement import SegmentInfo, place_asset
from scene_compose import build_scene, export_glb, scene_report

W, H = 800, 600
FLOOR_Y = -1.4  # camera is 1.4 m above the floor


def _synth_depth_and_segments(cam: PinholeCamera):
    """Build a synthetic metric depth map + a few labelled object segments."""
    depth = np.full((H, W), 50.0, dtype=np.float32)  # far background

    # --- Floor: depth consistent with a flat plane at world y = FLOOR_Y ------
    vs, us = np.mgrid[0:H, 0:W]
    below = vs > (cam.cy + 5)
    dir_y = -(vs - cam.cy) / cam.fy           # world-space ray Y component
    with np.errstate(divide="ignore", invalid="ignore"):
        floor_d = FLOOR_Y / dir_y             # solve d for world_y == FLOOR_Y
    floor_mask = below & (floor_d > 0) & (floor_d < 50)
    depth[floor_mask] = floor_d[floor_mask].astype(np.float32)

    # --- Three objects: (label, bbox, depth) ---------------------------------
    objects = [
        ("cabinet",    (120, 120, 260, 430), 5.0),
        ("microscope", (360, 300, 470, 400), 3.0),
        ("stool",      (560, 330, 690, 470), 2.2),
    ]
    segments: list[SegmentInfo] = []
    for label, bbox, dz in objects:
        x0, y0, x1, y1 = bbox
        mask = np.zeros((H, W), dtype=bool)
        mask[y0:y1, x0:x1] = True
        depth[mask] = dz
        segments.append(SegmentInfo(label=label, mask=mask, bbox=bbox, depth_z=dz))

    return depth, floor_mask, segments


def _placeholder_mesh(label: str) -> trimesh.Trimesh:
    """Stand-in for a Trellis asset (unit-ish, upright, origin-centred)."""
    if label == "cabinet":
        return trimesh.creation.box(extents=(0.6, 1.0, 0.4))
    if label == "microscope":
        return trimesh.creation.cylinder(radius=0.25, height=1.0, sections=24)
    return trimesh.creation.icosphere(subdivisions=2, radius=0.5)  # stool blob


def main() -> int:
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    cam = PinholeCamera(W, H, vfov_deg=estimate_vfov(W, H))
    depth, floor_mask, segments = _synth_depth_and_segments(cam)

    # --- Ground fit from unprojected floor pixels ----------------------------
    floor_world = cam.unproject_depth_map(depth)[floor_mask]
    ground = fit_ground_plane(floor_world)
    assert ground is not None, "ground fit failed"
    ground_y = float(ground.point[1])
    print(f"[verify] Ground plane: normal={np.round(ground.normal, 3)}, "
          f"y≈{ground_y:.3f} m (expected ≈ {FLOOR_Y})")
    assert ground.normal[1] > 0.95, "ground normal is not pointing up"
    assert abs(ground_y - FLOOR_Y) < 0.3, "ground height off"

    # --- Place each asset ----------------------------------------------------
    # build_scene stores each raw mesh under its node transform, so we keep the
    # original (untransformed) mesh and only apply the transform locally to
    # verify it rests on the floor.
    placed_assets = []
    for seg in segments:
        mesh = _placeholder_mesh(seg.label)
        placed = place_asset(mesh, seg, cam, ground)
        seated = mesh.copy()
        seated.apply_transform(placed.transform)
        min_y = float(seated.bounds[0, 1])
        print(f"[verify] {seg.label:11s} depth={seg.depth_z:.1f}m  "
              f"real_h={placed.real_height_m:.2f}m  base_y={min_y:.3f}m  "
              f"pos=({placed.world_position[0]:.2f}, {placed.world_position[2]:.2f})")
        # The base must rest on the floor.
        assert abs(min_y - ground_y) < 0.10, f"{seg.label} not resting on floor"
        assert placed.real_height_m > 0.1, f"{seg.label} degenerate scale"
        placed_assets.append((mesh, placed))

    scene = build_scene(placed_assets)
    glb_path = export_glb(scene, str(out_dir / "demo_scene.glb"))
    print("\n" + scene_report(scene))
    print(f"\n[verify] GLB exported → {glb_path}")

    # Reload to confirm it's a valid, named, multi-node GLB.
    reloaded = trimesh.load(glb_path)
    assert len(reloaded.geometry) == len(segments), "node count mismatch on reload"
    print(f"[verify] Reloaded GLB OK — {len(reloaded.geometry)} named nodes.")

    _render_preview(scene, out_dir / "demo_scene.png", ground_y)
    print("[verify] ALL CHECKS PASSED ✓")
    return 0


def _render_preview(scene: trimesh.Scene, path: Path, ground_y: float) -> None:
    """Headless matplotlib preview so we can *see* the assembled scene."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("ortho")  # no perspective → floor contact reads true

    colors = {"cabinet": "#c97b4a", "microscope": "#4a82c9", "stool": "#5bc94a"}
    for node in scene.graph.nodes_geometry:
        T, gname = scene.graph[node]
        m = scene.geometry[gname].copy()
        m.apply_transform(T)
        base = node.rsplit("_", 1)[0]
        tris = m.vertices[m.faces]
        ax.add_collection3d(Poly3DCollection(
            tris, facecolor=colors.get(base, "#999999"),
            edgecolor="k", linewidths=0.1, alpha=0.95))

    # Floor patch
    g = np.array([[-3, ground_y, -7], [3, ground_y, -7],
                  [3, ground_y, 0], [-3, ground_y, 0]])
    ax.add_collection3d(Poly3DCollection([g], facecolor="#dddddd", alpha=0.5))

    xlim, ylim, zlim = (-3, 3), (ground_y, ground_y + 3.5), (-7, 0)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
    # Preserve real-world proportions so the upright cabinet reads as upright.
    ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))
    ax.set_xlabel("X"); ax.set_ylabel("Y (up)"); ax.set_zlabel("Z (depth)")
    ax.view_init(elev=15, azim=-65)
    ax.set_title("Reconstructed scene — assets seated on fitted floor")
    fig.tight_layout()
    fig.savefig(str(path), dpi=120)
    plt.close(fig)
    print(f"[verify] Preview render → {path.resolve()}")


if __name__ == "__main__":
    raise SystemExit(main())
