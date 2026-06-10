"""CPU test of the Stage-A preview geometry (no ML models needed).

Feeds synthetic segments + a synthetic depth map into compose_preview_scene
and checks that it produces a valid, named, multi-node GLB with a floor and
billboards seated on the fitted ground.

Run:  python3 test_preview_geometry.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh

from camera import PinholeCamera, estimate_vfov
from scene_preview import compose_preview_scene

W, H = 800, 600
FLOOR_Y = -1.4


@dataclass
class FakeSegment:
    label: str
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    is_thing: bool
    area_ratio: float
    masked_image: Image.Image


def _rgba(color):
    img = np.zeros((H, W, 4), dtype=np.uint8)
    img[..., :3] = color
    img[..., 3] = 255
    return Image.fromarray(img, "RGBA")


def main() -> int:
    cam = PinholeCamera(W, H, vfov_deg=estimate_vfov(W, H))

    # Synthetic depth: a flat floor at y=FLOOR_Y + objects at set depths.
    depth = np.full((H, W), 50.0, dtype=np.float32)
    vs, _ = np.mgrid[0:H, 0:W]
    dir_y = -(vs - cam.cy) / cam.fy
    with np.errstate(divide="ignore", invalid="ignore"):
        floor_d = FLOOR_Y / dir_y
    floor_region = (vs > cam.cy + 5) & (floor_d > 0) & (floor_d < 50)
    depth[floor_region] = floor_d[floor_region].astype(np.float32)

    segments: list[FakeSegment] = []

    # Ground segment (drives the floor fit + colour).
    gmask = floor_region.copy()
    segments.append(FakeSegment("floor", gmask, (0, int(cam.cy), W - 1, H - 1),
                                False, float(gmask.mean()), _rgba((90, 110, 70))))

    # Two "things" → billboards.
    for label, bbox, dz, col in [
        ("dog",   (320, 320, 430, 430), 3.0, (180, 140, 90)),
        ("person",(560, 180, 660, 470), 4.0, (90, 90, 180)),
    ]:
        x0, y0, x1, y1 = bbox
        m = np.zeros((H, W), dtype=bool)
        m[y0:y1, x0:x1] = True
        depth[m] = dz
        segments.append(FakeSegment(label, m, bbox, True,
                                    float(m.mean()), _rgba(col)))

    image = Image.fromarray(np.full((H, W, 3), 200, np.uint8))
    scene = compose_preview_scene(segments, depth, image)

    nodes = list(scene.graph.nodes_geometry)
    print(f"[test] scene nodes: {nodes}")
    assert "room_shell" in nodes, "floor (room_shell) missing"
    billboards = [n for n in nodes if n.startswith(("dog", "person"))]
    assert len(billboards) == 2, f"expected 2 billboards, got {billboards}"

    out = Path("outputs"); out.mkdir(exist_ok=True)
    path = out / "preview_test.glb"
    scene.export(str(path))
    reloaded = trimesh.load(str(path))
    assert len(reloaded.geometry) == 3, "reloaded node count wrong"

    # After normalization the floor sits at y=0; billboards must stand on it.
    floor_T, floor_g = scene.graph["room_shell"]
    floor_mesh = scene.geometry[floor_g].copy(); floor_mesh.apply_transform(floor_T)
    floor_y = float(floor_mesh.bounds[0, 1])
    print(f"[test] floor y={floor_y:.3f}")
    for node in billboards:
        T, gname = scene.graph[node]
        m = scene.geometry[gname].copy(); m.apply_transform(T)
        base_y = float(m.bounds[0, 1])
        print(f"[test] {node}: base_y={base_y:.3f}")
        assert abs(base_y - floor_y) < 0.15, f"{node} not on floor"

    print(f"[test] GLB → {path.resolve()}")
    print("[test] ALL CHECKS PASSED ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
