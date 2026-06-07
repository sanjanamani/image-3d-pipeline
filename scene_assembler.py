"""Scene assembly: build ground planes, water planes, billboards, and assemble scene."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import trimesh

from scene_texturer import make_ground_material, make_water_material


# ---------------------------------------------------------------------------
# Ground plane
# ---------------------------------------------------------------------------

def make_ground_plane(
    mask: np.ndarray,
    image: Image.Image,
    depth_z: float,
    scene_width: float,
    scene_height: float,
    subdivisions: int = 20,
) -> trimesh.Trimesh:
    """Create a subdivided ground plane textured from the masked image region.

    Parameters
    ----------
    mask:
        Boolean H×W mask of the ground region.
    image:
        Full RGB PIL image.
    depth_z:
        Z position (depth, metres) for the plane.
    scene_width, scene_height:
        World-space extent of the scene in X and Y axes.
    subdivisions:
        Number of grid subdivisions along each axis.

    Returns
    -------
    trimesh.Trimesh
        Flat subdivided mesh with tiled ground texture.
    """
    # Sample the average color of the masked region, then build a small
    # solid-color tile — avoids dumping the whole photo as a flat rectangle.
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    ys, xs = np.where(mask)
    if len(xs) > 0:
        avg = arr[ys, xs].mean(axis=0).astype(np.uint8)
        # Add slight brightness variation to make it look less flat
        tile_size = 128
        tile = np.full((tile_size, tile_size, 3), avg, dtype=np.uint8)
        # Add a subtle noise pattern
        noise = np.random.randint(-15, 15, tile.shape, dtype=np.int16)
        tile = np.clip(tile.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        crop = Image.fromarray(tile)
    else:
        crop = Image.new("RGB", (64, 64), (100, 120, 80))

    # Build subdivided XZ grid (Y-up coordinate system)
    n = subdivisions + 1
    xs_lin = np.linspace(-scene_width / 2, scene_width / 2, n, dtype=np.float32)
    zs_lin = np.linspace(-scene_height / 2, scene_height / 2, n, dtype=np.float32)
    xg, zg = np.meshgrid(xs_lin, zs_lin)
    xg = xg.ravel()
    zg = zg.ravel()
    yg = np.full_like(xg, -depth_z)  # place at depth

    vertices = np.stack([xg, yg, zg], axis=1)  # (N², 3)

    # Build faces (two triangles per quad)
    faces = []
    for row in range(subdivisions):
        for col in range(subdivisions):
            i0 = row * n + col
            i1 = i0 + 1
            i2 = i0 + n
            i3 = i2 + 1
            faces.append([i0, i2, i1])
            faces.append([i1, i2, i3])
    faces = np.array(faces, dtype=np.uint32)

    # UV: tiled from 0..subdivisions so texture tiles naturally
    u = (xg - xg.min()) / (xg.max() - xg.min() + 1e-8)
    v = (zg - zg.min()) / (zg.max() - zg.min() + 1e-8)
    uvs = np.stack([u, v], axis=1).astype(np.float32)

    material = trimesh.visual.texture.SimpleMaterial(image=crop)
    visual = trimesh.visual.TextureVisuals(uv=uvs, image=crop, material=material)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)
    return mesh


# ---------------------------------------------------------------------------
# Water plane
# ---------------------------------------------------------------------------

def make_water_plane(
    mask: np.ndarray,
    image: Image.Image,
    depth_z: float,
    scene_width: float,
    scene_height: float,
) -> trimesh.Trimesh:
    """Create a flat water plane with PBR water material.

    The water colour is sampled as the median of masked pixels.
    """
    arr = np.array(image.convert("RGB"))
    if mask.any():
        pixels = arr[mask]
        color = tuple(int(c) for c in np.median(pixels, axis=0).astype(int))
    else:
        color = (30, 80, 160)

    hw = scene_width / 2
    hh = scene_height / 2
    y = -depth_z

    vertices = np.array([
        [-hw, y, -hh],
        [ hw, y, -hh],
        [ hw, y,  hh],
        [-hw, y,  hh],
    ], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    uvs = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)

    material = make_water_material(color)
    visual = trimesh.visual.TextureVisuals(uv=uvs, material=material)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)
    return mesh


# ---------------------------------------------------------------------------
# Billboard
# ---------------------------------------------------------------------------

def make_billboard(
    masked_image: Image.Image,
    depth_z: float,
    bbox: tuple[int, int, int, int],
    scene_width: float,
    scene_height: float,
) -> trimesh.Trimesh:
    """Create a camera-facing quad with the masked image as texture.

    The quad is placed at depth_z and sized proportionally to the bounding box.

    Parameters
    ----------
    masked_image:
        RGBA PIL image (background transparent).
    depth_z:
        Z depth in metres.
    bbox:
        Pixel bounding box (x0, y0, x1, y1) in the original image.
    scene_width, scene_height:
        World-space extents for scaling.
    """
    x0, y0, x1, y1 = bbox
    bw = x1 - x0
    bh = y1 - y0
    if bw <= 0 or bh <= 0:
        bw = bh = 1

    # Scale to world units
    w = (bw / scene_width) * 2.0
    h = (bh / scene_height) * 2.0

    cx = ((x0 + x1) / 2 - scene_width / 2) / scene_width * 2.0
    cy = (scene_height / 2 - (y0 + y1) / 2) / scene_height * 2.0

    z = -depth_z

    vertices = np.array([
        [cx - w/2, cy - h/2, z],
        [cx + w/2, cy - h/2, z],
        [cx + w/2, cy + h/2, z],
        [cx - w/2, cy + h/2, z],
    ], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    uvs = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)

    rgb_image = masked_image.convert("RGBA")
    material = trimesh.visual.texture.SimpleMaterial(image=rgb_image)
    visual = trimesh.visual.TextureVisuals(uv=uvs, image=rgb_image, material=material)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)
    return mesh


# ---------------------------------------------------------------------------
# Scene assembly
# ---------------------------------------------------------------------------

def assemble_scene(objects: list[dict]) -> trimesh.Scene:
    """Assemble multiple meshes into a trimesh.Scene.

    Parameters
    ----------
    objects:
        List of dicts with keys:
        - ``mesh``: trimesh.Trimesh
        - ``label``: str (used as geometry name)
        - ``xyz_position``: [x, y, z] translation (metres, Y-up)
        - ``scale``: float or [sx, sy, sz] uniform/anisotropic scale

    Returns
    -------
    trimesh.Scene
        Y-up scene with all objects placed at their requested positions.
    """
    scene = trimesh.Scene()

    for i, obj in enumerate(objects):
        mesh: trimesh.Trimesh = obj["mesh"]
        label: str = obj.get("label", f"object_{i}")
        xyz = obj.get("xyz_position", [0.0, 0.0, 0.0])
        scale = obj.get("scale", 1.0)

        # Build transform: scale then translate
        if isinstance(scale, (int, float)):
            s = float(scale)
            scale_vec = [s, s, s]
        else:
            scale_vec = list(scale)

        transform = np.eye(4, dtype=np.float64)
        transform[0, 0] = scale_vec[0]
        transform[1, 1] = scale_vec[1]
        transform[2, 2] = scale_vec[2]
        transform[0, 3] = float(xyz[0])
        transform[1, 3] = float(xyz[1])
        transform[2, 3] = float(xyz[2])

        # Use unique name to avoid collisions
        name = f"{label}_{i}"
        scene.add_geometry(mesh, node_name=name, transform=transform)

    return scene


def export_scene(scene: trimesh.Scene, output_path: str) -> str:
    """Export a trimesh.Scene as a GLB file.

    Parameters
    ----------
    scene:
        The assembled scene.
    output_path:
        Destination path (should end in ``.glb``).

    Returns
    -------
    str
        Absolute resolved path of the written file.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    scene.export(output_path)
    print(f"[scene_assembler] Scene exported → {output_path}")
    return str(Path(output_path).resolve())
