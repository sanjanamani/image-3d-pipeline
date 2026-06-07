"""Image-projection texturing: UV-unwrap with xatlas, project front-view image.

Now supports multi-view texture projection via Zero123++ when available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import trimesh
import xatlas

# ---------------------------------------------------------------------------
# Optional multi-view synthesis
# ---------------------------------------------------------------------------
_novel_view_synth_available: bool | None = None


def _try_multiview(
    mesh: trimesh.Trimesh,
    front_image: Image.Image,
    atlas_size: int,
) -> trimesh.Trimesh | None:
    """Try Zero123++ multi-view texturing; return None on any failure."""
    try:
        from novel_view_synth import generate_views, VIEW_DIRECTIONS
        from scene_texturer import project_multiview

        views = generate_views(front_image)
        return project_multiview(mesh, views, VIEW_DIRECTIONS, atlas_size=atlas_size)
    except Exception as exc:
        print(f"[texturer] Multi-view texturing failed ({exc}); falling back to single-view.")
        return None


def project_multiview_from_paths(
    mesh_path: str,
    image_path: str,
    output_path: str,
    atlas_size: int = 1024,
) -> str:
    """Load a mesh and image, apply multi-view texture projection, export GLB.

    If Zero123++ is unavailable or fails, falls back to single-view projection.

    Parameters
    ----------
    mesh_path:
        Path to source GLB/OBJ mesh.
    image_path:
        Path to RGBA or RGB front-view image.
    output_path:
        Destination GLB path.
    atlas_size:
        Texture atlas resolution.

    Returns
    -------
    str
        Absolute path to the exported textured GLB.
    """
    scene_or_mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(scene_or_mesh, trimesh.Scene):
        meshes = [g for g in scene_or_mesh.geometry.values()
                  if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No triangle mesh found in {mesh_path}")
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = scene_or_mesh

    front_image = Image.open(image_path).convert("RGBA")
    out_mesh = _try_multiview(mesh, front_image, atlas_size)

    if out_mesh is None:
        # Fall back to single-view
        return project_texture(mesh_path, image_path, output_path, atlas_size)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_mesh.export(output_path)
    print(f"[texturer] Multi-view textured GLB saved → {output_path}")
    return str(Path(output_path).resolve())


def project_texture(
    mesh_path: str,
    image_path: str,
    output_path: str,
    atlas_size: int = 1024,
) -> str:
    """UV-unwrap a GLB mesh and project the front-view image as a texture.

    The camera is assumed to be at +Z looking toward -Z.  Front-facing triangles
    get their colour sampled from the source image; back-facing triangles receive
    the average foreground colour.
    """
    # ------------------------------------------------------------------
    # 1. Load mesh
    # ------------------------------------------------------------------
    scene_or_mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(scene_or_mesh, trimesh.Scene):
        meshes = [g for g in scene_or_mesh.geometry.values()
                  if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No triangle mesh found in {mesh_path}")
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = scene_or_mesh

    vertices = np.array(mesh.vertices, dtype=np.float32)   # (V, 3)
    faces = np.array(mesh.faces, dtype=np.uint32)           # (F, 3)

    # ------------------------------------------------------------------
    # 2. UV unwrap with xatlas
    # ------------------------------------------------------------------
    print("[texturer] Running xatlas UV unwrap …")
    vmapping, indices, uvs = xatlas.parametrize(vertices, faces)
    new_vertices = vertices[vmapping]   # (V_new, 3)
    print(f"[texturer] UV unwrap done: {len(new_vertices)} verts, {len(indices)} faces")

    # ------------------------------------------------------------------
    # 3. Load source image; compute average foreground colour
    # ------------------------------------------------------------------
    src_img = Image.open(image_path).convert("RGBA")
    src_arr = np.array(src_img, dtype=np.float32)   # (H, W, 4)
    img_h, img_w = src_arr.shape[:2]

    alpha_mask = src_arr[:, :, 3] > 10
    if alpha_mask.any():
        avg_color = src_arr[alpha_mask, :3].mean(axis=0)
    else:
        avg_color = np.array([128.0, 128.0, 128.0])

    src_rgb = src_arr[:, :, :3]  # (H, W, 3)

    # ------------------------------------------------------------------
    # 4. Compute per-face normals and front-facing mask
    # ------------------------------------------------------------------
    v0 = new_vertices[indices[:, 0]]
    v1 = new_vertices[indices[:, 1]]
    v2 = new_vertices[indices[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    face_normals = cross / norms
    front_facing = face_normals[:, 2] > 0   # (F,) bool

    # ------------------------------------------------------------------
    # 5. Compute image-projection bounds from front-facing vertices
    # ------------------------------------------------------------------
    front_idx = np.where(front_facing)[0]
    print(f"[texturer] {len(front_idx)} front-facing faces out of {len(indices)}")

    if len(front_idx) > 0:
        fv = indices[front_idx].ravel()
        x_min, x_max = new_vertices[fv, 0].min(), new_vertices[fv, 0].max()
        y_min, y_max = new_vertices[fv, 1].min(), new_vertices[fv, 1].max()
        x_range = x_max - x_min if x_max > x_min else 1.0
        y_range = y_max - y_min if y_max > y_min else 1.0
    else:
        x_min, x_range = 0.0, 1.0
        y_min, y_range = 0.0, 1.0

    # ------------------------------------------------------------------
    # 6. Rasterise front-facing triangles into atlas — vectorized
    # ------------------------------------------------------------------
    atlas = np.full((atlas_size, atlas_size, 3), avg_color, dtype=np.float32)

    if len(front_idx) > 0:
        # Atlas UV coords in pixel units for all front-facing tris
        uv_px = uvs * (atlas_size - 1)   # (V_new, 2)

        # Process in batches to avoid OOM on very large meshes
        batch_size = 5000
        for batch_start in range(0, len(front_idx), batch_size):
            batch = front_idx[batch_start:batch_start + batch_size]
            tri_idx = indices[batch]   # (B, 3)

            uv0 = uv_px[tri_idx[:, 0]]   # (B, 2)
            uv1 = uv_px[tri_idx[:, 1]]
            uv2 = uv_px[tri_idx[:, 2]]

            wx = new_vertices[tri_idx, 0]   # (B, 3)
            wy = new_vertices[tri_idx, 1]

            # Bounding box per triangle in atlas space
            u_lo = np.floor(np.minimum(uv0[:, 0], np.minimum(uv1[:, 0], uv2[:, 0]))).astype(int)
            u_hi = np.ceil( np.maximum(uv0[:, 0], np.maximum(uv1[:, 0], uv2[:, 0]))).astype(int)
            v_lo = np.floor(np.minimum(uv0[:, 1], np.minimum(uv1[:, 1], uv2[:, 1]))).astype(int)
            v_hi = np.ceil( np.maximum(uv0[:, 1], np.maximum(uv1[:, 1], uv2[:, 1]))).astype(int)

            u_lo = np.clip(u_lo, 0, atlas_size - 1)
            u_hi = np.clip(u_hi, 0, atlas_size - 1)
            v_lo = np.clip(v_lo, 0, atlas_size - 1)
            v_hi = np.clip(v_hi, 0, atlas_size - 1)

            denom = ((uv1[:, 1] - uv2[:, 1]) * (uv0[:, 0] - uv2[:, 0]) +
                     (uv2[:, 0] - uv1[:, 0]) * (uv0[:, 1] - uv2[:, 1]))   # (B,)
            valid = np.abs(denom) > 1e-6

            for i in np.where(valid)[0]:
                fi = i  # local index in this batch
                d = denom[fi]
                uv0i, uv1i, uv2i = uv0[fi], uv1[fi], uv2[fi]

                us = np.arange(u_lo[fi], u_hi[fi] + 1, dtype=np.float32)
                vs = np.arange(v_lo[fi], v_hi[fi] + 1, dtype=np.float32)
                if len(us) == 0 or len(vs) == 0:
                    continue
                ug, vg = np.meshgrid(us, vs)
                uf = ug.ravel()
                vf = vg.ravel()

                w0 = ((uv1i[1] - uv2i[1]) * (uf - uv2i[0]) +
                      (uv2i[0] - uv1i[0]) * (vf - uv2i[1])) / d
                w1 = ((uv2i[1] - uv0i[1]) * (uf - uv2i[0]) +
                      (uv0i[0] - uv2i[0]) * (vf - uv2i[1])) / d
                w2 = 1.0 - w0 - w1

                inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
                if not inside.any():
                    continue

                w0i, w1i, w2i = w0[inside], w1[inside], w2[inside]
                ui = uf[inside].astype(int)
                vi = vf[inside].astype(int)

                # Project world XY to image coords
                interp_x = w0i * wx[fi, 0] + w1i * wx[fi, 1] + w2i * wx[fi, 2]
                interp_y = w0i * wy[fi, 0] + w1i * wy[fi, 1] + w2i * wy[fi, 2]

                px = np.clip(((interp_x - x_min) / x_range * (img_w - 1)).astype(int), 0, img_w - 1)
                py = np.clip(((1.0 - (interp_y - y_min) / y_range) * (img_h - 1)).astype(int), 0, img_h - 1)

                atlas[vi, ui] = src_rgb[py, px]

    # ------------------------------------------------------------------
    # 7. Save atlas and export textured GLB
    # ------------------------------------------------------------------
    atlas_img = Image.fromarray(np.clip(atlas, 0, 255).astype(np.uint8), "RGB")
    atlas_path = Path(output_path).with_suffix(".atlas.png")
    atlas_img.save(str(atlas_path))
    print(f"[texturer] Atlas saved → {atlas_path}")

    material = trimesh.visual.texture.SimpleMaterial(image=atlas_img)
    texture_visuals = trimesh.visual.TextureVisuals(uv=uvs, image=atlas_img, material=material)
    out_mesh = trimesh.Trimesh(vertices=new_vertices, faces=indices,
                               visual=texture_visuals, process=False)
    out_mesh.export(output_path)
    print(f"[texturer] Textured GLB saved → {output_path}")
    return str(Path(output_path).resolve())
