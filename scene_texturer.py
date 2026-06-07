"""Multi-view texture projection for 3D meshes.

Projects multiple views onto a mesh, selecting the best view per face based on
dot-product alignment between face normal and camera direction.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
import trimesh
import xatlas


# ---------------------------------------------------------------------------
# Multi-view projection
# ---------------------------------------------------------------------------

def project_multiview(
    mesh: trimesh.Trimesh,
    views: list[Image.Image],
    view_directions: list[list[float]],
    atlas_size: int = 1024,
) -> trimesh.Trimesh:
    """Project multiple views onto a mesh using best-view selection per face.

    For each face the view whose direction has the highest dot product with the
    face normal (i.e. is most "face-on") is used to sample the colour.

    Parameters
    ----------
    mesh:
        Input trimesh.Trimesh.
    views:
        List of PIL images, one per view direction.
    view_directions:
        Unit vectors (list of [x, y, z]) for each view, same order as *views*.
    atlas_size:
        Square resolution of the output UV atlas.

    Returns
    -------
    trimesh.Trimesh
        New mesh with UV coordinates and baked texture atlas.
    """
    assert len(views) == len(view_directions), (
        f"views ({len(views)}) and view_directions ({len(view_directions)}) must match."
    )

    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.uint32)

    # UV-unwrap
    print("[scene_texturer] Running xatlas UV unwrap …")
    vmapping, indices, uvs = xatlas.parametrize(vertices, faces)
    new_verts = vertices[vmapping]  # (V_new, 3)
    print(f"[scene_texturer] UV unwrap: {len(new_verts)} verts, {len(indices)} faces")

    # Convert view directions to unit numpy vectors
    dirs = np.array(view_directions, dtype=np.float32)  # (N_views, 3)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    dirs = dirs / norms

    # Preload view arrays as float32 (H, W, 3)
    view_arrays: list[np.ndarray] = []
    for v in views:
        arr = np.array(v.convert("RGB"), dtype=np.float32)
        view_arrays.append(arr)

    # Per-face normals
    v0 = new_verts[indices[:, 0]]
    v1 = new_verts[indices[:, 1]]
    v2 = new_verts[indices[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    fn = np.linalg.norm(cross, axis=1, keepdims=True)
    fn = np.where(fn < 1e-10, 1.0, fn)
    face_normals = cross / fn  # (F, 3)

    # Dot product of each face normal with each view direction → pick best view
    dots = face_normals @ dirs.T  # (F, N_views)
    best_view_idx = np.argmax(dots, axis=1)  # (F,)

    # Average colour of all views (fallback for backfaces)
    all_avg = np.mean([arr.mean(axis=(0, 1)) for arr in view_arrays], axis=0)

    atlas = np.full((atlas_size, atlas_size, 3), all_avg, dtype=np.float32)

    uv_px = uvs * (atlas_size - 1)  # (V_new, 2)

    # Compute per-view projection bounds from vertices visible in that view
    # (use x, y in world space for orthographic front projection)
    print("[scene_texturer] Rasterising faces into atlas …")
    batch_size = 4000
    for batch_start in range(0, len(indices), batch_size):
        batch_end = min(batch_start + batch_size, len(indices))
        batch = np.arange(batch_start, batch_end)

        tri_idx = indices[batch]  # (B, 3)
        uv0 = uv_px[tri_idx[:, 0]]
        uv1 = uv_px[tri_idx[:, 1]]
        uv2 = uv_px[tri_idx[:, 2]]

        bv_idx = best_view_idx[batch]  # (B,) which view each face uses

        # Bounding boxes in atlas space
        u_lo = np.floor(np.minimum(uv0[:, 0], np.minimum(uv1[:, 0], uv2[:, 0]))).astype(int)
        u_hi = np.ceil( np.maximum(uv0[:, 0], np.maximum(uv1[:, 0], uv2[:, 0]))).astype(int)
        v_lo = np.floor(np.minimum(uv0[:, 1], np.minimum(uv1[:, 1], uv2[:, 1]))).astype(int)
        v_hi = np.ceil( np.maximum(uv0[:, 1], np.maximum(uv1[:, 1], uv2[:, 1]))).astype(int)
        u_lo = np.clip(u_lo, 0, atlas_size - 1)
        u_hi = np.clip(u_hi, 0, atlas_size - 1)
        v_lo = np.clip(v_lo, 0, atlas_size - 1)
        v_hi = np.clip(v_hi, 0, atlas_size - 1)

        denom = ((uv1[:, 1] - uv2[:, 1]) * (uv0[:, 0] - uv2[:, 0]) +
                 (uv2[:, 0] - uv1[:, 0]) * (uv0[:, 1] - uv2[:, 1]))
        valid = np.abs(denom) > 1e-6

        for i in np.where(valid)[0]:
            d = denom[i]
            uv0i, uv1i, uv2i = uv0[i], uv1[i], uv2[i]
            vi_idx = bv_idx[i]
            view_arr = view_arrays[vi_idx]
            img_h, img_w = view_arr.shape[:2]

            # World-space XY for this face's vertices (for UV projection)
            wx = new_verts[tri_idx[i], 0]
            wy = new_verts[tri_idx[i], 1]

            us = np.arange(u_lo[i], u_hi[i] + 1, dtype=np.float32)
            vs = np.arange(v_lo[i], v_hi[i] + 1, dtype=np.float32)
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
            vi2 = vf[inside].astype(int)

            interp_x = w0i * wx[0] + w1i * wx[1] + w2i * wx[2]
            interp_y = w0i * wy[0] + w1i * wy[1] + w2i * wy[2]

            # Compute projection bounds for this specific view
            x_min_v = new_verts[:, 0].min()
            x_range_v = new_verts[:, 0].max() - x_min_v or 1.0
            y_min_v = new_verts[:, 1].min()
            y_range_v = new_verts[:, 1].max() - y_min_v or 1.0

            px = np.clip(((interp_x - x_min_v) / x_range_v * (img_w - 1)).astype(int), 0, img_w - 1)
            py = np.clip(((1.0 - (interp_y - y_min_v) / y_range_v) * (img_h - 1)).astype(int), 0, img_h - 1)

            atlas[vi2, ui] = view_arr[py, px]

    atlas_img = Image.fromarray(np.clip(atlas, 0, 255).astype(np.uint8), "RGB")
    material = trimesh.visual.texture.SimpleMaterial(image=atlas_img)
    visual = trimesh.visual.TextureVisuals(uv=uvs, image=atlas_img, material=material)
    out_mesh = trimesh.Trimesh(vertices=new_verts, faces=indices, visual=visual, process=False)
    print("[scene_texturer] Multi-view texture projection complete.")
    return out_mesh


# ---------------------------------------------------------------------------
# PBR material helpers
# ---------------------------------------------------------------------------

def make_ground_material(color: tuple[int, int, int] = (100, 120, 80)) -> trimesh.visual.material.PBRMaterial:
    """Return a matte PBR material suitable for ground/grass."""
    return trimesh.visual.material.PBRMaterial(
        baseColorFactor=[color[0] / 255, color[1] / 255, color[2] / 255, 1.0],
        roughnessFactor=0.9,
        metallicFactor=0.0,
    )


def make_water_material(color: tuple[int, int, int] = (30, 80, 160)) -> trimesh.visual.material.PBRMaterial:
    """Return a glossy PBR material suitable for water."""
    return trimesh.visual.material.PBRMaterial(
        baseColorFactor=[color[0] / 255, color[1] / 255, color[2] / 255, 0.85],
        roughnessFactor=0.02,
        metallicFactor=0.3,
    )
