"""Top-level orchestrator for multi-object 3D scene generation."""

from __future__ import annotations

import traceback
import warnings
from pathlib import Path

import numpy as np
from PIL import Image


def run_scene_pipeline(
    image_path: str,
    output_dir: str = "outputs",
    seed: int = 42,
    simplify: float = 0.95,
    texture_size: int = 1024,
) -> str:
    """Generate a multi-object 3D scene from a single photograph.

    Steps
    -----
    1. Load image.
    2. OneFormer panoptic + SAM2 edge-refine → segments.
    3. Depth-Anything-V2 → metric depth map.
    4. Per-segment depth (median).
    5. Route each segment → ReconstructionMethod.
    6. Reconstruct each segment (Trellis / plane / billboard / sky).
    7. Assemble scene → scene.glb.

    Parameters
    ----------
    image_path:
        Path to input photo.
    output_dir:
        Root output directory.
    seed:
        Random seed for Trellis.
    simplify:
        Trellis mesh simplification ratio.
    texture_size:
        Texture atlas resolution in pixels.

    Returns
    -------
    str
        Absolute path to the exported ``scene.glb``.
    """
    # ------------------------------------------------------------------
    # Imports (lazy, so this module is importable even if deps missing)
    # ------------------------------------------------------------------
    from scene_detector import detect_segments
    from object_router import route_segment, ReconstructionMethod
    from depth_estimator import estimate_depth, get_segment_depth
    from novel_view_synth import generate_views, VIEW_DIRECTIONS
    from scene_texturer import project_multiview
    from scene_assembler import (
        make_ground_plane,
        make_water_plane,
        make_billboard,
        assemble_scene,
        export_scene,
    )

    import trimesh

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    out_root = Path(output_dir)
    scene_dir = out_root / "scene"
    scene_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size

    # ------------------------------------------------------------------
    # Step 2 — Detect segments
    # ------------------------------------------------------------------
    print("[scene_pipeline] Step 1/5 — Detecting segments …")
    segments = detect_segments(image_path)
    if not segments:
        raise RuntimeError("No segments detected. Ensure OneFormer is installed and accessible.")

    # ------------------------------------------------------------------
    # Step 3 — Estimate depth
    # ------------------------------------------------------------------
    print("[scene_pipeline] Step 2/5 — Estimating depth …")
    depth_map = estimate_depth(image_path)

    # ------------------------------------------------------------------
    # Step 4 — Per-segment depth + routing
    # ------------------------------------------------------------------
    print("[scene_pipeline] Step 3/5 — Routing segments …")
    routed: list[dict] = []
    for seg in segments:
        seg_depth = get_segment_depth(depth_map, seg.mask)
        method = route_segment(seg.label, seg.is_thing, seg.area_ratio)
        routed.append({
            "seg": seg,
            "depth_z": seg_depth,
            "method": method,
        })
        print(f"[scene_pipeline]   {seg.label!r}: depth={seg_depth:.2f}m, method={method.name}")

    # ------------------------------------------------------------------
    # Step 5 — Reconstruct each segment
    # ------------------------------------------------------------------
    print("[scene_pipeline] Step 4/5 — Reconstructing segments …")

    # Lazy Trellis import
    _trellis_pipe = None

    def _get_trellis():
        nonlocal _trellis_pipe
        if _trellis_pipe is not None:
            return _trellis_pipe
        try:
            import torch
            from trellis.pipelines import TrellisImageTo3DPipeline

            device = "cuda" if torch.cuda.is_available() else "cpu"
            print("[scene_pipeline] Loading Trellis pipeline …")
            _trellis_pipe = TrellisImageTo3DPipeline.from_pretrained(
                "JeffreyXiang/TRELLIS-image-large"
            )
            if device == "cuda":
                _trellis_pipe.cuda()
            else:
                _trellis_pipe.to("cpu")
            print(f"[scene_pipeline] Trellis loaded on {device.upper()}.")
            return _trellis_pipe
        except ImportError as exc:
            raise RuntimeError(
                "Trellis is not installed. Run setup.sh to install it."
            ) from exc

    assembled_objects: list[dict] = []
    label_counts: dict[str, int] = {}
    sky_color: tuple[int, int, int] | None = None

    for item in routed:
        seg = item["seg"]
        depth_z = item["depth_z"]
        method = item["method"]

        # Unique filename stem
        base = seg.label.replace(" ", "_")
        count = label_counts.get(base, 0)
        label_counts[base] = count + 1
        stem = f"{base}_{count}"

        # Approximate world-space X/Y from bounding box centre
        x0, y0, x1, y1 = seg.bbox
        cx_px = (x0 + x1) / 2
        cy_px = (y0 + y1) / 2
        world_x = (cx_px / img_w - 0.5) * 4.0   # ±2 m range
        world_y = (0.5 - cy_px / img_h) * 3.0   # ±1.5 m range

        xyz = [world_x, world_y, -depth_z]

        # ---- SKY ----
        if method == ReconstructionMethod.SKY:
            arr = np.array(image)
            sky_pixels = arr[seg.mask]
            if len(sky_pixels) > 0:
                sky_color = tuple(int(c) for c in np.median(sky_pixels, axis=0).astype(int))
            print(f"[scene_pipeline]   {stem}: SKY — colour={sky_color}")
            continue

        # ---- SKIP ----
        if method == ReconstructionMethod.SKIP:
            print(f"[scene_pipeline]   {stem}: SKIP")
            continue

        try:
            mesh: trimesh.Trimesh | None = None

            # ---- TRELLIS ----
            if method == ReconstructionMethod.TRELLIS:
                print(f"[scene_pipeline]   {stem}: TRELLIS reconstruction …")
                try:
                    from trellis.utils import postprocessing_utils

                    trellis = _get_trellis()

                    outputs = trellis.run(
                        seg.masked_image,
                        seed=seed,
                        sparse_structure_sampler_params={"steps": 12},
                        slat_sampler_params={"steps": 12},
                    )

                    obj_glb_path = scene_dir / f"{stem}.glb"
                    try:
                        glb = postprocessing_utils.to_glb(
                            outputs["gaussian"][0],
                            outputs["mesh"][0],
                            simplify=simplify,
                            texture_size=texture_size,
                        )
                        glb.export(str(obj_glb_path))
                        mesh = trimesh.load(str(obj_glb_path), force="mesh", process=False)
                        if isinstance(mesh, trimesh.Scene):
                            geoms = [g for g in mesh.geometry.values()
                                     if isinstance(g, trimesh.Trimesh)]
                            mesh = trimesh.util.concatenate(geoms) if geoms else None
                    except Exception as e:
                        print(f"[scene_pipeline]   Trellis GLB export failed ({e}); "
                              "using raw mesh.")
                        mesh_data = outputs["mesh"][0]
                        verts = mesh_data.vertices.cpu().float().numpy()
                        faces = mesh_data.faces.cpu().numpy()
                        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

                    if mesh is not None and len(mesh.faces) > 0:
                        # Apply multi-view texture
                        print(f"[scene_pipeline]   {stem}: Applying Zero123++ texturing …")
                        try:
                            views = generate_views(seg.masked_image)
                            mesh = project_multiview(
                                mesh, views, VIEW_DIRECTIONS, atlas_size=texture_size
                            )
                        except Exception as e:
                            print(f"[scene_pipeline]   Texturing failed ({e}); "
                                  "keeping untextured mesh.")

                        # Save individual GLB
                        mesh.export(str(obj_glb_path))
                        print(f"[scene_pipeline]   Saved {obj_glb_path}")

                except Exception as exc:
                    print(f"[scene_pipeline]   TRELLIS failed for {stem}: {exc}")
                    traceback.print_exc()
                    # Fall back to billboard
                    method = ReconstructionMethod.BILLBOARD

            # ---- GROUND_PLANE ----
            if method == ReconstructionMethod.GROUND_PLANE:
                print(f"[scene_pipeline]   {stem}: GROUND_PLANE …")
                mesh = make_ground_plane(
                    mask=seg.mask,
                    image=image,
                    depth_z=depth_z,
                    scene_width=float(img_w),
                    scene_height=float(img_h),
                )
                xyz = [0.0, 0.0, 0.0]  # ground plane centred at origin

            # ---- WATER_PLANE ----
            elif method == ReconstructionMethod.WATER_PLANE:
                print(f"[scene_pipeline]   {stem}: WATER_PLANE …")
                mesh = make_water_plane(
                    mask=seg.mask,
                    image=image,
                    depth_z=depth_z,
                    scene_width=float(img_w),
                    scene_height=float(img_h),
                )
                xyz = [0.0, 0.0, 0.0]

            # ---- BILLBOARD ----
            elif method == ReconstructionMethod.BILLBOARD:
                print(f"[scene_pipeline]   {stem}: BILLBOARD …")
                mesh = make_billboard(
                    masked_image=seg.masked_image,
                    depth_z=depth_z,
                    bbox=seg.bbox,
                    scene_width=float(img_w),
                    scene_height=float(img_h),
                )

            if mesh is not None:
                assembled_objects.append({
                    "mesh": mesh,
                    "label": seg.label,
                    "xyz_position": xyz,
                    "scale": 1.0,
                })

        except Exception as exc:
            print(f"[scene_pipeline]   ERROR processing {stem}: {exc}")
            traceback.print_exc()
            # Skip this object and continue

    # ------------------------------------------------------------------
    # Step 6 — Assemble + export
    # ------------------------------------------------------------------
    print(f"[scene_pipeline] Step 5/5 — Assembling {len(assembled_objects)} objects …")

    if not assembled_objects:
        warnings.warn(
            "[scene_pipeline] No objects were successfully reconstructed. "
            "The scene will be empty.",
            RuntimeWarning,
        )

    scene = assemble_scene(assembled_objects)
    scene_path = out_root / "scene.glb"
    result = export_scene(scene, str(scene_path))
    print(f"[scene_pipeline] Scene complete → {result}")
    return result
