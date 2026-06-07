import os
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Optional CUDA detection
# ---------------------------------------------------------------------------
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cpu":
        warnings.warn(
            "CUDA is not available. The Trellis pipeline will run on CPU, "
            "which is extremely slow. A CUDA-capable GPU is strongly recommended.",
            RuntimeWarning,
            stacklevel=2,
        )
except ImportError as exc:
    raise ImportError(
        "PyTorch is not installed. Run setup.sh to install all dependencies."
    ) from exc

# ---------------------------------------------------------------------------
# rembg (fallback segmenter)
# ---------------------------------------------------------------------------
try:
    from rembg import remove as rembg_remove
    _REMBG_AVAILABLE = True
except ImportError:
    _REMBG_AVAILABLE = False

# ---------------------------------------------------------------------------
# SAM2 segmenter (preferred)
# ---------------------------------------------------------------------------
try:
    from segmenter import segment_with_sam2, _SAM2_AVAILABLE
except ImportError:
    _SAM2_AVAILABLE = False
    segment_with_sam2 = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Hole capper
# ---------------------------------------------------------------------------
try:
    from hole_capper import cap_holes
    _HOLE_CAPPER_AVAILABLE = True
except ImportError:
    _HOLE_CAPPER_AVAILABLE = False
    cap_holes = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Texturer
# ---------------------------------------------------------------------------
try:
    from texturer import project_texture
    _TEXTURER_AVAILABLE = True
except ImportError:
    _TEXTURER_AVAILABLE = False
    project_texture = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Trellis — imported lazily so that the rest of the module is importable even
# when the TRELLIS repo has not been cloned yet.
# ---------------------------------------------------------------------------
_trellis_error: str | None = None
try:
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import postprocessing_utils
except ImportError as exc:
    _trellis_error = (
        "Trellis is not installed or not found on sys.path.\n"
        "Run setup.sh to clone the TRELLIS repository and install it:\n"
        "  bash setup.sh\n"
        f"Original error: {exc}"
    )
    TrellisImageTo3DPipeline = None  # type: ignore[assignment,misc]
    postprocessing_utils = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def remove_background(
    image_path: str,
    use_sam2: bool = True,
    click_points: list[tuple[int, int]] | None = None,
    click_labels: list[int] | None = None,
) -> Image.Image:
    """Remove the background from *image_path* and return a PIL RGBA image.

    Parameters
    ----------
    image_path:
        Path to the source image.
    use_sam2:
        When True (and SAM2 is installed), use SAM2 segmentation.
        Falls back to rembg automatically if SAM2 is unavailable.
    click_points:
        SAM2 point prompts as list of (x, y) pixel coords.
    click_labels:
        SAM2 point labels (1=fg, 0=bg).  Defaults to all-fg.
    """
    if use_sam2 and _SAM2_AVAILABLE and segment_with_sam2 is not None:
        print("[pipeline] Using SAM2 for segmentation …")
        return segment_with_sam2(image_path, points=click_points, labels=click_labels)

    if not _REMBG_AVAILABLE:
        raise RuntimeError(
            "Neither SAM2 nor rembg is available. "
            "Install at least one: pip install rembg"
        )
    print("[pipeline] Using rembg for background removal …")
    img = Image.open(image_path).convert("RGBA")
    return rembg_remove(img)


def run_pipeline(
    image_path: str,
    output_dir: str = "outputs",
    seed: int = 42,
    simplify: float = 0.95,
    texture_size: int = 1024,
    use_sam2: bool = True,
    click_points: list[tuple[int, int]] | None = None,
    click_labels: list[int] | None = None,
) -> str:
    """Full image-to-3D pipeline.

    Steps
    -----
    1. Segment / remove background (SAM2 if available, else rembg).
    2. Run Trellis inference to obtain a 3D representation.
    3. Cap holes in the extracted mesh.
    4. Export as GLB.
    5. Project front-view image texture onto the mesh.

    Parameters
    ----------
    image_path:
        Path to the source image (any format supported by Pillow).
    output_dir:
        Directory where outputs are written.  Created if it does not exist.
    seed:
        Random seed passed to the Trellis pipeline for reproducibility.
    simplify:
        Mesh simplification ratio (0–1).  Lower values produce simpler meshes.
    texture_size:
        Resolution of the baked texture atlas (pixels, square).
    use_sam2:
        Whether to attempt SAM2 segmentation (falls back to rembg on failure).
    click_points:
        Optional SAM2 point prompts: list of (x, y) pixel coordinates.
    click_labels:
        SAM2 point labels (1=fg, 0=bg).

    Returns
    -------
    str
        Absolute path to the exported ``.glb`` file.
    """
    if _trellis_error is not None:
        raise RuntimeError(_trellis_error)

    # -- 1. Background removal -----------------------------------------------
    print("[pipeline] Removing background …")
    fg_image: Image.Image = remove_background(
        image_path,
        use_sam2=use_sam2,
        click_points=click_points,
        click_labels=click_labels,
    )

    # -- 2. Persist the transparent preview so app.py can show it -----------
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    stem = Path(image_path).stem
    preview_path = output_dir_path / f"{stem}_nobg.png"
    fg_image.save(preview_path)
    print(f"[pipeline] Transparent preview saved → {preview_path}")

    # -- 3. Load Trellis pipeline --------------------------------------------
    print("[pipeline] Loading Trellis model (this may take a while on first run) …")
    pipe: TrellisImageTo3DPipeline = TrellisImageTo3DPipeline.from_pretrained(
        "JeffreyXiang/TRELLIS-image-large"
    )
    if DEVICE == "cuda":
        pipe.cuda()
    else:
        pipe.to("cpu")
    print(f"[pipeline] Model loaded on {DEVICE.upper()}.")

    # -- 4. Inference --------------------------------------------------------
    print("[pipeline] Running Trellis inference …")
    outputs = pipe.run(
        fg_image,
        seed=seed,
        sparse_structure_sampler_params={"steps": 12},
        slat_sampler_params={"steps": 12},
    )
    print("[pipeline] Inference complete.")

    # -- 5. Export as GLB (with optional hole capping) ----------------------
    glb_path = output_dir_path / f"{stem}.glb"
    print("[pipeline] Exporting GLB mesh …")
    try:
        # Try full Trellis GLB export first
        glb = postprocessing_utils.to_glb(
            outputs["gaussian"][0],
            outputs["mesh"][0],
            simplify=simplify,
            texture_size=texture_size,
        )
        glb.export(str(glb_path))
    except Exception as e:
        print(f"[pipeline] Full GLB export failed ({e}), using trimesh fallback …")
        import trimesh
        mesh_data = outputs["mesh"][0]
        verts = mesh_data.vertices.cpu().float().numpy()
        faces = mesh_data.faces.cpu().numpy()
        tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        # Keep only the largest connected component (removes floating debris)
        components = tm.split(only_watertight=False)
        if components:
            tm = max(components, key=lambda m: len(m.faces))
            print(f"[pipeline] Kept largest component: {len(tm.faces)} faces")

        # Cap holes before export
        if _HOLE_CAPPER_AVAILABLE and cap_holes is not None:
            print("[pipeline] Capping holes …")
            tm = cap_holes(tm)

        tm.export(str(glb_path))

    print(f"[pipeline] GLB exported → {glb_path}")

    # -- 6. Project image texture onto the exported mesh --------------------
    if _TEXTURER_AVAILABLE and project_texture is not None:
        try:
            print("[pipeline] Projecting image texture …")
            textured_path = output_dir_path / f"{stem}_textured.glb"
            project_texture(
                mesh_path=str(glb_path),
                image_path=str(preview_path),  # use bg-removed image
                output_path=str(textured_path),
                atlas_size=texture_size,
            )
            print(f"[pipeline] Textured GLB → {textured_path}")
            # Return the textured version as the primary output
            return str(textured_path.resolve())
        except Exception as e:
            print(f"[pipeline] Texture projection failed ({e}); returning untextured GLB.")

    return str(glb_path.resolve())


def preview_path_for(image_path: str, output_dir: str = "outputs") -> str:
    """Return the expected path of the no-background preview PNG."""
    stem = Path(image_path).stem
    return str((Path(output_dir) / f"{stem}_nobg.png").resolve())
