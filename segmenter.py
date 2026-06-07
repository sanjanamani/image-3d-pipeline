"""SAM2-based segmentation with rembg fallback."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Try to import SAM2; gracefully degrade to rembg if not available
# ---------------------------------------------------------------------------
_SAM2_AVAILABLE = False
_sam2_predictor = None  # cached predictor

try:
    import torch
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    _SAM2_AVAILABLE = True
except ImportError:
    pass


def _get_predictor():
    """Load (and cache) the SAM2 predictor."""
    global _sam2_predictor
    if _sam2_predictor is not None:
        return _sam2_predictor

    import torch
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    # from_pretrained downloads the model from HuggingFace automatically
    _sam2_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
    return _sam2_predictor


def _apply_mask(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Return an RGBA image where background pixels have alpha=0."""
    rgba = image.convert("RGBA")
    arr = np.array(rgba)
    # mask is boolean (H, W); True = foreground
    arr[:, :, 3] = np.where(mask, 255, 0).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def segment_with_sam2(
    image_path: str,
    points: list[tuple[int, int]] | None = None,
    labels: list[int] | None = None,
) -> Image.Image:
    """Segment an object using SAM2 and return a background-removed RGBA image.

    Parameters
    ----------
    image_path:
        Path to the source image.
    points:
        List of (x, y) pixel coordinates used as point prompts.
        If None, automatic mask generation is used via a center-point heuristic.
    labels:
        Corresponding labels for each point: 1 = foreground, 0 = background.
        Defaults to all-foreground (1) when not supplied.

    Returns
    -------
    PIL.Image
        RGBA image with the background set to transparent.
    """
    if not _SAM2_AVAILABLE:
        warnings.warn(
            "SAM2 is not installed; falling back to rembg. "
            "Install SAM2 with: pip install git+https://github.com/facebookresearch/sam2.git",
            RuntimeWarning,
            stacklevel=2,
        )
        from rembg import remove as rembg_remove
        img = Image.open(image_path).convert("RGBA")
        return rembg_remove(img)

    import torch

    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    predictor = _get_predictor()
    predictor.set_image(image_np)

    if points is not None:
        pts_arr = np.array(points, dtype=np.float32)   # (N, 2)
        if labels is None:
            lbl_arr = np.ones(len(points), dtype=np.int32)
        else:
            lbl_arr = np.array(labels, dtype=np.int32)
    else:
        # Default: center-point heuristic — assume object is roughly centered
        h, w = image_np.shape[:2]
        pts_arr = np.array([[w // 2, h // 2]], dtype=np.float32)
        lbl_arr = np.array([1], dtype=np.int32)

    with torch.inference_mode():
        masks, scores, _ = predictor.predict(
            point_coords=pts_arr,
            point_labels=lbl_arr,
            multimask_output=True,
        )

    # Pick the highest-scoring mask
    best = int(np.argmax(scores))
    mask = masks[best].astype(bool)

    return _apply_mask(image, mask)
