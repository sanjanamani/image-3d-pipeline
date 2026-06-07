"""Metric depth estimation using Depth-Anything-V2."""

from __future__ import annotations

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Lazy model cache
# ---------------------------------------------------------------------------
_depth_pipeline = None


def _get_pipeline():
    """Lazy-load and cache the Depth-Anything-V2 pipeline."""
    global _depth_pipeline
    if _depth_pipeline is not None:
        return _depth_pipeline

    try:
        from transformers import pipeline as hf_pipeline
    except ImportError as exc:
        raise ImportError(
            "transformers>=4.40 is required for Depth-Anything-V2. "
            "Install with: pip install transformers>=4.40"
        ) from exc

    print("[depth_estimator] Loading Depth-Anything-V2-Metric-Outdoor-Large …")
    try:
        import torch
        device = 0 if torch.cuda.is_available() else -1
    except ImportError:
        device = -1

    _depth_pipeline = hf_pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
        device=device,
    )
    print("[depth_estimator] Depth-Anything-V2 loaded.")
    return _depth_pipeline


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_depth(image_path: str) -> np.ndarray:
    """Estimate metric depth for an image.

    Parameters
    ----------
    image_path:
        Path to the source image.

    Returns
    -------
    np.ndarray
        Depth map as float32 H×W array, values in metres.
    """
    pipe = _get_pipeline()
    image = Image.open(image_path).convert("RGB")

    print("[depth_estimator] Running depth estimation …")
    result = pipe(image)
    depth = result["depth"]

    # Ensure numpy float32
    if isinstance(depth, Image.Image):
        depth_arr = np.array(depth, dtype=np.float32)
    else:
        depth_arr = np.array(depth, dtype=np.float32)

    print(f"[depth_estimator] Depth map: shape={depth_arr.shape}, "
          f"min={depth_arr.min():.2f}m, max={depth_arr.max():.2f}m")
    return depth_arr


def get_segment_depth(depth_map: np.ndarray, mask: np.ndarray) -> float:
    """Return the median depth (metres) of the pixels covered by *mask*.

    Parameters
    ----------
    depth_map:
        Float32 H×W depth map (metres), from :func:`estimate_depth`.
    mask:
        Boolean H×W mask (True = pixels belonging to the segment).

    Returns
    -------
    float
        Median depth in metres. Falls back to the overall image median if
        the mask is empty.
    """
    if mask.any():
        return float(np.median(depth_map[mask]))
    # Fallback: overall median
    return float(np.median(depth_map))
