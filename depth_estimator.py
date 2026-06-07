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
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = AutoImageProcessor.from_pretrained(
            "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"
        )
        model = AutoModelForDepthEstimation.from_pretrained(
            "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"
        ).to(device)
        model.eval()
        _depth_pipeline = (processor, model, device)
        print("[depth_estimator] Depth-Anything-V2 loaded.")
        return _depth_pipeline
    except Exception as exc:
        raise RuntimeError(f"Failed to load Depth-Anything-V2: {exc}") from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_depth(image_path: str) -> np.ndarray:
    """Estimate metric depth for an image.

    Returns float32 H×W array in metres (real-world scale).
    """
    import torch

    processor, model, device = _get_pipeline()
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size

    print("[depth_estimator] Running depth estimation …")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # predicted_depth: (1, H', W') tensor in metres
    predicted_depth = outputs.predicted_depth  # (1, H', W')

    # Resize back to original image size
    import torch.nn.functional as F
    depth_up = F.interpolate(
        predicted_depth.unsqueeze(1),
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze().cpu().numpy().astype(np.float32)

    print(f"[depth_estimator] Depth map: shape={depth_up.shape}, "
          f"min={depth_up.min():.2f}m, max={depth_up.max():.2f}m")
    return depth_up


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
