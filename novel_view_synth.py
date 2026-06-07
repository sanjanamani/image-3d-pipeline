"""Novel-view synthesis using Zero123++ for multi-view generation."""

from __future__ import annotations

import warnings

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Lazy model cache
# ---------------------------------------------------------------------------
_zero123_pipe = None
_ZERO123_AVAILABLE: bool | None = None  # None = not yet checked


def _check_zero123() -> bool:
    global _ZERO123_AVAILABLE
    if _ZERO123_AVAILABLE is not None:
        return _ZERO123_AVAILABLE

    try:
        import diffusers  # noqa: F401
        import torch  # noqa: F401
        _ZERO123_AVAILABLE = True
    except ImportError:
        _ZERO123_AVAILABLE = False

    return _ZERO123_AVAILABLE


def _get_zero123_pipeline():
    """Lazy-load and cache the Zero123++ diffusion pipeline."""
    global _zero123_pipe
    if _zero123_pipe is not None:
        return _zero123_pipe

    if not _check_zero123():
        return None

    try:
        import torch
        from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

        print("[novel_view_synth] Loading Zero123++ (sudo-ai/zero123plus-v1.1) …")
        _zero123_pipe = DiffusionPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.1",
            custom_pipeline="sudo-ai/zero123plus-pipeline",
            torch_dtype=torch.float16,
        )
        _zero123_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            _zero123_pipe.scheduler.config,
            timestep_spacing="trailing",
        )
        if torch.cuda.is_available():
            _zero123_pipe = _zero123_pipe.cuda()
        print("[novel_view_synth] Zero123++ loaded.")
        return _zero123_pipe
    except Exception as exc:
        warnings.warn(
            f"[novel_view_synth] Failed to load Zero123++: {exc}. "
            "Falling back to single-view duplication.",
            RuntimeWarning,
            stacklevel=2,
        )
        _zero123_pipe = None
        return None


# ---------------------------------------------------------------------------
# Grid splitting helpers
# ---------------------------------------------------------------------------

def _split_2x3_grid(grid_image: Image.Image) -> list[Image.Image]:
    """Split a 2-row × 3-column grid image into 6 individual views.

    Zero123++ outputs a grid where views are arranged as:
        [front-right | right      | back     ]
        [left        | front-left | front-top]
    """
    w, h = grid_image.size
    col_w = w // 3
    row_h = h // 2

    views = []
    for row in range(2):
        for col in range(3):
            x0 = col * col_w
            y0 = row * row_h
            x1 = x0 + col_w
            y1 = y0 + row_h
            views.append(grid_image.crop((x0, y0, x1, y1)))
    return views  # 6 views


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

#: Approximate camera directions (unit vectors) for the 7 views returned by
#: generate_views:  [original-front, fr, right, back, left, fl, top]
VIEW_DIRECTIONS: list[list[float]] = [
    [0.0,  0.0,  1.0],   # 0: front  (original)
    [0.7,  0.0,  0.7],   # 1: front-right
    [1.0,  0.0,  0.0],   # 2: right
    [0.0,  0.0, -1.0],   # 3: back
    [-1.0, 0.0,  0.0],   # 4: left
    [-0.7, 0.0,  0.7],   # 5: front-left
    [0.0,  1.0,  0.0],   # 6: top
]


def generate_views(masked_image: Image.Image) -> list[Image.Image]:
    """Generate 6 novel views of an object using Zero123++.

    The input *masked_image* should be an RGBA PIL image with the background
    transparent (or a plain RGB image).  The returned list contains 7 views:
    the original front view followed by the 6 generated views.

    If Zero123++ is not available, the original image is repeated 7 times so
    that downstream code (``project_multiview``) can still run in single-view
    mode.

    Parameters
    ----------
    masked_image:
        RGBA (or RGB) PIL image of the object.

    Returns
    -------
    list[PIL.Image]
        List of 7 PIL images: [front, front-right, right, back, left, front-left, top].
    """
    pipe = _get_zero123_pipeline()

    if pipe is None:
        warnings.warn(
            "[novel_view_synth] Zero123++ unavailable — returning single-view fallback.",
            RuntimeWarning,
            stacklevel=2,
        )
        return [masked_image] * 7

    try:
        # Zero123++ expects RGB
        rgb_input = masked_image.convert("RGB")

        print("[novel_view_synth] Generating novel views …")
        result = pipe(rgb_input, num_inference_steps=36)
        grid_image: Image.Image = result.images[0]

        six_views = _split_2x3_grid(grid_image)
        print(f"[novel_view_synth] Generated {len(six_views)} views from grid.")

        # Return: original front + 6 generated
        return [masked_image] + six_views

    except Exception as exc:
        warnings.warn(
            f"[novel_view_synth] Zero123++ inference failed ({exc}); "
            "falling back to single-view duplication.",
            RuntimeWarning,
            stacklevel=2,
        )
        return [masked_image] * 7
