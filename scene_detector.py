"""Panoptic scene detection using OneFormer + SAM2 refinement."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Cached model references
# ---------------------------------------------------------------------------
_oneformer_processor = None
_oneformer_model = None
_sam2_predictor = None


def _get_oneformer():
    """Lazy-load and cache OneFormer processor + model."""
    global _oneformer_processor, _oneformer_model
    if _oneformer_processor is not None:
        return _oneformer_processor, _oneformer_model

    try:
        from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
    except ImportError as exc:
        raise ImportError(
            "transformers>=4.40 is required for OneFormer. "
            "Install with: pip install transformers>=4.40"
        ) from exc

    print("[scene_detector] Loading OneFormer (shi-labs/oneformer_ade20k_swin_large) …")
    _oneformer_processor = OneFormerProcessor.from_pretrained(
        "shi-labs/oneformer_ade20k_swin_large"
    )
    _oneformer_model = OneFormerForUniversalSegmentation.from_pretrained(
        "shi-labs/oneformer_ade20k_swin_large"
    )
    _oneformer_model.eval()
    print("[scene_detector] OneFormer loaded.")
    return _oneformer_processor, _oneformer_model


def _get_sam2():
    """Lazy-load and cache SAM2 predictor."""
    global _sam2_predictor
    if _sam2_predictor is not None:
        return _sam2_predictor

    try:
        import torch
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        return None

    print("[scene_detector] Loading SAM2 for edge refinement …")
    _sam2_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
    print("[scene_detector] SAM2 loaded.")
    return _sam2_predictor


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class DetectedSegment:
    label: str
    mask: np.ndarray          # bool H×W
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1
    is_thing: bool
    masked_image: Image.Image  # RGBA PIL, background transparent
    segment_id: int = 0
    area_ratio: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    return int(x0), int(y0), int(x1), int(y1)


def _make_masked_image(image: Image.Image, mask: np.ndarray) -> Image.Image:
    rgba = image.convert("RGBA")
    arr = np.array(rgba)
    arr[:, :, 3] = np.where(mask, 255, 0).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def _refine_with_sam2(image_np: np.ndarray, coarse_mask: np.ndarray) -> np.ndarray:
    """Use SAM2 to refine a coarse mask using the mask bounding box as prompt."""
    predictor = _get_sam2()
    if predictor is None:
        return coarse_mask

    try:
        import torch

        rows = np.where(np.any(coarse_mask, axis=1))[0]
        cols = np.where(np.any(coarse_mask, axis=0))[0]
        if len(rows) == 0 or len(cols) == 0:
            return coarse_mask

        x0, y0, x1, y1 = int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])
        box = np.array([[x0, y0, x1, y1]], dtype=np.float32)

        predictor.set_image(image_np)
        with torch.inference_mode():
            masks, scores, _ = predictor.predict(
                box=box,
                multimask_output=True,
            )
        best = int(np.argmax(scores))
        return masks[best].astype(bool)
    except Exception as e:
        print(f"[scene_detector] SAM2 refinement failed: {e}; using coarse mask.")
        return coarse_mask


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_segments(image_path: str, min_area_ratio: float = 0.01) -> list[DetectedSegment]:
    """Detect all segments in an image using OneFormer panoptic segmentation.

    Parameters
    ----------
    image_path:
        Path to the source image.
    min_area_ratio:
        Minimum segment area as a fraction of total image area (default 1%).

    Returns
    -------
    list[DetectedSegment]
        Detected segments, filtered to those above min_area_ratio.
    """
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    total_pixels = image_np.shape[0] * image_np.shape[1]

    processor, model = _get_oneformer()

    print("[scene_detector] Running OneFormer panoptic segmentation …")
    try:
        import torch
        inputs = processor(images=image, task_inputs=["panoptic"], return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        panoptic_seg = processor.post_process_panoptic_segmentation(
            outputs,
            target_sizes=[(image_np.shape[0], image_np.shape[1])],
        )[0]
    except Exception as exc:
        raise RuntimeError(
            f"OneFormer inference failed: {exc}\n"
            "Make sure transformers>=4.40 and the model weights are accessible."
        ) from exc

    seg_map = panoptic_seg["segmentation"].cpu().numpy()   # H×W int
    segments_info = panoptic_seg["segments_info"]

    # Build id → label_name mapping from model config
    id2label = model.config.id2label  # int → str

    results: list[DetectedSegment] = []

    for seg_info in segments_info:
        seg_id = seg_info["id"]
        label_id = seg_info["label_id"]
        is_thing = bool(seg_info.get("isthing", False))

        label_name = id2label.get(label_id, f"class_{label_id}").lower()
        # Normalize label: replace spaces/hyphens with underscores
        label_name = label_name.strip().replace("-", "_").replace(" ", "_")

        mask = seg_map == seg_id
        area = int(mask.sum())
        area_ratio = area / total_pixels

        if area_ratio < min_area_ratio:
            continue

        if not mask.any():
            continue

        bbox = _mask_to_bbox(mask)

        # SAM2 refinement for "thing" segments (objects)
        refined_mask = mask
        if is_thing:
            refined_mask = _refine_with_sam2(image_np, mask)

        masked_image = _make_masked_image(image, refined_mask)

        seg = DetectedSegment(
            label=label_name,
            mask=refined_mask,
            bbox=bbox,
            is_thing=is_thing,
            masked_image=masked_image,
            segment_id=seg_id,
            area_ratio=area_ratio,
        )
        results.append(seg)
        print(f"[scene_detector]  segment: {label_name!r}  thing={is_thing}  area={area_ratio:.2%}")

    print(f"[scene_detector] Found {len(results)} segments (after area filter).")
    return results
