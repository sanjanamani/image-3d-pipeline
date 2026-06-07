"""Panoptic scene detection using OneFormer + DETR + SAM2 refinement.

OneFormer handles background stuff (grass, trees, water, sky).
DETR handles foreground things (dog, person, cat, etc.) using COCO classes.
SAM2 refines all masks to pixel-perfect edges.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Cached model references
# ---------------------------------------------------------------------------
_oneformer_processor = None
_oneformer_model = None
_detr_processor = None
_detr_model = None
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


def _get_detr():
    """Lazy-load DETR for COCO object detection (catches dogs, people, etc.)."""
    global _detr_processor, _detr_model
    if _detr_processor is not None:
        return _detr_processor, _detr_model

    from transformers import DetrImageProcessor, DetrForObjectDetection
    import torch

    print("[scene_detector] Loading DETR (facebook/detr-resnet-50) for object detection …")
    _detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    _detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _detr_model = _detr_model.to(device)
    _detr_model.eval()
    print("[scene_detector] DETR loaded.")
    return _detr_processor, _detr_model


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

    # -----------------------------------------------------------------------
    # Secondary pass: DETR for foreground objects OneFormer may have missed
    # (dogs, cats, people, etc. — COCO 80-class detector)
    # -----------------------------------------------------------------------
    detr_segments = _detect_things_detr(image, image_np, total_pixels,
                                        existing_labels=[s.label for s in results])
    results.extend(detr_segments)

    print(f"[scene_detector] Found {len(results)} segments (after area filter).")
    return results


def _detect_things_detr(
    image: Image.Image,
    image_np: np.ndarray,
    total_pixels: int,
    existing_labels: list[str],
    threshold: float = 0.7,
    min_area_ratio: float = 0.005,
) -> list[DetectedSegment]:
    """Run DETR to catch foreground objects that OneFormer missed."""
    try:
        import torch
        detr_proc, detr_model = _get_detr()
    except Exception as exc:
        print(f"[scene_detector] DETR unavailable ({exc}); skipping thing detection.")
        return []

    try:
        import torch
        inputs = detr_proc(images=image, return_tensors="pt")
        inputs = {k: v.to(next(detr_model.parameters()).device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = detr_model(**inputs)

        target_sizes = torch.tensor([[image_np.shape[0], image_np.shape[1]]])
        results_detr = detr_proc.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]
    except Exception as exc:
        print(f"[scene_detector] DETR inference failed: {exc}")
        return []

    segments: list[DetectedSegment] = []
    label_counts: dict[str, int] = {}

    for score, label_id, box in zip(
        results_detr["scores"], results_detr["labels"], results_detr["boxes"]
    ):
        label_name = detr_model.config.id2label[int(label_id)].lower().replace(" ", "_")

        # Skip if already detected by OneFormer
        if label_name in existing_labels:
            continue

        # Skip non-thing background classes
        skip_classes = {"sky", "floor", "ground", "grass", "water", "wall",
                        "ceiling", "tree", "plant", "field"}
        if label_name in skip_classes:
            continue

        box_np = box.cpu().numpy().astype(int)
        x0, y0, x1, y1 = box_np
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(image_np.shape[1] - 1, x1), min(image_np.shape[0] - 1, y1)

        if x1 <= x0 or y1 <= y0:
            continue

        # Use SAM2 to get a precise mask from the bounding box
        coarse_mask = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=bool)
        coarse_mask[y0:y1, x0:x1] = True
        refined_mask = _refine_with_sam2(image_np, coarse_mask)

        area_ratio = refined_mask.sum() / total_pixels
        if area_ratio < min_area_ratio:
            continue

        count = label_counts.get(label_name, 0)
        label_counts[label_name] = count + 1

        masked_image = _make_masked_image(image, refined_mask)
        bbox = _mask_to_bbox(refined_mask)

        seg = DetectedSegment(
            label=label_name,
            mask=refined_mask,
            bbox=bbox,
            is_thing=True,
            masked_image=masked_image,
            segment_id=-(len(segments) + 1),
            area_ratio=area_ratio,
        )
        segments.append(seg)
        print(f"[scene_detector]  DETR segment: {label_name!r}  "
              f"score={float(score):.2f}  area={area_ratio:.2%}")

    return segments
