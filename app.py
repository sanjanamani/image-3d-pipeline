"""Gradio front-end for the image-to-3D pipeline (single-object + scene)."""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Any

import gradio as gr

# ---------------------------------------------------------------------------
# Attempt to import the pipeline; surface a helpful error in the UI if it
# is not yet set up.
# ---------------------------------------------------------------------------
_pipeline_import_error: str | None = None
try:
    from pipeline import remove_background, run_pipeline, preview_path_for
except Exception as exc:  # noqa: BLE001
    _pipeline_import_error = (
        f"Could not import pipeline:\n{exc}\n\n"
        "Make sure you have run setup.sh and activated the virtual environment."
    )

# ---------------------------------------------------------------------------
# Attempt to import scene pipeline
# ---------------------------------------------------------------------------
_scene_pipeline_error: str | None = None
try:
    from scene_pipeline import run_scene_pipeline
except Exception as exc:  # noqa: BLE001
    _scene_pipeline_error = (
        f"Could not import scene_pipeline:\n{exc}\n\n"
        "Make sure transformers>=4.40, diffusers>=0.27 are installed."
    )

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs")
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# SAM2 availability probe (for UI hint)
# ---------------------------------------------------------------------------
_SAM2_AVAILABLE = False
try:
    from segmenter import _SAM2_AVAILABLE  # type: ignore[assignment]
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Segmentation preview (fast, no Trellis)
# ---------------------------------------------------------------------------

def run_segmentation_preview(
    image_path: str | None,
    use_sam2: bool,
    points_json: str,
) -> tuple[str | None, str]:
    """Remove background only and return (preview_png_path, status)."""
    if _pipeline_import_error:
        return None, f"SETUP ERROR\n\n{_pipeline_import_error}"

    if image_path is None:
        return None, "Please upload an image first."

    click_points: list[tuple[int, int]] | None = None
    try:
        raw = json.loads(points_json) if points_json.strip() else []
        if raw:
            click_points = [(int(p["x"]), int(p["y"])) for p in raw]
    except Exception:
        click_points = None

    try:
        bg_removed = remove_background(
            image_path,
            use_sam2=use_sam2,
            click_points=click_points,
        )
        stem = Path(image_path).stem
        out_dir = Path(OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        preview_save_path = str(out_dir / f"{stem}_nobg.png")
        bg_removed.save(preview_save_path)
        return preview_save_path, "Segmentation preview ready."
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        return None, f"Segmentation error: {exc}\n\n{tb}"


# ---------------------------------------------------------------------------
# Full pipeline handler
# ---------------------------------------------------------------------------

def process_image(
    image_path: str | None,
    seed: int,
    simplify: float,
    texture_size: int,
    use_sam2: bool,
    points_json: str,
) -> tuple[str | None, str | None, str]:
    """Run the full pipeline and return (preview_png, glb_path, status)."""

    if _pipeline_import_error:
        return None, None, f"SETUP ERROR\n\n{_pipeline_import_error}"

    if image_path is None:
        return None, None, "Please upload an image before running the pipeline."

    click_points: list[tuple[int, int]] | None = None
    try:
        raw = json.loads(points_json) if points_json.strip() else []
        if raw:
            click_points = [(int(p["x"]), int(p["y"])) for p in raw]
    except Exception:
        click_points = None

    status_lines: list[str] = []

    def log(msg: str) -> None:
        status_lines.append(msg)
        print(msg)

    try:
        log(f"Step 1/3 — Removing background ({'SAM2' if use_sam2 else 'rembg'}) …")
        if click_points:
            log(f"  Using {len(click_points)} click point(s) as SAM2 prompt.")

        stem = Path(image_path).stem
        out_dir = Path(OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)

        bg_removed = remove_background(
            image_path,
            use_sam2=use_sam2,
            click_points=click_points,
        )
        preview_save_path = str(out_dir / f"{stem}_nobg.png")
        bg_removed.save(preview_save_path)
        log(f"  Preview saved: {preview_save_path}")

        log("Step 2/3 — Running Trellis inference (may take several minutes) …")
        glb_path = run_pipeline(
            image_path,
            output_dir=OUTPUT_DIR,
            seed=seed,
            simplify=simplify,
            texture_size=texture_size,
            use_sam2=use_sam2,
            click_points=click_points,
        )
        log(f"Step 3/3 — GLB exported: {glb_path}")
        log("\nDone! Download your .glb file using the link below.")

        return preview_save_path, glb_path, "\n".join(status_lines)

    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        error_msg = f"ERROR: {exc}\n\nTraceback:\n{tb}"
        log(error_msg)
        return None, None, "\n".join(status_lines)


# ---------------------------------------------------------------------------
# Click handler: accumulate click coordinates from the Gradio Image component
# ---------------------------------------------------------------------------

def record_click(
    points_json: str,
    evt: gr.SelectData,
) -> tuple[str, str]:
    """Append the clicked pixel coordinate to the points list."""
    try:
        pts: list[dict[str, int]] = json.loads(points_json) if points_json.strip() else []
    except Exception:
        pts = []

    x, y = int(evt.index[0]), int(evt.index[1])
    pts.append({"x": x, "y": y})
    label = f"Click points ({len(pts)}): " + ", ".join(f"({p['x']},{p['y']})" for p in pts)
    return json.dumps(pts), label


def clear_points() -> tuple[str, str]:
    """Reset the recorded click points."""
    return "[]", "No click points recorded. Click on the image to add prompts."


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

_CSS = """
#status-box textarea {
    font-family: monospace;
    font-size: 0.85rem;
}
#click-info {
    font-size: 0.85rem;
    color: #555;
    margin-top: 4px;
}
"""

_SAM2_HINT = (
    "SAM2 is available — click on the object in the image to add point prompts, "
    "then run 'Preview segmentation' or 'Generate 3D model'."
    if _SAM2_AVAILABLE
    else "SAM2 is NOT installed. The SAM2 toggle will fall back to rembg automatically. "
         "Install with: pip install git+https://github.com/facebookresearch/sam2.git"
)

with gr.Blocks(title="Image → 3D (Trellis)", css=_CSS) as demo:
    gr.Markdown("# Image → 3D Pipeline")

    with gr.Tabs():

        # ==================================================================
        # Tab 1 — Single Object
        # ==================================================================
        with gr.Tab("Single Object"):
            gr.Markdown(
                """
                Upload a photo of an object and get a textured **GLB** file back.

                The pipeline will:
                1. Remove the background with **SAM2** (point-click) or **rembg** (auto)
                2. Run **Microsoft Trellis** to reconstruct the 3D shape
                3. Cap mesh holes and project multi-view image texture (Zero123++ if available)
                4. Export a ready-to-use **.glb** mesh
                """
            )

            # Hidden state: JSON array of {x, y} dicts
            points_state = gr.State("[]")

            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        label="Input image  ·  Click on the object to add SAM2 point prompts",
                        type="filepath",
                        image_mode="RGB",
                    )

                    # SAM2 controls
                    with gr.Group():
                        use_sam2_toggle = gr.Checkbox(
                            label="Use SAM2 (click to select object)",
                            value=_SAM2_AVAILABLE,
                            info=_SAM2_HINT,
                        )
                        click_info_box = gr.Markdown(
                            "No click points recorded. Click on the image above to add prompts.",
                            elem_id="click-info",
                        )
                        with gr.Row():
                            preview_seg_btn = gr.Button("Preview segmentation", variant="secondary")
                            clear_pts_btn = gr.Button("Clear click points", variant="secondary")

                    with gr.Accordion("Advanced options", open=False):
                        seed_input = gr.Slider(
                            label="Seed",
                            minimum=0,
                            maximum=9999,
                            step=1,
                            value=42,
                        )
                        simplify_input = gr.Slider(
                            label="Mesh simplification (lower = simpler mesh)",
                            minimum=0.1,
                            maximum=1.0,
                            step=0.05,
                            value=0.95,
                        )
                        texture_input = gr.Slider(
                            label="Texture size (px)",
                            minimum=256,
                            maximum=2048,
                            step=256,
                            value=1024,
                        )

                    run_btn = gr.Button("Generate 3D model", variant="primary")

                with gr.Column(scale=1):
                    preview_output = gr.Image(
                        label="Background-removed preview",
                        type="filepath",
                        interactive=False,
                    )
                    glb_output = gr.File(
                        label="Download GLB",
                        interactive=False,
                    )
                    status_output = gr.Textbox(
                        label="Status / log",
                        lines=12,
                        interactive=False,
                        elem_id="status-box",
                    )

            # --- Wire up events ---

            image_input.select(
                fn=record_click,
                inputs=[points_state],
                outputs=[points_state, click_info_box],
            )

            clear_pts_btn.click(
                fn=clear_points,
                inputs=[],
                outputs=[points_state, click_info_box],
            )

            preview_seg_btn.click(
                fn=run_segmentation_preview,
                inputs=[image_input, use_sam2_toggle, points_state],
                outputs=[preview_output, status_output],
            )

            run_btn.click(
                fn=process_image,
                inputs=[
                    image_input,
                    seed_input,
                    simplify_input,
                    texture_input,
                    use_sam2_toggle,
                    points_state,
                ],
                outputs=[preview_output, glb_output, status_output],
            )

            gr.Markdown(
                """
                ---
                **Tips**
                - Click on the object in the image to provide SAM2 point prompts for better segmentation.
                - Multiple clicks on the foreground object improve mask quality.
                - Use "Preview segmentation" to check the mask before running the full pipeline.
                - Works best on isolated objects (single subject, plain or simple background).
                - A dedicated NVIDIA GPU with at least 8 GB VRAM is recommended for reasonable speed.
                - If Trellis is not installed, run `bash setup.sh` first.
                """
            )

        # ==================================================================
        # Tab 2 — Scene Generator
        # ==================================================================
        with gr.Tab("Scene Generator"):
            gr.Markdown(
                """
                ## Multi-Object Scene Generator

                Upload a photograph and the pipeline will automatically:
                1. Detect all objects and regions with **OneFormer** panoptic segmentation
                2. Refine object masks with **SAM2**
                3. Estimate per-object depth with **Depth-Anything-V2**
                4. Reconstruct each object/region (Trellis / ground plane / billboard)
                5. Apply **Zero123++** multi-view texturing per object
                6. Assemble everything into a single **scene.glb**

                No prompts or clicks required — fully automatic.
                """
            )

            if _scene_pipeline_error:
                gr.Markdown(
                    f"> **Setup Error:** `{_scene_pipeline_error}`\n\n"
                    "Install missing dependencies: `pip install transformers>=4.40 diffusers>=0.27 accelerate>=0.28`"
                )

            with gr.Row():
                with gr.Column(scale=1):
                    scene_image_input = gr.Image(
                        label="Input photograph",
                        type="filepath",
                        image_mode="RGB",
                    )

                    with gr.Accordion("Advanced options", open=False):
                        scene_seed_input = gr.Slider(
                            label="Seed",
                            minimum=0,
                            maximum=9999,
                            step=1,
                            value=42,
                        )
                        scene_simplify_input = gr.Slider(
                            label="Mesh simplification",
                            minimum=0.1,
                            maximum=1.0,
                            step=0.05,
                            value=0.95,
                        )
                        scene_texture_input = gr.Slider(
                            label="Texture size (px)",
                            minimum=256,
                            maximum=2048,
                            step=256,
                            value=1024,
                        )

                    scene_run_btn = gr.Button("Generate Scene", variant="primary")

                with gr.Column(scale=1):
                    scene_glb_output = gr.File(
                        label="Download scene.glb",
                        interactive=False,
                    )
                    scene_status_output = gr.Textbox(
                        label="Status / log",
                        lines=14,
                        interactive=False,
                        elem_id="status-box",
                    )
                    scene_segments_display = gr.Markdown(
                        label="Detected segments",
                        value="",
                    )

            # --- Scene pipeline handler ---

            def run_scene(
                image_path: str | None,
                seed: int,
                simplify: float,
                texture_size: int,
            ) -> tuple[str | None, str, str]:
                """Run the multi-object scene pipeline."""
                if _scene_pipeline_error:
                    return None, f"SETUP ERROR\n\n{_scene_pipeline_error}", ""

                if image_path is None:
                    return None, "Please upload an image first.", ""

                status_lines: list[str] = []

                def log(msg: str) -> None:
                    status_lines.append(msg)
                    print(msg)

                try:
                    log("Detecting segments with OneFormer + SAM2 …")

                    # Import here to get segment info for display
                    from scene_detector import detect_segments
                    from object_router import route_segment

                    segs = detect_segments(image_path)

                    seg_lines = ["**Detected segments:**\n"]
                    for s in segs:
                        method = route_segment(s.label, s.is_thing, s.area_ratio)
                        seg_lines.append(
                            f"- `{s.label}` — {s.area_ratio:.1%} of image "
                            f"({'thing' if s.is_thing else 'stuff'}) → **{method.name}**"
                        )
                    segments_md = "\n".join(seg_lines)

                    log(f"Found {len(segs)} segments. Running full scene pipeline …")

                    glb_path = run_scene_pipeline(
                        image_path,
                        output_dir=OUTPUT_DIR,
                        seed=seed,
                        simplify=simplify,
                        texture_size=texture_size,
                    )

                    log(f"\nScene exported: {glb_path}")
                    log("Done!")

                    return glb_path, "\n".join(status_lines), segments_md

                except Exception as exc:  # noqa: BLE001
                    tb = traceback.format_exc()
                    error_msg = f"ERROR: {exc}\n\nTraceback:\n{tb}"
                    log(error_msg)
                    return None, "\n".join(status_lines), ""

            scene_run_btn.click(
                fn=run_scene,
                inputs=[
                    scene_image_input,
                    scene_seed_input,
                    scene_simplify_input,
                    scene_texture_input,
                ],
                outputs=[scene_glb_output, scene_status_output, scene_segments_display],
            )

            gr.Markdown(
                """
                ---
                **Notes**
                - Scene generation is slower than single-object mode (multiple Trellis inferences).
                - A CUDA GPU with 16+ GB VRAM is recommended for scenes with several objects.
                - Segments below 1% of image area are automatically skipped.
                - Individual per-object GLBs are saved to `outputs/scene/` alongside `scene.glb`.
                """
            )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
    )

