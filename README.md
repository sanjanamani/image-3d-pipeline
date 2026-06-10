# image-3d-pipeline

Turn a single photo into a 3D scene. Two stages:

- **Stage A — Layout preview** (no Trellis): detects every object, estimates
  depth, and places each one correctly in 3D as a billboard standing on a
  fitted floor. Runs on **native Windows + NVIDIA GPU** with no compiled CUDA
  ops. This is the fast "see what it produces" path.
- **Stage B — Full assets** (Trellis): swaps each billboard for a real
  generated 3D mesh. Needs **Linux / WSL2 / Colab** (Trellis builds custom
  CUDA kernels that don't compile on native Windows).

## Quickstart — Stage A (Windows, PowerShell, RTX GPU)

```powershell
# from the repo folder
.\setup.ps1
.venv\Scripts\Activate.ps1
python run.py path\to\photo.jpg --mode preview
```

Open `outputs\scene_preview.glb` in the Windows **3D Viewer** app, in
[gltf-viewer](https://gltf-viewer.donmccurdy.com/), or Blender.

First run downloads the detection + depth models (a few GB) and caches them.

## What you get

A `scene_preview.glb` containing:
- a `room_shell` floor fitted to the photo's ground plane,
- one named node per detected object (`dog_00`, `chair_01`, …), each a
  camera-facing billboard scaled to its real-world size and seated on the floor.

It's a real 3D layout of the photo you can orbit — a pop-up-book / diorama view.

## Modes

```text
python run.py IMAGE --mode preview   # Stage A: billboard layout (no Trellis)
python run.py IMAGE --mode scene     # Stage B: full multi-object scene (Trellis)
python run.py IMAGE --mode single    # single-object reconstruction (Trellis)
```

## Stage B (full Trellis assets)

Trellis needs a Linux toolchain. From PowerShell you can drop into WSL2:

```powershell
wsl --install            # one-time, if you don't have WSL2 yet
wsl                      # enter Ubuntu
# inside Ubuntu:
bash setup.sh            # installs CUDA torch + Trellis (Stage B)
python run.py photo.jpg --mode scene
```

Colab is an alternative if WSL gives trouble or you OOM on large scenes.

## Architecture

| File | Role |
|------|------|
| `run.py` | CLI entrypoint (`--mode preview/scene/single`) |
| `scene_detector.py` | OneFormer panoptic + DETR object detection + SAM2 refine |
| `depth_estimator.py` | Depth-Anything-V2 metric depth |
| `camera.py` | Pinhole camera; unproject depth → Y-up world points |
| `ground_fit.py` | RANSAC ground/floor plane fit |
| `placement.py` | Per-asset world position, real-world scale, base-snap |
| `scene_preview.py` | Stage-A billboard layout composition |
| `scene_compose.py` | Game-ready named scene-graph GLB export |
| `scene_pipeline.py` | Stage-B full multi-object Trellis scene |
| `pipeline.py` | Single-object Trellis pipeline |
| `app.py` | Gradio UI |

## Tests (no GPU needed)

```bash
python3 verify_scene.py            # placement / ground-fit / scene-graph proof
python3 test_preview_geometry.py   # Stage-A composition wiring
```

## Requirements

- `requirements-preview.txt` — Stage A (Windows-friendly, no Trellis)
- `requirements.txt` — full stack (includes Stage B / Trellis deps)
- NVIDIA GPU recommended. 8 GB VRAM (e.g. RTX 4060) works for Stage A and
  single objects; large Trellis scenes prefer 16 GB.
