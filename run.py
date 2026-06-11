"""Unified command-line entrypoint for the image-to-3D pipeline.

Examples
--------
  # Stage A — layout preview (no Trellis; runs on native Windows + CUDA):
  python run.py photo.jpg --mode preview

  # Full multi-object scene (needs Trellis):
  python run.py photo.jpg --mode scene

  # Single object (needs Trellis):
  python run.py object.jpg --mode single
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Image → 3D pipeline")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("--mode", choices=["preview", "build", "scene", "single"],
                        default="preview",
                        help="preview = billboard layout (no Trellis); "
                             "build = real 3D assets placed by the scene engine "
                             "(Trellis, Stage B); scene = legacy multi-object; "
                             "single = one object")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--simplify", type=float, default=0.95)
    parser.add_argument("--texture-size", type=int, default=1024)
    parser.add_argument("--generator", choices=["triposr", "trellis"],
                        default="triposr",
                        help="Stage-B (--mode build) asset generator: triposr "
                             "(Colab-friendly) or trellis (WSL2/local).")
    args = parser.parse_args(argv)

    if not Path(args.image).exists():
        print(f"ERROR: image not found: {args.image}", file=sys.stderr)
        return 2

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.mode == "preview":
        from scene_preview import run_preview
        out = run_preview(args.image, output_dir=args.output_dir)
    elif args.mode == "build":
        from scene_build import run_build
        out = run_build(args.image, output_dir=args.output_dir, seed=args.seed,
                        simplify=args.simplify, texture_size=args.texture_size,
                        backend=args.generator)
    elif args.mode == "scene":
        from scene_pipeline import run_scene_pipeline
        out = run_scene_pipeline(args.image, output_dir=args.output_dir,
                                 seed=args.seed, simplify=args.simplify,
                                 texture_size=args.texture_size)
    else:
        from pipeline import run_pipeline
        out = run_pipeline(args.image, output_dir=args.output_dir,
                           seed=args.seed, simplify=args.simplify,
                           texture_size=args.texture_size)

    print(f"\nDone → {out}")
    print("Open the .glb in Windows 3D Viewer, Blender, or https://gltf-viewer.donmccurdy.com/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
