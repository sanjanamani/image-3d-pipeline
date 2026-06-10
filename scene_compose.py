"""Compose placed assets into a game-ready, named scene graph and export GLB.

Unlike the old ``assemble_scene`` (which fused everything by guessed position),
this keeps every asset as its own named node with a real transform, so the
output drops cleanly into a game engine or a web viewer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from placement import PlacedAsset


def build_scene(
    assets: list[tuple[trimesh.Trimesh, PlacedAsset]],
    room_shell: trimesh.Trimesh | None = None,
) -> trimesh.Scene:
    """Build a Y-up scene graph from placed assets (+ optional room shell)."""
    scene = trimesh.Scene()

    if room_shell is not None:
        scene.add_geometry(room_shell, node_name="room_shell")

    name_counts: dict[str, int] = {}
    for mesh, placed in assets:
        base = placed.label or "asset"
        idx = name_counts.get(base, 0)
        name_counts[base] = idx + 1
        node_name = f"{base}_{idx:02d}"
        scene.add_geometry(mesh, node_name=node_name, transform=placed.transform)

    return scene


def export_glb(scene: trimesh.Scene, output_path: str) -> str:
    """Export a scene graph as GLB and return the absolute path."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(out))
    return str(out.resolve())


def scene_report(scene: trimesh.Scene) -> str:
    """Human-readable summary of the assembled scene graph."""
    nodes = list(scene.graph.nodes_geometry)
    lines = [f"Scene graph: {len(nodes)} named nodes"]
    for node in nodes:
        T, gname = scene.graph[node]
        geom = scene.geometry[gname].copy()
        geom.apply_transform(T)
        try:
            nfaces = len(geom.faces)
        except Exception:
            nfaces = 0
        lo, hi = np.asarray(geom.bounds)
        size = hi - lo
        lines.append(
            f"  - {node}: {nfaces} faces, "
            f"world size=({size[0]:.2f}×{size[1]:.2f}×{size[2]:.2f}) m"
        )
    return "\n".join(lines)
