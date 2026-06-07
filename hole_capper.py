"""Mesh hole-capping utilities using trimesh repair functions."""

from __future__ import annotations

import trimesh
import trimesh.repair


def cap_holes(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Close open holes in a mesh and return the repaired mesh.

    Steps applied in order:
    1. ``trimesh.repair.fill_holes``  — fills topological holes
    2. ``trimesh.repair.fix_winding`` — makes face windings consistent
    3. ``trimesh.repair.fix_normals`` — ensures outward-pointing normals

    Parameters
    ----------
    mesh:
        Input trimesh.Trimesh (may be modified in-place by trimesh).

    Returns
    -------
    trimesh.Trimesh
        The repaired mesh (same object, mutated).
    """
    # fill_holes returns True if the mesh is now watertight
    trimesh.repair.fill_holes(mesh)

    # Ensure winding / normal consistency regardless of fill result
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh)

    return mesh
