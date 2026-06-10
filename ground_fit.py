"""Fit a ground / floor plane to unprojected depth points (RANSAC).

The fitted plane gives every object a real surface to rest on, so assets stop
floating.  Pure NumPy — no sklearn dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GroundPlane:
    """An infinite plane ``normal · (p - point) = 0`` with an up-pointing normal."""

    normal: np.ndarray  # unit vector (3,)
    point: np.ndarray   # a point on the plane (3,)

    def signed_distance(self, pts: np.ndarray) -> np.ndarray:
        """Signed distance of points to the plane (positive = above)."""
        return (np.asarray(pts) - self.point) @ self.normal

    def project_onto(self, pts: np.ndarray) -> np.ndarray:
        """Orthogonally project points onto the plane."""
        pts = np.asarray(pts, dtype=np.float64)
        d = self.signed_distance(pts)
        return pts - np.outer(d, self.normal)

    def height_at_origin_axis(self) -> float:
        """Return the plane's Y intercept along a vertical drop (approx)."""
        # Solve for Y where (x=0, z=0): normal·([0,Y,0]-point)=0
        ny = self.normal[1]
        if abs(ny) < 1e-6:
            return float(self.point[1])
        return float(self.point[1] - (self.normal[0] * (-self.point[0]) +
                                       self.normal[2] * (-self.point[2])) / ny)


def fit_ground_plane(
    points: np.ndarray,
    iterations: int = 200,
    threshold: float = 0.05,
    rng: np.random.Generator | None = None,
) -> GroundPlane | None:
    """RANSAC-fit a plane to a world-space point cloud.

    Parameters
    ----------
    points:
        ``(N, 3)`` world points (e.g. the unprojected floor/ground pixels).
    iterations:
        Number of RANSAC trials.
    threshold:
        Inlier distance threshold in metres.
    rng:
        Optional NumPy random generator (for reproducibility).

    Returns
    -------
    GroundPlane | None
        Best-fit plane with an up-pointing (+Y) normal, or ``None`` if there
        are too few points.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 3:
        return None

    rng = rng or np.random.default_rng(0)
    best_inliers = -1
    best_plane: GroundPlane | None = None

    n = len(pts)
    for _ in range(iterations):
        idx = rng.choice(n, size=3, replace=False)
        a, b, c = pts[idx]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        # Orient normal upward (+Y) — ground normals point up.
        if normal[1] < 0:
            normal = -normal

        dist = np.abs((pts - a) @ normal)
        inliers = int((dist < threshold).sum())
        if inliers > best_inliers:
            best_inliers = inliers
            best_plane = GroundPlane(normal=normal, point=a.copy())

    if best_plane is None:
        return None

    # Refit on inliers via least squares (centroid + smallest-eigenvector normal).
    dist = np.abs((pts - best_plane.point) @ best_plane.normal)
    inlier_pts = pts[dist < threshold]
    if len(inlier_pts) >= 3:
        centroid = inlier_pts.mean(axis=0)
        cov = np.cov((inlier_pts - centroid).T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]
        if normal[1] < 0:
            normal = -normal
        best_plane = GroundPlane(normal=normal / np.linalg.norm(normal),
                                 point=centroid)

    return best_plane


def fallback_ground_plane(points: np.ndarray, percentile: float = 5.0) -> GroundPlane:
    """A flat horizontal floor at the low-Y percentile of the points.

    Used when there is no labelled ground region to fit against.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    y = np.percentile(pts[:, 1], percentile) if len(pts) else 0.0
    return GroundPlane(normal=np.array([0.0, 1.0, 0.0]),
                       point=np.array([0.0, y, 0.0]))
