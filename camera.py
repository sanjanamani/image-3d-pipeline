"""Pinhole camera model: unproject metric depth into a Y-up world frame.

This is the missing piece that turns the scene pipeline from a "cutout
compositor" into a real scene generator.  Instead of guessing world X/Y from
bounding-box centres with magic constants, we model the capturing camera as a
simple pinhole and unproject each pixel using its metric depth.

World convention (matches GLB / game engines):
    +X right, +Y up, -Z forward (into the scene).
Image convention:
    u → right (0..W), v → down (0..H).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class PinholeCamera:
    """A minimal pinhole camera defined by image size and vertical FOV."""

    width: int
    height: int
    vfov_deg: float = 55.0  # vertical field of view (degrees)

    @property
    def fy(self) -> float:
        return (self.height / 2.0) / math.tan(math.radians(self.vfov_deg) / 2.0)

    @property
    def fx(self) -> float:
        # Square pixels: horizontal focal length equals vertical focal length.
        return self.fy

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0

    # ------------------------------------------------------------------
    # Unprojection
    # ------------------------------------------------------------------
    def unproject(self, u: np.ndarray, v: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Unproject pixel coordinates + metric depth to world points.

        Parameters
        ----------
        u, v:
            Pixel coordinates (any broadcastable shape).
        depth:
            Metric depth (metres) at each pixel, same shape as ``u``/``v``.

        Returns
        -------
        np.ndarray
            World points of shape ``(..., 3)`` in the Y-up convention.
        """
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        d = np.asarray(depth, dtype=np.float64)

        # Camera-space ray (Z forward, into the scene).
        x_cam = (u - self.cx) * d / self.fx
        y_cam = (v - self.cy) * d / self.fy
        z_cam = d

        # Camera → world (Y-up, -Z forward): flip Y (image down → world down)
        # and flip Z (scene is in front along -Z).
        world = np.stack([x_cam, -y_cam, -z_cam], axis=-1)
        return world

    def unproject_depth_map(self, depth_map: np.ndarray) -> np.ndarray:
        """Unproject a full H×W depth map → (H, W, 3) world point grid."""
        h, w = depth_map.shape[:2]
        vs, us = np.mgrid[0:h, 0:w]
        return self.unproject(us, vs, depth_map)


def estimate_vfov(width: int, height: int) -> float:
    """Heuristic vertical FOV for an uncalibrated photo.

    Most phone / consumer cameras land around a 60–70° *horizontal* FOV.  We
    pick a moderate value and convert to vertical FOV given the aspect ratio,
    which is good enough for plausible scene scale.
    """
    hfov_deg = 65.0
    aspect = width / max(height, 1)
    hfov = math.radians(hfov_deg)
    vfov = 2.0 * math.atan(math.tan(hfov / 2.0) / max(aspect, 1e-6))
    return math.degrees(vfov)
