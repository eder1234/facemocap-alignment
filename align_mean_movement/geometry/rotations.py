from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def rotation_matrix_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    n = float(np.linalg.norm(axis))
    if n < 1e-12:
        return np.eye(3, dtype=float)
    axis = axis / n
    x, y, z = axis
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=float,
    )


def fixed_rot_xyz(deg_xyz: Tuple[float, float, float]) -> np.ndarray:
    rx, ry, rz = (math.radians(float(d)) for d in deg_xyz)
    Rx = rotation_matrix_from_axis_angle(np.array([1.0, 0.0, 0.0]), rx)
    Ry = rotation_matrix_from_axis_angle(np.array([0.0, 1.0, 0.0]), ry)
    Rz = rotation_matrix_from_axis_angle(np.array([0.0, 0.0, 1.0]), rz)
    return Rz @ Ry @ Rx
