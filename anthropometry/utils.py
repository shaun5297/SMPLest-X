"""Utilities for first-pass anthropometry on canonical SMPL-X meshes."""

from __future__ import annotations

from pathlib import Path

import numpy as np


AXIS_NAMES = ("x", "y", "z")

# Native SMPL-X joint indices used only to verify coordinate semantics.
PELVIS_INDEX = 0
LEFT_FOOT_INDEX = 10
RIGHT_FOOT_INDEX = 11
HEAD_INDEX = 15
LEFT_SHOULDER_INDEX = 16
RIGHT_SHOULDER_INDEX = 17

# Surface shoulder landmarks used by SMPL-Anthropometry and the CVPR 2025
# A2B implementation for SMPL-X shoulder breadth. This asymmetric pair is
# retained as a literature baseline, not as the bilateral acromion proxy.
PUBLISHED_LEFT_SHOULDER_SURFACE_VERTEX_ID = 4442
PUBLISHED_RIGHT_SHOULDER_SURFACE_VERTEX_ID = 7218

# Bilaterally consistent external-surface proxy, frozen after neutral-template,
# five-shape, local-normal, and manual MeshLab confirmation. This is not a bony
# acromion ground-truth annotation.
ACROMION_SURFACE_PROXY_V1_LEFT_VERTEX_ID = 4482
ACROMION_SURFACE_PROXY_V1_RIGHT_VERTEX_ID = 7218

SHOULDER_LANDMARKS = {
    "literature_shoulder_breadth": {
        "left_vertex_id": PUBLISHED_LEFT_SHOULDER_SURFACE_VERTEX_ID,
        "right_vertex_id": PUBLISHED_RIGHT_SHOULDER_SURFACE_VERTEX_ID,
        "status": "literature_baseline",
    },
    "acromion_surface_proxy_v1": {
        "left_vertex_id": ACROMION_SURFACE_PROXY_V1_LEFT_VERTEX_ID,
        "right_vertex_id": ACROMION_SURFACE_PROXY_V1_RIGHT_VERTEX_ID,
        "status": "frozen_v1",
    },
}


def load_canonical_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate one zero-pose canonical SMPL-X mesh."""
    with np.load(path, allow_pickle=False) as data:
        if "vertices" not in data or "joints" not in data:
            raise KeyError(f"{path} must contain vertices and joints")
        vertices = np.asarray(data["vertices"], dtype=np.float32)
        joints = np.asarray(data["joints"], dtype=np.float32)
        units = str(data["units"]) if "units" in data else ""

    if vertices.shape != (10475, 3):
        raise ValueError(f"expected vertices shape (10475, 3), got {vertices.shape}")
    if joints.ndim != 2 or joints.shape[0] <= RIGHT_SHOULDER_INDEX or joints.shape[1] != 3:
        raise ValueError(f"expected native SMPL-X joints shaped (N, 3), got {joints.shape}")
    if units and units != "meters":
        raise ValueError(f"expected mesh units to be meters, got {units!r}")
    if not np.isfinite(vertices).all() or not np.isfinite(joints).all():
        raise ValueError("canonical vertices or joints contain NaN or Inf")
    return vertices, joints


def dominant_axis(vector: np.ndarray) -> str:
    """Return the axis carrying the largest absolute component."""
    return AXIS_NAMES[int(np.argmax(np.abs(vector)))]


def measure_axis_ranges(vertices: np.ndarray) -> dict[str, float]:
    minima = vertices.min(axis=0)
    maxima = vertices.max(axis=0)
    ranges = maxima - minima
    result: dict[str, float] = {}
    for index, axis in enumerate(AXIS_NAMES):
        result[f"{axis}_min_m"] = float(minima[index])
        result[f"{axis}_max_m"] = float(maxima[index])
        result[f"{axis}_range_m"] = float(ranges[index])
    return result


def euclidean_distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    """Return the straight-line distance between two 3D points in metres."""
    return float(np.linalg.norm(point_a - point_b))


def verify_smplx_axes(vertices: np.ndarray, joints: np.ndarray) -> dict[str, object]:
    """Infer axis semantics from SMPL-X landmarks and reject mismatches."""
    shoulder_delta = joints[LEFT_SHOULDER_INDEX] - joints[RIGHT_SHOULDER_INDEX]
    feet_midpoint = 0.5 * (joints[LEFT_FOOT_INDEX] + joints[RIGHT_FOOT_INDEX])
    head_to_feet = joints[HEAD_INDEX] - feet_midpoint
    ranges = np.ptp(vertices, axis=0)

    left_right_axis = dominant_axis(shoulder_delta)
    vertical_axis = dominant_axis(head_to_feet)
    depth_axis = AXIS_NAMES[int(np.argmin(ranges))]
    checks = {
        "left_right_is_x": left_right_axis == "x",
        "vertical_is_y": vertical_axis == "y",
        "depth_is_z": depth_axis == "z",
        "left_is_positive_x": float(shoulder_delta[0]) > 0.0,
        "head_is_positive_y_from_feet": float(head_to_feet[1]) > 0.0,
    }
    if not all(checks.values()):
        raise ValueError(
            "canonical mesh axis validation failed: "
            f"left_right={left_right_axis}, vertical={vertical_axis}, "
            f"depth={depth_axis}, checks={checks}"
        )

    return {
        "left_right_axis": left_right_axis,
        "vertical_axis": vertical_axis,
        "depth_axis": depth_axis,
        "left_right_landmarks": "SMPL-X left/right shoulder joints",
        "vertical_landmarks": "SMPL-X head joint and mean of left/right foot joints",
        "checks": checks,
    }


def infer_sample_name(path: Path) -> str:
    """Infer the source sample name from the current result directory layout."""
    if path.parent.name == "canonical":
        source_name = path.parent.parent.name
    else:
        source_name = path.stem
    return source_name.split("_", 1)[0]
