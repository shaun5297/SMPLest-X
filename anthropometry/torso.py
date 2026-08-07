#!/usr/bin/env python3
"""Skeleton-constrained torso contour selection for Step 2.5A.

This module identifies a torso contour at a supplied height.  It does not scan
for, define, or measure a waist.
"""

from __future__ import annotations

import numpy as np


PELVIS_JOINT_INDEX = 0
SPINE1_JOINT_INDEX = 3
SPINE2_JOINT_INDEX = 6
SPINE3_JOINT_INDEX = 9
TORSO_CENTERLINE_JOINT_INDICES = (
    PELVIS_JOINT_INDEX,
    SPINE1_JOINT_INDEX,
    SPINE2_JOINT_INDEX,
    SPINE3_JOINT_INDEX,
)
DEFAULT_CENTERLINE_PROXIMITY_M = 0.03


class TorsoSelectionError(ValueError):
    """Raised when no contour is compatible with the skeleton centerline."""


def _validate_joints(joints: np.ndarray) -> np.ndarray:
    joints = np.asarray(joints, dtype=np.float64)
    if joints.ndim != 2 or joints.shape[1] != 3:
        raise ValueError(f"joints must have shape (J, 3), got {joints.shape}")
    if len(joints) <= SPINE3_JOINT_INDEX:
        raise ValueError("joints do not contain the native SMPL-X spine chain")
    if not np.isfinite(joints).all():
        raise ValueError("joints contain NaN or Inf")
    return joints


def compute_torso_vertical_interval(joints: np.ndarray) -> dict[str, object]:
    """Return the skeleton-bounded lower-torso interval [pelvis, spine2]."""
    joints = _validate_joints(joints)
    pelvis_y = float(joints[PELVIS_JOINT_INDEX, 1])
    spine1_y = float(joints[SPINE1_JOINT_INDEX, 1])
    spine2_y = float(joints[SPINE2_JOINT_INDEX, 1])
    spine3_y = float(joints[SPINE3_JOINT_INDEX, 1])
    if not pelvis_y < spine1_y < spine2_y < spine3_y:
        raise ValueError(
            "expected canonical SMPL-X spine ordering pelvis < spine1 < spine2 < spine3"
        )
    return {
        "y_min_m": pelvis_y,
        "y_max_m": spine2_y,
        "lower_bound": {"joint": "pelvis", "joint_index": PELVIS_JOINT_INDEX},
        "upper_bound": {"joint": "spine2", "joint_index": SPINE2_JOINT_INDEX},
        "purpose": (
            "lower-torso contour-selection interval; excludes the crotch/upper legs "
            "below pelvis and the upper thorax/shoulders above spine2"
        ),
    }


def interpolate_spine_centerline_xz(joints: np.ndarray, plane_y: float) -> np.ndarray:
    """Piecewise-linearly interpolate the pelvis-to-spine3 centerline at ``plane_y``."""
    joints = _validate_joints(joints)
    if not np.isfinite(plane_y):
        raise ValueError("plane_y must be finite")
    chain = joints[list(TORSO_CENTERLINE_JOINT_INDICES)]
    order = np.argsort(chain[:, 1])
    chain = chain[order]
    if np.any(np.diff(chain[:, 1]) <= 0.0):
        raise ValueError("spine centerline joint heights must be strictly increasing")
    if plane_y < chain[0, 1] or plane_y > chain[-1, 1]:
        raise ValueError("plane_y lies outside the pelvis-to-spine3 centerline")
    x = np.interp(plane_y, chain[:, 1], chain[:, 0])
    z = np.interp(plane_y, chain[:, 1], chain[:, 2])
    return np.asarray([x, z], dtype=np.float64)


def _contour_xz(contour: dict[str, object]) -> np.ndarray:
    if "ordered_points_m" not in contour:
        raise KeyError("contour must contain ordered_points_m")
    points = np.asarray(contour["ordered_points_m"], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("contour ordered_points_m must contain at least three 3D points")
    if not np.isfinite(points).all():
        raise ValueError("contour points contain NaN or Inf")
    return points[:, [0, 2]]


def point_in_polygon_xz(point_xz: np.ndarray, polygon_xz: np.ndarray) -> bool:
    """Return whether a point lies inside a simple X-Z polygon by ray casting."""
    point = np.asarray(point_xz, dtype=np.float64)
    polygon = np.asarray(polygon_xz, dtype=np.float64)
    if point.shape != (2,):
        raise ValueError(f"point_xz must have shape (2,), got {point.shape}")
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        raise ValueError("polygon_xz must have shape (N, 2), N >= 3")
    if not np.isfinite(point).all() or not np.isfinite(polygon).all():
        raise ValueError("point or polygon contains NaN or Inf")

    x, z = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, z1 = previous
        x2, z2 = current
        if (z1 > z) != (z2 > z):
            crossing_x = x1 + (z - z1) * (x2 - x1) / (z2 - z1)
            if x < crossing_x:
                inside = not inside
        previous = current
    return inside


def distance_to_polygon_boundary_xz(point_xz: np.ndarray, polygon_xz: np.ndarray) -> float:
    """Return the shortest Euclidean X-Z distance from a point to polygon edges."""
    point = np.asarray(point_xz, dtype=np.float64)
    polygon = np.asarray(polygon_xz, dtype=np.float64)
    if point.shape != (2,) or polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 2:
        raise ValueError("invalid point or polygon shape")
    starts = polygon
    ends = np.roll(polygon, -1, axis=0)
    edges = ends - starts
    squared_lengths = np.einsum("ij,ij->i", edges, edges)
    offsets = point - starts
    parameters = np.divide(
        np.einsum("ij,ij->i", offsets, edges),
        squared_lengths,
        out=np.zeros_like(squared_lengths),
        where=squared_lengths > 0.0,
    )
    parameters = np.clip(parameters, 0.0, 1.0)
    projections = starts + parameters[:, None] * edges
    return float(np.linalg.norm(projections - point, axis=1).min())


def select_torso_contour(
    contours: list[dict[str, object]],
    centerline_xz_m: np.ndarray,
    *,
    max_centerline_proximity_m: float = DEFAULT_CENTERLINE_PROXIMITY_M,
) -> dict[str, object]:
    """Select a torso contour using centerline containment before area.

    A spine-centerline-containing contour is always preferred.  Area and
    centroid distance only rank contours after that anatomical/geometric gate.
    If no contour contains the centerline, a bounded nearest-boundary fallback
    is allowed and recorded explicitly.
    """
    centerline = np.asarray(centerline_xz_m, dtype=np.float64)
    if centerline.shape != (2,) or not np.isfinite(centerline).all():
        raise ValueError("centerline_xz_m must be a finite two-vector")
    if max_centerline_proximity_m <= 0.0:
        raise ValueError("max_centerline_proximity_m must be positive")
    if not contours:
        raise TorsoSelectionError("no contours were supplied")

    candidates = []
    for list_index, contour in enumerate(contours):
        polygon = _contour_xz(contour)
        boundary_distance = distance_to_polygon_boundary_xz(centerline, polygon)
        inside = point_in_polygon_xz(centerline, polygon) or boundary_distance <= 1e-10
        centroid = np.asarray(contour["centroid_xz_m"], dtype=np.float64)
        area = float(contour["area_m2"])
        perimeter = float(contour["perimeter_m"])
        if centroid.shape != (2,) or not np.isfinite([*centroid, area, perimeter]).all():
            raise ValueError("contour metrics contain NaN or Inf")
        candidates.append(
            {
                "list_index": int(list_index),
                "contour_id": int(contour["id"]),
                "centerline_inside": bool(inside),
                "centerline_boundary_distance_m": float(boundary_distance),
                "centroid_distance_m": float(np.linalg.norm(centroid - centerline)),
                "area_m2": area,
                "perimeter_m": perimeter,
            }
        )

    containing = [item for item in candidates if item["centerline_inside"]]
    if containing:
        selected_metrics = max(
            containing,
            key=lambda item: (
                item["area_m2"],
                -item["centroid_distance_m"],
                -item["contour_id"],
            ),
        )
        method = "spine_centerline_containment_then_area"
    else:
        selected_metrics = min(
            candidates,
            key=lambda item: (
                item["centerline_boundary_distance_m"],
                item["centroid_distance_m"],
                -item["area_m2"],
            ),
        )
        if selected_metrics["centerline_boundary_distance_m"] > max_centerline_proximity_m:
            raise TorsoSelectionError(
                "no contour contains or lies within "
                f"{max_centerline_proximity_m:.4f} m of the spine centerline"
            )
        method = "bounded_centerline_proximity_fallback"

    selected_contour = contours[selected_metrics["list_index"]]
    return {
        "selected_contour": selected_contour,
        "selected_contour_id": selected_metrics["contour_id"],
        "selection_method": method,
        "centerline_xz_m": centerline.tolist(),
        "max_centerline_proximity_m": float(max_centerline_proximity_m),
        "selected_metrics": selected_metrics,
        "candidate_metrics": candidates,
    }
