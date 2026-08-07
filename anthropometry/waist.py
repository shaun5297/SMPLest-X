#!/usr/bin/env python3
"""Geometry-defined waist baseline on canonical SMPL-X meshes.

``geometry_waist_v0`` is the minimum raw horizontal torso circumference in the
native pelvis-to-spine2 interval.  No smoothing, curve fitting, calibration, or
ISO/anatomical waist semantics are applied.
"""

from __future__ import annotations

import numpy as np

from slicing import ENDPOINT_CLUSTER_TOLERANCE_M, EPS, slice_mesh
from torso import (
    DEFAULT_CENTERLINE_PROXIMITY_M,
    compute_torso_vertical_interval,
    interpolate_spine_centerline_xz,
    select_torso_contour,
)


GEOMETRY_WAIST_DEFINITION = "geometry_waist_v0"
GEOMETRY_WAIST_STATUS = "baseline"
DEFAULT_STEP_NORMALIZED_HEIGHT = 0.002
DEFAULT_LOCAL_JUMP_WARNING_CM = 3.0


def build_scan_heights(y_min: float, y_max: float, step_m: float) -> np.ndarray:
    """Build an inclusive discrete scan with exact lower and upper boundaries."""
    if not np.isfinite([y_min, y_max, step_m]).all():
        raise ValueError("scan bounds and step must be finite")
    if y_max <= y_min:
        raise ValueError("y_max must be greater than y_min")
    if step_m <= 0.0:
        raise ValueError("step_m must be positive")
    count = int(np.floor((y_max - y_min) / step_m))
    heights = y_min + np.arange(count + 1, dtype=np.float64) * step_m
    tolerance = max(1e-12, step_m * 1e-9)
    if y_max - heights[-1] > tolerance:
        heights = np.append(heights, y_max)
    else:
        heights[-1] = y_max
    return heights


def select_profile_minimum(records: list[dict[str, object]]) -> dict[str, object]:
    """Select the raw discrete minimum without replacing boundary minima."""
    if not records:
        raise ValueError("profile records must be non-empty")
    perimeters = np.asarray([record["perimeter_m"] for record in records], dtype=np.float64)
    if not np.isfinite(perimeters).all() or np.any(perimeters <= 0.0):
        raise ValueError("profile perimeters must be finite and positive")
    selected_index = int(np.argmin(perimeters))
    return {
        "selected_index": selected_index,
        "boundary_minimum": selected_index in (0, len(records) - 1),
        "selected_record": records[selected_index],
    }


def build_local_stability_window(
    records: list[dict[str, object]],
    selected_index: int,
    *,
    radius: int = 2,
) -> dict[str, object]:
    """Return the raw minimum neighborhood and adjacent circumference changes."""
    if radius < 1:
        raise ValueError("radius must be at least one")
    if not 0 <= selected_index < len(records):
        raise IndexError("selected_index lies outside profile records")
    start = max(0, selected_index - radius)
    stop = min(len(records), selected_index + radius + 1)
    neighborhood = []
    for index in range(start, stop):
        record = records[index]
        neighborhood.append(
            {
                "index": index,
                "offset_steps": index - selected_index,
                "plane_y_m": record["plane_y_m"],
                "normalized_height": record["normalized_height"],
                "perimeter_m": record["perimeter_m"],
                "perimeter_cm": record["perimeter_cm"],
            }
        )
    adjacent_changes = [
        {
            "from_index": index,
            "to_index": index + 1,
            "absolute_change_cm": abs(
                records[index + 1]["perimeter_cm"] - records[index]["perimeter_cm"]
            ),
        }
        for index in range(len(records) - 1)
    ]
    return {
        "radius_steps": radius,
        "neighborhood": neighborhood,
        "adjacent_changes": adjacent_changes,
        "max_adjacent_change_cm": max(
            (item["absolute_change_cm"] for item in adjacent_changes), default=0.0
        ),
    }


def scan_geometry_waist(
    vertices: np.ndarray,
    faces: np.ndarray,
    joints: np.ndarray,
    *,
    step_normalized_height: float = DEFAULT_STEP_NORMALIZED_HEIGHT,
    eps: float = EPS,
    endpoint_tolerance: float = ENDPOINT_CLUSTER_TOLERANCE_M,
    centerline_proximity_m: float = DEFAULT_CENTERLINE_PROXIMITY_M,
    local_jump_warning_cm: float = DEFAULT_LOCAL_JUMP_WARNING_CM,
) -> dict[str, object]:
    """Compute the raw discrete ``geometry_waist_v0`` profile and minimum."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    joints = np.asarray(joints, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("vertices must be a finite array shaped (N, 3)")
    if step_normalized_height <= 0.0:
        raise ValueError("step_normalized_height must be positive")
    if local_jump_warning_cm <= 0.0:
        raise ValueError("local_jump_warning_cm must be positive")

    mesh_min_y = float(vertices[:, 1].min())
    mesh_max_y = float(vertices[:, 1].max())
    raw_height = mesh_max_y - mesh_min_y
    if raw_height <= 0.0:
        raise ValueError("mesh must have positive raw height")
    step_m = step_normalized_height * raw_height
    interval = compute_torso_vertical_interval(joints)
    heights = build_scan_heights(interval["y_min_m"], interval["y_max_m"], step_m)

    records = []
    fallback_count = 0
    for index, plane_y in enumerate(heights):
        sliced = slice_mesh(
            vertices,
            faces,
            float(plane_y),
            eps=eps,
            endpoint_tolerance=endpoint_tolerance,
        )
        connectivity = sliced["diagnostics"]["connectivity"]
        if connectivity["invalid_component_count"] != 0:
            raise RuntimeError(f"slice {index} contains an invalid contour component")
        centerline = interpolate_spine_centerline_xz(joints, float(plane_y))
        selection = select_torso_contour(
            sliced["contours"],
            centerline,
            max_centerline_proximity_m=centerline_proximity_m,
        )
        selected_contour = selection["selected_contour"]
        selected_metrics = selection["selected_metrics"]
        fallback_used = selection["selection_method"] != "spine_centerline_containment_then_area"
        fallback_count += int(fallback_used)
        records.append(
            {
                "index": index,
                "plane_y_m": float(plane_y),
                "normalized_height": float((plane_y - mesh_min_y) / raw_height),
                "perimeter_m": selected_contour["perimeter_m"],
                "perimeter_cm": selected_contour["perimeter_cm"],
                "area_m2": selected_contour["area_m2"],
                "area_cm2": selected_contour["area_cm2"],
                "centroid_xz_m": selected_contour["centroid_xz_m"],
                "centerline_xz_m": centerline.tolist(),
                "centerline_to_centroid_m": selected_metrics["centroid_distance_m"],
                "centerline_to_centroid_cm": selected_metrics["centroid_distance_m"] * 100.0,
                "centerline_inside": selected_metrics["centerline_inside"],
                "num_contours": len(sliced["contours"]),
                "selected_contour_id": selection["selected_contour_id"],
                "selection_mode": selection["selection_method"],
                "fallback_used": fallback_used,
                "unique_segment_count": connectivity["unique_segment_count"],
                "remaining_duplicate_segments": connectivity["remaining_duplicate_segments"],
                "invalid_component_count": connectivity["invalid_component_count"],
            }
        )

    minimum = select_profile_minimum(records)
    selected_index = minimum["selected_index"]
    selected_record = minimum["selected_record"]
    selected_slice = slice_mesh(
        vertices,
        faces,
        selected_record["plane_y_m"],
        eps=eps,
        endpoint_tolerance=endpoint_tolerance,
    )
    selected_centerline = np.asarray(selected_record["centerline_xz_m"], dtype=np.float64)
    selected_again = select_torso_contour(
        selected_slice["contours"],
        selected_centerline,
        max_centerline_proximity_m=centerline_proximity_m,
    )
    selected_contour = selected_again["selected_contour"]
    local_stability = build_local_stability_window(records, selected_index, radius=2)
    warnings = []
    if minimum["boundary_minimum"]:
        warnings.append("minimum lies on the pelvis-to-spine2 search boundary")
    if fallback_count:
        warnings.append(f"torso selector used bounded fallback on {fallback_count} slices")
    if local_stability["max_adjacent_change_cm"] > local_jump_warning_cm:
        warnings.append(
            "raw profile contains an adjacent-layer circumference change above "
            f"{local_jump_warning_cm:.3f} cm"
        )

    interval = {
        **interval,
        "normalized_y_min": float((interval["y_min_m"] - mesh_min_y) / raw_height),
        "normalized_y_max": float((interval["y_max_m"] - mesh_min_y) / raw_height),
    }
    return {
        "definition": GEOMETRY_WAIST_DEFINITION,
        "status": GEOMETRY_WAIST_STATUS,
        "definition_text": (
            "The minimum horizontal torso circumference within the pelvis-to-spine2 "
            "skeleton-constrained interval of the zero-pose canonical SMPL-X mesh."
        ),
        "measurement_space": "raw canonical SMPL-X geometry",
        "metric_calibration": "none",
        "search_interval": interval,
        "scan_parameters": {
            "step_normalized_height": float(step_normalized_height),
            "step_m": float(step_m),
            "num_slices": int(len(records)),
            "smoothing": "none",
            "curve_fitting": "none",
            "minimum_method": "raw_discrete_argmin",
            "local_jump_warning_cm": float(local_jump_warning_cm),
        },
        "profile": records,
        "selected": {
            **selected_record,
            "ordered_points_m": selected_contour["ordered_points_m"],
            "num_points": selected_contour["num_points"],
        },
        "selected_index": selected_index,
        "boundary_minimum": minimum["boundary_minimum"],
        "fallback_count": fallback_count,
        "local_stability": local_stability,
        "warnings": warnings,
    }
