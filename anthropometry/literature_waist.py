#!/usr/bin/env python3
"""Independent landmark-defined waist reference for canonical SMPL-X.

``literature_waist_v1`` adapts the public SMPL-Anthropometry SMPL-X waist
landmarks to the horizontal slicing engine.  It performs exactly one slice at
the mean Y coordinate of the published front/back belly-button vertices and
never reads or searches a circumference profile.
"""

from __future__ import annotations

import numpy as np

from slicing import ENDPOINT_CLUSTER_TOLERANCE_M, EPS, slice_mesh
from torso import (
    DEFAULT_CENTERLINE_PROXIMITY_M,
    PELVIS_JOINT_INDEX,
    SPINE3_JOINT_INDEX,
    compute_torso_vertical_interval,
    interpolate_spine_centerline_xz,
    select_torso_contour,
)


LITERATURE_WAIST_DEFINITION = "literature_waist_v1"
LITERATURE_WAIST_STATUS = "baseline"
BELLY_BUTTON_VERTEX_ID = 5939
BACK_BELLY_BUTTON_VERTEX_ID = 5941

LITERATURE_SOURCES = [
    {
        "name": "SMPL-Anthropometry landmark definitions",
        "url": (
            "https://github.com/DavidBoja/SMPL-Anthropometry/blob/master/"
            "landmark_definitions.py"
        ),
        "relevance": "SMPL-X BELLY_BUTTON=5939 and BACK_BELLY_BUTTON=5941",
    },
    {
        "name": "SMPL-Anthropometry measurement definitions",
        "url": (
            "https://github.com/DavidBoja/SMPL-Anthropometry/blob/master/"
            "measurement_definitions.py"
        ),
        "relevance": (
            "waist uses both belly-button landmarks and the pelvis-to-spine3 plane normal"
        ),
    },
    {
        "name": "SMPL-Anthropometry plane construction",
        "url": (
            "https://github.com/DavidBoja/SMPL-Anthropometry/blob/master/measure.py"
        ),
        "relevance": (
            "plane origin is the landmark mean and plane normal is the selected joint vector"
        ),
    },
    {
        "name": "CDC/NCHS Waist Circumference Measurement Methodology Study",
        "url": "https://www.cdc.gov/nchs/data/series/sr_02/sr02_182-508.pdf",
        "relevance": (
            "documents that WHO waist is measured midway between the uppermost lateral "
            "ilium and the lower margin of the last palpable rib"
        ),
    },
]


def _validate_vertices(vertices: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), got {vertices.shape}")
    if len(vertices) <= BACK_BELLY_BUTTON_VERTEX_ID:
        raise ValueError("vertices do not contain the published SMPL-X waist landmarks")
    if not np.isfinite(vertices).all():
        raise ValueError("vertices contain NaN or Inf")
    return vertices


def compute_literature_waist_plane(vertices: np.ndarray) -> dict[str, object]:
    """Return the horizontal plane adapted from the published landmark origin."""
    vertices = _validate_vertices(vertices)
    front = vertices[BELLY_BUTTON_VERTEX_ID]
    back = vertices[BACK_BELLY_BUTTON_VERTEX_ID]
    origin = 0.5 * (front + back)
    return {
        "plane_y_m": float(origin[1]),
        "plane_origin_m": origin.tolist(),
        "front_landmark": {
            "name": "BELLY_BUTTON",
            "vertex_id": BELLY_BUTTON_VERTEX_ID,
            "coordinate_m": front.tolist(),
        },
        "back_landmark": {
            "name": "BACK_BELLY_BUTTON",
            "vertex_id": BACK_BELLY_BUTTON_VERTEX_ID,
            "coordinate_m": back.tolist(),
        },
        "landmark_absolute_y_mismatch_m": float(abs(front[1] - back[1])),
        "landmark_absolute_y_mismatch_mm": float(abs(front[1] - back[1]) * 1000.0),
        "origin_rule": "mean of published front/back SMPL-X belly-button vertices",
        "horizontal_adaptation": True,
    }


def measure_literature_waist(
    vertices: np.ndarray,
    faces: np.ndarray,
    joints: np.ndarray,
    *,
    eps: float = EPS,
    endpoint_tolerance: float = ENDPOINT_CLUSTER_TOLERANCE_M,
    centerline_proximity_m: float = DEFAULT_CENTERLINE_PROXIMITY_M,
) -> dict[str, object]:
    """Measure one independent horizontal landmark-defined waist slice."""
    vertices = _validate_vertices(vertices)
    joints = np.asarray(joints, dtype=np.float64)
    if joints.ndim != 2 or joints.shape[1] != 3 or len(joints) <= SPINE3_JOINT_INDEX:
        raise ValueError("joints do not contain native pelvis and spine3 joints")
    if not np.isfinite(joints).all():
        raise ValueError("joints contain NaN or Inf")

    plane = compute_literature_waist_plane(vertices)
    plane_y = plane["plane_y_m"]
    interval = compute_torso_vertical_interval(joints)
    if not interval["y_min_m"] <= plane_y <= interval["y_max_m"]:
        raise ValueError("literature waist plane lies outside the pelvis-to-spine2 interval")

    published_normal = joints[PELVIS_JOINT_INDEX] - joints[SPINE3_JOINT_INDEX]
    published_normal /= np.linalg.norm(published_normal)
    tilt_degrees = float(np.degrees(np.arccos(np.clip(abs(published_normal[1]), 0.0, 1.0))))
    origin = np.asarray(plane["plane_origin_m"], dtype=np.float64)
    landmark_points = vertices[[BELLY_BUTTON_VERTEX_ID, BACK_BELLY_BUTTON_VERTEX_ID]]
    published_plane_residuals = np.abs((landmark_points - origin) @ published_normal)

    sliced = slice_mesh(
        vertices,
        faces,
        plane_y,
        eps=eps,
        endpoint_tolerance=endpoint_tolerance,
    )
    connectivity = sliced["diagnostics"]["connectivity"]
    if connectivity["invalid_component_count"] != 0:
        raise RuntimeError("literature waist slice contains an invalid contour component")
    centerline = interpolate_spine_centerline_xz(joints, plane_y)
    selection = select_torso_contour(
        sliced["contours"],
        centerline,
        max_centerline_proximity_m=centerline_proximity_m,
    )
    contour = selection["selected_contour"]
    metrics = selection["selected_metrics"]
    mesh_min_y = float(vertices[:, 1].min())
    raw_height = float(np.ptp(vertices[:, 1]))

    return {
        "definition": LITERATURE_WAIST_DEFINITION,
        "status": LITERATURE_WAIST_STATUS,
        "definition_text": (
            "A single horizontal canonical SMPL-X slice at the mean Y coordinate of "
            "published front/back belly-button landmarks v5939 and v5941."
        ),
        "independent_of_circumference_minimum": True,
        "measurement_space": "raw canonical SMPL-X geometry",
        "metric_calibration": "none",
        "plane_definition": {
            **plane,
            "normalized_height": float((plane_y - mesh_min_y) / raw_height),
            "published_source_plane_normal_unit": published_normal.tolist(),
            "published_source_normal_tilt_from_vertical_degrees": tilt_degrees,
            "published_landmark_to_source_plane_residual_mm": (
                published_plane_residuals * 1000.0
            ).tolist(),
            "adaptation_note": (
                "The source implementation uses the landmark mean as a 3D plane origin "
                "and pelvis-to-spine3 as its normal. This baseline preserves the source "
                "origin height but uses y=constant for the canonical horizontal engine."
            ),
        },
        "search_or_scan": "none; one direct landmark-defined slice",
        "num_contours": len(sliced["contours"]),
        "selected_contour_id": selection["selected_contour_id"],
        "selection_mode": selection["selection_method"],
        "fallback_used": selection["selection_method"] != "spine_centerline_containment_then_area",
        "centerline_xz_m": centerline.tolist(),
        "centerline_inside": metrics["centerline_inside"],
        "centerline_to_centroid_m": metrics["centroid_distance_m"],
        "perimeter_m": contour["perimeter_m"],
        "perimeter_cm": contour["perimeter_cm"],
        "area_m2": contour["area_m2"],
        "area_cm2": contour["area_cm2"],
        "centroid_xz_m": contour["centroid_xz_m"],
        "ordered_points_m": contour["ordered_points_m"],
        "num_points": contour["num_points"],
        "diagnostics": {
            "unique_segment_count": connectivity["unique_segment_count"],
            "remaining_duplicate_segments": connectivity["remaining_duplicate_segments"],
            "invalid_component_count": connectivity["invalid_component_count"],
        },
        "scientific_boundary": (
            "This is a reproducible horizontal adaptation of an open-source SMPL-X "
            "landmark definition. It is not an ISO or WHO waist measurement because "
            "SMPL-X does not explicitly annotate the last palpable rib and iliac crest."
        ),
        "sources": LITERATURE_SOURCES,
    }
