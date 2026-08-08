#!/usr/bin/env python3
"""Direct landmark-plane chest validation for Step 2.7B.

The two public definitions remain independent.  This module never scans for an
extreme, clamps a plane, or moves a landmark-defined plane.
"""

from __future__ import annotations

import math

import numpy as np

from slicing import EPS, slice_mesh
from torso import (
    DEFAULT_CENTERLINE_PROXIMITY_M,
    PELVIS_JOINT_INDEX,
    SPINE1_JOINT_INDEX,
    SPINE2_JOINT_INDEX,
    SPINE3_JOINT_INDEX,
    select_torso_contour,
)


LITERATURE_CHEST_DEFINITION = "literature_chest_v1"
LITERATURE_CHEST_CANDIDATE = LITERATURE_CHEST_DEFINITION
LITERATURE_CHEST_STATUS = "baseline"
FOCUSED_SHAPY_CHEST_CONTROL = "focused_shapy_chest_control"
LEFT_NIPPLE_VERTEX_ID = 3572
RIGHT_NIPPLE_VERTEX_ID = 8340
FOCUSED_NIPPLE_FACE_ID = 18402
FOCUSED_NIPPLE_BARYCENTRIC = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
PROBE_OFFSETS_HEIGHT_FRACTION = (-0.004, -0.002, 0.0, 0.002, 0.004)

# The public single-vertex plane can pass within one micrometre of a second
# surface crossing.  The generic 1e-6 m endpoint tolerance would erase that
# valid edge.  A tighter clustering tolerance preserves the exact plane.
CHEST_ENDPOINT_CLUSTER_TOLERANCE_M = 1e-7
ARM_TORSO_MERGE_XSPAN_TO_SHOULDER_RATIO = 1.45
LEFT_SHOULDER_JOINT_INDEX = 16
RIGHT_SHOULDER_JOINT_INDEX = 17
NECK_JOINT_INDEX = 12


class ChestSelectionError(ValueError):
    """Raised when a thoracic contour cannot be selected safely."""


def _validate_inputs(
    vertices: np.ndarray, faces: np.ndarray, joints: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    joints = np.asarray(joints, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if len(vertices) <= RIGHT_NIPPLE_VERTEX_ID:
        raise ValueError("vertices do not contain the public nipple landmarks")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) <= FOCUSED_NIPPLE_FACE_ID:
        raise ValueError("faces do not contain the Focused/SHAPY chest anchor face")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("faces contain an invalid vertex index")
    if joints.ndim != 2 or joints.shape[1] != 3 or len(joints) <= RIGHT_SHOULDER_JOINT_INDEX:
        raise ValueError("joints do not contain the thoracic spine and shoulders")
    if not np.isfinite(vertices).all() or not np.isfinite(joints).all():
        raise ValueError("vertices or joints contain NaN or Inf")
    return vertices, faces, joints


def compute_candidate_planes(
    vertices: np.ndarray, faces: np.ndarray
) -> dict[str, dict[str, object]]:
    """Evaluate both public landmarks on one subject canonical mesh."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    left = vertices[LEFT_NIPPLE_VERTEX_ID]
    right = vertices[RIGHT_NIPPLE_VERTEX_ID]
    focused_face_vertices = vertices[faces[FOCUSED_NIPPLE_FACE_ID]]
    focused = np.sum(
        focused_face_vertices * FOCUSED_NIPPLE_BARYCENTRIC[:, None], axis=0
    )
    return {
        LITERATURE_CHEST_CANDIDATE: {
            "definition": LITERATURE_CHEST_CANDIDATE,
            "status": LITERATURE_CHEST_STATUS,
            "plane_y_m": float(0.5 * (left[1] + right[1])),
            "plane_rule": "y = mean(Y(v3572), Y(v8340))",
            "landmarks": [
                {"name": "LEFT_NIPPLE", "vertex_id": LEFT_NIPPLE_VERTEX_ID, "coordinate_m": left.tolist()},
                {"name": "RIGHT_NIPPLE", "vertex_id": RIGHT_NIPPLE_VERTEX_ID, "coordinate_m": right.tolist()},
            ],
            "bilateral_y_mismatch_m": float(abs(left[1] - right[1])),
            "representation": "subject-deformed fixed-topology vertices",
        },
        FOCUSED_SHAPY_CHEST_CONTROL: {
            "definition": FOCUSED_SHAPY_CHEST_CONTROL,
            "status": "control",
            "plane_y_m": float(focused[1]),
            "plane_rule": "y = Y(face18402, barycentric=[0,0,1])",
            "landmarks": [
                {
                    "name": "NippleRight",
                    "face_id": FOCUSED_NIPPLE_FACE_ID,
                    "face_vertex_ids": faces[FOCUSED_NIPPLE_FACE_ID].tolist(),
                    "barycentric": FOCUSED_NIPPLE_BARYCENTRIC.tolist(),
                    "coordinate_m": focused.tolist(),
                }
            ],
            "representation": "subject-deformed face+barycentric surface anchor",
        },
    }


def thoracic_centerline_xz(
    joints: np.ndarray, plane_y: float
) -> tuple[np.ndarray, str]:
    """Evaluate the spine-chain X-Z centerline at a chest plane.

    The chain extends from spine3 to the native neck joint so nipple and
    shoulder-level planes remain true skeleton interpolation.  The measurement
    plane itself is never altered.
    """
    joints = np.asarray(joints, dtype=np.float64)
    indices = (
        PELVIS_JOINT_INDEX,
        SPINE1_JOINT_INDEX,
        SPINE2_JOINT_INDEX,
        SPINE3_JOINT_INDEX,
        NECK_JOINT_INDEX,
    )
    chain = joints[list(indices)]
    if np.any(np.diff(chain[:, 1]) <= 0.0):
        raise ValueError("expected pelvis < spine1 < spine2 < spine3 < neck")
    if plane_y < chain[0, 1]:
        raise ChestSelectionError("chest plane lies below the pelvis-to-spine3 chain")
    if plane_y <= chain[-1, 1]:
        x = np.interp(plane_y, chain[:, 1], chain[:, 0])
        z = np.interp(plane_y, chain[:, 1], chain[:, 2])
        return np.asarray([x, z]), "piecewise_thoracic_skeleton_interpolation"
    first, second = chain[-2], chain[-1]
    fraction = (plane_y - first[1]) / (second[1] - first[1])
    point = first + fraction * (second - first)
    return point[[0, 2]], "spine3_to_neck_linear_extrapolation"


def _compactness(area_m2: float, perimeter_m: float) -> float:
    if perimeter_m <= 0.0:
        return 0.0
    return float(4.0 * math.pi * area_m2 / (perimeter_m * perimeter_m))


def _enrich_contour(
    contour: dict[str, object], centerline_xz: np.ndarray
) -> dict[str, object]:
    points = np.asarray(contour["ordered_points_m"], dtype=np.float64)
    centroid = np.asarray(contour["centroid_xz_m"], dtype=np.float64)
    area = float(contour["area_m2"])
    perimeter = float(contour["perimeter_m"])
    return {
        "contour_id": int(contour["id"]),
        "perimeter_m": perimeter,
        "perimeter_cm": perimeter * 100.0,
        "area_m2": area,
        "area_cm2": area * 10_000.0,
        "centroid_xz_m": centroid.tolist(),
        "centerline_to_centroid_m": float(np.linalg.norm(centroid - centerline_xz)),
        "compactness": _compactness(area, perimeter),
        "x_span_m": float(np.ptp(points[:, 0])),
        "z_span_m": float(np.ptp(points[:, 2])),
        "num_points": int(len(points)),
    }


def select_chest_torso_contour(
    contours: list[dict[str, object]],
    joints: np.ndarray,
    plane_y: float,
    *,
    max_centerline_proximity_m: float = DEFAULT_CENTERLINE_PROXIMITY_M,
) -> dict[str, object]:
    """Select the thoracic loop by centerline containment before area."""
    centerline, centerline_mode = thoracic_centerline_xz(joints, plane_y)
    selection = select_torso_contour(
        contours,
        centerline,
        max_centerline_proximity_m=max_centerline_proximity_m,
    )
    selected_id = int(selection["selected_contour_id"])
    all_metrics = [_enrich_contour(item, centerline) for item in contours]
    selected_metrics = next(item for item in all_metrics if item["contour_id"] == selected_id)
    left_shoulder_x = float(joints[LEFT_SHOULDER_JOINT_INDEX, 0])
    right_shoulder_x = float(joints[RIGHT_SHOULDER_JOINT_INDEX, 0])
    shoulder_x_min = min(left_shoulder_x, right_shoulder_x)
    shoulder_x_max = max(left_shoulder_x, right_shoulder_x)
    for item in all_metrics:
        if item["contour_id"] == selected_id:
            item["role"] = "thoracic_torso"
        elif item["centroid_xz_m"][0] > shoulder_x_max:
            item["role"] = "left_lateral_upper_limb_loop"
        elif item["centroid_xz_m"][0] < shoulder_x_min:
            item["role"] = "right_lateral_upper_limb_loop"
        else:
            item["role"] = "central_auxiliary_loop"
    shoulder_x_span = float(
        abs(joints[LEFT_SHOULDER_JOINT_INDEX, 0] - joints[RIGHT_SHOULDER_JOINT_INDEX, 0])
    )
    merge_ratio = selected_metrics["x_span_m"] / shoulder_x_span
    possible_merge = merge_ratio > ARM_TORSO_MERGE_XSPAN_TO_SHOULDER_RATIO
    return {
        "selected_contour": selection["selected_contour"],
        "selected_contour_id": selected_id,
        "selection_mode": selection["selection_method"],
        "fallback_used": selection["selection_method"] != "spine_centerline_containment_then_area",
        "centerline_xz_m": centerline.tolist(),
        "centerline_mode": centerline_mode,
        "centerline_inside": bool(selection["selected_metrics"]["centerline_inside"]),
        "selected_metrics": selected_metrics,
        "all_contour_metrics": all_metrics,
        "shoulder_joint_x_span_m": shoulder_x_span,
        "selected_x_span_to_shoulder_ratio": float(merge_ratio),
        "possible_arm_torso_merge": bool(possible_merge),
        "merge_flag_rule": (
            f"selected x-span / shoulder-joint x-span > "
            f"{ARM_TORSO_MERGE_XSPAN_TO_SHOULDER_RATIO:.2f}"
        ),
    }


def evaluate_chest_plane(
    vertices: np.ndarray,
    faces: np.ndarray,
    joints: np.ndarray,
    plane_y: float,
) -> dict[str, object]:
    """Slice and select one exact horizontal chest plane."""
    sliced = slice_mesh(
        vertices,
        faces,
        plane_y,
        eps=EPS,
        endpoint_tolerance=CHEST_ENDPOINT_CLUSTER_TOLERANCE_M,
    )
    connectivity = sliced["diagnostics"]["connectivity"]
    if not sliced["contours"]:
        raise ChestSelectionError("chest plane produced no closed contour")
    selection = select_chest_torso_contour(sliced["contours"], joints, plane_y)
    return {
        "plane_y_m": float(plane_y),
        "num_contours": int(len(sliced["contours"])),
        "topology_class": (
            "single_thoracic_loop"
            if len(sliced["contours"]) == 1
            else "thoracic_plus_auxiliary_loops"
        ),
        **{key: value for key, value in selection.items() if key != "selected_contour"},
        "selected_ordered_points_m": selection["selected_contour"]["ordered_points_m"],
        "diagnostics": {
            "invalid_component_count": int(connectivity["invalid_component_count"]),
            "remaining_duplicate_segments": int(connectivity["remaining_duplicate_segments"]),
            "endpoint_cluster_tolerance_m": CHEST_ENDPOINT_CLUSTER_TOLERANCE_M,
        },
    }


def evaluate_chest_candidate(
    vertices: np.ndarray,
    faces: np.ndarray,
    joints: np.ndarray,
    plane_definition: dict[str, object],
) -> dict[str, object]:
    """Evaluate one main plane plus non-optimizing local topology probes."""
    vertices, faces, joints = _validate_inputs(vertices, faces, joints)
    base_y = float(plane_definition["plane_y_m"])
    raw_height = float(np.ptp(vertices[:, 1]))
    mesh_min_y = float(vertices[:, 1].min())
    probes = []
    for offset in PROBE_OFFSETS_HEIGHT_FRACTION:
        plane_y = base_y + offset * raw_height
        result = evaluate_chest_plane(vertices, faces, joints, plane_y)
        result["offset_height_fraction"] = float(offset)
        result["offset_mm"] = float(offset * raw_height * 1000.0)
        result["normalized_height"] = float((plane_y - mesh_min_y) / raw_height)
        probes.append(result)
    main = next(item for item in probes if item["offset_height_fraction"] == 0.0)
    topology_changed = len({item["num_contours"] for item in probes}) > 1
    possible_merge = any(item["possible_arm_torso_merge"] for item in probes)
    auxiliary_roles = {
        metric["role"]
        for item in probes
        for metric in item["all_contour_metrics"]
        if metric["role"] != "thoracic_torso"
    }
    lateral_only = bool(auxiliary_roles) and auxiliary_roles <= {
        "left_lateral_upper_limb_loop",
        "right_lateral_upper_limb_loop",
    }
    return {
        **plane_definition,
        "normalized_height": float((base_y - mesh_min_y) / raw_height),
        "search_or_optimization": "none",
        "plane_moved_or_clamped": False,
        "main_slice": main,
        "local_topology_probes": probes,
        "probe_summary": {
            "topology_changed_within_plus_minus_0_004H": bool(topology_changed),
            "num_contours_sequence": [item["num_contours"] for item in probes],
            "possible_arm_torso_merge": bool(possible_merge),
            "auxiliary_loop_roles": sorted(auxiliary_roles),
            "topology_change_explained_by_lateral_upper_limb_loops": bool(
                topology_changed and lateral_only and not possible_merge
            ),
            "perimeter_range_cm": float(
                max(item["selected_metrics"]["perimeter_cm"] for item in probes)
                - min(item["selected_metrics"]["perimeter_cm"] for item in probes)
            ),
            "interpretation": (
                "distal upper-limb loops appear or disappear while the independent "
                "centerline-containing thoracic loop remains the measured contour"
                if topology_changed and lateral_only and not possible_merge
                else "non-lateral auxiliary-loop topology change requires review"
                if topology_changed and not possible_merge
                else "no auxiliary-loop topology change detected"
                if not topology_changed and not possible_merge
                else "possible arm-torso merge requires manual review"
            ),
        },
    }
