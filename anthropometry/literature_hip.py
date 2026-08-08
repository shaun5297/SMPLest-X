#!/usr/bin/env python3
"""Independent literature-defined hip plane on canonical SMPL-X meshes.

``literature_hip_v1`` preserves the public SMPL-Anthropometry SMPL-X hip
landmark (PUBIC_BONE, vertex 5949) while using the shared horizontal slicing
and closed-loop perimeter engine.  It performs one direct slice and never
searches the pelvic circumference profile.
"""

from __future__ import annotations

import numpy as np

from hip import classify_pelvic_topology, contour_compactness, pelvic_centerline_xz
from slicing import ENDPOINT_CLUSTER_TOLERANCE_M, EPS, slice_mesh
from torso import (
    DEFAULT_CENTERLINE_PROXIMITY_M,
    PELVIS_JOINT_INDEX,
    SPINE3_JOINT_INDEX,
    select_torso_contour,
)


LITERATURE_HIP_DEFINITION = "literature_hip_v1"
LITERATURE_HIP_STATUS = "baseline"
PUBIC_BONE_VERTEX_ID = 5949

LITERATURE_SOURCES = [
    {
        "name": "A2B official B2A SMPL-X measurer",
        "url": (
            "https://github.com/kaulquappe23/a2b_human_mesh/blob/main/"
            "anthro/measurements/measure.py"
        ),
        "relevance": (
            "the official A2B implementation imports PUBIC_BONE=5949 from "
            "SMPL-Anthropometry and defines hip with that landmark, a "
            "pelvis-to-spine3 normal, and hips/upper-leg face regions"
        ),
    },
    {
        "name": "SMPL-Anthropometry landmark definitions",
        "url": (
            "https://github.com/DavidBoja/SMPL-Anthropometry/blob/master/"
            "landmark_definitions.py"
        ),
        "relevance": "SMPL-X PUBIC_BONE is topology vertex 5949",
    },
    {
        "name": "SMPL-Anthropometry measurement definitions",
        "url": (
            "https://github.com/DavidBoja/SMPL-Anthropometry/blob/master/"
            "measurement_definitions.py"
        ),
        "relevance": (
            "hip circumference uses PUBIC_BONE, the pelvis-to-spine3 normal, "
            "and the hips body-part segmentation"
        ),
    },
    {
        "name": "SMPL-Anthropometry plane construction",
        "url": (
            "https://github.com/DavidBoja/SMPL-Anthropometry/blob/master/measure.py"
        ),
        "relevance": (
            "the source plane origin is the landmark coordinate and its normal "
            "is the selected joint vector"
        ),
    },
]


def _validate_inputs(
    vertices: np.ndarray,
    faces: np.ndarray | None = None,
    joints: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), got {vertices.shape}")
    if len(vertices) <= PUBIC_BONE_VERTEX_ID:
        raise ValueError("vertices do not contain the published SMPL-X hip landmark")
    if not np.isfinite(vertices).all():
        raise ValueError("vertices contain NaN or Inf")

    validated_faces = None
    if faces is not None:
        validated_faces = np.asarray(faces, dtype=np.int64)
        if validated_faces.ndim != 2 or validated_faces.shape[1] != 3:
            raise ValueError("faces must have shape (F, 3)")
        if (
            validated_faces.size == 0
            or validated_faces.min() < 0
            or validated_faces.max() >= len(vertices)
        ):
            raise ValueError("faces are empty or contain an invalid vertex index")

    validated_joints = None
    if joints is not None:
        validated_joints = np.asarray(joints, dtype=np.float64)
        if (
            validated_joints.ndim != 2
            or validated_joints.shape[1] != 3
            or len(validated_joints) <= SPINE3_JOINT_INDEX
        ):
            raise ValueError("joints do not contain native pelvis and spine3 joints")
        if not np.isfinite(validated_joints).all():
            raise ValueError("joints contain NaN or Inf")
    return vertices, validated_faces, validated_joints


def compute_literature_hip_plane(vertices: np.ndarray) -> dict[str, object]:
    """Return the horizontal plane at the published pubic-bone vertex height."""
    vertices, _, _ = _validate_inputs(vertices)
    landmark = vertices[PUBIC_BONE_VERTEX_ID]
    return {
        "plane_y_m": float(landmark[1]),
        "plane_origin_m": landmark.tolist(),
        "landmark": {
            "name": "PUBIC_BONE",
            "vertex_id": PUBIC_BONE_VERTEX_ID,
            "coordinate_m": landmark.tolist(),
            "representation": "fixed topology vertex evaluated on subject beta mesh",
        },
        "origin_rule": "Y coordinate of published SMPL-X PUBIC_BONE vertex v5949",
        "horizontal_adaptation": True,
        "shape_adaptation": "topology vertex deforms with subject beta",
    }


def measure_literature_hip(
    vertices: np.ndarray,
    faces: np.ndarray,
    joints: np.ndarray,
    *,
    eps: float = EPS,
    endpoint_tolerance: float = ENDPOINT_CLUSTER_TOLERANCE_M,
    centerline_proximity_m: float = DEFAULT_CENTERLINE_PROXIMITY_M,
) -> dict[str, object]:
    """Measure one direct landmark-defined horizontal hip slice."""
    vertices, validated_faces, validated_joints = _validate_inputs(
        vertices, faces, joints
    )
    assert validated_faces is not None and validated_joints is not None
    faces = validated_faces
    joints = validated_joints

    plane = compute_literature_hip_plane(vertices)
    plane_y = float(plane["plane_y_m"])
    published_normal = joints[PELVIS_JOINT_INDEX] - joints[SPINE3_JOINT_INDEX]
    normal_length = float(np.linalg.norm(published_normal))
    if normal_length <= 0.0:
        raise ValueError("pelvis-to-spine3 source normal has zero length")
    published_normal /= normal_length
    tilt_degrees = float(
        np.degrees(np.arccos(np.clip(abs(published_normal[1]), 0.0, 1.0)))
    )

    sliced = slice_mesh(
        vertices,
        faces,
        plane_y,
        eps=eps,
        endpoint_tolerance=endpoint_tolerance,
    )
    connectivity = sliced["diagnostics"]["connectivity"]
    if connectivity["invalid_component_count"] != 0:
        raise RuntimeError("literature hip slice contains an invalid contour component")
    topology = classify_pelvic_topology(
        sliced["contours"],
        float(joints[PELVIS_JOINT_INDEX, 0]),
        invalid_component_count=int(connectivity["invalid_component_count"]),
    )
    if topology["state"] != "joined_pelvis":
        raise RuntimeError(
            "published hip landmark plane is not a joined-pelvis topology: "
            f"{topology['state']}"
        )

    centerline = pelvic_centerline_xz(joints, plane_y)
    selection = select_torso_contour(
        sliced["contours"],
        centerline,
        max_centerline_proximity_m=centerline_proximity_m,
    )
    contour = selection["selected_contour"]
    metrics = selection["selected_metrics"]
    mesh_min_y = float(vertices[:, 1].min())
    raw_height = float(np.ptp(vertices[:, 1]))
    fallback = selection["selection_method"] != "spine_centerline_containment_then_area"

    return {
        "definition": LITERATURE_HIP_DEFINITION,
        "status": LITERATURE_HIP_STATUS,
        "definition_text": (
            "A single horizontal canonical SMPL-X slice at the Y coordinate of "
            "the published PUBIC_BONE surface vertex v5949."
        ),
        "independent_of_circumference_maximum": True,
        "measurement_space": "raw canonical SMPL-X geometry",
        "metric_calibration": "none",
        "plane_definition": {
            **plane,
            "normalized_height": float((plane_y - mesh_min_y) / raw_height),
            "published_source_plane_normal_unit": published_normal.tolist(),
            "published_source_normal_tilt_from_vertical_degrees": tilt_degrees,
            "adaptation_note": (
                "The source uses v5949 as a 3D plane origin and pelvis-to-spine3 "
                "as its normal. On the zero-pose canonical mesh this baseline "
                "preserves the landmark height and uses y=constant so all hip "
                "definitions share the same horizontal slicing engine."
            ),
        },
        "search_or_scan": "none; one direct landmark-defined slice",
        "source_perimeter_adaptation": (
            "The source filters hips faces and applies a convex hull. This baseline "
            "uses the shared stitched closed-loop perimeter so the comparison isolates "
            "plane definition rather than mixing slicing engines."
        ),
        "topology": topology,
        "num_contours": len(sliced["contours"]),
        "selected_contour_id": int(selection["selected_contour_id"]),
        "selection_mode": selection["selection_method"],
        "fallback_used": bool(fallback),
        "centerline_xz_m": centerline.tolist(),
        "centerline_inside": bool(metrics["centerline_inside"]),
        "centerline_to_centroid_m": float(metrics["centroid_distance_m"]),
        "perimeter_m": float(contour["perimeter_m"]),
        "perimeter_cm": float(contour["perimeter_cm"]),
        "area_m2": float(contour["area_m2"]),
        "area_cm2": float(contour["area_cm2"]),
        "compactness": contour_compactness(contour),
        "centroid_xz_m": contour["centroid_xz_m"],
        "ordered_points_m": contour["ordered_points_m"],
        "num_points": int(contour["num_points"]),
        "diagnostics": {
            "unique_segment_count": int(connectivity["unique_segment_count"]),
            "remaining_duplicate_segments": int(
                connectivity["remaining_duplicate_segments"]
            ),
            "invalid_component_count": int(connectivity["invalid_component_count"]),
        },
        "scientific_boundary": (
            "This is a reproducible horizontal adaptation of an open-source SMPL-X "
            "landmark plane. It is a literature baseline, not clinical palpation, "
            "ISO ground truth, or evidence of measurement accuracy."
        ),
        "sources": LITERATURE_SOURCES,
    }
