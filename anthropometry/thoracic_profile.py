#!/usr/bin/env python3
"""Skeleton-bounded thoracic profile characterization for Step 2.7C."""

from __future__ import annotations

import numpy as np

from chest import (
    FOCUSED_SHAPY_CHEST_CONTROL,
    LEFT_SHOULDER_JOINT_INDEX,
    LITERATURE_CHEST_CANDIDATE,
    RIGHT_SHOULDER_JOINT_INDEX,
    compute_candidate_planes,
    evaluate_chest_plane,
)
from torso import SPINE2_JOINT_INDEX


THORACIC_INTERVAL_DEFINITION = "skeletal_thoracic_diagnostic_interval_v1"
THORACIC_PROFILE_STEP_HEIGHT_FRACTION = 0.002
GEOMETRY_CHEST_EXTREME_NAME = "geometry_chest_extreme"
GEOMETRY_CHEST_REJECTED_STATUS = "rejected_as_measurement_definition"
MONOTONIC_TOLERANCE_CM = 1e-6


def compute_thoracic_diagnostic_interval(
    vertices: np.ndarray,
    faces: np.ndarray,
    joints: np.ndarray,
) -> dict[str, object]:
    """Return the spine2-to-mean-shoulder diagnostic interval.

    Both public landmark planes must lie strictly inside.  The function flags
    violations and never clamps either plane.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    joints = np.asarray(joints, dtype=np.float64)
    lower = float(joints[SPINE2_JOINT_INDEX, 1])
    left_shoulder = float(joints[LEFT_SHOULDER_JOINT_INDEX, 1])
    right_shoulder = float(joints[RIGHT_SHOULDER_JOINT_INDEX, 1])
    upper = 0.5 * (left_shoulder + right_shoulder)
    if not lower < upper:
        raise ValueError("expected spine2 below mean shoulder height")
    definitions = compute_candidate_planes(vertices, faces)
    public_planes = {
        name: float(definitions[name]["plane_y_m"])
        for name in (LITERATURE_CHEST_CANDIDATE, FOCUSED_SHAPY_CHEST_CONTROL)
    }
    inside = {name: lower < value < upper for name, value in public_planes.items()}
    return {
        "definition": THORACIC_INTERVAL_DEFINITION,
        "purpose": "profile characterization only; not an anatomical chest measurement interval",
        "lower_y_m": lower,
        "lower_bound": {"joint": "spine2", "joint_index": SPINE2_JOINT_INDEX},
        "upper_y_m": upper,
        "upper_bound": {
            "rule": "mean(left_shoulder_y, right_shoulder_y)",
            "left_joint_index": LEFT_SHOULDER_JOINT_INDEX,
            "right_joint_index": RIGHT_SHOULDER_JOINT_INDEX,
        },
        "left_shoulder_y_m": left_shoulder,
        "right_shoulder_y_m": right_shoulder,
        "shoulder_y_mismatch_m": abs(left_shoulder - right_shoulder),
        "public_plane_y_m": public_planes,
        "public_plane_inside_diagnostic_interval": inside,
        "public_plane_outside_diagnostic_interval": not all(inside.values()),
        "plane_clamping": "none",
    }


def make_sweep_heights(lower_y: float, upper_y: float, raw_height: float) -> list[float]:
    """Create a fixed-step sweep with both skeleton boundaries included."""
    if not lower_y < upper_y or raw_height <= 0.0:
        raise ValueError("invalid thoracic interval or raw height")
    step = THORACIC_PROFILE_STEP_HEIGHT_FRACTION * raw_height
    values = list(np.arange(lower_y, upper_y, step, dtype=np.float64))
    if not values or abs(values[0] - lower_y) > 1e-12:
        values.insert(0, float(lower_y))
    if upper_y - values[-1] > 1e-12:
        values.append(float(upper_y))
    else:
        values[-1] = float(upper_y)
    return [float(value) for value in values]


def classify_profile(perimeters_cm: list[float]) -> dict[str, object]:
    """Classify a raw, unsmoothed discrete profile."""
    values = np.asarray(perimeters_cm, dtype=np.float64)
    if values.ndim != 1 or len(values) < 3 or not np.isfinite(values).all():
        raise ValueError("profile needs at least three finite perimeter values")
    differences = np.diff(values)
    maximum_index = int(np.argmax(values))
    if np.all(differences >= -MONOTONIC_TOLERANCE_CM):
        profile_type = "monotonic_increasing"
    elif np.all(differences <= MONOTONIC_TOLERANCE_CM):
        profile_type = "monotonic_decreasing"
    elif maximum_index == 0:
        profile_type = "lower_boundary_peak"
    elif maximum_index == len(values) - 1:
        profile_type = "upper_boundary_peak"
    else:
        local_left = values[maximum_index] - values[maximum_index - 1]
        local_right = values[maximum_index] - values[maximum_index + 1]
        if local_left > MONOTONIC_TOLERANCE_CM and local_right > MONOTONIC_TOLERANCE_CM:
            profile_type = "internal_peak"
        else:
            profile_type = "plateau_or_complex"
    return {
        "profile_type": profile_type,
        "argmax_index": maximum_index,
        "argmax_value_cm": float(values[maximum_index]),
        "positive_step_count": int(np.sum(differences > MONOTONIC_TOLERANCE_CM)),
        "negative_step_count": int(np.sum(differences < -MONOTONIC_TOLERANCE_CM)),
        "flat_step_count": int(np.sum(np.abs(differences) <= MONOTONIC_TOLERANCE_CM)),
    }


def characterize_thoracic_profile(
    vertices: np.ndarray,
    faces: np.ndarray,
    joints: np.ndarray,
) -> dict[str, object]:
    """Measure and classify C(y) without defining a chest measurement."""
    vertices = np.asarray(vertices, dtype=np.float64)
    raw_height = float(np.ptp(vertices[:, 1]))
    mesh_min_y = float(vertices[:, 1].min())
    interval = compute_thoracic_diagnostic_interval(vertices, faces, joints)
    if interval["public_plane_outside_diagnostic_interval"]:
        raise ValueError("a public chest plane lies outside the diagnostic interval")
    heights = make_sweep_heights(
        float(interval["lower_y_m"]), float(interval["upper_y_m"]), raw_height
    )
    profile = []
    for index, plane_y in enumerate(heights):
        result = evaluate_chest_plane(vertices, faces, joints, plane_y)
        roles = [item["role"] for item in result["all_contour_metrics"]]
        lateral_count = sum("lateral_upper_limb_loop" in role for role in roles)
        central_auxiliary_count = sum(role == "central_auxiliary_loop" for role in roles)
        profile.append(
            {
                "index": index,
                "plane_y_m": plane_y,
                "y_norm": float((plane_y - mesh_min_y) / raw_height),
                "interval_fraction": float(
                    (plane_y - interval["lower_y_m"])
                    / (interval["upper_y_m"] - interval["lower_y_m"])
                ),
                "torso_perimeter_m": result["selected_metrics"]["perimeter_m"],
                "torso_perimeter_cm": result["selected_metrics"]["perimeter_cm"],
                "torso_area_m2": result["selected_metrics"]["area_m2"],
                "torso_compactness": result["selected_metrics"]["compactness"],
                "torso_centroid_xz_m": result["selected_metrics"]["centroid_xz_m"],
                "centerline_xz_m": result["centerline_xz_m"],
                "centerline_to_centroid_m": result["selected_metrics"]["centerline_to_centroid_m"],
                "num_total_loops": result["num_contours"],
                "selected_contour_id": result["selected_contour_id"],
                "selection_mode": result["selection_mode"],
                "fallback_used": result["fallback_used"],
                "lateral_upper_limb_loop_count": int(lateral_count),
                "central_auxiliary_loop_count": int(central_auxiliary_count),
                "arm_torso_merge": result["possible_arm_torso_merge"],
                "selected_x_span_to_shoulder_ratio": result["selected_x_span_to_shoulder_ratio"],
                "invalid_component_count": result["diagnostics"]["invalid_component_count"],
                "remaining_duplicate_segments": result["diagnostics"]["remaining_duplicate_segments"],
            }
        )

    raw_classification = classify_profile(
        [item["torso_perimeter_cm"] for item in profile]
    )
    first_merge_index = next(
        (item["index"] for item in profile if item["arm_torso_merge"]), None
    )
    clean_prefix = profile if first_merge_index is None else profile[:first_merge_index]
    clean_classification = (
        classify_profile([item["torso_perimeter_cm"] for item in clean_prefix])
        if len(clean_prefix) >= 3
        else None
    )
    raw_argmax = profile[raw_classification["argmax_index"]]
    public_references = interval["public_plane_y_m"]
    interpretation_type = (
        "complex_arm_torso_transition"
        if first_merge_index is not None
        else raw_classification["profile_type"]
    )
    return {
        "interval": interval,
        "step": {
            "height_fraction": THORACIC_PROFILE_STEP_HEIGHT_FRACTION,
            "step_m": THORACIC_PROFILE_STEP_HEIGHT_FRACTION * raw_height,
            "smoothing": "none",
            "curve_fitting": "none",
            "metric_calibration": "none",
        },
        "public_reference_planes": {
            name: {
                "plane_y_m": plane_y,
                "y_norm": float((plane_y - mesh_min_y) / raw_height),
            }
            for name, plane_y in public_references.items()
        },
        "profile": profile,
        "classification": {
            "profile_type": interpretation_type,
            "raw_profile": raw_classification,
            "clean_pre_merge_profile": clean_classification,
            "first_arm_torso_merge_index": first_merge_index,
            "first_arm_torso_merge_y_norm": (
                profile[first_merge_index]["y_norm"] if first_merge_index is not None else None
            ),
            "argmax_y_norm": raw_argmax["y_norm"],
            "argmax_perimeter_cm": raw_argmax["torso_perimeter_cm"],
            "argmax_at_upper_boundary": raw_classification["argmax_index"] == len(profile) - 1,
            "argmax_at_lower_boundary": raw_classification["argmax_index"] == 0,
            "argmax_has_arm_torso_merge": raw_argmax["arm_torso_merge"],
        },
        "geometry_extreme_assessment": {
            "name": GEOMETRY_CHEST_EXTREME_NAME,
            "status": (
                GEOMETRY_CHEST_REJECTED_STATUS
                if first_merge_index is not None
                and clean_classification is not None
                and clean_classification["profile_type"] == "monotonic_increasing"
                and raw_argmax["arm_torso_merge"]
                else "undetermined"
            ),
            "reason": (
                "clean thoracic C(y) increases toward an arm-torso topology transition; "
                "the raw internal maximum is dominated by merged upper limbs"
                if first_merge_index is not None
                and clean_classification is not None
                and clean_classification["profile_type"] == "monotonic_increasing"
                and raw_argmax["arm_torso_merge"]
                else "profile does not satisfy the rejection rule"
            ),
        },
    }
