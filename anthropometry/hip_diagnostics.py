#!/usr/bin/env python3
"""Diagnostics for Step 2.6C without changing ``geometry_hip_v0``.

The diagnostics test whether the frozen pelvic-region lower bound truncates a
larger stable perimeter, quantify lower-bound and scan-step sensitivity, and
describe the near-maximum plateau around the raw discrete argmax.
"""

from __future__ import annotations

import numpy as np

from hip import (
    DEFAULT_BILATERAL_AREA_RATIO,
    DEFAULT_DOMINANT_AREA_FRACTION,
    DEFAULT_STABLE_RELATIVE_PERIMETER_CHANGE,
    classify_pelvic_topology,
    contour_compactness,
    pelvic_centerline_xz,
    scan_geometry_hip,
)
from slicing import ENDPOINT_CLUSTER_TOLERANCE_M, EPS, slice_mesh
from torso import DEFAULT_CENTERLINE_PROXIMITY_M, select_torso_contour


DEFAULT_DOWNWARD_PROBE_LAYERS = 8
DEFAULT_NEAR_MAXIMUM_RELATIVE_TOLERANCE = 0.001
DEFAULT_STEP_SIZES = (0.001, 0.002, 0.004)


def near_maximum_plateau(
    profile: list[dict[str, object]],
    selected_index: int,
    *,
    relative_tolerance: float = DEFAULT_NEAR_MAXIMUM_RELATIVE_TOLERANCE,
) -> dict[str, object]:
    """Describe the contiguous C >= Cmax - tolerance plateau around argmax."""
    if not profile or not 0 <= selected_index < len(profile):
        raise ValueError("profile and selected_index are inconsistent")
    if not 0.0 < relative_tolerance < 1.0:
        raise ValueError("relative_tolerance must lie in (0, 1)")
    circumferences = np.asarray(
        [record["perimeter_cm"] for record in profile], dtype=np.float64
    )
    if not np.isfinite(circumferences).all() or np.any(circumferences <= 0.0):
        raise ValueError("profile circumferences must be finite and positive")
    cmax = float(circumferences[selected_index])
    threshold = cmax * (1.0 - relative_tolerance)
    qualifies = circumferences >= threshold

    left = selected_index
    while left > 0 and qualifies[left - 1]:
        left -= 1
    right = selected_index
    while right + 1 < len(profile) and qualifies[right + 1]:
        right += 1

    y_min = float(profile[left]["plane_y_m"])
    y_max = float(profile[right]["plane_y_m"])
    return {
        "definition": "C >= Cmax - 0.1% * Cmax",
        "relative_tolerance": relative_tolerance,
        "cmax_cm": cmax,
        "threshold_cm": threshold,
        "plateau_start_index": left,
        "plateau_end_index": right,
        "plateau_point_count": right - left + 1,
        "plateau_y_min_m": y_min,
        "plateau_y_max_m": y_max,
        "plateau_width_m": y_max - y_min,
        "plateau_width_mm": (y_max - y_min) * 1000.0,
        "plateau_y_norm_min": float(profile[left]["normalized_height"]),
        "plateau_y_norm_max": float(profile[right]["normalized_height"]),
        "touches_lower_boundary": left == 0,
        "all_qualifying_indices": np.flatnonzero(qualifies).astype(int).tolist(),
    }


def _slice_with_micro_retry(
    vertices: np.ndarray,
    faces: np.ndarray,
    requested_plane_y: float,
    *,
    eps: float,
    endpoint_tolerance: float,
) -> tuple[dict[str, object], float]:
    retry = 5.0 * endpoint_tolerance
    sliced: dict[str, object] | None = None
    actual_y = requested_plane_y
    for adjustment in (0.0, retry, -retry):
        actual_y = requested_plane_y + adjustment
        candidate = slice_mesh(
            vertices,
            faces,
            actual_y,
            eps=eps,
            endpoint_tolerance=endpoint_tolerance,
        )
        connectivity = candidate["diagnostics"]["connectivity"]
        sliced = candidate
        if connectivity["invalid_component_count"] == 0 and candidate["contours"]:
            break
    assert sliced is not None
    return sliced, actual_y


def probe_below_stable_lower(
    vertices: np.ndarray,
    faces: np.ndarray,
    joints: np.ndarray,
    baseline_result: dict[str, object],
    *,
    layers: int = DEFAULT_DOWNWARD_PROBE_LAYERS,
    eps: float = EPS,
    endpoint_tolerance: float = ENDPOINT_CLUSTER_TOLERANCE_M,
    centerline_proximity_m: float = DEFAULT_CENTERLINE_PROXIMITY_M,
    dominant_area_fraction: float = DEFAULT_DOMINANT_AREA_FRACTION,
    bilateral_area_ratio: float = DEFAULT_BILATERAL_AREA_RATIO,
) -> list[dict[str, object]]:
    """Probe fixed-step layers below the current stable lower boundary."""
    if layers < 1:
        raise ValueError("layers must be positive")
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    joints = np.asarray(joints, dtype=np.float64)
    mesh_min_y = float(vertices[:, 1].min())
    raw_height = float(np.ptp(vertices[:, 1]))
    step_m = float(baseline_result["scan_parameters"]["step_m"])
    lower_y = float(
        baseline_result["search_region"]["stable_lower_bound"]["plane_y_m"]
    )
    compactness_threshold = float(
        baseline_result["search_region"]["stability_gate"]["minimum_compactness"]
    )
    relative_change_limit = float(
        baseline_result["search_region"]["stability_gate"][
            "maximum_relative_adjacent_perimeter_change"
        ]
    )
    higher_c = float(baseline_result["profile"][0]["perimeter_cm"])
    records: list[dict[str, object]] = []
    for offset in range(1, layers + 1):
        requested_y = lower_y - offset * step_m
        sliced, actual_y = _slice_with_micro_retry(
            vertices,
            faces,
            requested_y,
            eps=eps,
            endpoint_tolerance=endpoint_tolerance,
        )
        connectivity = sliced["diagnostics"]["connectivity"]
        topology = classify_pelvic_topology(
            sliced["contours"],
            float(joints[0, 0]),
            invalid_component_count=int(connectivity["invalid_component_count"]),
            dominant_area_fraction=dominant_area_fraction,
            bilateral_area_ratio=bilateral_area_ratio,
        )
        record: dict[str, object] = {
            "offset_layers_below_lower": offset,
            "distance_below_lower_m": lower_y - actual_y,
            "distance_below_lower_mm": (lower_y - actual_y) * 1000.0,
            "requested_plane_y_m": requested_y,
            "plane_y_m": actual_y,
            "plane_adjustment_m": actual_y - requested_y,
            "normalized_height": (actual_y - mesh_min_y) / raw_height,
            "topology_state": topology["state"],
            "num_contours": topology["num_contours"],
            "dominant_contour_count": topology["dominant_contour_count"],
            "invalid_component_count": int(connectivity["invalid_component_count"]),
            "remaining_duplicate_segments": int(
                connectivity["remaining_duplicate_segments"]
            ),
            "measurement_valid": False,
        }
        if topology["state"] == "joined_pelvis":
            centerline = pelvic_centerline_xz(joints, actual_y)
            selection = select_torso_contour(
                sliced["contours"],
                centerline,
                max_centerline_proximity_m=centerline_proximity_m,
            )
            contour = selection["selected_contour"]
            metrics = selection["selected_metrics"]
            fallback = (
                selection["selection_method"]
                != "spine_centerline_containment_then_area"
            )
            circumference = float(contour["perimeter_cm"])
            relative_change = abs(circumference - higher_c) / higher_c
            joined_safe = (
                not fallback
                and bool(metrics["centerline_inside"])
                and connectivity["invalid_component_count"] == 0
                and connectivity["remaining_duplicate_segments"] == 0
            )
            compactness = contour_compactness(contour)
            record.update(
                {
                    "measurement_valid": True,
                    "perimeter_m": float(contour["perimeter_m"]),
                    "perimeter_cm": circumference,
                    "area_m2": float(contour["area_m2"]),
                    "area_cm2": float(contour["area_cm2"]),
                    "compactness": compactness,
                    "centroid_xz_m": contour["centroid_xz_m"],
                    "centerline_xz_m": centerline.tolist(),
                    "centerline_inside": bool(metrics["centerline_inside"]),
                    "selection_mode": selection["selection_method"],
                    "fallback_used": fallback,
                    "delta_c_cm_from_next_higher_layer": circumference - higher_c,
                    "absolute_relative_c_change_from_next_higher": relative_change,
                    "joined_safe": joined_safe,
                    "compactness_gate_passed": compactness >= compactness_threshold,
                    "continuity_gate_passed": relative_change <= relative_change_limit,
                    "stable_safe": (
                        joined_safe
                        and compactness >= compactness_threshold
                        and relative_change <= relative_change_limit
                    ),
                }
            )
            higher_c = circumference
        records.append(record)
    return records


def lower_bound_sensitivity(
    baseline_result: dict[str, object],
    downward_probe: list[dict[str, object]],
    *,
    material_relative_tolerance: float = DEFAULT_NEAR_MAXIMUM_RELATIVE_TOLERANCE,
    joined_extension_depth_mm: float = 20.0,
) -> dict[str, object]:
    """Compare current Cmax with the deepest contiguous stable-safe extension."""
    current_lower = baseline_result["profile"][0]
    current_selected = baseline_result["selected"]
    current_cmax = float(current_selected["perimeter_cm"])
    material_threshold_cm = current_cmax * material_relative_tolerance

    contiguous_safe: list[dict[str, object]] = []
    for record in downward_probe:
        if not record.get("stable_safe", False):
            break
        contiguous_safe.append(record)
    extended_profile = list(reversed(contiguous_safe)) + list(
        baseline_result["profile"]
    )
    extended_selected = max(extended_profile, key=lambda item: item["perimeter_cm"])
    delta_c = float(extended_selected["perimeter_cm"] - current_cmax)

    joined_records = [
        record for record in downward_probe if record.get("joined_safe", False)
    ]
    joined_within_depth = [
        record
        for record in joined_records
        if record["distance_below_lower_mm"] <= joined_extension_depth_mm
    ]
    joined_depth_selected = max(
        [current_selected, *joined_within_depth],
        key=lambda item: item["perimeter_cm"],
    )
    joined_depth_delta = float(
        joined_depth_selected["perimeter_cm"] - current_cmax
    )
    joined_all_selected = max(
        [current_selected, *joined_records], key=lambda item: item["perimeter_cm"]
    )
    joined_all_delta = float(joined_all_selected["perimeter_cm"] - current_cmax)
    case = "B" if delta_c > material_threshold_cm else "A"
    return {
        "case": case,
        "case_definition": {
            "A": "no material larger perimeter exists in the contiguous stable-safe extension",
            "B": "a material larger stable-safe perimeter exists below the current lower bound",
        }[case],
        "material_relative_tolerance": material_relative_tolerance,
        "material_delta_c_threshold_cm": material_threshold_cm,
        "current": {
            "lower_plane_y_m": current_lower["plane_y_m"],
            "lower_y_norm": current_lower["normalized_height"],
            "cmax_cm": current_cmax,
            "cmax_plane_y_m": current_selected["plane_y_m"],
            "cmax_y_norm": current_selected["normalized_height"],
        },
        "extended_stable_safe": {
            "extended_layer_count": len(contiguous_safe),
            "lower_plane_y_m": (
                contiguous_safe[-1]["plane_y_m"]
                if contiguous_safe
                else current_lower["plane_y_m"]
            ),
            "lower_y_norm": (
                contiguous_safe[-1]["normalized_height"]
                if contiguous_safe
                else current_lower["normalized_height"]
            ),
            "cmax_cm": float(extended_selected["perimeter_cm"]),
            "cmax_plane_y_m": float(extended_selected["plane_y_m"]),
            "cmax_y_norm": float(extended_selected["normalized_height"]),
            "delta_c_cm_vs_current": delta_c,
            "delta_y_mm_vs_current_cmax": (
                float(extended_selected["plane_y_m"])
                - float(current_selected["plane_y_m"])
            ) * 1000.0,
        },
        "extended_joined_safe_within_20mm": {
            "target_depth_mm": joined_extension_depth_mm,
            "included_layer_count": len(joined_within_depth),
            "lower_plane_y_m": (
                joined_within_depth[-1]["plane_y_m"]
                if joined_within_depth
                else current_lower["plane_y_m"]
            ),
            "lower_y_norm": (
                joined_within_depth[-1]["normalized_height"]
                if joined_within_depth
                else current_lower["normalized_height"]
            ),
            "maximum_c_cm": float(joined_depth_selected["perimeter_cm"]),
            "delta_c_cm_vs_current": joined_depth_delta,
            "maximum_plane_y_m": float(joined_depth_selected["plane_y_m"]),
            "maximum_y_norm": float(joined_depth_selected["normalized_height"]),
            "maximum_is_stable_safe": bool(
                joined_depth_selected.get("stable_safe", True)
            ),
        },
        "joined_only_full_probe_diagnostic": {
            "lowest_joined_safe_probe_y_norm": (
                joined_records[-1]["normalized_height"] if joined_records else None
            ),
            "maximum_c_cm": float(joined_all_selected["perimeter_cm"]),
            "delta_c_cm_vs_current": joined_all_delta,
            "maximum_is_stable_safe": bool(
                joined_all_selected.get("stable_safe", True)
            ),
            "interpretation": (
                "joined-only layers may include the compactness-rejected topology "
                "merge recovery and do not redefine the safe search interval"
            ),
        },
        "hidden_stable_maximum": case == "B",
    }


def step_size_sensitivity(
    vertices: np.ndarray,
    faces: np.ndarray,
    joints: np.ndarray,
    *,
    step_sizes: tuple[float, ...] = DEFAULT_STEP_SIZES,
    reference_step: float = 0.002,
) -> dict[str, object]:
    """Rerun the unchanged baseline at multiple normalized-height steps."""
    if reference_step not in step_sizes:
        raise ValueError("reference_step must be included in step_sizes")
    results: dict[str, dict[str, object]] = {}
    for step in step_sizes:
        result = scan_geometry_hip(
            vertices,
            faces,
            joints,
            step_normalized_height=step,
        )
        selected = result["selected"]
        results[f"{step:.3f}H"] = {
            "step_normalized_height": step,
            "step_m": result["scan_parameters"]["step_m"],
            "stable_lower_y_norm": result["search_region"]["stable_lower_bound"][
                "normalized_height"
            ],
            "selected_y_norm": selected["normalized_height"],
            "selected_plane_y_m": selected["plane_y_m"],
            "selected_c_cm": selected["perimeter_cm"],
            "boundary_maximum": result["boundary_maximum"],
        }
    reference = results[f"{reference_step:.3f}H"]
    for record in results.values():
        record["delta_c_cm_vs_0.002H"] = (
            record["selected_c_cm"] - reference["selected_c_cm"]
        )
        record["absolute_delta_c_cm_vs_0.002H"] = abs(
            record["delta_c_cm_vs_0.002H"]
        )
        record["relative_delta_c_percent_vs_0.002H"] = (
            record["delta_c_cm_vs_0.002H"] / reference["selected_c_cm"] * 100.0
        )
        record["delta_y_mm_vs_0.002H"] = (
            record["selected_plane_y_m"] - reference["selected_plane_y_m"]
        ) * 1000.0
    return {
        "definition_unchanged": True,
        "smoothing": "none",
        "curve_fitting": "none",
        "metric_calibration": "none",
        "reference_step": reference_step,
        "results": results,
        "maximum_absolute_delta_c_cm": max(
            record["absolute_delta_c_cm_vs_0.002H"] for record in results.values()
        ),
        "maximum_absolute_relative_delta_c_percent": max(
            abs(record["relative_delta_c_percent_vs_0.002H"])
            for record in results.values()
        ),
        "maximum_absolute_delta_y_mm": max(
            abs(record["delta_y_mm_vs_0.002H"]) for record in results.values()
        ),
    }
