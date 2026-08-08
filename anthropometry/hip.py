#!/usr/bin/env python3
"""Topology-constrained geometry hip baseline on canonical SMPL-X meshes.

Step 2.6A detects the bilateral-leg to joined-pelvis transition and excludes
the transient crotch-merge contour. Step 2.6B selects the raw discrete maximum
perimeter in the resulting stable pelvic interval. No smoothing, fitting,
calibration, or anatomical/ISO hip claim is applied.
"""

from __future__ import annotations

import numpy as np

from slicing import ENDPOINT_CLUSTER_TOLERANCE_M, EPS, slice_mesh
from torso import (
    DEFAULT_CENTERLINE_PROXIMITY_M,
    PELVIS_JOINT_INDEX,
    SPINE1_JOINT_INDEX,
    interpolate_spine_centerline_xz,
    select_torso_contour,
)


LEFT_HIP_JOINT_INDEX = 1
RIGHT_HIP_JOINT_INDEX = 2
LEFT_KNEE_JOINT_INDEX = 4
RIGHT_KNEE_JOINT_INDEX = 5

GEOMETRY_HIP_DEFINITION = "geometry_hip_v0"
GEOMETRY_HIP_STATUS = "baseline"
DEFAULT_STEP_NORMALIZED_HEIGHT = 0.002
DEFAULT_STABLE_TOPOLOGY_LAYERS = 3
DEFAULT_STABLE_PELVIS_LAYERS = 4
DEFAULT_DOMINANT_AREA_FRACTION = 0.10
DEFAULT_BILATERAL_AREA_RATIO = 0.50
DEFAULT_COMPACTNESS_REFERENCE_RATIO = 0.95
DEFAULT_STABLE_RELATIVE_PERIMETER_CHANGE = 0.01
DEFAULT_LOCAL_JUMP_WARNING_CM = 1.0


class PelvicRegionError(ValueError):
    """Raised when a stable topology-constrained pelvic interval is unavailable."""


def _validate_inputs(
    vertices: np.ndarray, faces: np.ndarray, joints: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    joints = np.asarray(joints, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (F, 3)")
    if joints.ndim != 2 or joints.shape[1] != 3:
        raise ValueError("joints must have shape (J, 3)")
    if len(joints) <= max(SPINE1_JOINT_INDEX, RIGHT_KNEE_JOINT_INDEX):
        raise ValueError("joints do not contain the required SMPL-X pelvis chain")
    if not np.isfinite(vertices).all() or not np.isfinite(joints).all():
        raise ValueError("vertices and joints must be finite")
    if faces.size == 0 or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("faces are empty or contain an invalid vertex index")
    return vertices, faces, joints


def _inclusive_heights(y_min: float, y_max: float, step_m: float) -> np.ndarray:
    if not np.isfinite([y_min, y_max, step_m]).all() or y_max <= y_min:
        raise ValueError("scan bounds must be finite and strictly increasing")
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


def contour_compactness(contour: dict[str, object]) -> float:
    """Return dimensionless isoperimetric compactness ``4πA/P²``."""
    area = float(contour["area_m2"])
    perimeter = float(contour["perimeter_m"])
    if not np.isfinite([area, perimeter]).all() or area <= 0.0 or perimeter <= 0.0:
        raise ValueError("contour area and perimeter must be finite and positive")
    return float(4.0 * np.pi * area / perimeter**2)


def classify_pelvic_topology(
    contours: list[dict[str, object]],
    pelvis_center_x_m: float,
    *,
    invalid_component_count: int = 0,
    dominant_area_fraction: float = DEFAULT_DOMINANT_AREA_FRACTION,
    bilateral_area_ratio: float = DEFAULT_BILATERAL_AREA_RATIO,
) -> dict[str, object]:
    """Classify a slice as joined pelvis, bilateral split, or transitional.

    Tiny loops are retained in diagnostics but excluded from the dominant-loop
    topology using an area fraction relative to the largest contour.
    """
    if not 0.0 < dominant_area_fraction <= 1.0:
        raise ValueError("dominant_area_fraction must lie in (0, 1]")
    if not 0.0 < bilateral_area_ratio <= 1.0:
        raise ValueError("bilateral_area_ratio must lie in (0, 1]")
    if invalid_component_count:
        return {
            "state": "invalid",
            "num_contours": len(contours),
            "dominant_contour_ids": [],
            "dominant_contour_count": 0,
        }
    if not contours:
        return {
            "state": "empty",
            "num_contours": 0,
            "dominant_contour_ids": [],
            "dominant_contour_count": 0,
        }

    areas = np.asarray([contour["area_m2"] for contour in contours], dtype=np.float64)
    if not np.isfinite(areas).all() or np.any(areas <= 0.0):
        raise ValueError("contour areas must be finite and positive")
    threshold = float(areas.max() * dominant_area_fraction)
    dominant = [
        contour for contour, area in zip(contours, areas) if area >= threshold
    ]
    state = "transitional"
    bilateral_metrics: dict[str, object] = {}
    if len(dominant) == 1:
        state = "joined_pelvis"
    elif len(dominant) == 2:
        offsets = [
            float(contour["centroid_xz_m"][0]) - pelvis_center_x_m
            for contour in dominant
        ]
        area_ratio = min(float(contour["area_m2"]) for contour in dominant) / max(
            float(contour["area_m2"]) for contour in dominant
        )
        straddles_center = offsets[0] * offsets[1] < 0.0
        bilateral_metrics = {
            "centroid_x_offsets_m": offsets,
            "area_ratio_smaller_over_larger": area_ratio,
            "straddles_pelvis_center_x": bool(straddles_center),
        }
        if straddles_center and area_ratio >= bilateral_area_ratio:
            state = "bilateral_leg_split"

    return {
        "state": state,
        "num_contours": len(contours),
        "dominant_area_threshold_m2": threshold,
        "dominant_contour_ids": [int(contour["id"]) for contour in dominant],
        "dominant_contour_count": len(dominant),
        **bilateral_metrics,
    }


def find_first_stable_state(
    records: list[dict[str, object]], state: str, layers: int
) -> int:
    """Return the first index starting a consecutive stable topology run."""
    if layers < 2:
        raise ValueError("layers must be at least two")
    for index in range(len(records) - layers + 1):
        if all(record["topology_state"] == state for record in records[index:index + layers]):
            return index
    raise PelvicRegionError(f"no stable {state!r} run of {layers} layers was found")


def pelvic_centerline_xz(joints: np.ndarray, plane_y: float) -> np.ndarray:
    """Use the pelvis X-Z point below pelvis and the spine chain above it."""
    pelvis_y = float(joints[PELVIS_JOINT_INDEX, 1])
    if plane_y <= pelvis_y:
        return np.asarray(joints[PELVIS_JOINT_INDEX, [0, 2]], dtype=np.float64)
    return interpolate_spine_centerline_xz(joints, plane_y)


def _slice_topology_record(
    vertices: np.ndarray,
    faces: np.ndarray,
    joints: np.ndarray,
    plane_y: float,
    mesh_min_y: float,
    raw_height: float,
    *,
    eps: float,
    endpoint_tolerance: float,
    dominant_area_fraction: float,
    bilateral_area_ratio: float,
) -> tuple[dict[str, object], dict[str, object]]:
    requested_plane_y = float(plane_y)
    sliced = None
    actual_plane_y = requested_plane_y
    retry_adjustment = 5.0 * endpoint_tolerance
    for adjustment in (0.0, retry_adjustment, -retry_adjustment):
        actual_plane_y = requested_plane_y + adjustment
        candidate = slice_mesh(
            vertices,
            faces,
            actual_plane_y,
            eps=eps,
            endpoint_tolerance=endpoint_tolerance,
        )
        connectivity = candidate["diagnostics"]["connectivity"]
        sliced = candidate
        if connectivity["invalid_component_count"] == 0 and candidate["contours"]:
            break
    assert sliced is not None
    connectivity = sliced["diagnostics"]["connectivity"]
    topology = classify_pelvic_topology(
        sliced["contours"],
        float(joints[PELVIS_JOINT_INDEX, 0]),
        invalid_component_count=int(connectivity["invalid_component_count"]),
        dominant_area_fraction=dominant_area_fraction,
        bilateral_area_ratio=bilateral_area_ratio,
    )
    record = {
        "requested_plane_y_m": requested_plane_y,
        "plane_y_m": actual_plane_y,
        "plane_adjustment_m": actual_plane_y - requested_plane_y,
        "degeneracy_retry_used": actual_plane_y != requested_plane_y,
        "normalized_height": float((actual_plane_y - mesh_min_y) / raw_height),
        "topology_state": topology["state"],
        "num_contours": topology["num_contours"],
        "dominant_contour_count": topology["dominant_contour_count"],
        "dominant_contour_ids": topology["dominant_contour_ids"],
        "remaining_duplicate_segments": int(connectivity["remaining_duplicate_segments"]),
        "invalid_component_count": int(connectivity["invalid_component_count"]),
    }
    return record, sliced


def _pelvic_measurement_record(
    topology_record: dict[str, object],
    sliced: dict[str, object],
    joints: np.ndarray,
    *,
    centerline_proximity_m: float,
) -> dict[str, object]:
    if topology_record["topology_state"] != "joined_pelvis":
        return {**topology_record, "measurement_valid": False}
    centerline = pelvic_centerline_xz(joints, float(topology_record["plane_y_m"]))
    selection = select_torso_contour(
        sliced["contours"],
        centerline,
        max_centerline_proximity_m=centerline_proximity_m,
    )
    contour = selection["selected_contour"]
    metrics = selection["selected_metrics"]
    fallback = selection["selection_method"] != "spine_centerline_containment_then_area"
    return {
        **topology_record,
        "measurement_valid": True,
        "perimeter_m": float(contour["perimeter_m"]),
        "perimeter_cm": float(contour["perimeter_cm"]),
        "area_m2": float(contour["area_m2"]),
        "area_cm2": float(contour["area_cm2"]),
        "compactness": contour_compactness(contour),
        "centroid_xz_m": contour["centroid_xz_m"],
        "centerline_xz_m": centerline.tolist(),
        "centerline_to_centroid_m": float(metrics["centroid_distance_m"]),
        "centerline_inside": bool(metrics["centerline_inside"]),
        "selected_contour_id": int(selection["selected_contour_id"]),
        "selection_mode": selection["selection_method"],
        "fallback_used": bool(fallback),
    }


def find_stable_pelvis_lower_index(
    records: list[dict[str, object]],
    reference_compactness: float,
    *,
    layers: int = DEFAULT_STABLE_PELVIS_LAYERS,
    compactness_reference_ratio: float = DEFAULT_COMPACTNESS_REFERENCE_RATIO,
    max_relative_perimeter_change: float = DEFAULT_STABLE_RELATIVE_PERIMETER_CHANGE,
) -> int:
    """Find the first stable joined-pelvis window above the merge transient."""
    if layers < 2:
        raise ValueError("layers must be at least two")
    if not 0.0 < compactness_reference_ratio <= 1.0:
        raise ValueError("compactness_reference_ratio must lie in (0, 1]")
    if max_relative_perimeter_change <= 0.0:
        raise ValueError("max_relative_perimeter_change must be positive")
    threshold = reference_compactness * compactness_reference_ratio
    for index in range(len(records) - layers + 1):
        window = records[index:index + layers]
        if not all(
            item.get("measurement_valid", False)
            and not item["fallback_used"]
            and item["centerline_inside"]
            and item["compactness"] >= threshold
            for item in window
        ):
            continue
        relative_changes = [
            abs(window[offset + 1]["perimeter_m"] - window[offset]["perimeter_m"])
            / window[offset]["perimeter_m"]
            for offset in range(layers - 1)
        ]
        if max(relative_changes, default=0.0) <= max_relative_perimeter_change:
            return index
    raise PelvicRegionError("no stable joined-pelvis measurement window was found")


def detect_pelvic_search_region(
    vertices: np.ndarray,
    faces: np.ndarray,
    joints: np.ndarray,
    *,
    step_normalized_height: float = DEFAULT_STEP_NORMALIZED_HEIGHT,
    stable_topology_layers: int = DEFAULT_STABLE_TOPOLOGY_LAYERS,
    stable_pelvis_layers: int = DEFAULT_STABLE_PELVIS_LAYERS,
    dominant_area_fraction: float = DEFAULT_DOMINANT_AREA_FRACTION,
    bilateral_area_ratio: float = DEFAULT_BILATERAL_AREA_RATIO,
    compactness_reference_ratio: float = DEFAULT_COMPACTNESS_REFERENCE_RATIO,
    max_relative_perimeter_change: float = DEFAULT_STABLE_RELATIVE_PERIMETER_CHANGE,
    eps: float = EPS,
    endpoint_tolerance: float = ENDPOINT_CLUSTER_TOLERANCE_M,
    centerline_proximity_m: float = DEFAULT_CENTERLINE_PROXIMITY_M,
) -> dict[str, object]:
    """Detect the topology transition and stable pelvis-to-spine1 interval."""
    vertices, faces, joints = _validate_inputs(vertices, faces, joints)
    if step_normalized_height <= 0.0:
        raise ValueError("step_normalized_height must be positive")
    mesh_min_y = float(vertices[:, 1].min())
    mesh_max_y = float(vertices[:, 1].max())
    raw_height = mesh_max_y - mesh_min_y
    step_m = step_normalized_height * raw_height

    pelvis_y = float(joints[PELVIS_JOINT_INDEX, 1])
    spine1_y = float(joints[SPINE1_JOINT_INDEX, 1])
    knee_limit_y = max(
        float(joints[LEFT_KNEE_JOINT_INDEX, 1]),
        float(joints[RIGHT_KNEE_JOINT_INDEX, 1]),
    )
    hip_limit_y = max(
        float(joints[LEFT_HIP_JOINT_INDEX, 1]),
        float(joints[RIGHT_HIP_JOINT_INDEX, 1]),
    )
    if not knee_limit_y < hip_limit_y < pelvis_y < spine1_y:
        raise ValueError("expected knee < hip < pelvis < spine1 joint ordering")

    descending = pelvis_y - np.arange(
        int(np.floor((pelvis_y - knee_limit_y) / step_m)) + 1,
        dtype=np.float64,
    ) * step_m
    topology_probe: list[dict[str, object]] = []
    for plane_y in descending:
        record, _ = _slice_topology_record(
            vertices, faces, joints, float(plane_y), mesh_min_y, raw_height,
            eps=eps,
            endpoint_tolerance=endpoint_tolerance,
            dominant_area_fraction=dominant_area_fraction,
            bilateral_area_ratio=bilateral_area_ratio,
        )
        topology_probe.append(record)
    split_index = find_first_stable_state(
        topology_probe, "bilateral_leg_split", stable_topology_layers
    )
    joined_indices = [
        index for index in range(split_index)
        if topology_probe[index]["topology_state"] == "joined_pelvis"
    ]
    if not joined_indices:
        raise PelvicRegionError("no joined pelvis contour exists above the stable leg split")
    joined_index = max(joined_indices)
    joined_start_y = float(topology_probe[joined_index]["plane_y_m"])
    highest_split_y = float(topology_probe[split_index]["plane_y_m"])

    pelvis_topology, pelvis_slice = _slice_topology_record(
        vertices, faces, joints, pelvis_y, mesh_min_y, raw_height,
        eps=eps,
        endpoint_tolerance=endpoint_tolerance,
        dominant_area_fraction=dominant_area_fraction,
        bilateral_area_ratio=bilateral_area_ratio,
    )
    pelvis_reference = _pelvic_measurement_record(
        pelvis_topology,
        pelvis_slice,
        joints,
        centerline_proximity_m=centerline_proximity_m,
    )
    if not pelvis_reference.get("measurement_valid", False):
        raise PelvicRegionError("pelvis-joint reference slice is not a joined pelvis contour")

    ascending_records: list[dict[str, object]] = []
    for plane_y in _inclusive_heights(joined_start_y, spine1_y, step_m):
        topology, sliced = _slice_topology_record(
            vertices, faces, joints, float(plane_y), mesh_min_y, raw_height,
            eps=eps,
            endpoint_tolerance=endpoint_tolerance,
            dominant_area_fraction=dominant_area_fraction,
            bilateral_area_ratio=bilateral_area_ratio,
        )
        ascending_records.append(
            _pelvic_measurement_record(
                topology,
                sliced,
                joints,
                centerline_proximity_m=centerline_proximity_m,
            )
        )
    stable_lower_index = find_stable_pelvis_lower_index(
        ascending_records,
        float(pelvis_reference["compactness"]),
        layers=stable_pelvis_layers,
        compactness_reference_ratio=compactness_reference_ratio,
        max_relative_perimeter_change=max_relative_perimeter_change,
    )
    stable_profile = ascending_records[stable_lower_index:]
    if any(not record.get("measurement_valid", False) for record in stable_profile):
        raise PelvicRegionError("stable pelvis interval contains a non-joined slice")

    stable_lower = stable_profile[0]
    return {
        "definition": "topology_defined_pelvic_search_region_v0",
        "status": "baseline",
        "step_normalized_height": float(step_normalized_height),
        "step_m": float(step_m),
        "lower_probe_limit": {
            "joint": "higher knee",
            "plane_y_m": knee_limit_y,
            "normalized_height": float((knee_limit_y - mesh_min_y) / raw_height),
        },
        "upper_bound": {
            "joint": "spine1",
            "joint_index": SPINE1_JOINT_INDEX,
            "plane_y_m": spine1_y,
            "normalized_height": float((spine1_y - mesh_min_y) / raw_height),
        },
        "topology_transition": {
            "lowest_joined_plane_y_m": joined_start_y,
            "lowest_joined_normalized_height": float(
                (joined_start_y - mesh_min_y) / raw_height
            ),
            "highest_stable_split_plane_y_m": highest_split_y,
            "highest_stable_split_normalized_height": float(
                (highest_split_y - mesh_min_y) / raw_height
            ),
            "bracket_width_m": joined_start_y - highest_split_y,
            "stable_split_layers": stable_topology_layers,
        },
        "stability_gate": {
            "stable_layers": stable_pelvis_layers,
            "pelvis_reference_compactness": pelvis_reference["compactness"],
            "compactness_reference_ratio": compactness_reference_ratio,
            "minimum_compactness": (
                pelvis_reference["compactness"] * compactness_reference_ratio
            ),
            "maximum_relative_adjacent_perimeter_change": (
                max_relative_perimeter_change
            ),
            "purpose": "exclude the transient high-perimeter crotch-merge contour",
        },
        "stable_lower_bound": {
            "plane_y_m": stable_lower["plane_y_m"],
            "normalized_height": stable_lower["normalized_height"],
            "source": "first stable joined-pelvis window after topology transition",
        },
        "degeneracy_retry_count": sum(
            bool(record["degeneracy_retry_used"]) for record in ascending_records
        ),
        "maximum_plane_adjustment_m": max(
            abs(float(record["plane_adjustment_m"]))
            for record in ascending_records
        ),
        "topology_probe": topology_probe[: split_index + stable_topology_layers],
        "profile": stable_profile,
    }


def select_profile_maximum(records: list[dict[str, object]]) -> dict[str, object]:
    """Select the raw discrete maximum perimeter, never maximum area."""
    if not records:
        raise ValueError("profile records must be non-empty")
    perimeters = np.asarray([record["perimeter_m"] for record in records], dtype=np.float64)
    if not np.isfinite(perimeters).all() or np.any(perimeters <= 0.0):
        raise ValueError("profile perimeters must be finite and positive")
    selected_index = int(np.argmax(perimeters))
    return {
        "selected_index": selected_index,
        "boundary_maximum": selected_index in (0, len(records) - 1),
        "selected_record": records[selected_index],
    }


def _local_stability(
    records: list[dict[str, object]], selected_index: int, radius: int = 2
) -> dict[str, object]:
    start = max(0, selected_index - radius)
    stop = min(len(records), selected_index + radius + 1)
    neighborhood = [
        {
            "index": index,
            "offset_steps": index - selected_index,
            "plane_y_m": records[index]["plane_y_m"],
            "normalized_height": records[index]["normalized_height"],
            "perimeter_cm": records[index]["perimeter_cm"],
        }
        for index in range(start, stop)
    ]
    changes = [
        abs(records[index + 1]["perimeter_cm"] - records[index]["perimeter_cm"])
        for index in range(len(records) - 1)
    ]
    return {
        "radius_steps": radius,
        "neighborhood": neighborhood,
        "max_adjacent_change_cm": max(changes, default=0.0),
    }


def scan_geometry_hip(
    vertices: np.ndarray,
    faces: np.ndarray,
    joints: np.ndarray,
    *,
    step_normalized_height: float = DEFAULT_STEP_NORMALIZED_HEIGHT,
    local_jump_warning_cm: float = DEFAULT_LOCAL_JUMP_WARNING_CM,
    **region_options: object,
) -> dict[str, object]:
    """Compute ``geometry_hip_v0`` by raw perimeter argmax in the stable region."""
    vertices, faces, joints = _validate_inputs(vertices, faces, joints)
    if local_jump_warning_cm <= 0.0:
        raise ValueError("local_jump_warning_cm must be positive")
    region = detect_pelvic_search_region(
        vertices,
        faces,
        joints,
        step_normalized_height=step_normalized_height,
        **region_options,
    )
    profile = region["profile"]
    maximum = select_profile_maximum(profile)
    selected_index = maximum["selected_index"]
    selected = dict(maximum["selected_record"])

    selected_slice = slice_mesh(vertices, faces, float(selected["plane_y_m"]))
    selected_again = select_torso_contour(
        selected_slice["contours"],
        np.asarray(selected["centerline_xz_m"], dtype=np.float64),
    )
    selected_contour = selected_again["selected_contour"]
    selected.update(
        {
            "ordered_points_m": selected_contour["ordered_points_m"],
            "num_points": selected_contour["num_points"],
        }
    )
    stability = _local_stability(profile, selected_index)
    warnings: list[str] = []
    if maximum["boundary_maximum"]:
        warnings.append("maximum lies on the stable pelvic search boundary")
    fallback_count = sum(bool(record["fallback_used"]) for record in profile)
    if fallback_count:
        warnings.append(f"pelvic contour selector used fallback on {fallback_count} slices")
    if region["degeneracy_retry_count"]:
        warnings.append(
            "horizontal plane received an explicit micro-offset degeneracy retry on "
            f"{region['degeneracy_retry_count']} slice(s); maximum adjustment="
            f"{region['maximum_plane_adjustment_m'] * 1e6:.3f} µm"
        )
    if stability["max_adjacent_change_cm"] > local_jump_warning_cm:
        warnings.append(
            "stable pelvic profile contains an adjacent-layer circumference change "
            f"above {local_jump_warning_cm:.3f} cm"
        )

    region_summary = {key: value for key, value in region.items() if key != "profile"}
    return {
        "definition": GEOMETRY_HIP_DEFINITION,
        "status": GEOMETRY_HIP_STATUS,
        "definition_text": (
            "The maximum raw horizontal pelvic-contour perimeter within a "
            "topology- and stability-constrained lower-pelvis-to-spine1 interval "
            "of the zero-pose canonical SMPL-X mesh."
        ),
        "measurement_space": "raw canonical SMPL-X geometry",
        "metric_calibration": "none",
        "search_region": region_summary,
        "scan_parameters": {
            "step_normalized_height": step_normalized_height,
            "step_m": region["step_m"],
            "num_slices": len(profile),
            "smoothing": "none",
            "curve_fitting": "none",
            "maximum_method": "raw_discrete_argmax_perimeter",
            "area_selection": "not_used; area retained for diagnostics only",
            "local_jump_warning_cm": local_jump_warning_cm,
        },
        "profile": profile,
        "selected": selected,
        "selected_index": selected_index,
        "boundary_maximum": maximum["boundary_maximum"],
        "fallback_count": fallback_count,
        "local_stability": stability,
        "warnings": warnings,
    }
