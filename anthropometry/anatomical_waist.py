#!/usr/bin/env python3
"""Gender-aware anatomical waist proxy on canonical SMPL-X meshes.

The fixed-XYZ baseline and the surface-anchored proxy both perform one
horizontal slice at the bilateral mean of manually annotated lower-rib/
iliac-crest midpoint heights. Neither searches a circumference profile.
"""

from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np

from slicing import ENDPOINT_CLUSTER_TOLERANCE_M, EPS, slice_mesh
from torso import (
    DEFAULT_CENTERLINE_PROXIMITY_M,
    compute_torso_vertical_interval,
    interpolate_spine_centerline_xz,
    select_torso_contour,
)


ANATOMICAL_WAIST_DEFINITION = "anatomical_waist_proxy_v1"
SURFACE_ANCHORED_ANATOMICAL_WAIST_DEFINITION = (
    "anatomical_midpoint_waist_proxy_v1"
)
SURFACE_ANCHORED_ANATOMICAL_WAIST_STATUS = "frozen_v1"
ANATOMICAL_WAIST_STATUS = "baseline"
WAIST_LANDMARK_NAMES = (
    "left_lower_rib",
    "right_lower_rib",
    "left_iliac_crest",
    "right_iliac_crest",
)
GENDER_LABEL_MAP = {
    "0": "female",
    "female": "female",
    "f": "female",
    "1": "male",
    "male": "male",
    "m": "male",
    "neutral": "neutral",
    "n": "neutral",
}


def normalize_gender_label(value: object) -> str:
    """Normalize project gender labels without silently guessing unknown values."""
    key = str(value).strip().lower()
    if key.endswith(".0") and key[:-2] in {"0", "1"}:
        key = key[:-2]
    if key not in GENDER_LABEL_MAP:
        raise ValueError(f"unsupported gender label: {value!r}")
    return GENDER_LABEL_MAP[key]


def _xlsx_rows(path: Path) -> list[list[str]]:
    namespace = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as archive:
        shared_strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", namespace):
                shared_strings.append(
                    "".join((text.text or "") for text in item.findall(".//a:t", namespace))
                )
        root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows = []
        for row in root.findall(".//a:sheetData/a:row", namespace):
            values_by_column: dict[int, str] = {}
            for cell in row.findall("a:c", namespace):
                reference = cell.attrib.get("r", "A1")
                letters = "".join(character for character in reference if character.isalpha())
                column = 0
                for character in letters.upper():
                    column = column * 26 + ord(character) - ord("A") + 1
                column -= 1
                cell_type = cell.attrib.get("t")
                if cell_type == "inlineStr":
                    value = "".join(
                        (text.text or "") for text in cell.findall(".//a:t", namespace)
                    )
                else:
                    value_node = cell.find("a:v", namespace)
                    value = "" if value_node is None else value_node.text or ""
                    if cell_type == "s" and value:
                        value = shared_strings[int(value)]
                values_by_column[column] = value
            width = max(values_by_column, default=-1) + 1
            rows.append([values_by_column.get(index, "") for index in range(width)])
    return rows


def load_subject_gender_labels(path: Path) -> dict[str, str]:
    """Load subject-level gender labels and reject conflicting duplicates."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"gender label file not found: {path}")
    if path.suffix.lower() != ".xlsx":
        raise ValueError("gender label source must be an .xlsx workbook")
    rows = _xlsx_rows(path)
    if not rows:
        raise ValueError("gender label workbook is empty")
    header = [str(value).strip() for value in rows[0]]
    try:
        subject_index = header.index("subject_id")
        gender_index = header.index("gender_label")
    except ValueError as error:
        raise ValueError("gender workbook must contain subject_id and gender_label") from error

    labels: dict[str, str] = {}
    for row in rows[1:]:
        if len(row) <= max(subject_index, gender_index):
            continue
        subject = str(row[subject_index]).strip()
        if not subject:
            continue
        gender = normalize_gender_label(row[gender_index])
        if subject in labels and labels[subject] != gender:
            raise ValueError(f"conflicting gender labels for {subject}")
        labels[subject] = gender
    return labels


def load_anatomical_landmarks(path: Path) -> dict[str, object]:
    """Load and validate one gender-specific four-point landmark file."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    missing = [name for name in WAIST_LANDMARK_NAMES if name not in data]
    if missing:
        raise ValueError(f"{path} is missing landmarks: {', '.join(missing)}")
    points = {}
    for name in WAIST_LANDMARK_NAMES:
        point = np.asarray(data[name], dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError(f"{name} must be a finite XYZ coordinate")
        points[name] = point
    return {
        "path": str(path.resolve()),
        "landmark_type": data.get("landmark_type", "fixed_xyz_per_gender"),
        "points": points,
    }


def load_surface_anchored_landmarks(
    vertices: np.ndarray,
    faces: np.ndarray,
    path: Path,
) -> dict[str, object]:
    """Evaluate face+barycentric anchors on one subject-specific mesh."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "smplx_surface_landmarks_v1":
        raise ValueError(f"{path} is not an smplx_surface_landmarks_v1 file")
    anchors = data.get("anchors", {})
    missing = [name for name in WAIST_LANDMARK_NAMES if name not in anchors]
    if missing:
        raise ValueError(f"{path} is missing anchors: {', '.join(missing)}")

    points = {}
    evaluated_anchors = {}
    for name in WAIST_LANDMARK_NAMES:
        anchor = anchors[name]
        face_id = int(anchor["face_id"])
        if not 0 <= face_id < len(faces):
            raise ValueError(f"{name} face_id lies outside the mesh topology")
        expected_vertex_ids = np.asarray(anchor["vertex_ids"], dtype=np.int64)
        actual_vertex_ids = faces[face_id]
        if expected_vertex_ids.shape != (3,) or not np.array_equal(
            expected_vertex_ids, actual_vertex_ids
        ):
            raise ValueError(f"{name} face vertex IDs do not match the mesh topology")
        barycentric = np.asarray(anchor["barycentric"], dtype=np.float64)
        if barycentric.shape != (3,) or not np.isfinite(barycentric).all():
            raise ValueError(f"{name} barycentric coordinates must be finite length 3")
        if not np.isclose(barycentric.sum(), 1.0, atol=1e-8):
            raise ValueError(f"{name} barycentric coordinates do not sum to one")
        if np.any(barycentric < -1e-8) or np.any(barycentric > 1.0 + 1e-8):
            raise ValueError(f"{name} barycentric coordinates lie outside the face")
        point = barycentric @ vertices[actual_vertex_ids]
        points[name] = point
        evaluated_anchors[name] = {
            "face_id": face_id,
            "vertex_ids": actual_vertex_ids.tolist(),
            "barycentric": barycentric.tolist(),
            "evaluated_xyz_m": point.tolist(),
            "source_xyz_m": anchor["source_xyz_m"],
            "projected_template_xyz_m": anchor["projected_template_xyz_m"],
            "projection_distance_mm": anchor["projection_distance_mm"],
        }
    return {
        "path": str(path.resolve()),
        "landmark_type": "surface_face_barycentric",
        "gender": data.get("gender"),
        "template": data.get("template"),
        "projection": data.get("projection"),
        "points": points,
        "anchors": evaluated_anchors,
    }


def compute_anatomical_waist_plane(landmarks: dict[str, object]) -> dict[str, object]:
    """Compute the bilateral mean lower-rib/iliac-crest midpoint height."""
    points = landmarks["points"]
    left_midpoint = 0.5 * (points["left_lower_rib"] + points["left_iliac_crest"])
    right_midpoint = 0.5 * (points["right_lower_rib"] + points["right_iliac_crest"])
    origin = 0.5 * (left_midpoint + right_midpoint)
    return {
        "plane_y_m": float(origin[1]),
        "plane_origin_m": origin.tolist(),
        "left_midpoint_m": left_midpoint.tolist(),
        "right_midpoint_m": right_midpoint.tolist(),
        "left_right_midpoint_y_mismatch_mm": float(
            abs(left_midpoint[1] - right_midpoint[1]) * 1000.0
        ),
        "lower_rib_y_mismatch_mm": float(
            abs(points["left_lower_rib"][1] - points["right_lower_rib"][1]) * 1000.0
        ),
        "iliac_crest_y_mismatch_mm": float(
            abs(points["left_iliac_crest"][1] - points["right_iliac_crest"][1])
            * 1000.0
        ),
        "support_points_m": {name: points[name].tolist() for name in WAIST_LANDMARK_NAMES},
        "origin_rule": (
            "mean height of left/right lower-rib to iliac-crest midpoints"
        ),
    }


def _measure_anatomical_waist_from_landmarks(
    vertices: np.ndarray,
    faces: np.ndarray,
    joints: np.ndarray,
    *,
    gender: str,
    landmarks: dict[str, object],
    definition: str,
    status: str,
    surface_anchored: bool,
    eps: float = EPS,
    endpoint_tolerance: float = ENDPOINT_CLUSTER_TOLERANCE_M,
    centerline_proximity_m: float = DEFAULT_CENTERLINE_PROXIMITY_M,
) -> dict[str, object]:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    joints = np.asarray(joints, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("vertices must be a finite array shaped (N, 3)")
    gender = normalize_gender_label(gender)
    if gender not in {"female", "male", "neutral"}:
        raise ValueError(f"unsupported SMPL-X gender: {gender}")

    plane = compute_anatomical_waist_plane(landmarks)
    plane_y = plane["plane_y_m"]
    interval = compute_torso_vertical_interval(joints)
    if not interval["y_min_m"] <= plane_y <= interval["y_max_m"]:
        raise ValueError("anatomical waist plane lies outside the pelvis-to-spine2 interval")

    sliced = slice_mesh(
        vertices,
        faces,
        plane_y,
        eps=eps,
        endpoint_tolerance=endpoint_tolerance,
    )
    connectivity = sliced["diagnostics"]["connectivity"]
    if connectivity["invalid_component_count"] != 0:
        raise RuntimeError("anatomical waist slice contains an invalid contour component")
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
        "definition": definition,
        "status": status,
        "definition_text": (
            "A single horizontal canonical SMPL-X slice at the bilateral mean height "
            "of manually annotated lower-rib to iliac-crest midpoints."
        ),
        "independent_of_circumference_minimum": True,
        "measurement_space": "raw canonical SMPL-X geometry",
        "metric_calibration": "none",
        "gender": gender,
        "gender_fallback_used": False,
        "landmark_source": {
            "path": landmarks["path"],
            "landmark_type": landmarks["landmark_type"],
            "annotation_scope": "gender-specific standard-pose SMPL-X template",
            "surface_anchored": surface_anchored,
            "anchor_template": landmarks.get("template"),
            "projection": landmarks.get("projection"),
            "evaluated_anchors": landmarks.get("anchors"),
        },
        "plane_definition": {
            **plane,
            "normalized_height": float((plane_y - mesh_min_y) / raw_height),
        },
        "search_or_scan": "none; one direct anatomical-landmark-defined slice",
        "num_contours": len(sliced["contours"]),
        "selected_contour_id": selection["selected_contour_id"],
        "selection_mode": selection["selection_method"],
        "fallback_used": (
            selection["selection_method"]
            != "spine_centerline_containment_then_area"
        ),
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
            "remaining_duplicate_segments": connectivity[
                "remaining_duplicate_segments"
            ],
            "invalid_component_count": connectivity["invalid_component_count"],
        },
        "scientific_boundary": (
            "This is a reproducible landmark proxy based on manual annotations on "
            "gendered standard-pose SMPL-X templates. It is WHO-protocol-inspired, "
            "not a per-subject clinical palpation or certified WHO/ISO measurement."
        ),
    }


def measure_anatomical_waist(
    vertices: np.ndarray,
    faces: np.ndarray,
    joints: np.ndarray,
    *,
    gender: str,
    landmark_path: Path,
    eps: float = EPS,
    endpoint_tolerance: float = ENDPOINT_CLUSTER_TOLERANCE_M,
    centerline_proximity_m: float = DEFAULT_CENTERLINE_PROXIMITY_M,
) -> dict[str, object]:
    """Measure the retained fixed-XYZ anatomy baseline."""
    landmarks = load_anatomical_landmarks(landmark_path)
    return _measure_anatomical_waist_from_landmarks(
        vertices,
        faces,
        joints,
        gender=gender,
        landmarks=landmarks,
        definition=ANATOMICAL_WAIST_DEFINITION,
        status=ANATOMICAL_WAIST_STATUS,
        surface_anchored=False,
        eps=eps,
        endpoint_tolerance=endpoint_tolerance,
        centerline_proximity_m=centerline_proximity_m,
    )


def measure_surface_anchored_anatomical_waist(
    vertices: np.ndarray,
    faces: np.ndarray,
    joints: np.ndarray,
    *,
    gender: str,
    anchor_path: Path,
    eps: float = EPS,
    endpoint_tolerance: float = ENDPOINT_CLUSTER_TOLERANCE_M,
    centerline_proximity_m: float = DEFAULT_CENTERLINE_PROXIMITY_M,
) -> dict[str, object]:
    """Measure the final beta-deforming face+barycentric anatomy proxy."""
    landmarks = load_surface_anchored_landmarks(vertices, faces, anchor_path)
    normalized_gender = normalize_gender_label(gender)
    if landmarks["gender"] != normalized_gender:
        raise ValueError(
            f"anchor gender {landmarks['gender']} does not match sample gender "
            f"{normalized_gender}"
        )
    return _measure_anatomical_waist_from_landmarks(
        vertices,
        faces,
        joints,
        gender=normalized_gender,
        landmarks=landmarks,
        definition=SURFACE_ANCHORED_ANATOMICAL_WAIST_DEFINITION,
        status=SURFACE_ANCHORED_ANATOMICAL_WAIST_STATUS,
        surface_anchored=True,
        eps=eps,
        endpoint_tolerance=endpoint_tolerance,
        centerline_proximity_m=centerline_proximity_m,
    )
