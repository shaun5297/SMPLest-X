#!/usr/bin/env python3
"""Unified raw canonical SMPL-X anthropometric extractor for Step 2.8."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from anatomical_waist import (
    load_subject_gender_labels,
    measure_surface_anchored_anatomical_waist,
)
from chest import (
    FOCUSED_SHAPY_CHEST_CONTROL,
    LITERATURE_CHEST_DEFINITION,
    compute_candidate_planes,
    evaluate_chest_plane,
)
from definitions import PRIMARY_TARGETS, get_definition_registry
from hip import scan_geometry_hip
from literature_hip import measure_literature_hip
from literature_waist import measure_literature_waist
from utils import (
    ACROMION_SURFACE_PROXY_V1_LEFT_VERTEX_ID,
    ACROMION_SURFACE_PROXY_V1_RIGHT_VERTEX_ID,
    LEFT_SHOULDER_INDEX,
    PUBLISHED_LEFT_SHOULDER_SURFACE_VERTEX_ID,
    PUBLISHED_RIGHT_SHOULDER_SURFACE_VERTEX_ID,
    RIGHT_SHOULDER_INDEX,
    euclidean_distance,
    infer_sample_name,
    load_canonical_mesh,
    measure_axis_ranges,
    verify_smplx_axes,
)
from waist import scan_geometry_waist


EXTRACTOR_DEFINITION = "unified_anthropometric_extractor_v1"


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    path = path.resolve()
    vertices, joints = load_canonical_mesh(path)
    with np.load(path, allow_pickle=False) as data:
        if "faces" not in data:
            raise KeyError(f"{path} must contain faces")
        faces = np.asarray(data["faces"], dtype=np.int32)
        model_gender = str(data["model_gender"]) if "model_gender" in data else "unknown"
    axis_verification = verify_smplx_axes(vertices, joints)
    return vertices, joints, faces, {
        "source_npz": str(path),
        "model_gender": model_gender,
        "axis_verification": axis_verification,
    }


def _circumference_summary(
    result: dict[str, object], *, selected: bool = False
) -> dict[str, object]:
    value = result["selected"] if selected else result
    plane = value if selected else result.get("plane_definition", {})
    return {
        "definition": result["definition"],
        "status": result["status"],
        "perimeter_m": float(value["perimeter_m"]),
        "perimeter_cm": float(value["perimeter_cm"]),
        "plane_y_m": float(value.get("plane_y_m", plane["plane_y_m"])),
        "y_norm": float(value.get("normalized_height", plane.get("normalized_height"))),
        "area_m2": float(value["area_m2"]),
        "centroid_xz_m": value["centroid_xz_m"],
        "selected_contour_id": int(value["selected_contour_id"]),
        "selection_mode": value["selection_mode"],
        "fallback_used": bool(value["fallback_used"]),
        "num_contours": int(value.get("num_contours", result.get("num_contours", 0))),
        "diagnostics": {
            "invalid_component_count": int(
                value.get("invalid_component_count", result.get("diagnostics", {}).get("invalid_component_count", 0))
            ),
            "remaining_duplicate_segments": int(
                value.get("remaining_duplicate_segments", result.get("diagnostics", {}).get("remaining_duplicate_segments", 0))
            ),
        },
    }


def _chest_summary(
    vertices: np.ndarray,
    faces: np.ndarray,
    joints: np.ndarray,
    definition: dict[str, object],
) -> dict[str, object]:
    result = evaluate_chest_plane(vertices, faces, joints, float(definition["plane_y_m"]))
    raw_height = float(np.ptp(vertices[:, 1]))
    mesh_min_y = float(vertices[:, 1].min())
    metrics = result["selected_metrics"]
    return {
        "definition": definition["definition"],
        "status": definition["status"],
        "perimeter_m": float(metrics["perimeter_m"]),
        "perimeter_cm": float(metrics["perimeter_cm"]),
        "plane_y_m": float(definition["plane_y_m"]),
        "y_norm": float((definition["plane_y_m"] - mesh_min_y) / raw_height),
        "area_m2": float(metrics["area_m2"]),
        "compactness": float(metrics["compactness"]),
        "centroid_xz_m": metrics["centroid_xz_m"],
        "landmarks": definition["landmarks"],
        "plane_rule": definition["plane_rule"],
        "selected_contour_id": result["selected_contour_id"],
        "selection_mode": result["selection_mode"],
        "fallback_used": result["fallback_used"],
        "num_contours": result["num_contours"],
        "arm_torso_merge": result["possible_arm_torso_merge"],
        "diagnostics": result["diagnostics"],
    }


def extract_anthropometry(
    input_path: Path,
    *,
    gender: str,
    anchor_dir: Path,
) -> dict[str, object]:
    """Compute every frozen Phase 2 output from one canonical NPZ."""
    input_path = input_path.resolve()
    vertices, joints, faces, source = load_mesh(input_path)
    sample = infer_sample_name(input_path)
    raw_height_m = float(np.ptp(vertices[:, 1]))

    shoulder_points = {
        "left_joint": joints[LEFT_SHOULDER_INDEX],
        "right_joint": joints[RIGHT_SHOULDER_INDEX],
        "left_literature": vertices[PUBLISHED_LEFT_SHOULDER_SURFACE_VERTEX_ID],
        "right_literature": vertices[PUBLISHED_RIGHT_SHOULDER_SURFACE_VERTEX_ID],
        "left_acromion": vertices[ACROMION_SURFACE_PROXY_V1_LEFT_VERTEX_ID],
        "right_acromion": vertices[ACROMION_SURFACE_PROXY_V1_RIGHT_VERTEX_ID],
    }
    geometry_waist = scan_geometry_waist(vertices, faces, joints)
    literature_waist = measure_literature_waist(vertices, faces, joints)
    anchor_path = anchor_dir / f"landmarks_{gender}_surface.json"
    anatomical_waist = measure_surface_anchored_anatomical_waist(
        vertices, faces, joints, gender=gender, anchor_path=anchor_path
    )
    geometry_hip = scan_geometry_hip(vertices, faces, joints)
    literature_hip = measure_literature_hip(vertices, faces, joints)
    chest_planes = compute_candidate_planes(vertices, faces)

    result = {
        "extractor": {
            "definition": EXTRACTOR_DEFINITION,
            "status": "baseline_infrastructure",
            "primary_targets": PRIMARY_TARGETS,
        },
        "sample": sample,
        "source_npz": str(input_path),
        "geometry": {
            "coordinate_system": "smplx_native",
            "axis_semantics": {"x": "left_right", "y": "vertical", "z": "front_back"},
            "unit": "meter",
            "metric_calibrated": False,
            "canonical_pose": "zero_pose",
            "model_gender": source["model_gender"],
            "subject_gender_label": gender,
            "axis_ranges_m": measure_axis_ranges(vertices),
            "axis_verification": source["axis_verification"],
        },
        "height": {
            "raw_height_v0": {
                "definition": "raw_height_v0",
                "status": "baseline",
                "value_m": raw_height_m,
                "value_cm": raw_height_m * 100.0,
                "method": "max(vertices[:,1]) - min(vertices[:,1])",
            }
        },
        "shoulder": {
            "shoulder_joint_width": {
                "definition": "shoulder_joint_width",
                "status": "control",
                "value_m": euclidean_distance(shoulder_points["left_joint"], shoulder_points["right_joint"]),
            },
            "literature_shoulder_breadth": {
                "definition": "literature_shoulder_breadth",
                "status": "literature_baseline",
                "value_m": euclidean_distance(shoulder_points["left_literature"], shoulder_points["right_literature"]),
            },
            "acromion_surface_proxy_v1": {
                "definition": "acromion_surface_proxy_v1",
                "status": "frozen_v1",
                "value_m": euclidean_distance(shoulder_points["left_acromion"], shoulder_points["right_acromion"]),
            },
            "landmark_coordinates_m": {key: value.tolist() for key, value in shoulder_points.items()},
        },
        "waist": {
            "geometry_waist_v0": _circumference_summary(geometry_waist, selected=True),
            "literature_waist_v1": {
                **_circumference_summary(literature_waist),
                "landmarks": {
                    "front": literature_waist["plane_definition"]["front_landmark"],
                    "back": literature_waist["plane_definition"]["back_landmark"],
                },
            },
            "anatomical_midpoint_waist_proxy_v1": {
                **_circumference_summary(anatomical_waist),
                "gender": gender,
                "anchor_path": str(anchor_path.resolve()),
                "landmarks": anatomical_waist["plane_definition"]["support_points_m"],
            },
        },
        "hip": {
            "geometry_hip_v0": _circumference_summary(geometry_hip, selected=True),
            "literature_hip_v1": {
                **_circumference_summary(literature_hip),
                "landmarks": literature_hip["plane_definition"]["landmark"],
            },
        },
        "chest": {
            "literature_chest_v1": _chest_summary(
                vertices, faces, joints, chest_planes[LITERATURE_CHEST_DEFINITION]
            ),
            "focused_shapy_chest_control": _chest_summary(
                vertices, faces, joints, chest_planes[FOCUSED_SHAPY_CHEST_CONTROL]
            ),
            "geometry_extreme_status": get_definition_registry()["geometry_chest_extreme"],
        },
        "definition_registry": get_definition_registry(),
        "scientific_boundary": (
            "All values are raw zero-pose canonical SMPL-X anthropometry. They are "
            "not real-person ground-truth measurements and receive no metric calibration."
        ),
    }
    for item in result["shoulder"].values():
        if isinstance(item, dict) and "value_m" in item:
            item["value_cm"] = item["value_m"] * 100.0
    return result


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--gender-labels", required=True, type=Path)
    parser.add_argument(
        "--anchor-dir",
        type=Path,
        default=repo_root / "anthropometry" / "landmarks" / "anatomical_midpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "anthropometry" / "artifacts" / "unified_extractor",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gender_labels = load_subject_gender_labels(args.gender_labels)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for input_path in args.inputs:
        sample = infer_sample_name(input_path)
        if sample not in gender_labels:
            raise KeyError(f"gender label missing for {sample}")
        result = extract_anthropometry(
            input_path, gender=gender_labels[sample], anchor_dir=args.anchor_dir
        )
        output_path = args.output_dir / f"{sample}_anthropometry.json"
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"{sample}: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
