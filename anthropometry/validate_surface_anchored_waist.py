#!/usr/bin/env python3
"""Validate beta-deforming anatomical waist surface anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from anatomical_waist import (
    WAIST_LANDMARK_NAMES,
    compute_anatomical_waist_plane,
    load_anatomical_landmarks,
    load_surface_anchored_landmarks,
    load_subject_gender_labels,
    measure_anatomical_waist,
    measure_surface_anchored_anatomical_waist,
)
from utils import infer_sample_name, load_canonical_mesh, verify_smplx_axes


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Compare fixed template XYZ and face+barycentric anatomical waist proxies."
        )
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Canonical SMPL-X NPZ files")
    parser.add_argument("--gender-labels", required=True, type=Path)
    parser.add_argument("--fixed-landmark-dir", required=True, type=Path)
    parser.add_argument(
        "--neutral-model",
        required=True,
        type=Path,
        help="SMPLX_NEUTRAL.npz used to separate template-transfer and beta effects",
    )
    parser.add_argument(
        "--anchor-dir",
        type=Path,
        default=repo_root / "anthropometry" / "landmarks" / "anatomical_midpoint",
    )
    parser.add_argument("--max-template-projection-mm", type=float, default=2.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            repo_root
            / "anthropometry"
            / "artifacts"
            / "surface_anchored_waist_validation"
        ),
    )
    return parser.parse_args()


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices, joints = load_canonical_mesh(path)
    verify_smplx_axes(vertices, joints)
    with np.load(path, allow_pickle=False) as data:
        if "faces" not in data:
            raise KeyError(f"{path} must contain faces")
        faces = np.asarray(data["faces"], dtype=np.int32)
    return vertices, joints, faces


def closed_points(record: dict[str, object]) -> np.ndarray:
    points = np.asarray(record["ordered_points_m"], dtype=np.float64)
    return np.vstack([points, points[0]])


def landmark_displacements(
    fixed: dict[str, object],
    anchored: dict[str, object],
) -> dict[str, dict[str, object]]:
    fixed_points = fixed["plane_definition"]["support_points_m"]
    anchored_points = anchored["plane_definition"]["support_points_m"]
    records = {}
    for name in WAIST_LANDMARK_NAMES:
        fixed_point = np.asarray(fixed_points[name], dtype=np.float64)
        anchored_point = np.asarray(anchored_points[name], dtype=np.float64)
        delta = anchored_point - fixed_point
        records[name] = {
            "fixed_xyz_m": fixed_point.tolist(),
            "anchored_xyz_m": anchored_point.tolist(),
            "delta_xyz_mm_anchored_minus_fixed": (delta * 1000.0).tolist(),
            "euclidean_displacement_mm": float(np.linalg.norm(delta) * 1000.0),
        }
    return records


def plot_result(
    sample: str,
    vertices: np.ndarray,
    fixed: dict[str, object],
    anchored: dict[str, object],
    output_path: Path,
) -> None:
    fixed_plane = fixed["plane_definition"]
    anchored_plane = anchored["plane_definition"]
    fixed_points = np.asarray(
        [fixed_plane["support_points_m"][name] for name in WAIST_LANDMARK_NAMES],
        dtype=np.float64,
    )
    anchored_points = np.asarray(
        [anchored_plane["support_points_m"][name] for name in WAIST_LANDMARK_NAMES],
        dtype=np.float64,
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.5))
    sagittal_axis, contour_axis = axes
    sagittal_axis.scatter(
        vertices[:, 2], vertices[:, 1], s=0.35, color="0.78", alpha=0.18,
        rasterized=True, label="canonical mesh vertices",
    )
    sagittal_axis.axhline(
        fixed_plane["plane_y_m"], color="0.35", linestyle="--", linewidth=2.0,
        label=f"fixed XYZ y_norm={fixed_plane['normalized_height']:.3f}",
    )
    sagittal_axis.axhline(
        anchored_plane["plane_y_m"], color="#2ca02c", linewidth=2.4,
        label=f"surface anchor y_norm={anchored_plane['normalized_height']:.3f}",
    )
    sagittal_axis.scatter(
        fixed_points[:, 2], fixed_points[:, 1], marker="o", s=65,
        facecolors="none", edgecolors="0.25", linewidth=1.5,
        label="fixed template XYZ",
    )
    sagittal_axis.scatter(
        anchored_points[:, 2], anchored_points[:, 1], marker="x", s=75,
        color="#2ca02c", linewidths=2, label="evaluated surface anchors",
    )
    for fixed_point, anchored_point in zip(fixed_points, anchored_points):
        sagittal_axis.plot(
            [fixed_point[2], anchored_point[2]],
            [fixed_point[1], anchored_point[1]],
            color="#2ca02c", alpha=0.55, linewidth=1.0,
        )
    delta_y_mm = (
        anchored_plane["plane_y_m"] - fixed_plane["plane_y_m"]
    ) * 1000.0
    sagittal_axis.set_xlabel("Z (m, sagittal)")
    sagittal_axis.set_ylabel("Y (m, vertical)")
    sagittal_axis.set_title(
        f"{anchored['gender']} beta-deforming landmarks\n"
        f"anchored - fixed plane = {delta_y_mm:+.2f} mm"
    )
    sagittal_axis.grid(alpha=0.2)
    sagittal_axis.legend(frameon=False, fontsize=8)

    fixed_contour = closed_points(fixed)
    anchored_contour = closed_points(anchored)
    contour_axis.plot(
        fixed_contour[:, 0], fixed_contour[:, 2], color="0.35", linestyle="--",
        linewidth=2.0, label=f"fixed XYZ: {fixed['perimeter_cm']:.3f} cm",
    )
    contour_axis.plot(
        anchored_contour[:, 0], anchored_contour[:, 2], color="#2ca02c",
        linewidth=2.5, label=f"surface anchored: {anchored['perimeter_cm']:.3f} cm",
    )
    centerline = np.asarray(anchored["centerline_xz_m"], dtype=np.float64)
    contour_axis.scatter(
        [centerline[0]], [centerline[1]], marker="x", s=80, linewidths=2,
        color="black", label="spine centerline at anchored plane",
    )
    contour_axis.set_xlabel("X (m)")
    contour_axis.set_ylabel("Z (m)")
    contour_axis.set_title("Fixed versus beta-deforming waist contours")
    contour_axis.set_aspect("equal", adjustable="box")
    contour_axis.grid(alpha=0.2)
    contour_axis.legend(frameon=False, fontsize=8)
    fig.suptitle(f"{sample}: Step 2.5C.1 surface-anchored anatomical landmarks")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def validate_file(
    path: Path,
    gender: str,
    fixed_landmark_dir: Path,
    anchor_dir: Path,
    neutral_vertices: np.ndarray,
    neutral_faces: np.ndarray,
    max_template_projection_mm: float,
    output_dir: Path,
) -> dict[str, object]:
    path = path.resolve()
    vertices, joints, faces = load_mesh(path)
    sample = infer_sample_name(path)
    fixed = measure_anatomical_waist(
        vertices,
        faces,
        joints,
        gender=gender,
        landmark_path=fixed_landmark_dir / f"landmarks_{gender}.json",
    )
    anchored = measure_surface_anchored_anatomical_waist(
        vertices,
        faces,
        joints,
        gender=gender,
        anchor_path=anchor_dir / f"landmarks_{gender}_surface.json",
    )
    fixed_plane = fixed["plane_definition"]
    anchored_plane = anchored["plane_definition"]
    if not np.array_equal(faces, neutral_faces):
        raise ValueError("subject faces do not match the neutral SMPL-X topology")
    anchor_path = anchor_dir / f"landmarks_{gender}_surface.json"
    zero_beta_landmarks = load_surface_anchored_landmarks(
        neutral_vertices,
        neutral_faces,
        anchor_path,
    )
    zero_beta_plane = compute_anatomical_waist_plane(zero_beta_landmarks)
    fixed_landmarks = load_anatomical_landmarks(
        fixed_landmark_dir / f"landmarks_{gender}.json"
    )
    fixed_reference_plane = compute_anatomical_waist_plane(fixed_landmarks)
    displacements = landmark_displacements(fixed, anchored)
    comparison = {
        "fixed_definition": fixed["definition"],
        "anchored_definition": anchored["definition"],
        "fixed_plane_y_m": fixed_plane["plane_y_m"],
        "anchored_plane_y_m": anchored_plane["plane_y_m"],
        "fixed_y_norm": fixed_plane["normalized_height"],
        "anchored_y_norm": anchored_plane["normalized_height"],
        "delta_y_mm_anchored_minus_fixed": (
            anchored_plane["plane_y_m"] - fixed_plane["plane_y_m"]
        ) * 1000.0,
        "plane_y_decomposition_mm": {
            "gender_template_anchor_to_neutral_zero_beta_minus_fixed": (
                zero_beta_plane["plane_y_m"] - fixed_reference_plane["plane_y_m"]
            ) * 1000.0,
            "subject_beta_deformation_minus_neutral_zero_beta": (
                anchored_plane["plane_y_m"] - zero_beta_plane["plane_y_m"]
            ) * 1000.0,
            "total_subject_anchor_minus_fixed": (
                anchored_plane["plane_y_m"] - fixed_reference_plane["plane_y_m"]
            ) * 1000.0,
        },
        "fixed_circumference_cm": fixed["perimeter_cm"],
        "anchored_circumference_cm": anchored["perimeter_cm"],
        "delta_c_cm_anchored_minus_fixed": (
            anchored["perimeter_cm"] - fixed["perimeter_cm"]
        ),
        "landmark_displacements": displacements,
        "maximum_landmark_displacement_mm": max(
            item["euclidean_displacement_mm"] for item in displacements.values()
        ),
    }

    failures = []
    if anchored["definition"] != "anatomical_midpoint_waist_proxy_v1":
        failures.append("anchored result has the wrong definition name")
    if not anchored["landmark_source"]["surface_anchored"]:
        failures.append("anchored result does not declare surface anchoring")
    projection = anchored["landmark_source"]["projection"]
    if projection["maximum_projection_distance_mm"] > max_template_projection_mm:
        failures.append("template surface projection distance exceeds threshold")
    if anchored["search_or_scan"] != "none; one direct anatomical-landmark-defined slice":
        failures.append("anchored measurement performed a scan")
    if anchored["fallback_used"]:
        failures.append("anchored torso selector used fallback")
    if not anchored["centerline_inside"]:
        failures.append("anchored contour does not contain spine centerline")
    if anchored["diagnostics"]["invalid_component_count"] != 0:
        failures.append("anchored slice contains an invalid component")
    if anchored["diagnostics"]["remaining_duplicate_segments"] != 0:
        failures.append("anchored slice retains duplicate segments")
    if not np.isfinite(
        [comparison["delta_y_mm_anchored_minus_fixed"],
         comparison["delta_c_cm_anchored_minus_fixed"]]
    ).all():
        failures.append("fixed-to-anchored comparison contains NaN or Inf")

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"{sample}_fixed_vs_surface_anchored_waist.png"
    plot_result(sample, vertices, fixed, anchored, plot_path)
    return {
        "sample": sample,
        "source_npz": str(path),
        "gender": gender,
        "fixed_xyz": fixed,
        "surface_anchored": anchored,
        "comparison": comparison,
        "debug_plot": str(plot_path),
        "validation": {"passed": not failures, "failures": failures},
    }


def group_variation(results: list[dict[str, object]]) -> dict[str, object]:
    groups = {}
    for gender in ("female", "male"):
        group = [item for item in results if item["gender"] == gender]
        if not group:
            continue
        fixed_y = np.asarray(
            [item["comparison"]["fixed_plane_y_m"] for item in group], dtype=np.float64
        )
        anchored_y = np.asarray(
            [item["comparison"]["anchored_plane_y_m"] for item in group], dtype=np.float64
        )
        groups[gender] = {
            "sample_count": len(group),
            "fixed_plane_y_range_mm": float(np.ptp(fixed_y) * 1000.0),
            "anchored_plane_y_range_mm": float(np.ptp(anchored_y) * 1000.0),
            "anchored_plane_y_std_mm": float(np.std(anchored_y) * 1000.0),
        }
    return groups


def main() -> int:
    args = parse_args()
    if args.max_template_projection_mm <= 0.0:
        raise ValueError("--max-template-projection-mm must be positive")
    labels = load_subject_gender_labels(args.gender_labels.resolve())
    with np.load(args.neutral_model.resolve(), allow_pickle=False) as model:
        neutral_vertices = np.asarray(model["v_template"], dtype=np.float64)
        neutral_faces = np.asarray(model["f"], dtype=np.int64)
    inputs_by_sample = {infer_sample_name(path): path for path in args.inputs}
    missing = sorted(set(inputs_by_sample) - set(labels))
    if missing:
        raise KeyError(f"gender labels are missing samples: {', '.join(missing)}")
    results = [
        validate_file(
            path,
            labels[sample],
            args.fixed_landmark_dir.resolve(),
            args.anchor_dir.resolve(),
            neutral_vertices,
            neutral_faces,
            args.max_template_projection_mm,
            args.output_dir,
        )
        for sample, path in sorted(inputs_by_sample.items())
    ]
    passed = all(item["validation"]["passed"] for item in results)
    report = {
        "experiment": "Step 2.5C.1 surface-anchored anatomical landmarks",
        "definition": "anatomical_midpoint_waist_proxy_v1",
        "status": "frozen_v1",
        "scope": (
            "gender-template manual XYZ projected to face+barycentric anchors and "
            "evaluated on each beta-deformed neutral canonical SMPL-X mesh"
        ),
        "gender_label_source": str(args.gender_labels.resolve()),
        "fixed_landmark_dir": str(args.fixed_landmark_dir.resolve()),
        "anchor_dir": str(args.anchor_dir.resolve()),
        "neutral_model": str(args.neutral_model.resolve()),
        "maximum_template_projection_threshold_mm": args.max_template_projection_mm,
        "cross_beta_plane_variation": group_variation(results),
        "samples": results,
        "passed": passed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "surface_anchored_waist_validation.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("| Sample | Gender | Fixed C | Anchored C | Delta y (mm) | Delta C (cm) | Max point move (mm) |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for result in results:
        comparison = result["comparison"]
        print(
            f"| {result['sample']} | {result['gender']} | "
            f"{comparison['fixed_circumference_cm']:.3f} | "
            f"{comparison['anchored_circumference_cm']:.3f} | "
            f"{comparison['delta_y_mm_anchored_minus_fixed']:+.3f} | "
            f"{comparison['delta_c_cm_anchored_minus_fixed']:+.3f} | "
            f"{comparison['maximum_landmark_displacement_mm']:.3f} |"
        )
        for failure in result["validation"]["failures"]:
            print(f"  - failure: {failure}")
    print("\nCross-beta anchored plane variation:")
    for gender, stats in report["cross_beta_plane_variation"].items():
        print(
            f"  {gender}: fixed range={stats['fixed_plane_y_range_mm']:.3f} mm, "
            f"anchored range={stats['anchored_plane_y_range_mm']:.3f} mm, "
            f"anchored std={stats['anchored_plane_y_std_mm']:.3f} mm"
        )
    print("\nPlane-y decomposition (mm):")
    for result in results:
        decomposition = result["comparison"]["plane_y_decomposition_mm"]
        print(
            f"  {result['sample']}: template->neutral="
            f"{decomposition['gender_template_anchor_to_neutral_zero_beta_minus_fixed']:+.3f}, "
            f"beta={decomposition['subject_beta_deformation_minus_neutral_zero_beta']:+.3f}, "
            f"total={decomposition['total_subject_anchor_minus_fixed']:+.3f}"
        )
    print(f"\nValidation JSON: {output_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
