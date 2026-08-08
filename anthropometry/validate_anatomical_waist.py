#!/usr/bin/env python3
"""Validate the gender-aware anatomical waist proxy on canonical meshes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from anatomical_waist import (
    load_subject_gender_labels,
    measure_anatomical_waist,
)
from utils import infer_sample_name, load_canonical_mesh, verify_smplx_axes


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Validate one gender-routed lower-rib/iliac-crest waist slice and compare "
            "it with geometry_waist_v0 and literature_waist_v1."
        )
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Canonical SMPL-X NPZ files")
    parser.add_argument("--gender-labels", required=True, type=Path)
    parser.add_argument("--landmark-dir", required=True, type=Path)
    parser.add_argument(
        "--geometry-report",
        type=Path,
        default=(
            repo_root
            / "anthropometry"
            / "artifacts"
            / "waist_validation"
            / "geometry_waist_validation.json"
        ),
    )
    parser.add_argument(
        "--literature-report",
        type=Path,
        default=(
            repo_root
            / "anthropometry"
            / "artifacts"
            / "literature_waist_validation"
            / "literature_waist_validation.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            repo_root
            / "anthropometry"
            / "artifacts"
            / "anatomical_waist_validation"
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


def load_report(path: Path, definition: str) -> dict[str, dict[str, object]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("definition") != definition:
        raise ValueError(f"{path} is not a {definition} report")
    return {sample["sample"]: sample for sample in report["samples"]}


def closed_points(record: dict[str, object]) -> np.ndarray:
    points = np.asarray(record["ordered_points_m"], dtype=np.float64)
    return np.vstack([points, points[0]])


def plot_comparison(
    sample: str,
    vertices: np.ndarray,
    anatomical: dict[str, object],
    geometry: dict[str, object],
    literature: dict[str, object],
    output_path: Path,
) -> None:
    anatomy_plane = anatomical["plane_definition"]
    geometry_selected = geometry["selected"]
    literature_plane = literature["plane_definition"]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    sagittal_axis, contour_axis = axes
    sagittal_axis.scatter(
        vertices[:, 2], vertices[:, 1], s=0.35, color="0.78", alpha=0.18,
        rasterized=True, label="canonical mesh vertices",
    )
    sagittal_axis.axhline(
        anatomy_plane["plane_y_m"], color="#2ca02c", linewidth=2.4,
        label=f"anatomical proxy y_norm={anatomy_plane['normalized_height']:.3f}",
    )
    sagittal_axis.axhline(
        literature_plane["plane_y_m"], color="#ff7f0e", linewidth=2.0,
        linestyle=":", label=(
            f"literature y_norm={literature_plane['normalized_height']:.3f}"
        ),
    )
    sagittal_axis.axhline(
        geometry_selected["plane_y_m"], color="#d62728", linewidth=2.0,
        linestyle="--", label=(
            f"geometry minimum y_norm={geometry_selected['normalized_height']:.3f}"
        ),
    )
    support = anatomy_plane["support_points_m"]
    names = [
        "left_lower_rib", "right_lower_rib", "left_iliac_crest", "right_iliac_crest"
    ]
    support_points = np.asarray([support[name] for name in names], dtype=np.float64)
    sagittal_axis.scatter(
        support_points[:, 2], support_points[:, 1], s=72,
        color=["#1f77b4", "#1f77b4", "#9467bd", "#9467bd"],
        edgecolor="white", linewidth=0.8, label="manual rib / iliac landmarks",
    )
    midpoints = np.asarray(
        [anatomy_plane["left_midpoint_m"], anatomy_plane["right_midpoint_m"]],
        dtype=np.float64,
    )
    sagittal_axis.scatter(
        midpoints[:, 2], midpoints[:, 1], marker="x", s=85, linewidths=2,
        color="black", label="left/right anatomical midpoints",
    )
    sagittal_axis.set_xlabel("Z (m, sagittal)")
    sagittal_axis.set_ylabel("Y (m, vertical)")
    sagittal_axis.set_title(
        f"{anatomical['gender']} landmark-routed plane\n"
        f"bilateral midpoint mismatch="
        f"{anatomy_plane['left_right_midpoint_y_mismatch_mm']:.1f} mm"
    )
    sagittal_axis.grid(alpha=0.2)
    sagittal_axis.legend(frameon=False, fontsize=8)

    anatomy_contour = closed_points(anatomical)
    literature_contour = closed_points(literature)
    geometry_contour = closed_points(geometry_selected)
    contour_axis.plot(
        anatomy_contour[:, 0], anatomy_contour[:, 2], color="#2ca02c",
        linewidth=2.5, label=f"anatomical proxy: {anatomical['perimeter_cm']:.2f} cm",
    )
    contour_axis.plot(
        literature_contour[:, 0], literature_contour[:, 2], color="#ff7f0e",
        linewidth=2.0, linestyle=":",
        label=f"literature: {literature['perimeter_cm']:.2f} cm",
    )
    contour_axis.plot(
        geometry_contour[:, 0], geometry_contour[:, 2], color="#d62728",
        linewidth=2.0, linestyle="--",
        label=f"geometry minimum: {geometry_selected['perimeter_cm']:.2f} cm",
    )
    centerline = np.asarray(anatomical["centerline_xz_m"], dtype=np.float64)
    contour_axis.scatter(
        [centerline[0]], [centerline[1]], marker="x", s=80, linewidths=2,
        color="black", label="spine centerline at anatomical plane",
    )
    contour_axis.set_xlabel("X (m)")
    contour_axis.set_ylabel("Z (m)")
    contour_axis.set_title("Three independently selected waist contours")
    contour_axis.set_aspect("equal", adjustable="box")
    contour_axis.grid(alpha=0.2)
    contour_axis.legend(frameon=False, fontsize=8)
    fig.suptitle(f"{sample}: anatomical, literature, and geometry waist definitions")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def validate_file(
    path: Path,
    gender: str,
    landmark_dir: Path,
    geometry: dict[str, object],
    literature: dict[str, object],
    output_dir: Path,
) -> dict[str, object]:
    path = path.resolve()
    vertices, joints, faces = load_mesh(path)
    sample = infer_sample_name(path)
    landmark_path = landmark_dir / f"landmarks_{gender}.json"
    anatomical = measure_anatomical_waist(
        vertices,
        faces,
        joints,
        gender=gender,
        landmark_path=landmark_path,
    )
    geometry_selected = geometry["selected"]
    anatomy_plane = anatomical["plane_definition"]
    literature_plane = literature["plane_definition"]
    comparisons = {
        "geometry_waist_v0": {
            "other_y_norm": geometry_selected["normalized_height"],
            "delta_y_cm_anatomical_minus_other": (
                anatomy_plane["plane_y_m"] - geometry_selected["plane_y_m"]
            ) * 100.0,
            "other_circumference_cm": geometry_selected["perimeter_cm"],
            "delta_c_cm_anatomical_minus_other": (
                anatomical["perimeter_cm"] - geometry_selected["perimeter_cm"]
            ),
        },
        "literature_waist_v1": {
            "other_y_norm": literature_plane["normalized_height"],
            "delta_y_cm_anatomical_minus_other": (
                anatomy_plane["plane_y_m"] - literature_plane["plane_y_m"]
            ) * 100.0,
            "other_circumference_cm": literature["perimeter_cm"],
            "delta_c_cm_anatomical_minus_other": (
                anatomical["perimeter_cm"] - literature["perimeter_cm"]
            ),
        },
    }

    failures = []
    if gender not in {"female", "male"}:
        failures.append("validation sample did not resolve to female or male")
    if anatomical["gender_fallback_used"]:
        failures.append("gender routing used a fallback")
    if not anatomical["independent_of_circumference_minimum"]:
        failures.append("measurement is not independent from circumference minimum")
    if anatomical["search_or_scan"] != "none; one direct anatomical-landmark-defined slice":
        failures.append("measurement performed a scan instead of one direct slice")
    if not np.isfinite(anatomical["perimeter_m"]) or anatomical["perimeter_m"] <= 0.0:
        failures.append("anatomical circumference is non-finite or non-positive")
    if anatomical["fallback_used"]:
        failures.append("torso selector used bounded fallback")
    if not anatomical["centerline_inside"]:
        failures.append("selected contour does not contain the spine centerline")
    if anatomical["diagnostics"]["invalid_component_count"] != 0:
        failures.append("slice contains an invalid contour component")
    if anatomical["diagnostics"]["remaining_duplicate_segments"] != 0:
        failures.append("slice retains duplicate segments")

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"{sample}_anatomical_waist_comparison.png"
    plot_comparison(sample, vertices, anatomical, geometry, literature, plot_path)
    return {
        "sample": sample,
        "source_npz": str(path),
        **anatomical,
        "comparison": comparisons,
        "debug_plot": str(plot_path),
        "validation": {"passed": not failures, "failures": failures},
    }


def main() -> int:
    args = parse_args()
    gender_labels = load_subject_gender_labels(args.gender_labels.resolve())
    geometry_by_sample = load_report(args.geometry_report.resolve(), "geometry_waist_v0")
    literature_by_sample = load_report(
        args.literature_report.resolve(), "literature_waist_v1"
    )
    inputs_by_sample = {infer_sample_name(path): path for path in args.inputs}
    missing_gender = sorted(set(inputs_by_sample) - set(gender_labels))
    missing_geometry = sorted(set(inputs_by_sample) - set(geometry_by_sample))
    missing_literature = sorted(set(inputs_by_sample) - set(literature_by_sample))
    if missing_gender or missing_geometry or missing_literature:
        raise KeyError(
            "missing dependencies: "
            f"gender={missing_gender}, geometry={missing_geometry}, "
            f"literature={missing_literature}"
        )

    results = [
        validate_file(
            path,
            gender_labels[sample],
            args.landmark_dir.resolve(),
            geometry_by_sample[sample],
            literature_by_sample[sample],
            args.output_dir,
        )
        for sample, path in sorted(inputs_by_sample.items())
    ]
    passed = all(item["validation"]["passed"] for item in results)
    report = {
        "experiment": "Step 2.5C gender-aware anatomical waist proxy",
        "definition": "anatomical_waist_proxy_v1",
        "status": "baseline",
        "scope": (
            "single horizontal raw canonical slice at the bilateral mean of manual "
            "lower-rib/iliac-crest midpoint heights; gender-specific standard-pose "
            "SMPL-X landmarks; no circumference search, smoothing, fitting, or calibration"
        ),
        "gender_label_source": str(args.gender_labels.resolve()),
        "gender_mapping": {"0": "female", "1": "male"},
        "landmark_dir": str(args.landmark_dir.resolve()),
        "samples": results,
        "passed": passed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "anatomical_waist_validation.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(
        "| Sample | Gender | Anatomy y | Geometry y | Delta y A-G (cm) | "
        "Anatomy C | Geometry C | Delta C A-G | Literature C | Delta C A-L |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for result in results:
        geometry = result["comparison"]["geometry_waist_v0"]
        literature = result["comparison"]["literature_waist_v1"]
        print(
            f"| {result['sample']} | {result['gender']} | "
            f"{result['plane_definition']['normalized_height']:.4f} | "
            f"{geometry['other_y_norm']:.4f} | "
            f"{geometry['delta_y_cm_anatomical_minus_other']:+.3f} | "
            f"{result['perimeter_cm']:.3f} | "
            f"{geometry['other_circumference_cm']:.3f} | "
            f"{geometry['delta_c_cm_anatomical_minus_other']:+.3f} | "
            f"{literature['other_circumference_cm']:.3f} | "
            f"{literature['delta_c_cm_anatomical_minus_other']:+.3f} |"
        )
        for failure in result["validation"]["failures"]:
            print(f"  - failure: {failure}")
    print(f"\nValidation JSON: {output_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
