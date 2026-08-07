#!/usr/bin/env python3
"""Validate and compare the independent ``literature_waist_v1`` reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from literature_waist import measure_literature_waist
from utils import infer_sample_name, load_canonical_mesh, verify_smplx_axes


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Validate the single-slice literature landmark waist reference and compare "
            "it with an existing geometry_waist_v0 report."
        )
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Canonical SMPL-X NPZ files")
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
        "--output-dir",
        type=Path,
        default=(
            repo_root
            / "anthropometry"
            / "artifacts"
            / "literature_waist_validation"
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


def load_geometry_results(path: Path) -> dict[str, dict[str, object]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("definition") != "geometry_waist_v0":
        raise ValueError(f"{path} is not a geometry_waist_v0 report")
    return {sample["sample"]: sample for sample in report["samples"]}


def plot_comparison(
    sample: str,
    vertices: np.ndarray,
    literature: dict[str, object],
    geometry: dict[str, object],
    output_path: Path,
) -> None:
    plane = literature["plane_definition"]
    front = np.asarray(plane["front_landmark"]["coordinate_m"], dtype=np.float64)
    back = np.asarray(plane["back_landmark"]["coordinate_m"], dtype=np.float64)
    origin = np.asarray(plane["plane_origin_m"], dtype=np.float64)
    lit_y = float(plane["plane_y_m"])
    geometry_selected = geometry["selected"]
    geom_y = float(geometry_selected["plane_y_m"])

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.5))
    sagittal_axis, contour_axis = axes
    sagittal_axis.scatter(
        vertices[:, 2], vertices[:, 1], s=0.35, color="0.78", alpha=0.18,
        rasterized=True, label="canonical mesh vertices",
    )
    sagittal_axis.axhline(
        lit_y, color="#2ca02c", linewidth=2.2,
        label=f"literature y_norm={plane['normalized_height']:.3f}",
    )
    sagittal_axis.axhline(
        geom_y, color="#d62728", linewidth=2.0, linestyle="--",
        label=f"geometry minimum y_norm={geometry_selected['normalized_height']:.3f}",
    )
    sagittal_axis.scatter(
        [front[2], back[2]], [front[1], back[1]], s=75,
        color=["#1f77b4", "#9467bd"], edgecolor="white", linewidth=0.8,
        label="published v5939 / v5941",
    )
    sagittal_axis.scatter(
        [origin[2]], [origin[1]], marker="x", s=90, linewidths=2.2, color="black",
        label="landmark mean origin",
    )
    sagittal_axis.set_xlabel("Z (m, sagittal)")
    sagittal_axis.set_ylabel("Y (m, vertical)")
    sagittal_axis.set_title(
        "Independent landmark plane\n"
        f"landmark Y mismatch={plane['landmark_absolute_y_mismatch_mm']:.1f} mm"
    )
    sagittal_axis.grid(alpha=0.2)
    sagittal_axis.legend(frameon=False, fontsize=8)

    lit_points = np.asarray(literature["ordered_points_m"], dtype=np.float64)
    geom_points = np.asarray(geometry_selected["ordered_points_m"], dtype=np.float64)
    lit_closed = np.vstack([lit_points, lit_points[0]])
    geom_closed = np.vstack([geom_points, geom_points[0]])
    contour_axis.plot(
        lit_closed[:, 0], lit_closed[:, 2], color="#2ca02c", linewidth=2.4,
        label=f"literature: {literature['perimeter_cm']:.2f} cm",
    )
    contour_axis.plot(
        geom_closed[:, 0], geom_closed[:, 2], color="#d62728", linewidth=2.0,
        linestyle="--", label=f"geometry minimum: {geometry_selected['perimeter_cm']:.2f} cm",
    )
    lit_centerline = np.asarray(literature["centerline_xz_m"], dtype=np.float64)
    contour_axis.scatter(
        [lit_centerline[0]], [lit_centerline[1]], marker="x", s=80,
        linewidths=2, color="black", label="spine centerline at literature plane",
    )
    contour_axis.set_xlabel("X (m)")
    contour_axis.set_ylabel("Z (m)")
    contour_axis.set_title("X-Z contours at two independently selected heights")
    contour_axis.set_aspect("equal", adjustable="box")
    contour_axis.grid(alpha=0.2)
    contour_axis.legend(frameon=False, fontsize=8)
    fig.suptitle(f"{sample}: literature_waist_v1 vs geometry_waist_v0")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def validate_file(
    path: Path,
    geometry: dict[str, object],
    output_dir: Path,
) -> dict[str, object]:
    path = path.resolve()
    vertices, joints, faces = load_mesh(path)
    sample = infer_sample_name(path)
    literature = measure_literature_waist(vertices, faces, joints)
    geometry_selected = geometry["selected"]
    delta_y_m = literature["plane_definition"]["plane_y_m"] - geometry_selected["plane_y_m"]
    delta_c_cm = literature["perimeter_cm"] - geometry_selected["perimeter_cm"]
    comparison = {
        "geometry_y_norm": geometry_selected["normalized_height"],
        "literature_y_norm": literature["plane_definition"]["normalized_height"],
        "delta_y_m_literature_minus_geometry": delta_y_m,
        "delta_y_cm_literature_minus_geometry": delta_y_m * 100.0,
        "geometry_circumference_cm": geometry_selected["perimeter_cm"],
        "literature_circumference_cm": literature["perimeter_cm"],
        "delta_c_cm_literature_minus_geometry": delta_c_cm,
    }

    failures = []
    if not literature["independent_of_circumference_minimum"]:
        failures.append("measurement does not declare independence from circumference minimum")
    if literature["search_or_scan"] != "none; one direct landmark-defined slice":
        failures.append("measurement performed a scan instead of one direct slice")
    if not np.isfinite(literature["perimeter_m"]) or literature["perimeter_m"] <= 0.0:
        failures.append("literature circumference is non-finite or non-positive")
    if literature["fallback_used"]:
        failures.append("torso selector used bounded fallback")
    if not literature["centerline_inside"]:
        failures.append("selected contour does not contain the spine centerline")
    if literature["diagnostics"]["invalid_component_count"] != 0:
        failures.append("slice contains an invalid contour component")
    if literature["diagnostics"]["remaining_duplicate_segments"] != 0:
        failures.append("slice retains duplicate segments")

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"{sample}_literature_vs_geometry_waist.png"
    plot_comparison(sample, vertices, literature, geometry, plot_path)
    return {
        "sample": sample,
        "source_npz": str(path),
        **literature,
        "comparison_to_geometry_waist_v0": comparison,
        "debug_plot": str(plot_path),
        "validation": {"passed": not failures, "failures": failures},
    }


def main() -> int:
    args = parse_args()
    geometry_by_sample = load_geometry_results(args.geometry_report.resolve())
    inputs_by_sample = {infer_sample_name(path): path for path in args.inputs}
    missing = sorted(set(inputs_by_sample) - set(geometry_by_sample))
    if missing:
        raise KeyError(f"geometry report is missing samples: {', '.join(missing)}")
    results = [
        validate_file(path, geometry_by_sample[sample], args.output_dir)
        for sample, path in sorted(inputs_by_sample.items())
    ]
    passed = all(item["validation"]["passed"] for item in results)
    report = {
        "experiment": "Step 2.5C independent literature waist reference",
        "definition": "literature_waist_v1",
        "status": "baseline",
        "scope": (
            "single horizontal raw canonical slice at the mean Y of published SMPL-X "
            "belly-button landmarks; no circumference search, ISO/WHO claim, smoothing, "
            "fitting, or metric calibration"
        ),
        "geometry_report": str(args.geometry_report.resolve()),
        "samples": results,
        "passed": passed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "literature_waist_validation.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("| Sample | Geometry y | Literature y | Delta y (cm) | Geometry C | Literature C | Delta C (cm) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for result in results:
        comparison = result["comparison_to_geometry_waist_v0"]
        print(
            f"| {result['sample']} | {comparison['geometry_y_norm']:.4f} | "
            f"{comparison['literature_y_norm']:.4f} | "
            f"{comparison['delta_y_cm_literature_minus_geometry']:+.3f} | "
            f"{comparison['geometry_circumference_cm']:.3f} | "
            f"{comparison['literature_circumference_cm']:.3f} | "
            f"{comparison['delta_c_cm_literature_minus_geometry']:+.3f} |"
        )
        for failure in result["validation"]["failures"]:
            print(f"  - failure: {failure}")
    print(f"\nValidation JSON: {output_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
