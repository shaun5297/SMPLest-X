#!/usr/bin/env python3
"""Validate Step 2.6D ``literature_hip_v1`` against geometry_hip_v0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from literature_hip import measure_literature_hip
from utils import infer_sample_name, load_canonical_mesh, verify_smplx_axes


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--geometry-report",
        type=Path,
        default=(
            repo_root
            / "anthropometry"
            / "artifacts"
            / "hip_validation"
            / "geometry_hip_validation.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            repo_root
            / "anthropometry"
            / "artifacts"
            / "literature_hip_validation"
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
    if report.get("definition") != "geometry_hip_v0":
        raise ValueError(f"{path} is not a geometry_hip_v0 report")
    return {sample["sample"]: sample for sample in report["samples"]}


def plot_comparison(
    sample: str,
    vertices: np.ndarray,
    literature: dict[str, object],
    geometry: dict[str, object],
    output_path: Path,
) -> None:
    plane = literature["plane_definition"]
    landmark = np.asarray(plane["landmark"]["coordinate_m"], dtype=np.float64)
    lit_y = float(plane["plane_y_m"])
    selected = geometry["selected"]
    geom_y = float(selected["plane_y_m"])
    profile = geometry["profile"]

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.4))
    sagittal_axis, profile_axis, contour_axis = axes

    sagittal_axis.scatter(
        vertices[:, 2],
        vertices[:, 1],
        s=0.35,
        color="0.78",
        alpha=0.18,
        rasterized=True,
        label="canonical mesh vertices",
    )
    sagittal_axis.axhline(
        lit_y,
        color="#2ca02c",
        linewidth=2.2,
        label=f"literature y_norm={plane['normalized_height']:.3f}",
    )
    sagittal_axis.axhline(
        geom_y,
        color="#d62728",
        linewidth=2.0,
        linestyle="--",
        label=f"geometry max y_norm={selected['normalized_height']:.3f}",
    )
    sagittal_axis.scatter(
        [landmark[2]],
        [landmark[1]],
        s=85,
        color="#1f77b4",
        edgecolor="white",
        linewidth=0.8,
        label="published PUBIC_BONE v5949",
    )
    sagittal_axis.set_xlabel("Z (m, sagittal)")
    sagittal_axis.set_ylabel("Y (m, vertical)")
    sagittal_axis.set_title("Independent literature landmark plane")
    sagittal_axis.grid(alpha=0.2)
    sagittal_axis.legend(frameon=False, fontsize=8)

    profile_axis.plot(
        [item["normalized_height"] for item in profile],
        [item["perimeter_cm"] for item in profile],
        color="#1f77b4",
        marker="o",
        markersize=3.0,
        linewidth=1.5,
        label="geometry stable pelvic C(y)",
    )
    profile_axis.scatter(
        [selected["normalized_height"]],
        [selected["perimeter_cm"]],
        marker="*",
        s=170,
        color="#d62728",
        zorder=5,
        label=f"geometry max: {selected['perimeter_cm']:.2f} cm",
    )
    profile_axis.scatter(
        [plane["normalized_height"]],
        [literature["perimeter_cm"]],
        marker="D",
        s=62,
        color="#2ca02c",
        zorder=5,
        label=f"literature slice: {literature['perimeter_cm']:.2f} cm",
    )
    profile_axis.axvline(
        plane["normalized_height"], color="#2ca02c", linewidth=1.3, alpha=0.7
    )
    profile_axis.set_xlabel("Normalized height")
    profile_axis.set_ylabel("Raw circumference (cm)")
    profile_axis.set_title("Independent plane on geometry profile")
    profile_axis.grid(alpha=0.2)
    profile_axis.legend(frameon=False, fontsize=8)

    lit_points = np.asarray(literature["ordered_points_m"], dtype=np.float64)
    geom_points = np.asarray(selected["ordered_points_m"], dtype=np.float64)
    lit_closed = np.vstack([lit_points, lit_points[0]])
    geom_closed = np.vstack([geom_points, geom_points[0]])
    contour_axis.plot(
        lit_closed[:, 0],
        lit_closed[:, 2],
        color="#2ca02c",
        linewidth=2.4,
        label=f"literature: {literature['perimeter_cm']:.2f} cm",
    )
    contour_axis.plot(
        geom_closed[:, 0],
        geom_closed[:, 2],
        color="#d62728",
        linewidth=2.0,
        linestyle="--",
        label=f"geometry max: {selected['perimeter_cm']:.2f} cm",
    )
    centerline = np.asarray(literature["centerline_xz_m"], dtype=np.float64)
    contour_axis.scatter(
        [centerline[0]],
        [centerline[1]],
        marker="x",
        s=80,
        linewidths=2,
        color="black",
        label="pelvic centerline",
    )
    contour_axis.set_xlabel("X (m)")
    contour_axis.set_ylabel("Z (m)")
    contour_axis.set_aspect("equal", adjustable="box")
    contour_axis.set_title("X-Z contours")
    contour_axis.grid(alpha=0.2)
    contour_axis.legend(frameon=False, fontsize=8)

    fig.suptitle(f"{sample}: literature_hip_v1 vs geometry_hip_v0")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
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
    literature = measure_literature_hip(vertices, faces, joints)
    selected = geometry["selected"]
    plane = literature["plane_definition"]
    delta_y_m = float(plane["plane_y_m"] - selected["plane_y_m"])
    delta_c_cm = float(literature["perimeter_cm"] - selected["perimeter_cm"])
    stable_lower_y = float(
        geometry["search_region"]["stable_lower_bound"]["plane_y_m"]
    )
    upper_y = float(geometry["search_region"]["upper_bound"]["plane_y_m"])
    comparison = {
        "geometry_y_norm": float(selected["normalized_height"]),
        "literature_y_norm": float(plane["normalized_height"]),
        "delta_y_m_literature_minus_geometry": delta_y_m,
        "delta_y_mm_literature_minus_geometry": delta_y_m * 1000.0,
        "geometry_circumference_cm": float(selected["perimeter_cm"]),
        "literature_circumference_cm": float(literature["perimeter_cm"]),
        "delta_c_cm_literature_minus_geometry": delta_c_cm,
        "literature_plane_inside_geometry_stable_interval": (
            stable_lower_y <= float(plane["plane_y_m"]) <= upper_y
        ),
    }

    failures: list[str] = []
    if not literature["independent_of_circumference_maximum"]:
        failures.append("measurement does not declare independence from profile maximum")
    if literature["search_or_scan"] != "none; one direct landmark-defined slice":
        failures.append("measurement performed a scan instead of one direct slice")
    if literature["topology"]["state"] != "joined_pelvis":
        failures.append("literature plane does not produce joined-pelvis topology")
    if not np.isfinite(literature["perimeter_m"]) or literature["perimeter_m"] <= 0.0:
        failures.append("literature circumference is non-finite or non-positive")
    if literature["fallback_used"]:
        failures.append("pelvic contour selector used bounded fallback")
    if not literature["centerline_inside"]:
        failures.append("selected contour does not contain the pelvic centerline")
    if literature["diagnostics"]["invalid_component_count"] != 0:
        failures.append("slice contains an invalid contour component")
    if literature["diagnostics"]["remaining_duplicate_segments"] != 0:
        failures.append("slice retains duplicate segments")

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"{sample}_literature_vs_geometry_hip.png"
    plot_comparison(sample, vertices, literature, geometry, plot_path)
    return {
        "sample": sample,
        "source_npz": str(path),
        **literature,
        "comparison_to_geometry_hip_v0": comparison,
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
        "experiment": "Step 2.6D independent literature hip reference",
        "definition": "literature_hip_v1",
        "status": "baseline",
        "scope": (
            "single horizontal raw canonical slice at published SMPL-X PUBIC_BONE "
            "v5949; no circumference search, anatomical/ISO claim, smoothing, "
            "curve fitting, or metric calibration"
        ),
        "geometry_report": str(args.geometry_report.resolve()),
        "samples": results,
        "passed": passed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "literature_hip_validation.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(
        "| Sample | Geometry y | Literature y | Delta y (mm) | Geometry C | "
        "Literature C | Delta C (cm) | Stable interval? |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|:---:|")
    for result in results:
        comparison = result["comparison_to_geometry_hip_v0"]
        print(
            f"| {result['sample']} | {comparison['geometry_y_norm']:.4f} | "
            f"{comparison['literature_y_norm']:.4f} | "
            f"{comparison['delta_y_mm_literature_minus_geometry']:+.2f} | "
            f"{comparison['geometry_circumference_cm']:.3f} | "
            f"{comparison['literature_circumference_cm']:.3f} | "
            f"{comparison['delta_c_cm_literature_minus_geometry']:+.3f} | "
            f"{'yes' if comparison['literature_plane_inside_geometry_stable_interval'] else 'no'} |"
        )
        for failure in result["validation"]["failures"]:
            print(f"  - failure: {failure}")
    print(f"\nValidation JSON: {output_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
