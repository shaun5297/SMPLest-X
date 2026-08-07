#!/usr/bin/env python3
"""Run and validate the raw ``geometry_waist_v0`` baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils import infer_sample_name, load_canonical_mesh, verify_smplx_axes
from waist import (
    DEFAULT_LOCAL_JUMP_WARNING_CM,
    DEFAULT_STEP_NORMALIZED_HEIGHT,
    scan_geometry_waist,
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Validate the raw discrete minimum torso circumference baseline. "
            "No ISO definition, smoothing, fitting, or metric calibration is applied."
        )
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Canonical SMPL-X NPZ files")
    parser.add_argument(
        "--step-normalized-height",
        type=float,
        default=DEFAULT_STEP_NORMALIZED_HEIGHT,
    )
    parser.add_argument(
        "--local-jump-warning-cm",
        type=float,
        default=DEFAULT_LOCAL_JUMP_WARNING_CM,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "anthropometry" / "artifacts" / "waist_validation",
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


def plot_result(sample: str, result: dict[str, object], output_path: Path) -> None:
    profile = result["profile"]
    selected = result["selected"]
    interval = result["search_interval"]
    normalized_heights = np.asarray(
        [item["normalized_height"] for item in profile], dtype=np.float64
    )
    circumferences = np.asarray([item["perimeter_cm"] for item in profile], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    profile_axis, contour_axis = axes
    profile_axis.plot(
        normalized_heights,
        circumferences,
        color="#1f77b4",
        marker="o",
        markersize=3.2,
        linewidth=1.5,
        label="raw discrete torso circumference",
    )
    profile_axis.axvline(
        interval["normalized_y_min"], color="0.35", linestyle="--", label="pelvis boundary"
    )
    profile_axis.axvline(
        interval["normalized_y_max"], color="0.55", linestyle=":", label="spine2 boundary"
    )
    profile_axis.scatter(
        [selected["normalized_height"]],
        [selected["perimeter_cm"]],
        marker="*",
        s=180,
        color="#d62728",
        zorder=5,
        label=(
            f"minimum: y_norm={selected['normalized_height']:.3f}, "
            f"C={selected['perimeter_cm']:.2f} cm"
        ),
    )
    profile_axis.set_xlabel("Normalized height")
    profile_axis.set_ylabel("Raw circumference (cm)")
    profile_axis.set_title("Raw circumference profile")
    profile_axis.grid(alpha=0.25)
    profile_axis.legend(frameon=False, fontsize=9)

    points = np.asarray(selected["ordered_points_m"], dtype=np.float64)
    closed = np.vstack([points, points[0]])
    contour_axis.plot(closed[:, 0], closed[:, 2], color="#d62728", linewidth=2.5)
    centerline = np.asarray(selected["centerline_xz_m"], dtype=np.float64)
    centroid = np.asarray(selected["centroid_xz_m"], dtype=np.float64)
    contour_axis.scatter(
        [centerline[0]], [centerline[1]], marker="x", s=85, linewidths=2, color="black",
        label="spine centerline",
    )
    contour_axis.scatter(
        [centroid[0]], [centroid[1]], marker="+", s=85, linewidths=2, color="#1f77b4",
        label="contour centroid",
    )
    contour_axis.set_xlabel("X (m)")
    contour_axis.set_ylabel("Z (m)")
    contour_axis.set_title(
        f"Selected contour at y_norm={selected['normalized_height']:.3f}\n"
        f"C={selected['perimeter_cm']:.2f} cm, area={selected['area_cm2']:.1f} cm²"
    )
    contour_axis.set_aspect("equal", adjustable="box")
    contour_axis.grid(alpha=0.25)
    contour_axis.legend(frameon=False, fontsize=9)
    fig.suptitle(f"{sample}: geometry_waist_v0 (raw canonical geometry)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def validate_file(path: Path, args: argparse.Namespace) -> dict[str, object]:
    path = path.resolve()
    vertices, joints, faces = load_mesh(path)
    sample = infer_sample_name(path)
    result = scan_geometry_waist(
        vertices,
        faces,
        joints,
        step_normalized_height=args.step_normalized_height,
        local_jump_warning_cm=args.local_jump_warning_cm,
    )
    failures = []
    profile = result["profile"]
    if len(profile) < 5:
        failures.append("dense profile contains fewer than five slices")
    perimeters = np.asarray([item["perimeter_m"] for item in profile], dtype=np.float64)
    if not np.isfinite(perimeters).all() or np.any(perimeters <= 0.0):
        failures.append("profile contains a non-finite or non-positive circumference")
    if any(item["invalid_component_count"] != 0 for item in profile):
        failures.append("a profile slice contains an invalid contour component")
    if any(item["remaining_duplicate_segments"] != 0 for item in profile):
        failures.append("a profile slice retains duplicate segments")
    if any(not item["centerline_inside"] and not item["fallback_used"] for item in profile):
        failures.append("a non-fallback selection does not contain the spine centerline")
    if int(np.argmin(perimeters)) != result["selected_index"]:
        failures.append("selected index is not the raw discrete argmin")
    if result["selected"]["perimeter_m"] != float(perimeters.min()):
        failures.append("selected circumference differs from the profile minimum")
    neighborhood = result["local_stability"]["neighborhood"]
    if not neighborhood:
        failures.append("minimum local-stability neighborhood is empty")
    if any(not np.isfinite(item["perimeter_cm"]) for item in neighborhood):
        failures.append("minimum neighborhood contains NaN or Inf")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = args.output_dir / f"{sample}_geometry_waist_v0.png"
    plot_result(sample, result, plot_path)
    return {
        "sample": sample,
        "source_npz": str(path),
        **result,
        "debug_plot": str(plot_path),
        "validation": {
            "passed": not failures,
            "failures": failures,
            "warnings": result["warnings"],
        },
    }


def main() -> int:
    args = parse_args()
    results = [validate_file(path, args) for path in args.inputs]
    passed = all(item["validation"]["passed"] for item in results)
    report = {
        "experiment": "Step 2.5B geometry-defined waist baseline",
        "definition": "geometry_waist_v0",
        "status": "baseline",
        "scope": (
            "raw canonical circumference only; no ISO/anatomical waist claim, smoothing, "
            "curve fitting, or metric calibration"
        ),
        "samples": results,
        "passed": passed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "geometry_waist_validation.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("| Sample | Search range y_norm | Slices | Waist y_norm | Raw waist (cm) | Boundary? | Fallbacks |")
    print("|---|---:|---:|---:|---:|:---:|---:|")
    for result in results:
        interval = result["search_interval"]
        selected = result["selected"]
        print(
            f"| {result['sample']} | {interval['normalized_y_min']:.3f}–"
            f"{interval['normalized_y_max']:.3f} | {len(result['profile'])} | "
            f"{selected['normalized_height']:.3f} | {selected['perimeter_cm']:.2f} | "
            f"{'yes' if result['boundary_minimum'] else 'no'} | {result['fallback_count']} |"
        )
        for warning in result["validation"]["warnings"]:
            print(f"  - warning: {warning}")
        for failure in result["validation"]["failures"]:
            print(f"  - failure: {failure}")
    print(f"\nValidation JSON: {output_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
