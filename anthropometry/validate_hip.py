#!/usr/bin/env python3
"""Validate Step 2.6A pelvic-region detection and geometry_hip_v0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hip import (
    DEFAULT_LOCAL_JUMP_WARNING_CM,
    DEFAULT_STEP_NORMALIZED_HEIGHT,
    scan_geometry_hip,
)
from utils import infer_sample_name, load_canonical_mesh, verify_smplx_axes


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
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
        default=repo_root / "anthropometry" / "artifacts" / "hip_validation",
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
    region = result["search_region"]
    transition = region["topology_transition"]

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.2))
    topology_axis, profile_axis, contour_axis = axes

    state_value = {
        "bilateral_leg_split": 0,
        "transitional": 1,
        "empty": 1,
        "invalid": 1,
        "joined_pelvis": 2,
    }
    probe = region["topology_probe"]
    topology_axis.scatter(
        [item["normalized_height"] for item in probe],
        [state_value[item["topology_state"]] for item in probe],
        c=[
            {
                "bilateral_leg_split": "#1f77b4",
                "joined_pelvis": "#2ca02c",
            }.get(item["topology_state"], "#ff7f0e")
            for item in probe
        ],
        s=25,
    )
    topology_axis.axvspan(
        transition["highest_stable_split_normalized_height"],
        transition["lowest_joined_normalized_height"],
        color="#ffcc80",
        alpha=0.45,
        label="topology transition bracket",
    )
    topology_axis.set_yticks([0, 1, 2], ["bilateral split", "transitional", "joined pelvis"])
    topology_axis.set_xlabel("Normalized height")
    topology_axis.set_title("2.6A topology probe")
    topology_axis.grid(axis="x", alpha=0.25)
    topology_axis.legend(frameon=False, fontsize=8)

    normalized_heights = [item["normalized_height"] for item in profile]
    circumferences = [item["perimeter_cm"] for item in profile]
    profile_axis.plot(
        normalized_heights,
        circumferences,
        color="#1f77b4",
        marker="o",
        markersize=3.0,
        linewidth=1.5,
        label="stable pelvic perimeter",
    )
    profile_axis.axvline(
        region["stable_lower_bound"]["normalized_height"],
        color="#9467bd",
        linestyle="--",
        label="stable pelvic lower bound",
    )
    profile_axis.axvline(
        region["upper_bound"]["normalized_height"],
        color="0.45",
        linestyle=":",
        label="spine1 upper bound",
    )
    profile_axis.scatter(
        [selected["normalized_height"]],
        [selected["perimeter_cm"]],
        marker="*",
        s=180,
        color="#d62728",
        zorder=5,
        label=(
            f"perimeter argmax: y={selected['normalized_height']:.3f}, "
            f"C={selected['perimeter_cm']:.2f} cm"
        ),
    )
    profile_axis.set_xlabel("Normalized height")
    profile_axis.set_ylabel("Raw circumference (cm)")
    profile_axis.set_title("2.6B stable pelvic C(y)")
    profile_axis.grid(alpha=0.25)
    profile_axis.legend(frameon=False, fontsize=8)

    points = np.asarray(selected["ordered_points_m"], dtype=np.float64)
    closed = np.vstack([points, points[0]])
    contour_axis.plot(closed[:, 0], closed[:, 2], color="#d62728", linewidth=2.5)
    centerline = np.asarray(selected["centerline_xz_m"], dtype=np.float64)
    centroid = np.asarray(selected["centroid_xz_m"], dtype=np.float64)
    contour_axis.scatter(
        [centerline[0]], [centerline[1]], marker="x", s=80,
        linewidths=2, color="black", label="pelvic centerline"
    )
    contour_axis.scatter(
        [centroid[0]], [centroid[1]], marker="+", s=80,
        linewidths=2, color="#1f77b4", label="contour centroid"
    )
    contour_axis.set_xlabel("X (m)")
    contour_axis.set_ylabel("Z (m)")
    contour_axis.set_aspect("equal", adjustable="box")
    contour_axis.set_title(
        f"Selected contour\ny={selected['normalized_height']:.3f}, "
        f"C={selected['perimeter_cm']:.2f} cm, "
        f"compactness={selected['compactness']:.3f}"
    )
    contour_axis.grid(alpha=0.25)
    contour_axis.legend(frameon=False, fontsize=8)

    fig.suptitle(f"{sample}: geometry_hip_v0 (raw canonical geometry)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def validate_file(path: Path, args: argparse.Namespace) -> dict[str, object]:
    path = path.resolve()
    vertices, joints, faces = load_mesh(path)
    sample = infer_sample_name(path)
    result = scan_geometry_hip(
        vertices,
        faces,
        joints,
        step_normalized_height=args.step_normalized_height,
        local_jump_warning_cm=args.local_jump_warning_cm,
    )
    failures: list[str] = []
    profile = result["profile"]
    perimeters = np.asarray([item["perimeter_m"] for item in profile], dtype=np.float64)
    areas = np.asarray([item["area_m2"] for item in profile], dtype=np.float64)
    if len(profile) < 5:
        failures.append("stable pelvic profile contains fewer than five slices")
    if not np.isfinite(perimeters).all() or np.any(perimeters <= 0.0):
        failures.append("profile contains a non-finite or non-positive perimeter")
    if any(item["topology_state"] != "joined_pelvis" for item in profile):
        failures.append("stable pelvic profile contains a non-joined topology")
    if any(item["invalid_component_count"] for item in profile):
        failures.append("stable pelvic profile contains an invalid component")
    if any(item["remaining_duplicate_segments"] for item in profile):
        failures.append("stable pelvic profile retains duplicate segments")
    if any(item["fallback_used"] for item in profile):
        failures.append("pelvic contour selection used fallback")
    if any(not item["centerline_inside"] for item in profile):
        failures.append("pelvic centerline lies outside a selected contour")
    if int(np.argmax(perimeters)) != result["selected_index"]:
        failures.append("selected index is not the raw perimeter argmax")
    if result["selected"]["perimeter_m"] != float(perimeters.max()):
        failures.append("selected circumference differs from the profile maximum")
    transition = result["search_region"]["topology_transition"]
    if not (
        transition["highest_stable_split_plane_y_m"]
        < transition["lowest_joined_plane_y_m"]
        < result["search_region"]["stable_lower_bound"]["plane_y_m"]
    ):
        failures.append("topology transition and stable lower boundary are misordered")
    if result["fallback_count"] != 0:
        failures.append("fallback_count is non-zero")

    area_argmax_index = int(np.argmax(areas))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = args.output_dir / f"{sample}_geometry_hip_v0.png"
    plot_result(sample, result, plot_path)
    return {
        "sample": sample,
        "source_npz": str(path),
        **result,
        "area_sanity_check": {
            "area_argmax_index": area_argmax_index,
            "area_argmax_normalized_height": profile[area_argmax_index][
                "normalized_height"
            ],
            "same_as_perimeter_argmax": area_argmax_index == result["selected_index"],
            "selection_rule": "diagnostic only; area does not select geometry_hip_v0",
        },
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
        "experiment": "Step 2.6A pelvic region and 2.6B geometry hip baseline",
        "definition": "geometry_hip_v0",
        "status": "baseline",
        "scope": (
            "raw canonical maximum pelvic-contour perimeter; topology/stability "
            "constrained; no anatomical/ISO claim, smoothing, fitting, or calibration"
        ),
        "samples": results,
        "passed": passed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "geometry_hip_validation.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(
        "| Sample | Transition y_norm | Stable range y_norm | Hip y_norm | "
        "Raw hip (cm) | Boundary? | Fallbacks |"
    )
    print("|---|---:|---:|---:|---:|:---:|---:|")
    for result in results:
        transition = result["search_region"]["topology_transition"]
        lower = result["search_region"]["stable_lower_bound"]
        upper = result["search_region"]["upper_bound"]
        selected = result["selected"]
        print(
            f"| {result['sample']} | "
            f"{transition['highest_stable_split_normalized_height']:.3f}–"
            f"{transition['lowest_joined_normalized_height']:.3f} | "
            f"{lower['normalized_height']:.3f}–{upper['normalized_height']:.3f} | "
            f"{selected['normalized_height']:.3f} | "
            f"{selected['perimeter_cm']:.2f} | "
            f"{'yes' if result['boundary_maximum'] else 'no'} | "
            f"{result['fallback_count']} |"
        )
        for warning in result["validation"]["warnings"]:
            print(f"  - warning: {warning}")
        for failure in result["validation"]["failures"]:
            print(f"  - failure: {failure}")
    print(f"\nValidation JSON: {output_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
