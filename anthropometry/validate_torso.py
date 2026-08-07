#!/usr/bin/env python3
"""Validate Step 2.5A skeleton-constrained torso contour selection."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from slicing import ENDPOINT_CLUSTER_TOLERANCE_M, EPS, slice_mesh
from torso import (
    DEFAULT_CENTERLINE_PROXIMITY_M,
    TorsoSelectionError,
    compute_torso_vertical_interval,
    interpolate_spine_centerline_xz,
    select_torso_contour,
)
from utils import infer_sample_name, load_canonical_mesh, verify_smplx_axes


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Validate torso contour selection between native SMPL-X pelvis and spine2. "
            "This does not locate or measure a waist."
        )
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Canonical SMPL-X NPZ files")
    parser.add_argument("--num-heights", type=int, default=9)
    parser.add_argument("--eps", type=float, default=EPS)
    parser.add_argument(
        "--endpoint-tolerance",
        type=float,
        default=ENDPOINT_CLUSTER_TOLERANCE_M,
    )
    parser.add_argument(
        "--centerline-proximity",
        type=float,
        default=DEFAULT_CENTERLINE_PROXIMITY_M,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "anthropometry" / "artifacts" / "torso_validation",
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


def summarize_contour(contour: dict[str, object]) -> dict[str, object]:
    return {
        "id": contour["id"],
        "ordered_points_m": contour["ordered_points_m"],
        "num_points": contour["num_points"],
        "perimeter_m": contour["perimeter_m"],
        "perimeter_cm": contour["perimeter_cm"],
        "area_m2": contour["area_m2"],
        "area_cm2": contour["area_cm2"],
        "centroid_xz_m": contour["centroid_xz_m"],
    }


def plot_validation(
    vertices: np.ndarray,
    sample: str,
    records: list[dict[str, object]],
    output_path: Path,
) -> None:
    columns = 3
    rows = math.ceil(len(records) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(12.0, 3.8 * rows), squeeze=False)
    body_height = float(np.ptp(vertices[:, 1]))
    for axis, record in zip(axes.flat, records):
        plane_y = record["plane_y_m"]
        near_plane = np.abs(vertices[:, 1] - plane_y) <= max(0.01, body_height * 0.01)
        background = vertices[near_plane]
        axis.scatter(background[:, 0], background[:, 2], s=2, color="0.82", alpha=0.35)
        for contour in record["contours"]:
            points = np.asarray(contour["ordered_points_m"], dtype=np.float64)
            closed = np.vstack([points, points[0]])
            selected = contour["id"] == record["selected_contour_id"]
            axis.plot(
                closed[:, 0],
                closed[:, 2],
                color="#d62728" if selected else "0.45",
                linewidth=2.4 if selected else 1.2,
                linestyle="-" if selected else "--",
            )
        centerline = np.asarray(record["centerline_xz_m"], dtype=np.float64)
        axis.scatter(
            [centerline[0]],
            [centerline[1]],
            marker="x",
            s=70,
            linewidths=2,
            color="black",
            zorder=5,
        )
        selected = record["selected"]
        axis.set_title(
            f"y_norm={record['normalized_height']:.3f}, loops={record['num_contours']}\n"
            f"selected={record['selected_contour_id']}, "
            f"center dist={selected['centroid_distance_cm']:.2f} cm"
        )
        axis.set_xlabel("X (m)")
        axis.set_ylabel("Z (m)")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(records):]:
        axis.set_visible(False)
    fig.suptitle(
        f"{sample}: Step 2.5A torso contour selection\n"
        "red=selected contour, black x=interpolated spine centerline"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def validate_file(path: Path, args: argparse.Namespace) -> dict[str, object]:
    path = path.resolve()
    vertices, joints, faces = load_mesh(path)
    sample = infer_sample_name(path)
    interval = compute_torso_vertical_interval(joints)
    min_mesh_y = float(vertices[:, 1].min())
    body_height = float(np.ptp(vertices[:, 1]))
    interval["normalized_y_min"] = float((interval["y_min_m"] - min_mesh_y) / body_height)
    interval["normalized_y_max"] = float((interval["y_max_m"] - min_mesh_y) / body_height)

    records = []
    failures = []
    for plane_y in np.linspace(interval["y_min_m"], interval["y_max_m"], args.num_heights):
        normalized_height = float((plane_y - min_mesh_y) / body_height)
        sliced = slice_mesh(
            vertices,
            faces,
            float(plane_y),
            eps=args.eps,
            endpoint_tolerance=args.endpoint_tolerance,
        )
        connectivity = sliced["diagnostics"]["connectivity"]
        slice_failures = []
        if connectivity["invalid_component_count"] != 0:
            slice_failures.append("slicing returned an invalid contour component")
        if not sliced["contours"]:
            slice_failures.append("slicing returned no closed contours")
        centerline = interpolate_spine_centerline_xz(joints, float(plane_y))
        try:
            selection = select_torso_contour(
                sliced["contours"],
                centerline,
                max_centerline_proximity_m=args.centerline_proximity,
            )
        except TorsoSelectionError as error:
            slice_failures.append(str(error))
            failures.extend(
                f"y_norm={normalized_height:.4f}: {failure}" for failure in slice_failures
            )
            records.append(
                {
                    "normalized_height": normalized_height,
                    "plane_y_m": float(plane_y),
                    "num_contours": len(sliced["contours"]),
                    "contours": [summarize_contour(item) for item in sliced["contours"]],
                    "centerline_xz_m": centerline.tolist(),
                    "selection_error": str(error),
                    "failures": slice_failures,
                }
            )
            continue

        selected_metrics = selection["selected_metrics"]
        if selection["selection_method"] != "spine_centerline_containment_then_area":
            slice_failures.append("selection required the centerline-proximity fallback")
        if not selected_metrics["centerline_inside"]:
            slice_failures.append("selected contour does not contain the spine centerline")
        largest_area_id = max(sliced["contours"], key=lambda item: item["area_m2"])["id"]
        records.append(
            {
                "normalized_height": normalized_height,
                "plane_y_m": float(plane_y),
                "num_contours": len(sliced["contours"]),
                "contours": [summarize_contour(item) for item in sliced["contours"]],
                "centerline_xz_m": centerline.tolist(),
                "selected_contour_id": selection["selected_contour_id"],
                "largest_area_contour_id": largest_area_id,
                "selection_method": selection["selection_method"],
                "selected": {
                    **selected_metrics,
                    "centroid_distance_cm": selected_metrics["centroid_distance_m"] * 100.0,
                    "centerline_boundary_distance_cm": (
                        selected_metrics["centerline_boundary_distance_m"] * 100.0
                    ),
                },
                "candidate_metrics": selection["candidate_metrics"],
                "failures": slice_failures,
            }
        )
        failures.extend(
            f"y_norm={normalized_height:.4f}: {failure}" for failure in slice_failures
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = args.output_dir / f"{sample}_torso_selection.png"
    if all("selected_contour_id" in record for record in records):
        plot_validation(vertices, sample, records, plot_path)
    return {
        "sample": sample,
        "source_npz": str(path),
        "definition": "torso_contour_selector_v0",
        "status": "validation_passed" if not failures else "validation_failed",
        "search_interval": interval,
        "num_validation_heights": args.num_heights,
        "slices": records,
        "debug_plot": str(plot_path),
        "passed": not failures,
        "failures": failures,
    }


def main() -> int:
    args = parse_args()
    if args.num_heights < 3:
        raise ValueError("num-heights must be at least three")
    results = [validate_file(path, args) for path in args.inputs]
    report = {
        "experiment": "Step 2.5A torso contour selector validation",
        "scope": (
            "skeleton-constrained torso contour identification only; no waist search, "
            "minimum-circumference selection, calibration, or anatomical waist claim"
        ),
        "parameters": {
            "num_heights_per_mesh": args.num_heights,
            "eps": args.eps,
            "endpoint_tolerance_m": args.endpoint_tolerance,
            "centerline_proximity_m": args.centerline_proximity,
        },
        "samples": results,
        "passed": all(result["passed"] for result in results),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "torso_selection_validation.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("| Sample | Interval y_norm | Heights | Containment selections | Max center distance (cm) | Status |")
    print("|---|---|---:|---:|---:|---|")
    for result in results:
        interval = result["search_interval"]
        selected_records = [item for item in result["slices"] if "selected" in item]
        containment_count = sum(
            item["selection_method"] == "spine_centerline_containment_then_area"
            for item in selected_records
        )
        maximum_distance = max(
            (item["selected"]["centroid_distance_cm"] for item in selected_records),
            default=float("nan"),
        )
        print(
            f"| {result['sample']} | {interval['normalized_y_min']:.3f}–"
            f"{interval['normalized_y_max']:.3f} | {len(result['slices'])} | "
            f"{containment_count} | {maximum_distance:.3f} | "
            f"{'PASS' if result['passed'] else 'FAIL'} |"
        )
        for failure in result["failures"]:
            print(f"  - {failure}")
    print(f"\nValidation JSON: {output_path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
