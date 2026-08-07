#!/usr/bin/env python3
"""Validate generic horizontal slicing on canonical SMPL-X meshes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from slicing import ENDPOINT_CLUSTER_TOLERANCE_M, EPS, slice_mesh
from utils import infer_sample_name, load_canonical_mesh, verify_smplx_axes


DEFAULT_NORMALIZED_HEIGHTS = (0.45, 0.50, 0.55, 0.60, 0.70)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Slice canonical meshes at body-part-agnostic normalized heights and "
            "validate closed-contour reconstruction."
        )
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Canonical SMPL-X NPZ files")
    parser.add_argument(
        "--normalized-heights",
        nargs="+",
        type=float,
        default=DEFAULT_NORMALIZED_HEIGHTS,
    )
    parser.add_argument("--eps", type=float, default=EPS)
    parser.add_argument(
        "--endpoint-tolerance",
        type=float,
        default=ENDPOINT_CLUSTER_TOLERANCE_M,
    )
    parser.add_argument(
        "--continuity-delta-norm",
        type=float,
        default=0.002,
        help="Local normalized-height perturbation on each side of a validation plane",
    )
    parser.add_argument(
        "--max-local-perimeter-change-cm",
        type=float,
        default=10.0,
        help="Maximum allowed matched-loop change for each local perturbation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "anthropometry" / "artifacts" / "slicing_validation",
    )
    return parser.parse_args()


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices, joints = load_canonical_mesh(path)
    verify_smplx_axes(vertices, joints)
    with np.load(path, allow_pickle=False) as data:
        if "faces" not in data:
            raise KeyError(f"{path} must contain faces")
        faces = np.asarray(data["faces"], dtype=np.int32)
    return vertices, faces


def contour_summary(contour: dict[str, object]) -> dict[str, object]:
    return {
        "id": contour["id"],
        "ordered_points_m": contour["ordered_points_m"],
        "num_points": contour["num_points"],
        "perimeter_m": contour["perimeter_m"],
        "perimeter_cm": contour["perimeter_cm"],
        "area_m2": contour["area_m2"],
        "area_cm2": contour["area_cm2"],
        "centroid_xz_m": contour["centroid_xz_m"],
        "plane_y_m": contour["plane_y_m"],
        "y_span_m": contour["y_span_m"],
        "closure_error_m": contour["closure_error_m"],
        "all_node_degrees_two": contour["all_node_degrees_two"],
    }


def validate_slice(result: dict[str, object], endpoint_tolerance: float) -> list[str]:
    failures = []
    contours = result["contours"]
    connectivity = result["diagnostics"]["connectivity"]
    intersection = result["diagnostics"]["intersection"]
    if not contours:
        failures.append("no closed contours")
    if intersection["ambiguous_face_count"] != 0:
        failures.append("ambiguous triangle-plane intersection")
    if connectivity["remaining_duplicate_segments"] != 0:
        failures.append("duplicate segments remain after deduplication")
    if connectivity["invalid_component_count"] != 0:
        failures.append("non-closed or non-degree-2 segment component")
    if connectivity["zero_length_segments_removed"] != 0:
        failures.append("zero-length raw segments were generated")
    if connectivity["max_endpoint_cluster_error_m"] > endpoint_tolerance:
        failures.append("endpoint clustering exceeded tolerance")
    for contour in contours:
        numeric = np.asarray(
            [
                contour["perimeter_m"],
                contour["area_m2"],
                *contour["centroid_xz_m"],
                contour["y_span_m"],
                contour["closure_error_m"],
            ],
            dtype=np.float64,
        )
        if not np.isfinite(numeric).all():
            failures.append(f"contour {contour['id']} contains NaN or Inf")
        if contour["perimeter_m"] <= 0.0:
            failures.append(f"contour {contour['id']} has non-positive perimeter")
        if contour["area_m2"] <= 0.0:
            failures.append(f"contour {contour['id']} has non-positive area")
        if contour["num_points"] < 3:
            failures.append(f"contour {contour['id']} has fewer than three points")
        if contour["y_span_m"] > endpoint_tolerance:
            failures.append(f"contour {contour['id']} is not horizontal")
        if contour["closure_error_m"] > endpoint_tolerance:
            failures.append(f"contour {contour['id']} exceeds closure tolerance")
        if not contour["all_node_degrees_two"]:
            failures.append(f"contour {contour['id']} contains a non-degree-2 node")
    return failures


def _match_perimeter_changes(
    reference: list[dict[str, object]],
    neighbor: list[dict[str, object]],
) -> list[dict[str, float | int]]:
    if len(reference) != len(neighbor):
        return []
    remaining = set(range(len(neighbor)))
    changes = []
    for contour in reference:
        centroid = np.asarray(contour["centroid_xz_m"], dtype=np.float64)
        neighbor_index = min(
            remaining,
            key=lambda index: float(
                np.linalg.norm(
                    centroid
                    - np.asarray(neighbor[index]["centroid_xz_m"], dtype=np.float64)
                )
            ),
        )
        remaining.remove(neighbor_index)
        matched = neighbor[neighbor_index]
        changes.append(
            {
                "reference_contour_id": int(contour["id"]),
                "neighbor_contour_id": int(matched["id"]),
                "centroid_distance_cm": float(
                    np.linalg.norm(
                        centroid - np.asarray(matched["centroid_xz_m"], dtype=np.float64)
                    )
                    * 100.0
                ),
                "absolute_perimeter_change_cm": float(
                    abs(contour["perimeter_cm"] - matched["perimeter_cm"])
                ),
            }
        )
    return changes


def evaluate_continuity(
    vertices: np.ndarray,
    faces: np.ndarray,
    plane_y: float,
    body_height: float,
    center_contours: list[dict[str, object]],
    *,
    delta_norm: float,
    eps: float,
    endpoint_tolerance: float,
    maximum_change_cm: float,
) -> dict[str, object]:
    delta_y = body_height * delta_norm
    neighbors = {}
    all_changes = []
    topology_changed = False
    neighbors_structurally_valid = True
    for direction, neighbor_y in (("below", plane_y - delta_y), ("above", plane_y + delta_y)):
        neighbor_result = slice_mesh(
            vertices,
            faces,
            neighbor_y,
            eps=eps,
            endpoint_tolerance=endpoint_tolerance,
        )
        neighbor_contours = neighbor_result["contours"]
        structural_failures = validate_slice(neighbor_result, endpoint_tolerance)
        structurally_valid = not structural_failures
        neighbors_structurally_valid = neighbors_structurally_valid and structurally_valid
        changes = _match_perimeter_changes(center_contours, neighbor_contours)
        changed = len(center_contours) != len(neighbor_contours)
        topology_changed = topology_changed or changed
        all_changes.extend(changes)
        neighbors[direction] = {
            "plane_y_m": float(neighbor_y),
            "num_contours": int(len(neighbor_contours)),
            "topology_changed": bool(changed),
            "structurally_valid": bool(structurally_valid),
            "structural_failures": structural_failures,
            "matched_changes": changes,
        }
    maximum_observed = max(
        (item["absolute_perimeter_change_cm"] for item in all_changes),
        default=0.0,
    )
    if topology_changed:
        assessment = "valid_topology_event" if neighbors_structurally_valid else "invalid_topology_event"
    elif maximum_observed <= maximum_change_cm and neighbors_structurally_valid:
        assessment = "continuous_same_topology"
    else:
        assessment = "continuity_failure"
    return {
        "delta_normalized_height": float(delta_norm),
        "delta_y_m": float(delta_y),
        "neighbors": neighbors,
        "topology_changed": bool(topology_changed),
        "assessment": assessment,
        "max_matched_perimeter_change_cm": float(maximum_observed),
        "threshold_cm": float(maximum_change_cm),
        "passed": bool(
            neighbors_structurally_valid
            and (topology_changed or maximum_observed <= maximum_change_cm)
        ),
    }


def plot_debug(
    vertices: np.ndarray,
    sample: str,
    slices: list[dict[str, object]],
    output_path: Path,
) -> None:
    columns = len(slices)
    fig, axes = plt.subplots(1, columns, figsize=(4.2 * columns, 4.4), squeeze=False)
    colors = plt.get_cmap("tab10")
    body_height = float(np.ptp(vertices[:, 1]))
    for axis, record in zip(axes[0], slices):
        plane_y = record["plane_y_m"]
        near_plane = np.abs(vertices[:, 1] - plane_y) <= max(0.01, body_height * 0.01)
        background = vertices[near_plane]
        axis.scatter(background[:, 0], background[:, 2], s=2, color="0.75", alpha=0.35)
        for contour in record["full_result"]["contours"]:
            points = np.asarray(contour["ordered_points_m"], dtype=np.float64)
            closed = np.vstack([points, points[0]])
            color = colors(contour["id"] % 10)
            axis.plot(
                closed[:, 0],
                closed[:, 2],
                color=color,
                linewidth=2,
                label=(
                    f"loop {contour['id']}: {contour['perimeter_cm']:.1f} cm, "
                    f"{contour['area_cm2']:.0f} cm²"
                ),
            )
        axis.set_title(
            f"y_norm={record['normalized_height']:.2f}\n"
            f"segments={record['num_segments']}, loops={record['num_contours']}"
        )
        axis.set_xlabel("X (m)")
        axis.set_ylabel("Z (m)")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
        axis.legend(loc="best", fontsize=8, frameon=False)
    fig.suptitle(f"{sample}: generic horizontal mesh slicing (top view)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def validate_file(path: Path, args: argparse.Namespace) -> dict[str, object]:
    path = path.resolve()
    vertices, faces = load_mesh(path)
    sample = infer_sample_name(path)
    min_y = float(vertices[:, 1].min())
    max_y = float(vertices[:, 1].max())
    body_height = max_y - min_y
    slice_records = []
    failures = []
    for normalized_height in args.normalized_heights:
        plane_y = min_y + normalized_height * body_height
        result = slice_mesh(
            vertices,
            faces,
            plane_y,
            eps=args.eps,
            endpoint_tolerance=args.endpoint_tolerance,
        )
        slice_failures = validate_slice(result, args.endpoint_tolerance)
        continuity = evaluate_continuity(
            vertices,
            faces,
            plane_y,
            body_height,
            result["contours"],
            delta_norm=args.continuity_delta_norm,
            eps=args.eps,
            endpoint_tolerance=args.endpoint_tolerance,
            maximum_change_cm=args.max_local_perimeter_change_cm,
        )
        if not continuity["passed"]:
            slice_failures.append("local perimeter continuity check failed")
        failures.extend(
            f"y_norm={normalized_height:.3f}: {failure}" for failure in slice_failures
        )
        connectivity = result["diagnostics"]["connectivity"]
        slice_records.append(
            {
                "normalized_height": float(normalized_height),
                "plane_y_m": float(plane_y),
                "num_segments": connectivity["unique_segment_count"],
                "num_contours": len(result["contours"]),
                "contours": [contour_summary(item) for item in result["contours"]],
                "intersection_diagnostics": result["diagnostics"]["intersection"],
                "connectivity_diagnostics": connectivity,
                "continuity": continuity,
                "failures": slice_failures,
                "full_result": result,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = args.output_dir / f"{sample}_slicing_debug.png"
    plot_debug(vertices, sample, slice_records, figure_path)
    for record in slice_records:
        del record["full_result"]
    return {
        "sample": sample,
        "source_npz": str(path),
        "mesh": {
            "num_vertices": int(len(vertices)),
            "num_faces": int(len(faces)),
            "min_y_m": min_y,
            "max_y_m": max_y,
            "height_m": body_height,
        },
        "slices": slice_records,
        "debug_plot": str(figure_path),
        "passed": not failures,
        "failures": failures,
    }


def main() -> int:
    args = parse_args()
    if not args.normalized_heights or any(
        not 0.0 < value < 1.0 for value in args.normalized_heights
    ):
        raise ValueError("normalized heights must lie strictly between zero and one")
    if args.continuity_delta_norm <= 0.0:
        raise ValueError("continuity delta must be positive")
    results = [validate_file(path, args) for path in args.inputs]
    report = {
        "experiment": "Step 2.4 generic horizontal mesh slicing validation",
        "scope": "body-part-agnostic geometry; no torso/waist/chest/hip selection",
        "parameters": {
            "normalized_heights": list(args.normalized_heights),
            "eps": args.eps,
            "endpoint_tolerance_m": args.endpoint_tolerance,
            "continuity_delta_norm": args.continuity_delta_norm,
            "max_local_perimeter_change_cm": args.max_local_perimeter_change_cm,
        },
        "samples": results,
        "passed": all(result["passed"] for result in results),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "slicing_validation.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("| Sample | Slices | Loop counts | Max local perimeter change (cm) | Status |")
    print("|---|---:|---|---:|---|")
    for result in results:
        loop_counts = ", ".join(str(item["num_contours"]) for item in result["slices"])
        maximum_change = max(
            item["continuity"]["max_matched_perimeter_change_cm"]
            for item in result["slices"]
        )
        print(
            f"| {result['sample']} | {len(result['slices'])} | {loop_counts} | "
            f"{maximum_change:.3f} | {'PASS' if result['passed'] else 'FAIL'} |"
        )
        for failure in result["failures"]:
            print(f"  - {failure}")
    print(f"\nValidation JSON: {output_path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
