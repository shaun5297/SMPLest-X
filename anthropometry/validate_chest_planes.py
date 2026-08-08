#!/usr/bin/env python3
"""Validate both public chest planes and their local topology neighborhoods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from chest import (
    CHEST_ENDPOINT_CLUSTER_TOLERANCE_M,
    FOCUSED_SHAPY_CHEST_CONTROL,
    LITERATURE_CHEST_CANDIDATE,
    compute_candidate_planes,
    evaluate_chest_candidate,
)
from utils import infer_sample_name, load_canonical_mesh, verify_smplx_axes


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "anthropometry" / "artifacts" / "chest_plane_validation",
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


def candidate_failures(result: dict[str, object]) -> list[str]:
    failures: list[str] = []
    main = result["main_slice"]
    probes = result["local_topology_probes"]
    if result["search_or_optimization"] != "none":
        failures.append("candidate performed a scan or optimization")
    if result["plane_moved_or_clamped"]:
        failures.append("candidate plane was moved or clamped")
    if main["fallback_used"] or not main["centerline_inside"]:
        failures.append("main thoracic contour did not pass centerline containment")
    if main["possible_arm_torso_merge"]:
        failures.append("main plane indicates possible arm-torso merge")
    for probe in probes:
        offset = probe["offset_height_fraction"]
        if probe["fallback_used"]:
            failures.append(f"probe {offset:+.3f}H used fallback")
        if not probe["centerline_inside"]:
            failures.append(f"probe {offset:+.3f}H misses thoracic centerline")
        if probe["diagnostics"]["invalid_component_count"]:
            failures.append(f"probe {offset:+.3f}H has invalid/open component")
        if probe["diagnostics"]["remaining_duplicate_segments"]:
            failures.append(f"probe {offset:+.3f}H retains duplicate segments")
        if probe["possible_arm_torso_merge"]:
            failures.append(f"probe {offset:+.3f}H indicates possible arm-torso merge")
    summary = result["probe_summary"]
    if (
        summary["topology_changed_within_plus_minus_0_004H"]
        and not summary["topology_change_explained_by_lateral_upper_limb_loops"]
    ):
        failures.append("local topology change is not explained by lateral upper-limb loops")
    return failures


def plot_sample(
    sample: str,
    vertices: np.ndarray,
    results: dict[str, dict[str, object]],
    output_path: Path,
) -> None:
    names = (LITERATURE_CHEST_CANDIDATE, FOCUSED_SHAPY_CHEST_CONTROL)
    colors = ("#1f77b4", "#d95f02")
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.0))
    for row, (name, color) in enumerate(zip(names, colors)):
        result = results[name]
        main = result["main_slice"]
        probes = result["local_topology_probes"]
        front, profile, contour_axis = axes[row]

        front.scatter(
            vertices[:, 0], vertices[:, 1], s=0.3, color="0.72", alpha=0.16,
            rasterized=True,
        )
        front.axhline(result["plane_y_m"], color=color, linewidth=2.1)
        for landmark in result["landmarks"]:
            coordinate = np.asarray(landmark["coordinate_m"])
            front.scatter([coordinate[0]], [coordinate[1]], s=55, color=color,
                          edgecolor="white", linewidth=0.7, zorder=5)
        front.set_xlabel("X (m)")
        front.set_ylabel("Y (m)")
        front.set_title(f"{name}\nexact public landmark plane")
        front.grid(alpha=0.18)

        offsets = [item["offset_mm"] for item in probes]
        circumferences = [item["selected_metrics"]["perimeter_cm"] for item in probes]
        loop_counts = [item["num_contours"] for item in probes]
        profile.plot(offsets, circumferences, marker="o", color=color, linewidth=1.8)
        profile.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
        profile.scatter([0.0], [main["selected_metrics"]["perimeter_cm"]],
                        marker="*", s=130, color=color, zorder=5)
        for x, y, count in zip(offsets, circumferences, loop_counts):
            profile.annotate(f"{count} loop{'s' if count != 1 else ''}", (x, y),
                             xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7)
        profile.set_xlabel("Offset from public plane (mm)")
        profile.set_ylabel("Thoracic perimeter (cm)")
        profile.set_title("Local topology probe (diagnostic only)")
        profile.grid(alpha=0.2)

        points = np.asarray(main["selected_ordered_points_m"])
        closed = np.vstack([points, points[0]])
        contour_axis.plot(closed[:, 0], closed[:, 2], color=color, linewidth=2.3,
                          label=f"torso {main['selected_metrics']['perimeter_cm']:.2f} cm")
        centerline = np.asarray(main["centerline_xz_m"])
        contour_axis.scatter([centerline[0]], [centerline[1]], marker="x", s=80,
                             color="black", linewidth=2, label="thoracic centerline")
        contour_axis.set_xlabel("X (m)")
        contour_axis.set_ylabel("Z (m)")
        contour_axis.set_aspect("equal", adjustable="box")
        contour_axis.set_title(
            f"Main selected contour; {main['num_contours']} total loop(s)"
        )
        contour_axis.grid(alpha=0.2)
        contour_axis.legend(frameon=False, fontsize=8)

    fig.suptitle(f"{sample} — Step 2.7B direct chest-plane validation")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def validate_file(path: Path, output_dir: Path) -> dict[str, object]:
    path = path.resolve()
    vertices, joints, faces = load_mesh(path)
    sample = infer_sample_name(path)
    definitions = compute_candidate_planes(vertices, faces)
    candidates: dict[str, dict[str, object]] = {}
    for name, definition in definitions.items():
        result = evaluate_chest_candidate(vertices, faces, joints, definition)
        failures = candidate_failures(result)
        result["validation"] = {"passed": not failures, "failures": failures}
        candidates[name] = result

    literature = candidates[LITERATURE_CHEST_CANDIDATE]
    focused = candidates[FOCUSED_SHAPY_CHEST_CONTROL]
    comparison = {
        "focused_minus_literature_plane_y_mm": float(
            (focused["plane_y_m"] - literature["plane_y_m"]) * 1000.0
        ),
        "focused_minus_literature_circumference_cm": float(
            focused["main_slice"]["selected_metrics"]["perimeter_cm"]
            - literature["main_slice"]["selected_metrics"]["perimeter_cm"]
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"{sample}_chest_plane_validation.png"
    plot_sample(sample, vertices, candidates, plot_path)
    return {
        "sample": sample,
        "source_npz": str(path),
        "candidates": candidates,
        "candidate_comparison": comparison,
        "debug_plot": str(plot_path),
        "validation": {
            "passed": all(item["validation"]["passed"] for item in candidates.values()),
            "failures": [
                f"{name}: {failure}"
                for name, item in candidates.items()
                for failure in item["validation"]["failures"]
            ],
        },
    }


def main() -> int:
    args = parse_args()
    results = [validate_file(path, args.output_dir) for path in args.inputs]
    all_candidates = [candidate for sample in results for candidate in sample["candidates"].values()]
    main_successes = sum(candidate["validation"]["passed"] for candidate in all_candidates)
    probe_slices = [probe for candidate in all_candidates for probe in candidate["local_topology_probes"]]
    comparisons = [sample["candidate_comparison"] for sample in results]
    passed = all(sample["validation"]["passed"] for sample in results)
    report = {
        "experiment": "Step 2.7B direct public chest-plane validation",
        "status": "validated" if passed else "candidate_review_required",
        "scope": (
            "two independent landmark-to-plane definitions on raw zero-pose canonical "
            "SMPL-X meshes; no scan, argmax, clamp, plane movement, smoothing, fitting, "
            "calibration, ISO claim, or female-bust accuracy claim"
        ),
        "exact_plane_degeneracy_handling": {
            "plane_offset_or_clamp": "none",
            "endpoint_cluster_tolerance_m": CHEST_ENDPOINT_CLUSTER_TOLERANCE_M,
            "reason": (
                "The Focused control is exactly anchored at a mesh vertex. The generic "
                "1e-6 m clustering tolerance removed a valid sub-micrometre crossing "
                "edge in two samples; tightening endpoint clustering preserves the "
                "same exact public plane rather than perturbing its height."
            ),
        },
        "samples": results,
        "aggregate": {
            "main_slice_success_count": int(main_successes),
            "main_slice_total_count": int(len(all_candidates)),
            "probe_slice_total_count": int(len(probe_slices)),
            "fallback_count": int(sum(probe["fallback_used"] for probe in probe_slices)),
            "invalid_component_count": int(sum(probe["diagnostics"]["invalid_component_count"] for probe in probe_slices)),
            "possible_arm_torso_merge_count": int(sum(probe["possible_arm_torso_merge"] for probe in probe_slices)),
            "topology_change_candidate_count": int(sum(candidate["probe_summary"]["topology_changed_within_plus_minus_0_004H"] for candidate in all_candidates)),
            "topology_change_explained_by_lateral_upper_limb_count": int(sum(candidate["probe_summary"]["topology_change_explained_by_lateral_upper_limb_loops"] for candidate in all_candidates)),
            "mean_local_perimeter_range_cm": float(np.mean([candidate["probe_summary"]["perimeter_range_cm"] for candidate in all_candidates])),
            "max_local_perimeter_range_cm": float(np.max([candidate["probe_summary"]["perimeter_range_cm"] for candidate in all_candidates])),
            "mean_abs_plane_difference_mm": float(np.mean(np.abs([item["focused_minus_literature_plane_y_mm"] for item in comparisons]))),
            "max_abs_plane_difference_mm": float(np.max(np.abs([item["focused_minus_literature_plane_y_mm"] for item in comparisons]))),
            "mean_abs_circumference_difference_cm": float(np.mean(np.abs([item["focused_minus_literature_circumference_cm"] for item in comparisons]))),
            "max_abs_circumference_difference_cm": float(np.max(np.abs([item["focused_minus_literature_circumference_cm"] for item in comparisons]))),
        },
        "passed": passed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "chest_plane_validation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("| Sample | Candidate | y_norm | C (cm) | Main loops | Probe loops | Fallback | Merge flag |")
    print("|---|---|---:|---:|---:|---|:---:|:---:|")
    for sample in results:
        for name, candidate in sample["candidates"].items():
            main = candidate["main_slice"]
            print(
                f"| {sample['sample']} | {name} | {candidate['normalized_height']:.4f} | "
                f"{main['selected_metrics']['perimeter_cm']:.3f} | {main['num_contours']} | "
                f"{candidate['probe_summary']['num_contours_sequence']} | "
                f"{'yes' if main['fallback_used'] else 'no'} | "
                f"{'yes' if candidate['probe_summary']['possible_arm_torso_merge'] else 'no'} |"
            )
    print(json.dumps(report["aggregate"], indent=2))
    print(f"Validation JSON: {report_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
