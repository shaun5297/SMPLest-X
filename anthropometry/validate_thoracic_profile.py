#!/usr/bin/env python3
"""Run Step 2.7C skeleton-bounded thoracic profile characterization."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from chest import FOCUSED_SHAPY_CHEST_CONTROL, LITERATURE_CHEST_CANDIDATE
from thoracic_profile import characterize_thoracic_profile
from utils import infer_sample_name, load_canonical_mesh, verify_smplx_axes


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "anthropometry" / "artifacts" / "thoracic_profile",
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


def validate_result(result: dict[str, object]) -> list[str]:
    failures: list[str] = []
    interval = result["interval"]
    if interval["public_plane_outside_diagnostic_interval"]:
        failures.append("a public plane lies outside the diagnostic interval")
    for item in result["profile"]:
        if item["fallback_used"]:
            failures.append(f"slice {item['index']} used contour fallback")
        if item["invalid_component_count"]:
            failures.append(f"slice {item['index']} contains invalid/open component")
        if item["remaining_duplicate_segments"]:
            failures.append(f"slice {item['index']} retains duplicate segments")
        values = (
            item["plane_y_m"], item["torso_perimeter_cm"], item["torso_area_m2"],
            item["torso_compactness"], item["centerline_to_centroid_m"],
        )
        if not np.isfinite(values).all():
            failures.append(f"slice {item['index']} contains NaN or Inf")
    return failures


def plot_profile(sample: str, result: dict[str, object], output_path: Path) -> None:
    profile = result["profile"]
    y_norm = np.asarray([item["y_norm"] for item in profile])
    perimeter = np.asarray([item["torso_perimeter_cm"] for item in profile])
    loops = np.asarray([item["num_total_loops"] for item in profile])
    lateral_loops = np.asarray([item["lateral_upper_limb_loop_count"] for item in profile])
    merged = np.asarray([item["arm_torso_merge"] for item in profile], dtype=bool)
    classification = result["classification"]
    references = result["public_reference_planes"]

    fig, (profile_axis, topology_axis) = plt.subplots(2, 1, figsize=(11.5, 8.8), sharex=True)
    profile_axis.plot(y_norm, perimeter, color="0.55", linewidth=1.3, zorder=1)
    profile_axis.scatter(y_norm[~merged], perimeter[~merged], s=24, color="#1f77b4",
                         label="clean centerline-containing torso", zorder=3)
    if merged.any():
        profile_axis.scatter(y_norm[merged], perimeter[merged], s=28, marker="x",
                             color="#d62728", label="arm-torso merged contour", zorder=4)
    argmax_index = classification["raw_profile"]["argmax_index"]
    profile_axis.scatter([y_norm[argmax_index]], [perimeter[argmax_index]], marker="*",
                         s=170, color="#9467bd", label="raw discrete argmax", zorder=5)
    colors = {
        LITERATURE_CHEST_CANDIDATE: "#2ca02c",
        FOCUSED_SHAPY_CHEST_CONTROL: "#ff7f0e",
    }
    for name, reference in references.items():
        profile_axis.axvline(reference["y_norm"], color=colors[name], linewidth=1.6,
                             linestyle="--", label=name)
    first_merge = classification["first_arm_torso_merge_y_norm"]
    if first_merge is not None:
        profile_axis.axvline(first_merge, color="#d62728", linewidth=1.3, linestyle=":",
                             label="first arm-torso merge")
    profile_axis.set_ylabel("Torso contour perimeter (cm)")
    profile_axis.set_title(
        f"C(y): {classification['profile_type']} | "
        f"clean={classification['clean_pre_merge_profile']['profile_type']}"
    )
    profile_axis.grid(alpha=0.2)
    profile_axis.legend(frameon=False, fontsize=8, ncol=2)

    topology_axis.step(y_norm, loops, where="mid", color="#4c78a8", linewidth=1.8,
                       label="all closed loops")
    topology_axis.step(y_norm, lateral_loops, where="mid", color="#f58518", linewidth=1.5,
                       label="separate lateral upper-limb loops")
    topology_axis.fill_between(y_norm, 0, np.maximum(loops.max(), 1), where=merged,
                               color="#d62728", alpha=0.10, step="mid",
                               label="arm-torso merge region")
    topology_axis.set_xlabel("Normalized height")
    topology_axis.set_ylabel("Loop count")
    topology_axis.set_title("Topology transition map")
    topology_axis.grid(alpha=0.2)
    topology_axis.legend(frameon=False, fontsize=8)
    topology_axis.set_ylim(bottom=0)

    fig.suptitle(f"{sample} — Step 2.7C skeletal thoracic diagnostic profile")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def validate_file(path: Path, output_dir: Path) -> dict[str, object]:
    path = path.resolve()
    vertices, joints, faces = load_mesh(path)
    sample = infer_sample_name(path)
    result = characterize_thoracic_profile(vertices, faces, joints)
    failures = validate_result(result)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"{sample}_thoracic_profile.png"
    plot_profile(sample, result, plot_path)
    return {
        "sample": sample,
        "source_npz": str(path),
        **result,
        "debug_plot": str(plot_path),
        "validation": {"passed": not failures, "failures": failures},
    }


def main() -> int:
    args = parse_args()
    samples = [validate_file(path, args.output_dir) for path in args.inputs]
    profile_types = Counter(item["classification"]["profile_type"] for item in samples)
    raw_types = Counter(item["classification"]["raw_profile"]["profile_type"] for item in samples)
    clean_types = Counter(
        item["classification"]["clean_pre_merge_profile"]["profile_type"]
        for item in samples
    )
    rejected = [
        item for item in samples
        if item["geometry_extreme_assessment"]["status"]
        == "rejected_as_measurement_definition"
    ]
    merge_offsets = []
    for item in samples:
        first_merge_index = item["classification"]["first_arm_torso_merge_index"]
        if first_merge_index is not None:
            first_merge_y = item["profile"][first_merge_index]["plane_y_m"]
            literature_y = item["public_reference_planes"][LITERATURE_CHEST_CANDIDATE]["plane_y_m"]
            merge_offsets.append((first_merge_y - literature_y) * 1000.0)
    passed = all(item["validation"]["passed"] for item in samples)
    report = {
        "experiment": "Step 2.7C skeleton-bounded thoracic profile characterization",
        "definition": "skeletal_thoracic_diagnostic_interval_v1",
        "status": "completed" if passed else "validation_failed",
        "purpose": "characterize C(y) and topology; do not define a chest measurement",
        "samples": samples,
        "aggregate": {
            "sample_count": len(samples),
            "total_slice_count": sum(len(item["profile"]) for item in samples),
            "profile_type_counts": dict(profile_types),
            "raw_profile_type_counts": dict(raw_types),
            "clean_pre_merge_profile_type_counts": dict(clean_types),
            "geometry_extreme_rejected_count": len(rejected),
            "public_plane_outside_interval_count": sum(
                item["interval"]["public_plane_outside_diagnostic_interval"] for item in samples
            ),
            "fallback_count": sum(
                row["fallback_used"] for item in samples for row in item["profile"]
            ),
            "invalid_component_count": sum(
                row["invalid_component_count"] for item in samples for row in item["profile"]
            ),
            "remaining_duplicate_segments_count": sum(
                row["remaining_duplicate_segments"] for item in samples for row in item["profile"]
            ),
            "arm_torso_merge_slice_count": sum(
                row["arm_torso_merge"] for item in samples for row in item["profile"]
            ),
            "raw_argmax_at_upper_boundary_count": sum(
                item["classification"]["argmax_at_upper_boundary"] for item in samples
            ),
            "raw_argmax_has_arm_torso_merge_count": sum(
                item["classification"]["argmax_has_arm_torso_merge"] for item in samples
            ),
            "mean_shoulder_y_mismatch_mm": float(np.mean([
                item["interval"]["shoulder_y_mismatch_m"] * 1000.0 for item in samples
            ])),
            "max_shoulder_y_mismatch_mm": float(np.max([
                item["interval"]["shoulder_y_mismatch_m"] * 1000.0 for item in samples
            ])),
            "mean_first_merge_above_literature_plane_mm": float(np.mean(merge_offsets)),
            "range_first_merge_above_literature_plane_mm": [
                float(np.min(merge_offsets)), float(np.max(merge_offsets))
            ],
        },
        "decision": {
            "geometry_chest_extreme": {
                "status": (
                    "rejected_as_measurement_definition"
                    if len(rejected) > len(samples) / 2
                    else "undetermined"
                ),
                "reason": (
                    "All clean pre-merge thoracic profiles increase upward, while raw "
                    "internal maxima occur only after arm-torso topology merging."
                    if len(rejected) == len(samples)
                    else "sample-level assessments are not unanimous"
                ),
            },
            "recommended_primary_definition": (
                "literature_chest_v1"
                if len(rejected) > len(samples) / 2
                else "pending"
            ),
        },
        "passed": passed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "thoracic_profile_validation.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("| Sample | Profile | Clean pre-merge | Raw argmax | Merge above literature | Geometry extreme |")
    print("|---|---|---|---:|---:|---|")
    for item in samples:
        classification = item["classification"]
        merge_index = classification["first_arm_torso_merge_index"]
        merge_y = item["profile"][merge_index]["plane_y_m"]
        literature_y = item["public_reference_planes"][LITERATURE_CHEST_CANDIDATE]["plane_y_m"]
        print(
            f"| {item['sample']} | {classification['profile_type']} | "
            f"{classification['clean_pre_merge_profile']['profile_type']} | "
            f"{classification['argmax_perimeter_cm']:.2f} cm | "
            f"{(merge_y-literature_y)*1000.0:.1f} mm | "
            f"{item['geometry_extreme_assessment']['status']} |"
        )
    print(json.dumps(report["aggregate"], indent=2))
    print(json.dumps(report["decision"], indent=2))
    print(f"Validation JSON: {output_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
