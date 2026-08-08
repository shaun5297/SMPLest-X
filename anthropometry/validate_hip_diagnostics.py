#!/usr/bin/env python3
"""Run Step 2.6C lower-bound, plateau, and step-size diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hip import scan_geometry_hip
from hip_diagnostics import (
    DEFAULT_DOWNWARD_PROBE_LAYERS,
    lower_bound_sensitivity,
    near_maximum_plateau,
    probe_below_stable_lower,
    step_size_sensitivity,
)
from utils import infer_sample_name, load_canonical_mesh, verify_smplx_axes


STEP_C_STABILITY_LIMIT_PERCENT = 0.10
BOUNDARY_PROBE_MINIMUM_DEPTH_MM = 20.0


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--downward-probe-layers",
        type=int,
        default=DEFAULT_DOWNWARD_PROBE_LAYERS,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(repo_root / "anthropometry" / "artifacts" / "hip_diagnostics"),
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


def plot_boundary_diagnostic(
    sample: str,
    baseline: dict[str, object],
    probe: list[dict[str, object]],
    plateau: dict[str, object],
    output_path: Path,
) -> None:
    """Static research figure: focused profile, compactness, and plateau evidence."""
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    c_axis, compactness_axis, plateau_axis = axes
    blue = "#2F5D8A"
    orange = "#D47A24"
    neutral = "#46515C"

    distances = [0.0] + [record["distance_below_lower_mm"] for record in probe]
    circumferences = [baseline["profile"][0]["perimeter_cm"]] + [
        record.get("perimeter_cm", np.nan) for record in probe
    ]
    c_axis.plot(distances, circumferences, color=neutral, linewidth=1.4)
    c_axis.scatter([0.0], [circumferences[0]], color=blue, s=70, label="current lower")
    for record in probe:
        stable = bool(record.get("stable_safe", False))
        c_axis.scatter(
            [record["distance_below_lower_mm"]],
            [record.get("perimeter_cm", np.nan)],
            facecolors=blue if stable else "white",
            edgecolors=blue if stable else orange,
            marker="o" if record["topology_state"] == "joined_pelvis" else "x",
            s=55,
            linewidths=1.4,
        )
    c_axis.axhspan(
        plateau["threshold_cm"],
        plateau["cmax_cm"],
        color=blue,
        alpha=0.10,
        label="Cmax − 0.1% band",
    )
    c_axis.invert_xaxis()
    c_axis.set_xlabel("Distance below current stable lower bound (mm)")
    c_axis.set_ylabel("Raw circumference (cm)")
    c_axis.set_title("Downward extension probe")
    c_axis.grid(alpha=0.22)
    c_axis.legend(frameon=False, fontsize=8)

    valid_probe = [record for record in probe if record.get("measurement_valid", False)]
    compactness_axis.plot(
        [record["distance_below_lower_mm"] for record in valid_probe],
        [record["compactness"] for record in valid_probe],
        color=neutral,
        marker="o",
        markersize=4,
    )
    compactness_threshold = baseline["search_region"]["stability_gate"][
        "minimum_compactness"
    ]
    compactness_axis.axhline(
        compactness_threshold,
        color=orange,
        linestyle="--",
        label=f"stability threshold={compactness_threshold:.3f}",
    )
    compactness_axis.invert_xaxis()
    compactness_axis.set_xlabel("Distance below current stable lower bound (mm)")
    compactness_axis.set_ylabel("Contour compactness 4πA/P²")
    compactness_axis.set_title("Transient-recovery diagnostic")
    compactness_axis.grid(alpha=0.22)
    compactness_axis.legend(frameon=False, fontsize=8)

    profile = baseline["profile"]
    end = min(len(profile), max(15, plateau["plateau_end_index"] + 4))
    focused = profile[:end]
    plateau_axis.plot(
        [record["normalized_height"] for record in focused],
        [record["perimeter_cm"] for record in focused],
        color=blue,
        marker="o",
        markersize=3.5,
    )
    plateau_axis.axvspan(
        plateau["plateau_y_norm_min"],
        plateau["plateau_y_norm_max"],
        color=blue,
        alpha=0.14,
        label=f"near-max plateau ({plateau['plateau_width_mm']:.1f} mm)",
    )
    plateau_axis.axhline(
        plateau["threshold_cm"], color=orange, linestyle="--", linewidth=1.0
    )
    plateau_axis.set_xlabel("Normalized height")
    plateau_axis.set_ylabel("Raw circumference (cm)")
    plateau_axis.set_title("Near-maximum plateau")
    plateau_axis.grid(alpha=0.22)
    plateau_axis.legend(frameon=False, fontsize=8)

    fig.suptitle(
        f"{sample}: Step 2.6C boundary diagnostic\n"
        "Focused axes; raw canonical geometry; no smoothing or fitting",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_step_sensitivity(
    results: list[dict[str, object]], output_path: Path
) -> None:
    samples = [result["sample"] for result in results]
    x = np.arange(len(samples))
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.8), sharex=True)
    blue = "#2F5D8A"
    orange = "#D47A24"
    series = (("0.001H", blue), ("0.004H", orange))
    for key, color in series:
        delta_c = [
            result["step_size_sensitivity"]["results"][key][
                "delta_c_cm_vs_0.002H"
            ]
            for result in results
        ]
        delta_y = [
            result["step_size_sensitivity"]["results"][key][
                "delta_y_mm_vs_0.002H"
            ]
            for result in results
        ]
        axes[0].plot(x, delta_c, marker="o", color=color, label=f"{key} vs 0.002H")
        axes[1].plot(x, delta_y, marker="o", color=color, label=f"{key} vs 0.002H")
    axes[0].axhline(0.0, color="#56616B", linewidth=0.8)
    axes[1].axhline(0.0, color="#56616B", linewidth=0.8)
    axes[0].set_ylabel("ΔC (cm)")
    axes[1].set_ylabel("Δy (mm)")
    axes[1].set_xlabel("Sample")
    axes[1].set_xticks(x, samples)
    axes[0].set_title("Circumference sensitivity (primary validation target)")
    axes[1].set_title("Discrete argmax location sensitivity (secondary diagnostic)")
    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Step 2.6C scan-step sensitivity\n"
        "Unchanged raw discrete perimeter argmax; focused difference axes",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def validate_file(path: Path, args: argparse.Namespace) -> dict[str, object]:
    path = path.resolve()
    vertices, joints, faces = load_mesh(path)
    sample = infer_sample_name(path)
    baseline = scan_geometry_hip(vertices, faces, joints)
    plateau = near_maximum_plateau(
        baseline["profile"], baseline["selected_index"]
    )
    sensitivity = step_size_sensitivity(vertices, faces, joints)
    failures: list[str] = []
    warnings: list[str] = []

    downward_probe: list[dict[str, object]] = []
    lower_sensitivity: dict[str, object] | None = None
    diagnostic_plot: str | None = None
    if baseline["boundary_maximum"]:
        downward_probe = probe_below_stable_lower(
            vertices,
            faces,
            joints,
            baseline,
            layers=args.downward_probe_layers,
        )
        lower_sensitivity = lower_bound_sensitivity(baseline, downward_probe)
        if downward_probe[-1]["distance_below_lower_mm"] < BOUNDARY_PROBE_MINIMUM_DEPTH_MM:
            failures.append("downward probe covers less than 20 mm")
        if lower_sensitivity["case"] != "A":
            failures.append("a hidden material stable maximum exists below lower bound")
        if not plateau["touches_lower_boundary"]:
            failures.append("boundary argmax does not touch its near-maximum plateau")
        diagnostic_path = args.output_dir / f"{sample}_hip_boundary_diagnostic.png"
        plot_boundary_diagnostic(
            sample, baseline, downward_probe, plateau, diagnostic_path
        )
        diagnostic_plot = str(diagnostic_path)

    if (
        sensitivity["maximum_absolute_relative_delta_c_percent"]
        > STEP_C_STABILITY_LIMIT_PERCENT
    ):
        failures.append("step-size circumference sensitivity exceeds 0.10%")
    if sensitivity["maximum_absolute_delta_y_mm"] > 15.0:
        warnings.append("step-size argmax location varies by more than 15 mm")
    if baseline["definition"] != "geometry_hip_v0":
        failures.append("unexpected baseline definition")
    return {
        "sample": sample,
        "source_npz": str(path),
        "baseline_summary": {
            "definition": baseline["definition"],
            "status_before_2_6c": baseline["status"],
            "stable_lower_y_norm": baseline["profile"][0]["normalized_height"],
            "selected_y_norm": baseline["selected"]["normalized_height"],
            "selected_plane_y_m": baseline["selected"]["plane_y_m"],
            "selected_c_cm": baseline["selected"]["perimeter_cm"],
            "boundary_maximum": baseline["boundary_maximum"],
            "step_normalized_height": baseline["scan_parameters"][
                "step_normalized_height"
            ],
        },
        "near_maximum_plateau": plateau,
        "downward_probe": downward_probe,
        "lower_bound_sensitivity": lower_sensitivity,
        "step_size_sensitivity": sensitivity,
        "diagnostic_plot": diagnostic_plot,
        "validation": {
            "passed": not failures,
            "failures": failures,
            "warnings": warnings,
        },
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [validate_file(path, args) for path in args.inputs]
    boundary_results = [
        result for result in results if result["baseline_summary"]["boundary_maximum"]
    ]
    all_boundary_case_a = all(
        result["lower_bound_sensitivity"]["case"] == "A"
        for result in boundary_results
    )
    passed = (
        len(boundary_results) == 3
        and all_boundary_case_a
        and all(result["validation"]["passed"] for result in results)
    )
    aggregate_plot_path = args.output_dir / "hip_step_size_sensitivity.png"
    plot_step_sensitivity(results, aggregate_plot_path)

    report = {
        "experiment": "Step 2.6C geometry hip lower-bound and step sensitivity",
        "pelvic_search_region": {
            "definition": "topology_defined_pelvic_search_region_v0",
            "status": "frozen",
        },
        "geometry_hip": {
            "definition": "geometry_hip_v0",
            "status": "baseline" if passed else "pending_2_6c_review",
            "definition_modified": False,
            "step_normalized_height": 0.002,
            "maximum_method": "raw_discrete_argmax_perimeter",
            "smoothing": "none",
            "curve_fitting": "none",
            "metric_calibration": "none",
        },
        "decision_rule": {
            "boundary_case_acceptance": (
                "case A: no material larger perimeter in contiguous stable-safe "
                "downward extension"
            ),
            "material_c_threshold": "0.1% of current Cmax",
            "step_c_stability_limit_percent": STEP_C_STABILITY_LIMIT_PERCENT,
            "required_boundary_case_count": 3,
        },
        "samples": results,
        "aggregate": {
            "sample_count": len(results),
            "boundary_case_count": len(boundary_results),
            "boundary_case_a_count": sum(
                result["lower_bound_sensitivity"]["case"] == "A"
                for result in boundary_results
            ),
            "hidden_stable_maximum_count": sum(
                result["lower_bound_sensitivity"]["hidden_stable_maximum"]
                for result in boundary_results
            ),
            "maximum_absolute_step_delta_c_cm": max(
                result["step_size_sensitivity"]["maximum_absolute_delta_c_cm"]
                for result in results
            ),
            "maximum_absolute_step_delta_c_percent": max(
                result["step_size_sensitivity"][
                    "maximum_absolute_relative_delta_c_percent"
                ]
                for result in results
            ),
            "maximum_absolute_step_delta_y_mm": max(
                result["step_size_sensitivity"]["maximum_absolute_delta_y_mm"]
                for result in results
            ),
            "step_sensitivity_plot": str(aggregate_plot_path),
        },
        "conclusion": (
            "geometry_hip_v0 promoted to baseline; boundary maxima are shallow "
            "near-maximum plateaus with no hidden stable larger perimeter"
            if passed
            else "geometry_hip_v0 remains pending because Step 2.6C did not pass"
        ),
        "passed": passed,
    }
    output_path = args.output_dir / "geometry_hip_diagnostics.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("| Sample | Boundary | Case | Probe depth (mm) | ΔC joined≤20mm (cm) | ΔC stable (cm) | Plateau (mm) | Max |ΔC| step (cm) |")
    print("|---|:---:|:---:|---:|---:|---:|---:|---:|")
    for result in results:
        lower = result["lower_bound_sensitivity"]
        depth = result["downward_probe"][-1]["distance_below_lower_mm"] if lower else 0.0
        delta_joined = (
            lower["extended_joined_safe_within_20mm"]["delta_c_cm_vs_current"]
            if lower else 0.0
        )
        delta_stable = (
            lower["extended_stable_safe"]["delta_c_cm_vs_current"]
            if lower else 0.0
        )
        case = lower["case"] if lower else "n/a"
        print(
            f"| {result['sample']} | "
            f"{'yes' if result['baseline_summary']['boundary_maximum'] else 'no'} | "
            f"{case} | {depth:.2f} | {delta_joined:.4f} | {delta_stable:.4f} | "
            f"{result['near_maximum_plateau']['plateau_width_mm']:.2f} | "
            f"{result['step_size_sensitivity']['maximum_absolute_delta_c_cm']:.4f} |"
        )
        for failure in result["validation"]["failures"]:
            print(f"  - failure: {failure}")
        for warning in result["validation"]["warnings"]:
            print(f"  - warning: {warning}")
    print(f"\nValidation JSON: {output_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
