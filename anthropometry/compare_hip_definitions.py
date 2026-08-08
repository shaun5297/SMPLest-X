#!/usr/bin/env python3
"""Close Step 2.6 by comparing frozen geometry and literature hip baselines.

This module only summarizes existing Step 2.6B/2.6D validation reports.  It
does not recompute meshes, change either definition, smooth profiles, fit
curves, calibrate measurements, or rank accuracy without human ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_waist_definitions import signed_difference, summarize


GEOMETRY = "geometry_hip_v0"
LITERATURE = "literature_hip_v1"
EXPECTED_STATUS = {GEOMETRY: "baseline", LITERATURE: "baseline"}
PAIR_KEY = "geometry_minus_literature"


def parse_args() -> argparse.Namespace:
    artifact_root = Path(__file__).resolve().parent / "artifacts"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geometry-report",
        type=Path,
        default=artifact_root / "hip_validation" / "geometry_hip_validation.json",
    )
    parser.add_argument(
        "--literature-report",
        type=Path,
        default=(
            artifact_root
            / "literature_hip_validation"
            / "literature_hip_validation.json"
        ),
    )
    parser.add_argument(
        "--waist-comparison-report",
        type=Path,
        default=(
            artifact_root
            / "waist_definition_comparison"
            / "waist_definition_comparison.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=artifact_root / "hip_definition_comparison",
    )
    return parser.parse_args()


def load_report(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sample_index(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    indexed = {str(record["sample"]): record for record in report["samples"]}
    if len(indexed) != len(report["samples"]):
        raise ValueError("report contains duplicate sample identifiers")
    return indexed


def _measurement(definition: str, record: dict[str, Any]) -> dict[str, Any]:
    if definition == GEOMETRY:
        selected = record["selected"]
        return {
            "definition": definition,
            "status": record["status"],
            "plane_y_m": float(selected["plane_y_m"]),
            "y_norm": float(selected["normalized_height"]),
            "circumference_cm": float(selected["perimeter_cm"]),
            "fallback_used": bool(record["fallback_count"]),
            "centerline_inside": bool(selected["centerline_inside"]),
            "remaining_duplicate_segments": int(
                selected["remaining_duplicate_segments"]
            ),
            "invalid_component_count": int(selected["invalid_component_count"]),
            "topology_state": str(selected["topology_state"]),
            "validation_passed": bool(record["validation"]["passed"]),
        }

    plane = record["plane_definition"]
    diagnostics = record["diagnostics"]
    return {
        "definition": definition,
        "status": record["status"],
        "plane_y_m": float(plane["plane_y_m"]),
        "y_norm": float(plane["normalized_height"]),
        "circumference_cm": float(record["perimeter_cm"]),
        "fallback_used": bool(record["fallback_used"]),
        "centerline_inside": bool(record["centerline_inside"]),
        "remaining_duplicate_segments": int(
            diagnostics["remaining_duplicate_segments"]
        ),
        "invalid_component_count": int(diagnostics["invalid_component_count"]),
        "topology_state": str(record["topology"]["state"]),
        "validation_passed": bool(record["validation"]["passed"]),
    }


def _validate_measurement(measurement: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    definition = measurement["definition"]
    if measurement["status"] != EXPECTED_STATUS[definition]:
        failures.append(
            f"{definition}: expected status {EXPECTED_STATUS[definition]!r}, "
            f"got {measurement['status']!r}"
        )
    for key in ("plane_y_m", "y_norm", "circumference_cm"):
        if not np.isfinite(measurement[key]):
            failures.append(f"{definition}: {key} is not finite")
    if measurement["circumference_cm"] <= 0.0:
        failures.append(f"{definition}: circumference is not positive")
    if measurement["fallback_used"]:
        failures.append(f"{definition}: pelvic contour selector used fallback")
    if not measurement["centerline_inside"]:
        failures.append(f"{definition}: pelvic centerline lies outside contour")
    if measurement["remaining_duplicate_segments"]:
        failures.append(f"{definition}: duplicate slice segments remain")
    if measurement["invalid_component_count"]:
        failures.append(f"{definition}: invalid contour component detected")
    if measurement["topology_state"] != "joined_pelvis":
        failures.append(f"{definition}: topology is not joined_pelvis")
    if not measurement["validation_passed"]:
        failures.append(f"{definition}: source validation failed")
    return failures


def _waist_context(waist_report: dict[str, Any]) -> dict[str, Any]:
    if waist_report.get("status") != "waist_subsystem_closed":
        raise ValueError("waist comparison report is not closed")
    pair = waist_report["aggregate"]["pairwise_differences"][
        "geometry_minus_anatomical"
    ]
    return {
        "comparison_pair": (
            "geometry_waist_v0 minus anatomical_midpoint_waist_proxy_v1"
        ),
        "mean_absolute_delta_y_mm": float(
            pair["delta_y_mm_first_minus_second"]["mean_absolute"]
        ),
        "mean_absolute_delta_c_cm": float(
            pair["delta_c_cm_first_minus_second"]["mean_absolute"]
        ),
        "role": "methodological_context_not_accuracy_benchmark",
    }


def build_comparison(
    geometry_report: dict[str, Any],
    literature_report: dict[str, Any],
    waist_report: dict[str, Any],
) -> dict[str, Any]:
    reports = {GEOMETRY: geometry_report, LITERATURE: literature_report}
    for definition, report in reports.items():
        if report.get("definition") != definition:
            raise ValueError(
                f"expected {definition!r} report, got {report.get('definition')!r}"
            )
        if not report.get("passed"):
            raise ValueError(f"source report {definition!r} did not pass")

    indexed = {definition: sample_index(report) for definition, report in reports.items()}
    if set(indexed[GEOMETRY]) != set(indexed[LITERATURE]):
        raise ValueError("source reports do not contain identical sample sets")

    results: list[dict[str, Any]] = []
    all_failures: list[str] = []
    primary_successes = 0
    for sample in sorted(indexed[GEOMETRY]):
        measurements = {
            definition: _measurement(definition, indexed[definition][sample])
            for definition in (GEOMETRY, LITERATURE)
        }
        failures = [
            failure
            for measurement in measurements.values()
            for failure in _validate_measurement(measurement)
        ]
        primary_successes += sum(
            not _validate_measurement(measurement)
            for measurement in measurements.values()
        )
        all_failures.extend(f"{sample}: {failure}" for failure in failures)
        difference = signed_difference(
            measurements[GEOMETRY], measurements[LITERATURE]
        )
        results.append(
            {
                "sample": sample,
                "measurements": measurements,
                "pairwise_differences": {
                    PAIR_KEY: {
                        "direction": f"{GEOMETRY} minus {LITERATURE}",
                        **difference,
                    }
                },
                "validation": {"passed": not failures, "failures": failures},
            }
        )

    metrics = {
        metric: summarize(
            [
                record["pairwise_differences"][PAIR_KEY][metric]
                for record in results
            ]
        )
        for metric in (
            "delta_y_norm_first_minus_second",
            "delta_y_mm_first_minus_second",
            "delta_c_cm_first_minus_second",
        )
    }
    hip_context = {
        "comparison_pair": f"{GEOMETRY} minus {LITERATURE}",
        "mean_absolute_delta_y_mm": metrics[
            "delta_y_mm_first_minus_second"
        ]["mean_absolute"],
        "mean_absolute_delta_c_cm": metrics[
            "delta_c_cm_first_minus_second"
        ]["mean_absolute"],
    }
    waist_context = _waist_context(waist_report)

    return {
        "experiment": "step_2_6e_hip_definition_comparison",
        "status": "hip_subsystem_closed",
        "scope": {
            "purpose": (
                "Quantify plane-location and circumference differences between "
                "geometry and literature hip definitions without claiming equivalence "
                "or accuracy superiority."
            ),
            "sample_count": len(results),
            "main_definition_count": 2,
            "primary_measurement_count": len(results) * 2,
            "primary_measurement_success_count": primary_successes,
            "new_measurement_algorithm": False,
            "metric_calibration": "none; raw canonical SMPL-X geometry",
            "accuracy_claim": "none; no human ground truth is available",
            "frozen_definition_modified": False,
        },
        "definitions": {
            GEOMETRY: {
                "status": EXPECTED_STATUS[GEOMETRY],
                "role": "geometry_baseline",
                "selection": "raw discrete pelvic perimeter argmax",
            },
            LITERATURE: {
                "status": EXPECTED_STATUS[LITERATURE],
                "role": "literature_baseline",
                "selection": "single horizontal slice at beta-deformed v5949",
            },
        },
        "pairwise_delta_convention": "signed first definition minus second definition",
        "samples": results,
        "aggregate": {
            "pairwise_differences": {
                PAIR_KEY: {
                    "direction": f"{GEOMETRY} minus {LITERATURE}",
                    **metrics,
                }
            },
            "methodological_context": {
                "waist": waist_context,
                "hip": hip_context,
                "interpretation": (
                    "Definition-dependent plane displacement has a much larger "
                    "circumference effect for the retained waist comparison than for "
                    "the hip comparison. This is an observed canonical-sample "
                    "sensitivity contrast, not an accuracy ranking or proof that the "
                    "two hip definitions are equivalent."
                ),
                "measurement_aware_loss_implication": (
                    "Landmark-level sensitivity is measurement-specific; a uniform "
                    "assumption across anthropometric targets is unsupported."
                ),
            },
        },
        "scientific_conclusion": (
            "The two hip definitions select systematically different vertical "
            "locations but yield highly consistent circumferences on the current "
            "canonical SMPL-X samples; they are not claimed to be equivalent."
        ),
        "validation": {
            "passed": not all_failures and primary_successes == len(results) * 2,
            "failures": all_failures,
            "fallback_count": sum(
                measurement["fallback_used"]
                for record in results
                for measurement in record["measurements"].values()
            ),
            "bad_contour_count": sum(
                measurement["remaining_duplicate_segments"]
                + measurement["invalid_component_count"]
                for record in results
                for measurement in record["measurements"].values()
            ),
        },
    }


def write_csv(comparison: dict[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for record in comparison["samples"]:
        geometry = record["measurements"][GEOMETRY]
        literature = record["measurements"][LITERATURE]
        difference = record["pairwise_differences"][PAIR_KEY]
        rows.append(
            {
                "sample": record["sample"],
                "geometry_status": geometry["status"],
                "geometry_plane_y_m": geometry["plane_y_m"],
                "geometry_y_norm": geometry["y_norm"],
                "geometry_circumference_cm": geometry["circumference_cm"],
                "literature_status": literature["status"],
                "literature_plane_y_m": literature["plane_y_m"],
                "literature_y_norm": literature["y_norm"],
                "literature_circumference_cm": literature["circumference_cm"],
                "delta_y_norm_geometry_minus_literature": difference[
                    "delta_y_norm_first_minus_second"
                ],
                "delta_y_mm_geometry_minus_literature": difference[
                    "delta_y_mm_first_minus_second"
                ],
                "delta_c_cm_geometry_minus_literature": difference[
                    "delta_c_cm_first_minus_second"
                ],
                "primary_measurements_passed": record["validation"]["passed"],
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_aggregate(comparison: dict[str, Any], output_dir: Path) -> str:
    records = comparison["samples"]
    samples = [record["sample"] for record in records]
    short_samples = [sample.removeprefix("p000") for sample in samples]
    x = np.arange(len(samples))
    geometry = [record["measurements"][GEOMETRY] for record in records]
    literature = [record["measurements"][LITERATURE] for record in records]
    delta_y = [
        record["pairwise_differences"][PAIR_KEY][
            "delta_y_mm_first_minus_second"
        ]
        for record in records
    ]
    delta_c = [
        record["pairwise_differences"][PAIR_KEY][
            "delta_c_cm_first_minus_second"
        ]
        for record in records
    ]
    summary = comparison["aggregate"]["pairwise_differences"][PAIR_KEY]

    geometry_color = "#315C8C"
    literature_color = "#C7772E"
    delta_color = "#557A46"
    neutral = "#52616B"
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.4))

    axes[0, 0].plot(
        x,
        [item["y_norm"] for item in geometry],
        "o-",
        color=geometry_color,
        label="Geometry argmax",
    )
    axes[0, 0].plot(
        x,
        [item["y_norm"] for item in literature],
        "s--",
        color=literature_color,
        markerfacecolor="white",
        label="Literature v5949",
    )
    axes[0, 1].plot(
        x,
        [item["circumference_cm"] for item in geometry],
        "o-",
        color=geometry_color,
        label="Geometry argmax",
    )
    axes[0, 1].plot(
        x,
        [item["circumference_cm"] for item in literature],
        "s--",
        color=literature_color,
        markerfacecolor="white",
        label="Literature v5949",
    )
    axes[1, 0].bar(x, delta_y, color=delta_color, edgecolor="#35532C")
    axes[1, 0].axhline(
        summary["delta_y_mm_first_minus_second"]["mean"],
        color=neutral,
        linestyle="--",
        linewidth=1.2,
        label=(
            "Mean geometry − literature = "
            f"{summary['delta_y_mm_first_minus_second']['mean']:.2f} mm"
        ),
    )
    axes[1, 1].bar(x, delta_c, color=delta_color, edgecolor="#35532C")
    axes[1, 1].axhline(
        summary["delta_c_cm_first_minus_second"]["mean"],
        color=neutral,
        linestyle="--",
        linewidth=1.2,
        label=(
            "Mean geometry − literature = "
            f"{summary['delta_c_cm_first_minus_second']['mean']:.3f} cm"
        ),
    )

    titles = (
        "A. Plane location",
        "B. Raw circumference",
        "C. Signed plane difference",
        "D. Signed circumference difference",
    )
    ylabels = (
        "Normalized height",
        "Circumference (cm)",
        "Geometry − literature Δy (mm)",
        "Geometry − literature ΔC (cm)",
    )
    for panel_index, (axis, title, ylabel) in enumerate(
        zip(axes.flat, titles, ylabels)
    ):
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, short_samples)
        axis.set_xlabel("Sample (p000 prefix omitted)")
        if panel_index >= 2:
            axis.axhline(0.0, color="#90A4AE", linewidth=0.7)
        axis.grid(axis="y", color="#ECEFF1", linewidth=0.7)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Step 2.6E — Hip-definition comparison "
        "(no human GT; no equivalence or accuracy claim)",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.015, 0.07, 0.995, 0.94), h_pad=2.0, w_pad=1.5)
    fig.text(
        0.5,
        0.018,
        "Two definitions differ systematically in height but remain close in "
        "circumference on these five canonical SMPL-X samples.",
        ha="center",
        fontsize=9,
        color=neutral,
    )
    path = output_dir / "hip_definition_comparison_aggregate.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def main() -> int:
    args = parse_args()
    comparison = build_comparison(
        load_report(args.geometry_report),
        load_report(args.literature_report),
        load_report(args.waist_comparison_report),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison["figures"] = {
        "aggregate": plot_aggregate(comparison, args.output_dir)
    }
    json_path = args.output_dir / "hip_definition_comparison.json"
    csv_path = args.output_dir / "hip_definition_comparison.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    write_csv(comparison, csv_path)
    print(
        json.dumps(
            {
                "passed": comparison["validation"]["passed"],
                "primary_measurement_success_count": comparison["scope"][
                    "primary_measurement_success_count"
                ],
                "fallback_count": comparison["validation"]["fallback_count"],
                "bad_contour_count": comparison["validation"]["bad_contour_count"],
                "json": str(json_path),
                "csv": str(csv_path),
            },
            indent=2,
        )
    )
    return 0 if comparison["validation"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
