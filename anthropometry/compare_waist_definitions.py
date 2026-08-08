#!/usr/bin/env python3
"""Compare the three frozen waist definitions without ranking their accuracy.

This module consumes the validation reports produced by Steps 2.5B and 2.5C.
It does not recompute, calibrate, smooth, or modify any waist definition.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GEOMETRY = "geometry_waist_v0"
LITERATURE = "literature_waist_v1"
ANATOMICAL = "anatomical_midpoint_waist_proxy_v1"
MAIN_DEFINITIONS = (GEOMETRY, LITERATURE, ANATOMICAL)
EXPECTED_STATUS = {
    GEOMETRY: "baseline",
    LITERATURE: "baseline",
    ANATOMICAL: "frozen_v1",
}
PAIRWISE = (
    (GEOMETRY, ANATOMICAL),
    (GEOMETRY, LITERATURE),
    (LITERATURE, ANATOMICAL),
)


def parse_args() -> argparse.Namespace:
    artifact_root = Path(__file__).resolve().parent / "artifacts"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geometry-report",
        type=Path,
        default=artifact_root / "waist_validation" / "geometry_waist_validation.json",
    )
    parser.add_argument(
        "--literature-report",
        type=Path,
        default=(
            artifact_root
            / "literature_waist_validation"
            / "literature_waist_validation.json"
        ),
    )
    parser.add_argument(
        "--anatomical-report",
        type=Path,
        default=(
            artifact_root
            / "surface_anchored_waist_validation"
            / "surface_anchored_waist_validation.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=artifact_root / "waist_definition_comparison",
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


def signed_difference(
    first: dict[str, float], second: dict[str, float]
) -> dict[str, float]:
    """Return first-minus-second differences with explicit physical units."""
    return {
        "delta_y_norm_first_minus_second": first["y_norm"] - second["y_norm"],
        "delta_y_mm_first_minus_second":
            (first["plane_y_m"] - second["plane_y_m"]) * 1000.0,
        "delta_c_cm_first_minus_second":
            first["circumference_cm"] - second["circumference_cm"],
    }


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    return {
        "mean": fmean(values),
        "population_std": pstdev(values),
        "mean_absolute": fmean(abs(value) for value in values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _measurement(
    definition: str, record: dict[str, Any]
) -> dict[str, Any]:
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
        "validation_passed": bool(record.get("validation", {}).get("passed", True)),
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
        failures.append(f"{definition}: torso selector used fallback")
    if not measurement["centerline_inside"]:
        failures.append(f"{definition}: centerline lies outside selected contour")
    if measurement["remaining_duplicate_segments"]:
        failures.append(f"{definition}: duplicate slice segments remain")
    if measurement["invalid_component_count"]:
        failures.append(f"{definition}: invalid contour component detected")
    if not measurement["validation_passed"]:
        failures.append(f"{definition}: source validation failed")
    return failures


def _pair_key(first: str, second: str) -> str:
    short = {
        GEOMETRY: "geometry",
        LITERATURE: "literature",
        ANATOMICAL: "anatomical",
    }
    return f"{short[first]}_minus_{short[second]}"


def build_comparison(
    geometry_report: dict[str, Any],
    literature_report: dict[str, Any],
    anatomical_report: dict[str, Any],
) -> dict[str, Any]:
    reports = {
        GEOMETRY: geometry_report,
        LITERATURE: literature_report,
        ANATOMICAL: anatomical_report,
    }
    for definition, report in reports.items():
        if report["definition"] != definition:
            raise ValueError(
                f"expected {definition!r} report, got {report['definition']!r}"
            )
        if not report["passed"]:
            raise ValueError(f"source report {definition!r} did not pass")

    indexed = {definition: sample_index(report) for definition, report in reports.items()}
    sample_sets = [set(records) for records in indexed.values()]
    if not all(sample_set == sample_sets[0] for sample_set in sample_sets[1:]):
        raise ValueError("source reports do not contain identical sample sets")

    results: list[dict[str, Any]] = []
    all_failures: list[str] = []
    primary_successes = 0
    for sample in sorted(sample_sets[0]):
        source_records = {
            GEOMETRY: indexed[GEOMETRY][sample],
            LITERATURE: indexed[LITERATURE][sample],
            ANATOMICAL: indexed[ANATOMICAL][sample]["surface_anchored"],
        }
        measurements = {
            definition: _measurement(definition, source_records[definition])
            for definition in MAIN_DEFINITIONS
        }
        measurements[ANATOMICAL]["validation_passed"] = bool(
            indexed[ANATOMICAL][sample]["validation"]["passed"]
        )
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

        pairwise: dict[str, Any] = {}
        for first, second in PAIRWISE:
            key = _pair_key(first, second)
            pairwise[key] = {
                "direction": f"{first} minus {second}",
                **signed_difference(measurements[first], measurements[second]),
            }

        anchored_container = indexed[ANATOMICAL][sample]
        fixed = anchored_container["fixed_xyz"]
        results.append(
            {
                "sample": sample,
                "gender": anchored_container["gender"],
                "measurements": measurements,
                "pairwise_differences": pairwise,
                "fixed_xyz_control": {
                    "definition": fixed["definition"],
                    "status": fixed["status"],
                    "plane_y_m": fixed["plane_definition"]["plane_y_m"],
                    "y_norm": fixed["plane_definition"]["normalized_height"],
                    "circumference_cm": fixed["perimeter_cm"],
                    "role": "ablation_control_only",
                    "included_in_main_comparison": False,
                },
                "validation": {"passed": not failures, "failures": failures},
            }
        )

    pairwise_summary: dict[str, Any] = {}
    for first, second in PAIRWISE:
        key = _pair_key(first, second)
        metrics = {
            metric: summarize(
                [record["pairwise_differences"][key][metric] for record in results]
            )
            for metric in (
                "delta_y_norm_first_minus_second",
                "delta_y_mm_first_minus_second",
                "delta_c_cm_first_minus_second",
            )
        }
        pairwise_summary[key] = {
            "direction": f"{first} minus {second}",
            **metrics,
        }

    return {
        "experiment": "step_2_5d_waist_definition_comparison",
        "status": "waist_subsystem_closed",
        "scope": {
            "purpose": (
                "Quantify systematic plane-location and circumference differences "
                "among three waist definitions without claiming accuracy superiority."
            ),
            "sample_count": len(results),
            "main_definition_count": len(MAIN_DEFINITIONS),
            "primary_measurement_count": len(results) * len(MAIN_DEFINITIONS),
            "primary_measurement_success_count": primary_successes,
            "metric_calibration": "none; raw canonical SMPL-X geometry",
            "accuracy_claim": "none; no human ground truth is available",
            "frozen_definition_modified": False,
        },
        "definitions": {
            GEOMETRY: {
                "status": EXPECTED_STATUS[GEOMETRY],
                "role": "geometry_baseline",
                "selection": "raw discrete minimum of the torso C(y) profile",
                "landmark_required": False,
                "shape_adaptation": "determined directly from each beta-deformed mesh",
            },
            LITERATURE: {
                "status": EXPECTED_STATUS[LITERATURE],
                "role": "literature_baseline",
                "selection": "published topology-fixed surface landmark plane",
                "landmark_required": True,
                "shape_adaptation": "fixed vertex IDs with beta-deformed coordinates",
            },
            ANATOMICAL: {
                "status": EXPECTED_STATUS[ANATOMICAL],
                "role": "anatomical_proxy",
                "selection": "gender-specific face-barycentric surface anchors",
                "landmark_required": True,
                "shape_adaptation": "surface anchored and beta adaptive",
            },
        },
        "fixed_xyz_control": {
            "role": "ablation_control_only",
            "included_in_main_definition_competition": False,
            "reason_rejected_as_final_anatomical_representation": (
                "world-space XYZ landmarks are invariant to beta deformation"
            ),
            "cross_beta_surface_anchor_plane_range_mm": anatomical_report[
                "cross_beta_plane_variation"
            ],
        },
        "pairwise_delta_convention": "signed first definition minus second definition",
        "samples": results,
        "aggregate": {"pairwise_differences": pairwise_summary},
        "validation": {
            "passed": not all_failures and primary_successes == 15,
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


def csv_rows(comparison: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for record in comparison["samples"]:
        row: dict[str, Any] = {"sample": record["sample"], "gender": record["gender"]}
        for definition, prefix in (
            (GEOMETRY, "geometry"),
            (LITERATURE, "literature"),
            (ANATOMICAL, "anatomical"),
        ):
            measurement = record["measurements"][definition]
            row.update(
                {
                    f"{prefix}_status": measurement["status"],
                    f"{prefix}_plane_y_m": measurement["plane_y_m"],
                    f"{prefix}_y_norm": measurement["y_norm"],
                    f"{prefix}_circumference_cm": measurement["circumference_cm"],
                    f"{prefix}_fallback_used": measurement["fallback_used"],
                }
            )
        for key, difference in record["pairwise_differences"].items():
            row[f"delta_y_norm_{key}"] = difference[
                "delta_y_norm_first_minus_second"
            ]
            row[f"delta_y_mm_{key}"] = difference[
                "delta_y_mm_first_minus_second"
            ]
            row[f"delta_c_cm_{key}"] = difference[
                "delta_c_cm_first_minus_second"
            ]
        row["primary_measurements_passed"] = record["validation"]["passed"]
        rows.append(row)
    return list(rows[0]), rows


def write_csv(comparison: dict[str, Any], path: Path) -> None:
    fields, rows = csv_rows(comparison)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _profile_lookup(geometry_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return sample_index(geometry_report)


def plot_sample_profiles(
    comparison: dict[str, Any], geometry_report: dict[str, Any], output_dir: Path
) -> list[str]:
    geometry_samples = _profile_lookup(geometry_report)
    output_paths: list[str] = []
    colors = {
        GEOMETRY: "#C62828",
        LITERATURE: "#EF6C00",
        ANATOMICAL: "#2E7D32",
    }
    labels = {
        GEOMETRY: "Geometry minimum",
        LITERATURE: "Literature plane",
        ANATOMICAL: "Anatomical midpoint",
    }
    for record in comparison["samples"]:
        sample = record["sample"]
        profile = geometry_samples[sample]["profile"]
        x = np.asarray([item["normalized_height"] for item in profile])
        y = np.asarray([item["perimeter_cm"] for item in profile])
        fig, ax = plt.subplots(figsize=(8.2, 5.0), constrained_layout=True)
        ax.plot(x, y, color="#455A64", linewidth=1.8, marker="o", markersize=2.5)
        for definition in MAIN_DEFINITIONS:
            measurement = record["measurements"][definition]
            marker = "*" if definition == GEOMETRY else "o"
            size = 145 if definition == GEOMETRY else 58
            ax.axvline(
                measurement["y_norm"], color=colors[definition], linewidth=1.1,
                alpha=0.72, linestyle="--"
            )
            ax.scatter(
                [measurement["y_norm"]], [measurement["circumference_cm"]],
                color=colors[definition], edgecolor="white", linewidth=0.8,
                marker=marker, s=size, zorder=5,
                label=(
                    f"{labels[definition]}: y={measurement['y_norm']:.4f}, "
                    f"C={measurement['circumference_cm']:.2f} cm"
                ),
            )
        ax.set_title(f"{sample} waist-definition comparison")
        ax.set_xlabel("Normalized height")
        ax.set_ylabel("Raw canonical circumference (cm)")
        ax.grid(axis="y", color="#CFD8DC", linewidth=0.7, alpha=0.8)
        ax.legend(frameon=False, fontsize=8.5, loc="best")
        ax.text(
            0.01, 0.01,
            "Definition disagreement is not measurement error; no human GT/calibration.",
            transform=ax.transAxes, fontsize=7.5, color="#546E7A"
        )
        path = output_dir / f"{sample}_waist_definition_profile.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        output_paths.append(str(path))
    return output_paths


def plot_aggregate(comparison: dict[str, Any], output_dir: Path) -> str:
    samples = [record["sample"] for record in comparison["samples"]]
    short_samples = [sample.removeprefix("p000") for sample in samples]
    x = np.arange(len(samples))
    colors = {
        GEOMETRY: "#C62828",
        LITERATURE: "#EF6C00",
        ANATOMICAL: "#2E7D32",
    }
    labels = {
        GEOMETRY: "Geometry",
        LITERATURE: "Literature",
        ANATOMICAL: "Anatomical",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.1), constrained_layout=True)
    for definition in MAIN_DEFINITIONS:
        y_norm = [
            record["measurements"][definition]["y_norm"]
            for record in comparison["samples"]
        ]
        circumferences = [
            record["measurements"][definition]["circumference_cm"]
            for record in comparison["samples"]
        ]
        axes[0, 0].plot(x, y_norm, "o-", color=colors[definition], label=labels[definition])
        axes[0, 1].plot(
            x, circumferences, "o-", color=colors[definition], label=labels[definition]
        )

    pair_keys = [_pair_key(first, second) for first, second in PAIRWISE]
    pair_labels = ["Geometry − Anatomy", "Geometry − Literature", "Literature − Anatomy"]
    bar_colors = ["#5C6BC0", "#26A69A", "#AB47BC"]
    width = 0.24
    for index, (key, label, color) in enumerate(zip(pair_keys, pair_labels, bar_colors)):
        offset = (index - 1) * width
        axes[1, 0].bar(
            x + offset,
            [record["pairwise_differences"][key]["delta_y_mm_first_minus_second"] for record in comparison["samples"]],
            width=width,
            label=label,
            color=color,
        )
        axes[1, 1].bar(
            x + offset,
            [record["pairwise_differences"][key]["delta_c_cm_first_minus_second"] for record in comparison["samples"]],
            width=width,
            label=label,
            color=color,
        )

    titles = (
        "A. Plane location",
        "B. Raw circumference",
        "C. Pairwise plane difference",
        "D. Pairwise circumference difference",
    )
    ylabels = ("Normalized height", "Circumference (cm)", "Signed Δy (mm)", "Signed ΔC (cm)")
    for panel_index, (ax, title, ylabel) in enumerate(
        zip(axes.flat, titles, ylabels)
    ):
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, short_samples)
        ax.set_xlabel("Sample (p000 prefix omitted)")
        if panel_index >= 2:
            ax.axhline(0.0, color="#78909C", linewidth=0.7)
        ax.grid(axis="y", color="#ECEFF1", linewidth=0.7)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Step 2.5D — Waist-definition comparison (no human GT; no accuracy ranking)",
        fontsize=14,
    )
    path = output_dir / "waist_definition_comparison_aggregate.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def main() -> int:
    args = parse_args()
    geometry_report = load_report(args.geometry_report)
    literature_report = load_report(args.literature_report)
    anatomical_report = load_report(args.anatomical_report)
    comparison = build_comparison(
        geometry_report, literature_report, anatomical_report
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison["figures"] = {
        "sample_profiles": plot_sample_profiles(
            comparison, geometry_report, args.output_dir
        ),
        "aggregate": plot_aggregate(comparison, args.output_dir),
    }
    json_path = args.output_dir / "waist_definition_comparison.json"
    csv_path = args.output_dir / "waist_definition_comparison.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    write_csv(comparison, csv_path)
    print(json.dumps({
        "passed": comparison["validation"]["passed"],
        "primary_measurement_success_count": comparison["scope"]["primary_measurement_success_count"],
        "fallback_count": comparison["validation"]["fallback_count"],
        "bad_contour_count": comparison["validation"]["bad_contour_count"],
        "json": str(json_path),
        "csv": str(csv_path),
    }, indent=2))
    return 0 if comparison["validation"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
