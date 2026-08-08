#!/usr/bin/env python3
"""Final five-subject regression for the Step 2.8 unified extractor."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from anatomical_waist import load_subject_gender_labels
from definitions import PRIMARY_TARGETS
from extractor import extract_anthropometry
from utils import infer_sample_name


REGRESSION_TOLERANCE_CM = 1e-9


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    artifacts = repo_root / "anthropometry" / "artifacts"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--gender-labels", required=True, type=Path)
    parser.add_argument(
        "--anchor-dir",
        type=Path,
        default=repo_root / "anthropometry" / "landmarks" / "anatomical_midpoint",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=artifacts / "unified_extractor"
    )
    parser.add_argument(
        "--geometry-waist-report", type=Path,
        default=artifacts / "waist_validation" / "geometry_waist_validation.json",
    )
    parser.add_argument(
        "--literature-waist-report", type=Path,
        default=artifacts / "literature_waist_validation" / "literature_waist_validation.json",
    )
    parser.add_argument(
        "--anatomical-waist-report", type=Path,
        default=artifacts / "surface_anchored_waist_validation" / "surface_anchored_waist_validation.json",
    )
    parser.add_argument(
        "--geometry-hip-report", type=Path,
        default=artifacts / "hip_validation" / "geometry_hip_validation.json",
    )
    parser.add_argument(
        "--literature-hip-report", type=Path,
        default=artifacts / "literature_hip_validation" / "literature_hip_validation.json",
    )
    parser.add_argument(
        "--chest-report", type=Path,
        default=artifacts / "chest_plane_validation" / "chest_plane_validation.json",
    )
    return parser.parse_args()


def _samples(path: Path) -> dict[str, dict[str, object]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    return {item["sample"]: item for item in report["samples"]}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sidecar(path: Path, suffix: str) -> dict[str, object]:
    sidecar = path.with_name(f"{path.stem.removesuffix('_canonical')}_{suffix}.json")
    return json.loads(sidecar.read_text(encoding="utf-8"))


def main_values(result: dict[str, object]) -> dict[str, float]:
    return {
        "height_cm": result["height"]["raw_height_v0"]["value_cm"],
        "shoulder_joint_cm": result["shoulder"]["shoulder_joint_width"]["value_cm"],
        "shoulder_literature_cm": result["shoulder"]["literature_shoulder_breadth"]["value_cm"],
        "shoulder_acromion_cm": result["shoulder"]["acromion_surface_proxy_v1"]["value_cm"],
        "waist_geometry_cm": result["waist"]["geometry_waist_v0"]["perimeter_cm"],
        "waist_literature_cm": result["waist"]["literature_waist_v1"]["perimeter_cm"],
        "waist_anatomical_cm": result["waist"]["anatomical_midpoint_waist_proxy_v1"]["perimeter_cm"],
        "hip_geometry_cm": result["hip"]["geometry_hip_v0"]["perimeter_cm"],
        "hip_literature_cm": result["hip"]["literature_hip_v1"]["perimeter_cm"],
        "chest_literature_cm": result["chest"]["literature_chest_v1"]["perimeter_cm"],
        "chest_focused_control_cm": result["chest"]["focused_shapy_chest_control"]["perimeter_cm"],
    }


def expected_values(
    path: Path,
    sample: str,
    reports: dict[str, dict[str, dict[str, object]]],
) -> dict[str, float]:
    height = _sidecar(path, "height")
    shoulder = _sidecar(path, "shoulder_width")["measurements"]
    chest = reports["chest"][sample]["candidates"]
    return {
        "height_cm": height["raw_height_cm"],
        "shoulder_joint_cm": shoulder["shoulder_joint_distance_cm"],
        "shoulder_literature_cm": shoulder["literature_shoulder_breadth_cm"],
        "shoulder_acromion_cm": shoulder["acromion_surface_proxy_v1_distance_cm"],
        "waist_geometry_cm": reports["geometry_waist"][sample]["selected"]["perimeter_cm"],
        "waist_literature_cm": reports["literature_waist"][sample]["perimeter_cm"],
        "waist_anatomical_cm": reports["anatomical_waist"][sample]["surface_anchored"]["perimeter_cm"],
        "hip_geometry_cm": reports["geometry_hip"][sample]["selected"]["perimeter_cm"],
        "hip_literature_cm": reports["literature_hip"][sample]["perimeter_cm"],
        "chest_literature_cm": chest["literature_chest_v1"]["main_slice"]["selected_metrics"]["perimeter_cm"],
        "chest_focused_control_cm": chest["focused_shapy_chest_control"]["main_slice"]["selected_metrics"]["perimeter_cm"],
    }


def main() -> int:
    args = parse_args()
    gender_labels = load_subject_gender_labels(args.gender_labels)
    reports = {
        "geometry_waist": _samples(args.geometry_waist_report),
        "literature_waist": _samples(args.literature_waist_report),
        "anatomical_waist": _samples(args.anatomical_waist_report),
        "geometry_hip": _samples(args.geometry_hip_report),
        "literature_hip": _samples(args.literature_hip_report),
        "chest": _samples(args.chest_report),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_results = []
    for path in args.inputs:
        path = path.resolve()
        sample = infer_sample_name(path)
        if sample not in gender_labels:
            raise KeyError(f"gender label missing for {sample}")
        before_hash = _sha256(path)
        result = extract_anthropometry(
            path, gender=gender_labels[sample], anchor_dir=args.anchor_dir
        )
        after_hash = _sha256(path)
        actual = main_values(result)
        expected = expected_values(path, sample, reports)
        differences = {key: float(actual[key] - expected[key]) for key in actual}
        failures = [
            f"{key} drift={value:+.12g} cm"
            for key, value in differences.items()
            if abs(value) > REGRESSION_TOLERANCE_CM
        ]
        if before_hash != after_hash:
            failures.append("canonical NPZ hash changed during extraction")
        fallback_count = 0
        invalid_component_count = 0
        for section in ("waist", "hip", "chest"):
            for name, measurement in result[section].items():
                if not isinstance(measurement, dict) or "fallback_used" not in measurement:
                    continue
                if measurement["fallback_used"]:
                    fallback_count += 1
                    failures.append(f"{name} used contour fallback")
                invalid_count = int(measurement["diagnostics"]["invalid_component_count"])
                invalid_component_count += invalid_count
                if invalid_count:
                    failures.append(f"{name} contains invalid/open component")
        output_path = args.output_dir / f"{sample}_anthropometry.json"
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        sample_results.append(
            {
                "sample": sample,
                "gender": gender_labels[sample],
                "source_npz": str(path),
                "source_sha256_before": before_hash,
                "source_sha256_after": after_hash,
                "source_unchanged": before_hash == after_hash,
                "values_cm": actual,
                "regression_difference_cm": differences,
                "diagnostics": {
                    "fallback_count": fallback_count,
                    "invalid_component_count": invalid_component_count,
                },
                "output_json": str(output_path),
                "validation": {"passed": not failures, "failures": failures},
            }
        )

    columns = ["sample", "gender", *next(iter(sample_results))["values_cm"].keys()]
    csv_path = args.output_dir / "unified_anthropometry.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for item in sample_results:
            writer.writerow({"sample": item["sample"], "gender": item["gender"], **item["values_cm"]})

    all_differences = [
        abs(value)
        for item in sample_results
        for value in item["regression_difference_cm"].values()
    ]
    passed = all(item["validation"]["passed"] for item in sample_results)
    report = {
        "experiment": "Step 2.8 unified anthropometric extractor final regression",
        "definition": "unified_anthropometric_extractor_v1",
        "status": "completed" if passed else "validation_failed",
        "scope": "raw zero-pose canonical SMPL-X anthropometry; no GT or metric calibration",
        "primary_targets": PRIMARY_TARGETS,
        "regression_tolerance_cm": REGRESSION_TOLERANCE_CM,
        "samples": sample_results,
        "aggregate": {
            "sample_count": len(sample_results),
            "measurement_count": len(sample_results) * len(sample_results[0]["values_cm"]),
            "successful_measurement_count": sum(
                len(item["values_cm"]) if item["validation"]["passed"] else 0
                for item in sample_results
            ),
            "source_npz_unchanged_count": sum(item["source_unchanged"] for item in sample_results),
            "maximum_absolute_regression_drift_cm": max(all_differences, default=0.0),
            "fallback_count": sum(
                item["diagnostics"]["fallback_count"] for item in sample_results
            ),
            "invalid_component_count": sum(
                item["diagnostics"]["invalid_component_count"] for item in sample_results
            ),
        },
        "primary_table_columns": {
            "height": "height_cm",
            "shoulder": "shoulder_acromion_cm",
            "chest": "chest_literature_cm",
            "waist": "waist_anatomical_cm",
            "hip": "hip_literature_cm",
        },
        "csv": str(csv_path),
        "scientific_boundary": (
            "These are raw canonical mesh measurements, not real-person ground truth."
        ),
        "phase_2_status": "closed" if passed else "open",
        "passed": passed,
    }
    report_path = args.output_dir / "unified_extractor_validation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("| Sample | Height | Shoulder | Chest | Waist | Hip |")
    print("|---|---:|---:|---:|---:|---:|")
    for item in sample_results:
        values = item["values_cm"]
        print(
            f"| {item['sample']} | {values['height_cm']:.2f} | "
            f"{values['shoulder_acromion_cm']:.2f} | {values['chest_literature_cm']:.2f} | "
            f"{values['waist_anatomical_cm']:.2f} | {values['hip_literature_cm']:.2f} |"
        )
        for failure in item["validation"]["failures"]:
            print(f"  - failure: {failure}")
    print(json.dumps(report["aggregate"], indent=2))
    print(f"Validation JSON: {report_path}")
    print(f"CSV: {csv_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
