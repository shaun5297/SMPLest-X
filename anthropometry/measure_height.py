#!/usr/bin/env python3
"""Measure raw body height from zero-pose canonical SMPL-X meshes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils import (
    infer_sample_name,
    load_canonical_mesh,
    measure_axis_ranges,
    verify_smplx_axes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate SMPL-X axes and compute raw zero-pose mesh height."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Canonical SMPL-X NPZ files")
    return parser.parse_args()


def measure_file(input_path: Path) -> dict[str, object]:
    input_path = input_path.resolve()
    vertices, joints = load_canonical_mesh(input_path)
    axis = measure_axis_ranges(vertices)
    verification = verify_smplx_axes(vertices, joints)
    raw_height_m = axis["y_range_m"]

    result = {
        "sample": infer_sample_name(input_path),
        "source_npz": str(input_path),
        "axis": axis,
        "axis_verification": verification,
        "raw_height_m": raw_height_m,
        "raw_height_cm": raw_height_m * 100.0,
        "method": "max(vertices[:, 1]) - min(vertices[:, 1])",
    }
    output_path = input_path.with_name(f"{input_path.stem.removesuffix('_canonical')}_height.json")
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["output_json"] = str(output_path)
    return result


def print_markdown_table(results: list[dict[str, object]]) -> None:
    print("| Sample | X range (m) | Y range (m) | Z range (m) | Raw height (cm) |")
    print("|---|---:|---:|---:|---:|")
    for result in results:
        axis = result["axis"]
        print(
            f"| {result['sample']} | "
            f"{axis['x_range_m']:.4f} | "
            f"{axis['y_range_m']:.4f} | "
            f"{axis['z_range_m']:.4f} | "
            f"{result['raw_height_cm']:.2f} |"
        )


def main() -> int:
    args = parse_args()
    results = [measure_file(path) for path in args.inputs]
    print_markdown_table(results)
    print("\nAxis semantics verified for every sample: X=left/right, Y=vertical, Z=front/back.")
    for result in results:
        print(f"{result['sample']}: {result['output_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
