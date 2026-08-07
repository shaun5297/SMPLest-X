#!/usr/bin/env python3
"""Compare SMPL-X shoulder-joint and acromion-surface proxy distances."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils import (
    ACROMION_SURFACE_PROXY_V1_LEFT_VERTEX_ID,
    ACROMION_SURFACE_PROXY_V1_RIGHT_VERTEX_ID,
    LEFT_SHOULDER_INDEX,
    PUBLISHED_LEFT_SHOULDER_SURFACE_VERTEX_ID,
    PUBLISHED_RIGHT_SHOULDER_SURFACE_VERTEX_ID,
    RIGHT_SHOULDER_INDEX,
    euclidean_distance,
    infer_sample_name,
    load_canonical_mesh,
    verify_smplx_axes,
)


LANDMARK_SOURCES = [
    {
        "name": "SMPL-Anthropometry",
        "url": "https://github.com/DavidBoja/SMPL-Anthropometry",
        "definition": "SMPL-X shoulder breadth uses LEFT_SHOULDER=4442 and RIGHT_SHOULDER=7218",
    },
    {
        "name": "A2B Human Mesh (CVPR 2025 workshop)",
        "url": "https://github.com/kaulquappe23/a2b_human_mesh",
        "definition": "SMPL-X shoulder width uses lshoulder=4442 and rshoulder=7218",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure shoulder width using native SMPL-X shoulder joints and "
            "literature-backed surface landmark candidates."
        )
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Canonical SMPL-X NPZ files")
    return parser.parse_args()


def coordinates(values) -> list[float]:
    return [float(value) for value in values]


def measure_file(input_path: Path) -> dict[str, object]:
    input_path = input_path.resolve()
    vertices, joints = load_canonical_mesh(input_path)
    verify_smplx_axes(vertices, joints)

    left_joint = joints[LEFT_SHOULDER_INDEX]
    right_joint = joints[RIGHT_SHOULDER_INDEX]
    left_literature_surface = vertices[PUBLISHED_LEFT_SHOULDER_SURFACE_VERTEX_ID]
    right_literature_surface = vertices[PUBLISHED_RIGHT_SHOULDER_SURFACE_VERTEX_ID]
    left_acromion_proxy = vertices[ACROMION_SURFACE_PROXY_V1_LEFT_VERTEX_ID]
    right_acromion_proxy = vertices[ACROMION_SURFACE_PROXY_V1_RIGHT_VERTEX_ID]
    joint_distance_m = euclidean_distance(left_joint, right_joint)
    literature_distance_m = euclidean_distance(
        left_literature_surface, right_literature_surface
    )
    acromion_proxy_distance_m = euclidean_distance(
        left_acromion_proxy, right_acromion_proxy
    )

    result = {
        "sample": infer_sample_name(input_path),
        "source_npz": str(input_path),
        "units": "meters",
        "landmark_definition": {
            "shoulder_joints": {
                "left_joint_index": LEFT_SHOULDER_INDEX,
                "right_joint_index": RIGHT_SHOULDER_INDEX,
                "definition": "native SMPL-X left_shoulder/right_shoulder joints",
            },
            "literature_shoulder_breadth": {
                "left_vertex_id": PUBLISHED_LEFT_SHOULDER_SURFACE_VERTEX_ID,
                "right_vertex_id": PUBLISHED_RIGHT_SHOULDER_SURFACE_VERTEX_ID,
                "status": "literature_baseline",
                "definition": (
                    "surface shoulder landmarks reused from two open-source "
                    "SMPL-X anthropometry implementations"
                ),
                "caveat": (
                    "the vertices are not bilateral mirror counterparts; retain for "
                    "literature compatibility, not as the bilateral acromion proxy"
                ),
                "sources": LANDMARK_SOURCES,
            },
            "acromion_surface_proxy_v1": {
                "left_vertex_id": ACROMION_SURFACE_PROXY_V1_LEFT_VERTEX_ID,
                "right_vertex_id": ACROMION_SURFACE_PROXY_V1_RIGHT_VERTEX_ID,
                "status": "frozen_v1",
                "definition": (
                    "bilaterally consistent external shoulder-surface proxy selected by "
                    "reflection consistency, shoulder-joint-relative geometry, local "
                    "surface orientation, five-shape stability, and manual MeshLab review"
                ),
                "caveat": (
                    "external-surface proxy only; not a medically observed bony acromion "
                    "ground-truth annotation"
                ),
            },
        },
        "landmark_coordinates_m": {
            "left_shoulder_joint": coordinates(left_joint),
            "right_shoulder_joint": coordinates(right_joint),
            "left_literature_shoulder_surface": coordinates(left_literature_surface),
            "right_literature_shoulder_surface": coordinates(right_literature_surface),
            "left_acromion_surface_proxy_v1": coordinates(left_acromion_proxy),
            "right_acromion_surface_proxy_v1": coordinates(right_acromion_proxy),
        },
        "measurements": {
            "shoulder_joint_distance_m": joint_distance_m,
            "shoulder_joint_distance_cm": joint_distance_m * 100.0,
            "literature_shoulder_breadth_m": literature_distance_m,
            "literature_shoulder_breadth_cm": literature_distance_m * 100.0,
            "acromion_surface_proxy_v1_distance_m": acromion_proxy_distance_m,
            "acromion_surface_proxy_v1_distance_cm": acromion_proxy_distance_m * 100.0,
            "acromion_proxy_minus_joint_cm": (
                acromion_proxy_distance_m - joint_distance_m
            )
            * 100.0,
            "acromion_proxy_minus_literature_cm": (
                acromion_proxy_distance_m - literature_distance_m
            )
            * 100.0,
        },
    }
    output_path = input_path.with_name(
        f"{input_path.stem.removesuffix('_canonical')}_shoulder_width.json"
    )
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["output_json"] = str(output_path)
    return result


def print_markdown_table(results: list[dict[str, object]]) -> None:
    print(
        "| Sample | Shoulder joints (cm) | Literature 4442/7218 (cm) | "
        "Frozen proxy 4482/7218 (cm) |"
    )
    print("|---|---:|---:|---:|")
    for result in results:
        measurements = result["measurements"]
        print(
            f"| {result['sample']} | "
            f"{measurements['shoulder_joint_distance_cm']:.2f} | "
            f"{measurements['literature_shoulder_breadth_cm']:.2f} | "
            f"{measurements['acromion_surface_proxy_v1_distance_cm']:.2f} |"
        )


def main() -> int:
    args = parse_args()
    results = [measure_file(path) for path in args.inputs]
    print_markdown_table(results)
    print(
        f"\nFrozen acromion surface proxy v1: "
        f"left={ACROMION_SURFACE_PROXY_V1_LEFT_VERTEX_ID}, "
        f"right={ACROMION_SURFACE_PROXY_V1_RIGHT_VERTEX_ID}"
    )
    for result in results:
        print(f"{result['sample']}: {result['output_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
