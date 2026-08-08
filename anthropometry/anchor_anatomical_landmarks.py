#!/usr/bin/env python3
"""Project fixed anatomical XYZ landmarks onto gendered SMPL-X templates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import smplx
import torch
import trimesh

from anatomical_waist import WAIST_LANDMARK_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert fixed gender-template waist XYZ landmarks to topology-attached "
            "face IDs and barycentric coordinates. Original files are not modified."
        )
    )
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--fixed-landmark-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--genders", nargs="+", default=["neutral", "male", "female"],
        choices=["neutral", "male", "female"],
    )
    return parser.parse_args()


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def build_zero_beta_template(model_dir: Path, gender: str) -> tuple[np.ndarray, np.ndarray]:
    layer = smplx.create(
        str(model_dir),
        model_type="smplx",
        gender=gender,
        ext="npz",
        use_pca=False,
        flat_hand_mean=True,
    ).eval()
    with torch.no_grad():
        output = layer(
            betas=torch.zeros((1, 10), dtype=torch.float32),
            global_orient=torch.zeros((1, 3), dtype=torch.float32),
            body_pose=torch.zeros((1, 63), dtype=torch.float32),
            left_hand_pose=torch.zeros((1, 45), dtype=torch.float32),
            right_hand_pose=torch.zeros((1, 45), dtype=torch.float32),
            jaw_pose=torch.zeros((1, 3), dtype=torch.float32),
            leye_pose=torch.zeros((1, 3), dtype=torch.float32),
            reye_pose=torch.zeros((1, 3), dtype=torch.float32),
            expression=torch.zeros((1, 10), dtype=torch.float32),
            transl=torch.zeros((1, 3), dtype=torch.float32),
        )
    vertices = output.vertices[0].cpu().numpy().astype(np.float64, copy=False)
    faces = np.asarray(layer.faces, dtype=np.int64)
    return vertices, faces


def anchor_points(
    vertices: np.ndarray,
    faces: np.ndarray,
    fixed_points: dict[str, np.ndarray],
) -> dict[str, object]:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    query = np.asarray([fixed_points[name] for name in WAIST_LANDMARK_NAMES])
    closest, distances, triangle_ids = trimesh.proximity.closest_point_naive(mesh, query)
    anchors = {}
    for index, name in enumerate(WAIST_LANDMARK_NAMES):
        face_id = int(triangle_ids[index])
        vertex_ids = faces[face_id]
        triangle = vertices[vertex_ids][None, :, :]
        barycentric = trimesh.triangles.points_to_barycentric(
            triangle,
            closest[index][None, :],
        )[0]
        reconstructed = barycentric @ vertices[vertex_ids]
        anchors[name] = {
            "source_xyz_m": fixed_points[name].tolist(),
            "face_id": face_id,
            "vertex_ids": vertex_ids.tolist(),
            "barycentric": barycentric.tolist(),
            "projected_template_xyz_m": reconstructed.tolist(),
            "projection_distance_mm": float(distances[index] * 1000.0),
            "barycentric_sum": float(barycentric.sum()),
            "reconstruction_error_mm": float(
                np.linalg.norm(reconstructed - closest[index]) * 1000.0
            ),
        }
    return anchors


def plot_projection(
    gender: str,
    vertices: np.ndarray,
    fixed_points: dict[str, np.ndarray],
    anchors: dict[str, object],
    output_path: Path,
) -> None:
    projected = np.asarray(
        [anchors[name]["projected_template_xyz_m"] for name in WAIST_LANDMARK_NAMES],
        dtype=np.float64,
    )
    fixed = np.asarray([fixed_points[name] for name in WAIST_LANDMARK_NAMES])
    labels = ["L rib", "R rib", "L iliac", "R iliac"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    for axis, horizontal_index, horizontal_label, title in (
        (axes[0], 0, "X (m)", "Front view"),
        (axes[1], 2, "Z (m)", "Sagittal view"),
    ):
        axis.scatter(
            vertices[:, horizontal_index], vertices[:, 1], s=0.45,
            color="0.80", alpha=0.22, rasterized=True,
        )
        axis.scatter(
            fixed[:, horizontal_index], fixed[:, 1], s=72, marker="o",
            facecolors="none", edgecolors="0.25", linewidth=1.5,
            label="manual fixed XYZ",
        )
        axis.scatter(
            projected[:, horizontal_index], projected[:, 1], s=72, marker="x",
            color="#2ca02c", linewidths=2.0, label="surface projection",
        )
        for index, label in enumerate(labels):
            axis.annotate(
                label,
                (projected[index, horizontal_index], projected[index, 1]),
                xytext=(5, 5), textcoords="offset points", fontsize=8,
            )
        axis.set_xlabel(horizontal_label)
        axis.set_ylabel("Y (m)")
        axis.set_title(title)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle(f"{gender} zero-beta SMPL-X anatomical landmark surface projection")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def convert_gender(
    gender: str,
    model_dir: Path,
    fixed_landmark_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    fixed_path = fixed_landmark_dir / f"landmarks_{gender}.json"
    fixed = json.loads(fixed_path.read_text(encoding="utf-8"))
    fixed_points = {
        name: np.asarray(fixed[name], dtype=np.float64) for name in WAIST_LANDMARK_NAMES
    }
    vertices, faces = build_zero_beta_template(model_dir, gender)
    anchors = anchor_points(vertices, faces, fixed_points)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"landmarks_{gender}_surface_projection.png"
    plot_projection(gender, vertices, fixed_points, anchors, plot_path)
    model_file = model_dir / f"SMPLX_{gender.upper()}.npz"
    payload = {
        "schema": "smplx_surface_landmarks_v1",
        "definition": "anatomical_midpoint_waist_proxy_v1",
        "gender": gender,
        "source_fixed_landmarks": str(fixed_path.resolve()),
        "source_landmark_type": fixed.get("landmark_type", "fixed_xyz_per_gender"),
        "template": {
            "model_file": str(model_file.resolve()),
            "model_gender": gender,
            "betas": "zero",
            "pose": "zero",
            "expression": "zero",
            "translation": "zero",
            "vertex_count": int(len(vertices)),
            "face_count": int(len(faces)),
            "vertices_sha256_float64": array_sha256(vertices),
            "faces_sha256_int64": array_sha256(faces),
        },
        "projection": {
            "method": "closest_point_on_triangle_surface",
            "library": f"trimesh {trimesh.__version__}",
            "maximum_projection_distance_mm": max(
                anchor["projection_distance_mm"] for anchor in anchors.values()
            ),
        },
        "debug_plot": str(plot_path.resolve()),
        "anchors": anchors,
    }
    output_path = output_dir / f"landmarks_{gender}_surface.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {"path": str(output_path.resolve()), **payload}


def main() -> int:
    args = parse_args()
    results = [
        convert_gender(
            gender,
            args.model_dir.resolve(),
            args.fixed_landmark_dir.resolve(),
            args.output_dir.resolve(),
        )
        for gender in args.genders
    ]
    print("| Gender | Max projection (mm) | Face IDs | Output |")
    print("|---|---:|---|---|")
    for result in results:
        face_ids = [str(result["anchors"][name]["face_id"]) for name in WAIST_LANDMARK_NAMES]
        print(
            f"| {result['gender']} | "
            f"{result['projection']['maximum_projection_distance_mm']:.6f} | "
            f"{', '.join(face_ids)} | {result['path']} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
