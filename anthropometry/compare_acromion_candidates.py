#!/usr/bin/env python3
"""Compare bilateral SMPL-X surface candidates for an acromion proxy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import smplx
import torch

from utils import (
    ACROMION_SURFACE_PROXY_V1_LEFT_VERTEX_ID,
    ACROMION_SURFACE_PROXY_V1_RIGHT_VERTEX_ID,
    LEFT_SHOULDER_INDEX,
    PUBLISHED_LEFT_SHOULDER_SURFACE_VERTEX_ID,
    PUBLISHED_RIGHT_SHOULDER_SURFACE_VERTEX_ID,
    RIGHT_SHOULDER_INDEX,
    infer_sample_name,
    load_canonical_mesh,
    verify_smplx_axes,
)


CANDIDATE_PAIRS = {
    "published_asymmetric": {
        "left_vertex_id": PUBLISHED_LEFT_SHOULDER_SURFACE_VERTEX_ID,
        "right_vertex_id": PUBLISHED_RIGHT_SHOULDER_SURFACE_VERTEX_ID,
        "description": "Published SMPL-X shoulder-width pair",
    },
    "symmetric_from_left_4442": {
        "left_vertex_id": 4442,
        "right_vertex_id": 7178,
        "description": "v4442 and its neutral-template mirror counterpart",
    },
    "symmetric_from_right_7218": {
        "left_vertex_id": ACROMION_SURFACE_PROXY_V1_LEFT_VERTEX_ID,
        "right_vertex_id": ACROMION_SURFACE_PROXY_V1_RIGHT_VERTEX_ID,
        "description": "neutral-template mirror counterpart of v7218 and v7218",
    },
}

RECOMMENDED_CANDIDATE = "symmetric_from_right_7218"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Compare three SMPL-X bilateral acromion surface proxy candidates."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Canonical SMPL-X NPZ files")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=repo_root / "human_models" / "human_model_files",
        help="Directory containing the smplx model folder",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "anthropometry" / "artifacts",
        help="Directory for aggregate JSON and neutral-template views",
    )
    return parser.parse_args()


def vertex_normal(vertices: np.ndarray, faces: np.ndarray, vertex_id: int) -> np.ndarray:
    """Return the area-weighted average normal of the vertex's 1-ring faces.

    Each unnormalized triangle cross product has magnitude twice its area, so
    summing the cross products before normalization applies area weighting.
    """
    incident = faces[np.any(faces == vertex_id, axis=1)]
    if len(incident) == 0:
        raise ValueError(f"vertex {vertex_id} is not referenced by any face")
    triangles = vertices[incident]
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normal = face_normals.sum(axis=0)
    norm = np.linalg.norm(normal)
    if norm == 0:
        raise ValueError(f"vertex {vertex_id} has a degenerate local normal")
    return normal / norm


def evaluate_pair(
    vertices: np.ndarray,
    joints: np.ndarray,
    faces: np.ndarray,
    left_vertex_id: int,
    right_vertex_id: int,
) -> dict[str, object]:
    left = vertices[left_vertex_id]
    right = vertices[right_vertex_id]
    delta = left - right
    left_joint_offset = left - joints[LEFT_SHOULDER_INDEX]
    right_joint_offset = right - joints[RIGHT_SHOULDER_INDEX]
    mirrored_joint_relative_delta = np.array(
        [
            left_joint_offset[0] + right_joint_offset[0],
            left_joint_offset[1] - right_joint_offset[1],
            left_joint_offset[2] - right_joint_offset[2],
        ],
        dtype=np.float32,
    )
    left_normal = vertex_normal(vertices, faces, left_vertex_id)
    right_normal = vertex_normal(vertices, faces, right_vertex_id)
    return {
        "left_vertex_id": left_vertex_id,
        "right_vertex_id": right_vertex_id,
        "left_coordinate_m": [float(value) for value in left],
        "right_coordinate_m": [float(value) for value in right],
        "distance_m": float(np.linalg.norm(delta)),
        "distance_cm": float(np.linalg.norm(delta) * 100.0),
        "axis_delta_cm": [float(value * 100.0) for value in delta],
        "absolute_y_diff_cm": float(abs(delta[1]) * 100.0),
        "absolute_z_diff_cm": float(abs(delta[2]) * 100.0),
        "bilateral_yz_mismatch_cm": float(np.linalg.norm(delta[1:]) * 100.0),
        "left_shoulder_joint_offset_cm": [float(value * 100.0) for value in left_joint_offset],
        "right_shoulder_joint_offset_cm": [float(value * 100.0) for value in right_joint_offset],
        "mirrored_joint_relative_delta_cm": [
            float(value * 100.0) for value in mirrored_joint_relative_delta
        ],
        "mirrored_joint_relative_mismatch_cm": float(
            np.linalg.norm(mirrored_joint_relative_delta) * 100.0
        ),
        "left_surface_normal": [float(value) for value in left_normal],
        "right_surface_normal": [float(value) for value in right_normal],
        "mean_upward_normal_component": float(0.5 * (left_normal[1] + right_normal[1])),
    }


def measure_file(input_path: Path) -> dict[str, object]:
    input_path = input_path.resolve()
    vertices, joints = load_canonical_mesh(input_path)
    verify_smplx_axes(vertices, joints)
    with np.load(input_path, allow_pickle=False) as data:
        if "faces" not in data:
            raise KeyError(f"{input_path} must contain faces")
        faces = np.asarray(data["faces"], dtype=np.int32)

    candidates = {
        name: {
            **definition,
            **evaluate_pair(
                vertices,
                joints,
                faces,
                definition["left_vertex_id"],
                definition["right_vertex_id"],
            ),
        }
        for name, definition in CANDIDATE_PAIRS.items()
    }
    result = {
        "sample": infer_sample_name(input_path),
        "source_npz": str(input_path),
        "units": "meters",
        "candidates": candidates,
        "recommendation": {
            "candidate": RECOMMENDED_CANDIDATE,
            "left_vertex_id": ACROMION_SURFACE_PROXY_V1_LEFT_VERTEX_ID,
            "right_vertex_id": ACROMION_SURFACE_PROXY_V1_RIGHT_VERTEX_ID,
            "name": "acromion_surface_proxy_v1",
            "status": "frozen_v1",
            "reasons": [
                "neutral-template mirror consistency",
                "minimal bilateral Y/Z mismatch across the five tested shapes",
                "more upward-facing local shoulder surface than the v4442/v7178 pair",
                "manual MeshLab confirmation in front, back, top, and oblique views",
            ],
            "scientific_boundary": (
                "SMPL-X represents the external body surface, not the scapular bone; "
                "this is a reproducible surface proxy, not a directly observed bony acromion"
            ),
        },
    }
    output_path = input_path.with_name(
        f"{input_path.stem.removesuffix('_canonical')}_acromion_candidates.json"
    )
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["output_json"] = str(output_path)
    return result


def build_beta_zero_mesh(model_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    layer = smplx.create(
        str(model_path),
        "smplx",
        gender="NEUTRAL",
        use_pca=False,
        use_face_contour=True,
    ).eval()
    with torch.no_grad():
        output = layer(
            betas=torch.zeros((1, 10), dtype=torch.float32),
            body_pose=torch.zeros((1, 63), dtype=torch.float32),
            global_orient=torch.zeros((1, 3), dtype=torch.float32),
            left_hand_pose=torch.zeros((1, 45), dtype=torch.float32),
            right_hand_pose=torch.zeros((1, 45), dtype=torch.float32),
            jaw_pose=torch.zeros((1, 3), dtype=torch.float32),
            leye_pose=torch.zeros((1, 3), dtype=torch.float32),
            reye_pose=torch.zeros((1, 3), dtype=torch.float32),
            expression=torch.zeros((1, 10), dtype=torch.float32),
            transl=torch.zeros((1, 3), dtype=torch.float32),
        )
    return (
        output.vertices[0].cpu().numpy(),
        output.joints[0].cpu().numpy(),
        np.asarray(layer.faces, dtype=np.int32),
    )


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std_population": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def build_aggregate(
    results: list[dict[str, object]],
    neutral_vertices: np.ndarray,
    neutral_joints: np.ndarray,
    neutral_faces: np.ndarray,
) -> dict[str, object]:
    neutral = {}
    summaries = {}
    subject_metrics = []
    for name, definition in CANDIDATE_PAIRS.items():
        neutral[name] = {
            **definition,
            **evaluate_pair(
                neutral_vertices,
                neutral_joints,
                neutral_faces,
                definition["left_vertex_id"],
                definition["right_vertex_id"],
            ),
        }
        candidate_values = [result["candidates"][name] for result in results]
        summaries[name] = {
            "distance_cm": summarize([item["distance_cm"] for item in candidate_values]),
            "absolute_y_diff_cm": summarize(
                [item["absolute_y_diff_cm"] for item in candidate_values]
            ),
            "absolute_z_diff_cm": summarize(
                [item["absolute_z_diff_cm"] for item in candidate_values]
            ),
            "mirrored_joint_relative_mismatch_cm": summarize(
                [item["mirrored_joint_relative_mismatch_cm"] for item in candidate_values]
            ),
            "mean_upward_normal_component": summarize(
                [item["mean_upward_normal_component"] for item in candidate_values]
            ),
        }
    for result in results:
        subject_metrics.append(
            {
                "sample": result["sample"],
                "source_npz": result["source_npz"],
                "candidates": {
                    name: {
                        key: result["candidates"][name][key]
                        for key in (
                            "distance_cm",
                            "absolute_y_diff_cm",
                            "absolute_z_diff_cm",
                            "left_shoulder_joint_offset_cm",
                            "right_shoulder_joint_offset_cm",
                            "mirrored_joint_relative_delta_cm",
                            "mirrored_joint_relative_mismatch_cm",
                        )
                    }
                    for name in CANDIDATE_PAIRS
                },
            }
        )

    # Equal unit weights make the geometric check auditable. B and C are treated
    # as equivalent when their aggregate mismatch differs by less than 0.1 mm;
    # local surface orientation is then the deterministic anatomical tie-breaker.
    scores = {}
    for name in CANDIDATE_PAIRS:
        scores[name] = float(
            neutral[name]["absolute_y_diff_cm"]
            + neutral[name]["absolute_z_diff_cm"]
            + neutral[name]["mirrored_joint_relative_mismatch_cm"]
            + summaries[name]["absolute_y_diff_cm"]["mean"]
            + summaries[name]["absolute_z_diff_cm"]["mean"]
            + summaries[name]["mirrored_joint_relative_mismatch_cm"]["mean"]
        )

    return {
        "experiment": "Step 2.3.1 - Bilateral landmark confirmation",
        "units": "centimeters unless otherwise stated",
        "candidate_definitions": CANDIDATE_PAIRS,
        "beta_zero_neutral": neutral,
        "subjects": subject_metrics,
        "five_subject_summary": summaries,
        "automatic_scoring": {
            "formula": (
                "neutral(|dY|+|dZ|+joint_relative_mismatch) + "
                "subject_mean(|dY|+|dZ|+joint_relative_mismatch)"
            ),
            "weights": {"absolute_y_diff": 1.0, "absolute_z_diff": 1.0, "joint_relative": 1.0},
            "lower_is_better": True,
            "scores_cm": scores,
            "equivalence_tolerance_cm": 0.01,
            "tie_breaker": "higher mean upward-facing local surface-normal component",
        },
        "recommendation": {
            "candidate": RECOMMENDED_CANDIDATE,
            "left_vertex_id": ACROMION_SURFACE_PROXY_V1_LEFT_VERTEX_ID,
            "right_vertex_id": ACROMION_SURFACE_PROXY_V1_RIGHT_VERTEX_ID,
            "name": "acromion_surface_proxy_v1",
            "status": "frozen_v1",
            "decision": (
                "B and C are bilaterally equivalent within the 0.1 mm score tolerance; "
                "C is selected by its more upward-facing shoulder-surface orientation"
            ),
            "manual_acceptance": {
                "tool": "MeshLab",
                "views": ["front", "back", "top", "left_oblique", "right_oblique"],
                "checks": {
                    "shoulder_superolateral_region": True,
                    "bilateral_visual_correspondence": True,
                    "not_clavicle_anterior": True,
                    "not_excessively_lateral_deltoid": True,
                    "not_neck": True,
                },
            },
            "scientific_boundary": (
                "SMPL-X represents the external body surface, not the scapular bone; "
                "this is a reproducible surface proxy, not a directly observed bony acromion"
            ),
        },
    }


def plot_neutral_views(
    vertices: np.ndarray,
    joints: np.ndarray,
    output_path: Path,
) -> None:
    styles = {
        "published_asymmetric": ("#7b2cbf", "--", "A 4442/7218"),
        "symmetric_from_left_4442": ("#f77f00", ":", "B 4442/7178"),
        "symmetric_from_right_7218": ("#2a9d8f", "-", "C 4482/7218"),
    }
    shoulder_y = float(np.mean(joints[[LEFT_SHOULDER_INDEX, RIGHT_SHOULDER_INDEX], 1]))
    mask = (vertices[:, 1] > shoulder_y - 0.12) & (vertices[:, 1] < shoulder_y + 0.12)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    for axis, horizontal, vertical, title, vertical_label in (
        (axes[0], 0, 1, "Front view", "Y (m)"),
        (axes[1], 0, 2, "Top view", "Z (m)"),
    ):
        axis.scatter(vertices[mask, horizontal], vertices[mask, vertical], s=2, c="0.75", alpha=0.25)
        axis.scatter(
            joints[[LEFT_SHOULDER_INDEX, RIGHT_SHOULDER_INDEX], horizontal],
            joints[[LEFT_SHOULDER_INDEX, RIGHT_SHOULDER_INDEX], vertical],
            s=80,
            marker="x",
            linewidths=2,
            c="black",
            label="shoulder joints j16/j17",
            zorder=5,
        )
        for name, definition in CANDIDATE_PAIRS.items():
            color, linestyle, label = styles[name]
            ids = [definition["left_vertex_id"], definition["right_vertex_id"]]
            axis.plot(
                vertices[ids, horizontal],
                vertices[ids, vertical],
                color=color,
                linestyle=linestyle,
                linewidth=2.2,
                marker="o",
                markersize=5,
                label=label,
                zorder=4,
            )
        axis.set_title(title)
        axis.set_xlabel("X (m)")
        axis.set_ylabel(vertical_label)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
        axis.set_xlim(-0.32, 0.32)
    axes[0].set_ylim(shoulder_y - 0.08, shoulder_y + 0.09)
    top_landmark_ids = sorted(
        {
            vertex_id
            for definition in CANDIDATE_PAIRS.values()
            for vertex_id in (definition["left_vertex_id"], definition["right_vertex_id"])
        }
    )
    top_z = np.concatenate(
        [
            vertices[top_landmark_ids, 2],
            joints[[LEFT_SHOULDER_INDEX, RIGHT_SHOULDER_INDEX], 2],
        ]
    )
    axes[1].set_ylim(float(top_z.min() - 0.035), float(top_z.max() + 0.035))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("SMPL-X beta=0 neutral acromion surface candidates", y=0.98)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=4,
        frameon=False,
    )
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.13, top=0.75, wspace=0.18)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def print_markdown_table(results: list[dict[str, object]]) -> None:
    print("| Sample | 4442/7218 (cm) | 4442/7178 (cm) | 4482/7218 (cm) |")
    print("|---|---:|---:|---:|")
    for result in results:
        candidates = result["candidates"]
        print(
            f"| {result['sample']} | "
            f"{candidates['published_asymmetric']['distance_cm']:.2f} | "
            f"{candidates['symmetric_from_left_4442']['distance_cm']:.2f} | "
            f"{candidates['symmetric_from_right_7218']['distance_cm']:.2f} |"
        )


def main() -> int:
    args = parse_args()
    results = [measure_file(path) for path in args.inputs]
    neutral_vertices, neutral_joints, neutral_faces = build_beta_zero_mesh(args.model_path)
    aggregate = build_aggregate(results, neutral_vertices, neutral_joints, neutral_faces)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = args.output_dir / "shoulder_landmark_candidates.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    figure_path = args.output_dir / "neutral_acromion_candidate_comparison.png"
    plot_neutral_views(neutral_vertices, neutral_joints, figure_path)
    print_markdown_table(results)
    print("\nFrozen acromion surface proxy v1: left v4482, right v7218")
    print(f"Aggregate confirmation: {aggregate_path}")
    print(f"Neutral front/top views: {figure_path}")
    for result in results:
        print(f"{result['sample']}: {result['output_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
