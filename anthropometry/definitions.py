#!/usr/bin/env python3
"""Single source of truth for frozen Phase 2 anthropometric definitions."""

from __future__ import annotations

import copy


DEFINITION_REGISTRY: dict[str, dict[str, object]] = {
    "raw_height_v0": {
        "measurement": "height",
        "version": "v0",
        "type": "axis_extent",
        "status": "baseline",
        "algorithm": "max(vertices[:,1]) - min(vertices[:,1])",
    },
    "shoulder_joint_width": {
        "measurement": "shoulder",
        "type": "joint_distance",
        "status": "control",
        "landmarks": {"left_joint": 16, "right_joint": 17},
        "algorithm": "Euclidean distance",
    },
    "literature_shoulder_breadth": {
        "measurement": "shoulder",
        "type": "surface_vertex_distance",
        "status": "literature_baseline",
        "landmarks": {"left_vertex": 4442, "right_vertex": 7218},
        "algorithm": "Euclidean distance",
        "sources": [
            "https://github.com/DavidBoja/SMPL-Anthropometry",
            "https://github.com/kaulquappe23/a2b_human_mesh",
        ],
    },
    "acromion_surface_proxy_v1": {
        "measurement": "shoulder",
        "version": "v1",
        "type": "surface_vertex_distance",
        "status": "frozen_v1",
        "landmarks": {"left_vertex": 4482, "right_vertex": 7218},
        "algorithm": "Euclidean distance",
        "scientific_boundary": "external surface proxy; not bony acromion ground truth",
    },
    "geometry_waist_v0": {
        "measurement": "waist",
        "version": "v0",
        "type": "geometry_extreme",
        "status": "baseline",
        "algorithm": "raw discrete minimum torso perimeter in pelvis-to-spine2 interval",
    },
    "literature_waist_v1": {
        "measurement": "waist",
        "version": "v1",
        "type": "bilateral_surface_landmark_plane",
        "status": "baseline",
        "landmarks": {"front_vertex": 5939, "back_vertex": 5941},
        "algorithm": "horizontal slice at mean landmark Y",
    },
    "anatomical_midpoint_waist_proxy_v1": {
        "measurement": "waist",
        "version": "v1",
        "type": "surface_anchor_plane",
        "status": "frozen_v1",
        "landmarks": [
            "left_lower_rib", "right_lower_rib", "left_iliac_crest", "right_iliac_crest"
        ],
        "algorithm": "horizontal slice at bilateral lower-rib/iliac-crest midpoint height",
        "scientific_boundary": "WHO-protocol-inspired proxy; not clinical or certified ground truth",
    },
    "geometry_hip_v0": {
        "measurement": "hip",
        "version": "v0",
        "type": "geometry_extreme",
        "status": "baseline",
        "algorithm": "raw discrete maximum perimeter in stable joined-pelvis interval",
    },
    "literature_hip_v1": {
        "measurement": "hip",
        "version": "v1",
        "type": "surface_landmark_plane",
        "status": "baseline",
        "landmarks": {"vertex": 5949},
        "algorithm": "horizontal slice at PUBIC_BONE landmark Y",
    },
    "literature_chest_v1": {
        "measurement": "chest",
        "version": "v1",
        "type": "bilateral_surface_landmark_plane",
        "status": "baseline",
        "landmarks": {"left_vertex": 3572, "right_vertex": 8340},
        "algorithm": "horizontal slice at mean nipple landmark Y",
        "sources": [
            "https://github.com/DavidBoja/SMPL-Anthropometry",
            "https://github.com/kaulquappe23/a2b_human_mesh",
        ],
    },
    "focused_shapy_chest_control": {
        "measurement": "chest",
        "type": "surface_anchor_plane",
        "status": "control",
        "landmarks": {"face_id": 18402, "barycentric": [0.0, 0.0, 1.0]},
        "algorithm": "horizontal slice at single NippleRight surface-anchor Y",
    },
    "geometry_chest_extreme": {
        "measurement": "chest",
        "type": "geometry_extreme",
        "status": "rejected_as_measurement_definition",
        "algorithm": "maximum perimeter in skeleton-bounded thoracic diagnostic interval",
        "rejection_reason": (
            "clean torso C(y) increases toward arm-torso topology merging; raw maxima "
            "are dominated by merged upper limbs"
        ),
    },
}


PRIMARY_TARGETS = {
    "height": "raw_height_v0",
    "shoulder": "acromion_surface_proxy_v1",
    "waist": "anatomical_midpoint_waist_proxy_v1",
    "hip": "literature_hip_v1",
    "chest": "literature_chest_v1",
}


def get_definition_registry() -> dict[str, dict[str, object]]:
    """Return a defensive copy so callers cannot mutate the registry."""
    return copy.deepcopy(DEFINITION_REGISTRY)
