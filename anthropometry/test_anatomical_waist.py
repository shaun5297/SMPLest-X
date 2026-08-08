#!/usr/bin/env python3
"""Synthetic regression tests for ``anatomical_waist_proxy_v1``."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from anatomical_waist import (
    compute_anatomical_waist_plane,
    load_anatomical_landmarks,
    load_surface_anchored_landmarks,
    measure_anatomical_waist,
    measure_surface_anchored_anatomical_waist,
    normalize_gender_label,
)
from test_waist import synthetic_hourglass_mesh, synthetic_joints


def landmark_payload() -> dict[str, object]:
    return {
        "landmark_type": "fixed_xyz_per_gender",
        "left_lower_rib": [0.3, 0.72, 0.0],
        "right_lower_rib": [-0.3, 0.68, 0.0],
        "left_iliac_crest": [0.3, 0.32, 0.0],
        "right_iliac_crest": [-0.3, 0.28, 0.0],
    }


class AnatomicalWaistTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.landmark_path = Path(self.temporary.name) / "landmarks_female.json"
        self.landmark_path.write_text(json.dumps(landmark_payload()), encoding="utf-8")
        vertices, faces = synthetic_hourglass_mesh()
        face_id = 128
        vertex_ids = faces[face_id].tolist()
        barycentric = [0.5, 0.25, 0.25]
        point = np.asarray(barycentric) @ vertices[faces[face_id]]
        anchor = {
            "source_xyz_m": point.tolist(),
            "face_id": face_id,
            "vertex_ids": vertex_ids,
            "barycentric": barycentric,
            "projected_template_xyz_m": point.tolist(),
            "projection_distance_mm": 0.0,
        }
        self.anchor_path = Path(self.temporary.name) / "landmarks_female_surface.json"
        self.anchor_path.write_text(
            json.dumps(
                {
                    "schema": "smplx_surface_landmarks_v1",
                    "gender": "female",
                    "template": {"vertex_count": len(vertices), "face_count": len(faces)},
                    "projection": {"method": "synthetic"},
                    "anchors": {
                        name: anchor
                        for name in (
                            "left_lower_rib",
                            "right_lower_rib",
                            "left_iliac_crest",
                            "right_iliac_crest",
                        )
                    },
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_project_gender_mapping_is_explicit(self) -> None:
        self.assertEqual(normalize_gender_label(0), "female")
        self.assertEqual(normalize_gender_label("1"), "male")
        with self.assertRaisesRegex(ValueError, "unsupported gender label"):
            normalize_gender_label("unknown")

    def test_plane_is_bilateral_mean_of_rib_iliac_midpoints(self) -> None:
        landmarks = load_anatomical_landmarks(self.landmark_path)
        plane = compute_anatomical_waist_plane(landmarks)
        self.assertAlmostEqual(plane["plane_y_m"], 0.5)
        self.assertAlmostEqual(plane["left_midpoint_m"][1], 0.52)
        self.assertAlmostEqual(plane["right_midpoint_m"][1], 0.48)
        self.assertAlmostEqual(plane["left_right_midpoint_y_mismatch_mm"], 40.0)

    def test_end_to_end_performs_one_direct_slice(self) -> None:
        vertices, faces = synthetic_hourglass_mesh()
        result = measure_anatomical_waist(
            vertices,
            faces,
            synthetic_joints(),
            gender="female",
            landmark_path=self.landmark_path,
        )
        self.assertEqual(result["definition"], "anatomical_waist_proxy_v1")
        self.assertTrue(result["independent_of_circumference_minimum"])
        self.assertEqual(
            result["search_or_scan"],
            "none; one direct anatomical-landmark-defined slice",
        )
        self.assertAlmostEqual(result["plane_definition"]["plane_y_m"], 0.5)
        self.assertFalse(result["fallback_used"])
        self.assertTrue(result["centerline_inside"])

    def test_plane_is_independent_of_mesh_circumference(self) -> None:
        landmarks = load_anatomical_landmarks(self.landmark_path)
        first = compute_anatomical_waist_plane(landmarks)
        vertices, _ = synthetic_hourglass_mesh()
        vertices[:, [0, 2]] *= 2.0
        second = compute_anatomical_waist_plane(landmarks)
        self.assertEqual(first["plane_y_m"], second["plane_y_m"])

    def test_surface_anchors_evaluate_on_subject_vertices(self) -> None:
        vertices, faces = synthetic_hourglass_mesh()
        landmarks = load_surface_anchored_landmarks(vertices, faces, self.anchor_path)
        first = landmarks["points"]["left_lower_rib"]
        self.assertAlmostEqual(first[1], 0.5)
        altered = vertices.copy()
        altered[:, 0] *= 1.5
        moved = load_surface_anchored_landmarks(altered, faces, self.anchor_path)
        self.assertNotEqual(first[0], moved["points"]["left_lower_rib"][0])

    def test_surface_anchored_measurement_is_single_slice(self) -> None:
        vertices, faces = synthetic_hourglass_mesh()
        result = measure_surface_anchored_anatomical_waist(
            vertices,
            faces,
            synthetic_joints(),
            gender="female",
            anchor_path=self.anchor_path,
        )
        self.assertEqual(result["definition"], "anatomical_midpoint_waist_proxy_v1")
        self.assertEqual(result["status"], "frozen_v1")
        self.assertTrue(result["landmark_source"]["surface_anchored"])
        self.assertAlmostEqual(result["plane_definition"]["plane_y_m"], 0.5)

    def test_surface_anchor_topology_mismatch_is_rejected(self) -> None:
        vertices, faces = synthetic_hourglass_mesh()
        mismatched = faces.copy()
        mismatched[128] = mismatched[129]
        with self.assertRaisesRegex(ValueError, "do not match the mesh topology"):
            load_surface_anchored_landmarks(vertices, mismatched, self.anchor_path)


if __name__ == "__main__":
    unittest.main()
