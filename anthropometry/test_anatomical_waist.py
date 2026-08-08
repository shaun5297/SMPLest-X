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
    measure_anatomical_waist,
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


if __name__ == "__main__":
    unittest.main()
