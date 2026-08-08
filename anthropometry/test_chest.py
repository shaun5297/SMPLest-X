#!/usr/bin/env python3
"""Synthetic tests for the Step 2.7B chest definitions and selector."""

from __future__ import annotations

import unittest

import numpy as np

from chest import (
    FOCUSED_SHAPY_CHEST_CONTROL,
    LITERATURE_CHEST_CANDIDATE,
    compute_candidate_planes,
    select_chest_torso_contour,
    thoracic_centerline_xz,
)


def square(contour_id: int, x: float, z: float, half: float, y: float = 0.75):
    xz = np.asarray([[x-half,z-half],[x+half,z-half],[x+half,z+half],[x-half,z+half]])
    points = np.column_stack([xz[:, 0], np.full(4, y), xz[:, 1]])
    return {
        "id": contour_id,
        "ordered_points_m": points.tolist(),
        "centroid_xz_m": [x, z],
        "area_m2": float((2*half)**2),
        "perimeter_m": float(8*half),
    }


def joints():
    value = np.zeros((22, 3), dtype=np.float64)
    value[0] = [0.0, 0.40, 0.02]
    value[3] = [0.0, 0.52, 0.01]
    value[6] = [0.0, 0.64, 0.00]
    value[9] = [0.0, 0.72, -0.01]
    value[16] = [0.18, 0.82, -0.02]
    value[17] = [-0.18, 0.82, -0.02]
    return value


class ChestTests(unittest.TestCase):
    def test_public_planes_are_independent(self):
        vertices = np.zeros((9000, 3), dtype=np.float64)
        vertices[3572] = [0.1, 0.70, 0.1]
        vertices[8340] = [-0.1, 0.72, 0.1]
        faces = np.zeros((18403, 3), dtype=np.int64)
        faces[18402] = [10, 11, 12]
        vertices[12] = [-0.1, 0.735, 0.1]
        definitions = compute_candidate_planes(vertices, faces)
        self.assertAlmostEqual(definitions[LITERATURE_CHEST_CANDIDATE]["plane_y_m"], 0.71)
        self.assertAlmostEqual(definitions[FOCUSED_SHAPY_CHEST_CONTROL]["plane_y_m"], 0.735)

    def test_centerline_containment_beats_larger_side_loop(self):
        central = square(7, 0.0, -0.01, 0.10)
        arm = square(2, 0.50, 0.0, 0.20)
        result = select_chest_torso_contour([arm, central], joints(), 0.70)
        self.assertEqual(result["selected_contour_id"], 7)
        self.assertEqual(result["selection_mode"], "spine_centerline_containment_then_area")
        self.assertFalse(result["fallback_used"])

    def test_centerline_extrapolates_without_moving_plane(self):
        centerline, mode = thoracic_centerline_xz(joints(), 0.76)
        self.assertEqual(mode, "spine2_to_spine3_linear_extrapolation")
        np.testing.assert_allclose(centerline, [0.0, -0.015], atol=1e-12)

    def test_possible_arm_merge_is_flagged_by_x_span(self):
        merged = square(0, 0.0, -0.01, 0.30)
        result = select_chest_torso_contour([merged], joints(), 0.70)
        self.assertTrue(result["possible_arm_torso_merge"])


if __name__ == "__main__":
    unittest.main()
