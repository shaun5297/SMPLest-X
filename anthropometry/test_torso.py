#!/usr/bin/env python3
"""Synthetic regression tests for the Step 2.5A torso selector."""

from __future__ import annotations

import unittest

import numpy as np

from torso import (
    TorsoSelectionError,
    compute_torso_vertical_interval,
    distance_to_polygon_boundary_xz,
    interpolate_spine_centerline_xz,
    point_in_polygon_xz,
    select_torso_contour,
)


def square_contour(
    contour_id: int,
    center_x: float,
    center_z: float,
    half_size: float,
    plane_y: float = 0.5,
) -> dict[str, object]:
    xz = np.asarray(
        [
            [center_x - half_size, center_z - half_size],
            [center_x + half_size, center_z - half_size],
            [center_x + half_size, center_z + half_size],
            [center_x - half_size, center_z + half_size],
        ],
        dtype=np.float64,
    )
    points = np.column_stack([xz[:, 0], np.full(4, plane_y), xz[:, 1]])
    return {
        "id": contour_id,
        "ordered_points_m": points.tolist(),
        "centroid_xz_m": [center_x, center_z],
        "area_m2": (2.0 * half_size) ** 2,
        "perimeter_m": 8.0 * half_size,
    }


def synthetic_joints() -> np.ndarray:
    joints = np.zeros((22, 3), dtype=np.float64)
    joints[0] = [0.0, 0.50, 0.02]
    joints[3] = [0.01, 0.60, 0.01]
    joints[6] = [0.02, 0.72, 0.00]
    joints[9] = [0.03, 0.84, -0.01]
    return joints


class TorsoSelectionTests(unittest.TestCase):
    def test_skeleton_interval_uses_pelvis_and_spine2(self) -> None:
        interval = compute_torso_vertical_interval(synthetic_joints())
        self.assertEqual(interval["lower_bound"]["joint"], "pelvis")
        self.assertEqual(interval["upper_bound"]["joint"], "spine2")
        self.assertAlmostEqual(interval["y_min_m"], 0.50)
        self.assertAlmostEqual(interval["y_max_m"], 0.72)

    def test_piecewise_spine_centerline_interpolation(self) -> None:
        centerline = interpolate_spine_centerline_xz(synthetic_joints(), 0.66)
        np.testing.assert_allclose(centerline, [0.015, 0.005], atol=1e-12)

    def test_centerline_containment_beats_larger_side_contour(self) -> None:
        central = square_contour(7, 0.0, 0.0, 0.10)
        larger_side = square_contour(2, 0.45, 0.0, 0.20)
        selection = select_torso_contour([larger_side, central], np.asarray([0.0, 0.0]))
        self.assertEqual(selection["selected_contour_id"], 7)
        self.assertEqual(selection["selection_method"], "spine_centerline_containment_then_area")

    def test_area_only_ranks_centerline_containing_contours(self) -> None:
        inner = square_contour(0, 0.0, 0.0, 0.05)
        outer = square_contour(1, 0.0, 0.0, 0.10)
        selection = select_torso_contour([inner, outer], np.asarray([0.0, 0.0]))
        self.assertEqual(selection["selected_contour_id"], 1)

    def test_bounded_proximity_fallback_is_explicit(self) -> None:
        nearby = square_contour(3, 0.04, 0.0, 0.02)
        selection = select_torso_contour(
            [nearby], np.asarray([0.0, 0.0]), max_centerline_proximity_m=0.03
        )
        self.assertEqual(selection["selected_contour_id"], 3)
        self.assertEqual(selection["selection_method"], "bounded_centerline_proximity_fallback")
        self.assertAlmostEqual(selection["selected_metrics"]["centerline_boundary_distance_m"], 0.02)

    def test_far_contours_are_rejected(self) -> None:
        far = square_contour(4, 0.30, 0.0, 0.02)
        with self.assertRaises(TorsoSelectionError):
            select_torso_contour([far], np.asarray([0.0, 0.0]))

    def test_polygon_queries(self) -> None:
        polygon = np.asarray([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float64)
        self.assertTrue(point_in_polygon_xz(np.asarray([0.0, 0.0]), polygon))
        self.assertFalse(point_in_polygon_xz(np.asarray([2.0, 0.0]), polygon))
        self.assertAlmostEqual(
            distance_to_polygon_boundary_xz(np.asarray([0.0, 0.0]), polygon), 1.0
        )


if __name__ == "__main__":
    unittest.main()
