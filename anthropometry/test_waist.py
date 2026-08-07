#!/usr/bin/env python3
"""Synthetic regression tests for ``geometry_waist_v0``."""

from __future__ import annotations

import unittest

import numpy as np

from waist import (
    build_local_stability_window,
    build_scan_heights,
    scan_geometry_waist,
    select_profile_minimum,
)


def synthetic_hourglass_mesh(num_ring_points: int = 32) -> tuple[np.ndarray, np.ndarray]:
    heights = np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float64)
    radii = np.asarray([0.50, 0.40, 0.30, 0.20, 0.35, 0.50], dtype=np.float64)
    angles = np.linspace(0.0, 2.0 * np.pi, num_ring_points, endpoint=False)
    rings = []
    for height, radius in zip(heights, radii):
        rings.append(
            np.column_stack(
                [
                    radius * np.cos(angles),
                    np.full(num_ring_points, height),
                    radius * np.sin(angles),
                ]
            )
        )
    vertices = np.vstack(rings)
    faces = []
    for ring_index in range(len(heights) - 1):
        lower = ring_index * num_ring_points
        upper = (ring_index + 1) * num_ring_points
        for index in range(num_ring_points):
            following = (index + 1) % num_ring_points
            faces.append([lower + index, upper + index, upper + following])
            faces.append([lower + index, upper + following, lower + following])
    bottom_center = len(vertices)
    top_center = bottom_center + 1
    vertices = np.vstack([vertices, [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    top_start = (len(heights) - 1) * num_ring_points
    for index in range(num_ring_points):
        following = (index + 1) % num_ring_points
        faces.append([bottom_center, following, index])
        faces.append([top_center, top_start + index, top_start + following])
    return vertices, np.asarray(faces, dtype=np.int64)


def synthetic_joints() -> np.ndarray:
    joints = np.zeros((22, 3), dtype=np.float64)
    joints[0] = [0.0, 0.2, 0.0]
    joints[3] = [0.0, 0.4, 0.0]
    joints[6] = [0.0, 0.8, 0.0]
    joints[9] = [0.0, 0.9, 0.0]
    return joints


def profile(perimeters: list[float]) -> list[dict[str, float]]:
    return [
        {
            "plane_y_m": index * 0.1,
            "normalized_height": index * 0.1,
            "perimeter_m": value,
            "perimeter_cm": value * 100.0,
        }
        for index, value in enumerate(perimeters)
    ]


class WaistTests(unittest.TestCase):
    def test_scan_heights_include_exact_boundaries(self) -> None:
        heights = build_scan_heights(0.2, 0.83, 0.1)
        self.assertAlmostEqual(heights[0], 0.2)
        self.assertAlmostEqual(heights[-1], 0.83)
        np.testing.assert_allclose(np.diff(heights)[:-1], 0.1, atol=1e-12)
        self.assertLessEqual(heights[-1] - heights[-2], 0.1)

    def test_raw_argmin_is_not_replaced_when_on_boundary(self) -> None:
        records = profile([0.7, 0.8, 0.9])
        minimum = select_profile_minimum(records)
        self.assertEqual(minimum["selected_index"], 0)
        self.assertTrue(minimum["boundary_minimum"])

    def test_local_window_records_two_steps_each_side(self) -> None:
        records = profile([0.9, 0.8, 0.7, 0.8, 0.9])
        stability = build_local_stability_window(records, 2, radius=2)
        self.assertEqual([item["offset_steps"] for item in stability["neighborhood"]], [-2, -1, 0, 1, 2])
        self.assertAlmostEqual(stability["max_adjacent_change_cm"], 10.0)

    def test_hourglass_end_to_end_selects_internal_raw_minimum(self) -> None:
        vertices, faces = synthetic_hourglass_mesh()
        result = scan_geometry_waist(
            vertices,
            faces,
            synthetic_joints(),
            step_normalized_height=0.1,
            local_jump_warning_cm=100.0,
        )
        self.assertEqual(result["definition"], "geometry_waist_v0")
        self.assertEqual(result["status"], "baseline")
        self.assertEqual(result["scan_parameters"]["smoothing"], "none")
        self.assertEqual(result["scan_parameters"]["curve_fitting"], "none")
        self.assertAlmostEqual(result["selected"]["plane_y_m"], 0.6)
        self.assertFalse(result["boundary_minimum"])
        self.assertEqual(result["fallback_count"], 0)
        self.assertTrue(all(item["centerline_inside"] for item in result["profile"]))
        self.assertEqual(
            [item["offset_steps"] for item in result["local_stability"]["neighborhood"]],
            [-2, -1, 0, 1, 2],
        )


if __name__ == "__main__":
    unittest.main()
