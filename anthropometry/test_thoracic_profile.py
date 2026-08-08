#!/usr/bin/env python3
"""Unit tests for Step 2.7C interval and raw profile classification."""

from __future__ import annotations

import unittest

import numpy as np

from thoracic_profile import (
    classify_profile,
    compute_thoracic_diagnostic_interval,
    make_sweep_heights,
)


class ThoracicProfileTests(unittest.TestCase):
    @staticmethod
    def synthetic_interval_inputs(chest_y: float = 0.70):
        vertices = np.zeros((9000, 3), dtype=np.float64)
        vertices[3572] = [0.1, chest_y - 0.002, 0.1]
        vertices[8340] = [-0.1, chest_y + 0.002, 0.1]
        faces = np.zeros((18403, 3), dtype=np.int64)
        faces[18402] = [10, 11, 8340]
        joints = np.zeros((22, 3), dtype=np.float64)
        joints[6, 1] = 0.60
        joints[16, 1] = 0.81
        joints[17, 1] = 0.79
        return vertices, faces, joints

    def test_interval_contains_both_public_planes(self):
        interval = compute_thoracic_diagnostic_interval(
            *self.synthetic_interval_inputs()
        )
        self.assertEqual(interval["definition"], "skeletal_thoracic_diagnostic_interval_v1")
        self.assertAlmostEqual(interval["upper_y_m"], 0.80)
        self.assertFalse(interval["public_plane_outside_diagnostic_interval"])
        self.assertAlmostEqual(interval["shoulder_y_mismatch_m"], 0.02)

    def test_interval_flags_public_plane_outside_without_clamping(self):
        interval = compute_thoracic_diagnostic_interval(
            *self.synthetic_interval_inputs(chest_y=0.85)
        )
        self.assertTrue(interval["public_plane_outside_diagnostic_interval"])
        self.assertEqual(interval["plane_clamping"], "none")

    def test_sweep_includes_both_boundaries(self):
        values = make_sweep_heights(0.1, 0.21, 1.75)
        self.assertAlmostEqual(values[0], 0.1)
        self.assertAlmostEqual(values[-1], 0.21)
        self.assertTrue(np.all(np.diff(values) > 0.0))
        self.assertTrue(np.all(np.diff(values[:-1]) <= 0.002 * 1.75 + 1e-12))

    def test_monotonic_increasing(self):
        result = classify_profile([90.0, 91.0, 92.0, 93.0])
        self.assertEqual(result["profile_type"], "monotonic_increasing")

    def test_monotonic_decreasing(self):
        result = classify_profile([93.0, 92.0, 91.0, 90.0])
        self.assertEqual(result["profile_type"], "monotonic_decreasing")

    def test_internal_peak(self):
        result = classify_profile([90.0, 92.0, 94.0, 93.0, 91.0])
        self.assertEqual(result["profile_type"], "internal_peak")
        self.assertEqual(result["argmax_index"], 2)

    def test_upper_boundary_peak_with_non_monotonic_profile(self):
        result = classify_profile([90.0, 92.0, 91.0, 93.0])
        self.assertEqual(result["profile_type"], "upper_boundary_peak")


if __name__ == "__main__":
    unittest.main()
