#!/usr/bin/env python3
"""Regression tests for the Step 2.5D comparison layer."""

from __future__ import annotations

import unittest

from compare_waist_definitions import signed_difference, summarize


class WaistDefinitionComparisonTests(unittest.TestCase):
    def test_signed_difference_uses_first_minus_second(self) -> None:
        first = {"plane_y_m": 0.20, "y_norm": 0.60, "circumference_cm": 84.0}
        second = {"plane_y_m": 0.18, "y_norm": 0.58, "circumference_cm": 87.5}
        result = signed_difference(first, second)
        self.assertAlmostEqual(result["delta_y_norm_first_minus_second"], 0.02)
        self.assertAlmostEqual(result["delta_y_mm_first_minus_second"], 20.0)
        self.assertAlmostEqual(result["delta_c_cm_first_minus_second"], -3.5)

    def test_summary_records_signed_and_absolute_statistics(self) -> None:
        result = summarize([-2.0, 1.0, 4.0])
        self.assertAlmostEqual(result["mean"], 1.0)
        self.assertAlmostEqual(result["mean_absolute"], 7.0 / 3.0)
        self.assertEqual(result["minimum"], -2.0)
        self.assertEqual(result["maximum"], 4.0)

    def test_summary_rejects_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            summarize([])


if __name__ == "__main__":
    unittest.main()
