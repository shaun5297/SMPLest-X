#!/usr/bin/env python3
"""Regression tests for the Step 2.6E comparison layer."""

from __future__ import annotations

import unittest

from compare_hip_definitions import _waist_context


class HipDefinitionComparisonTests(unittest.TestCase):
    def test_waist_context_uses_geometry_anatomical_pair(self) -> None:
        report = {
            "status": "waist_subsystem_closed",
            "aggregate": {
                "pairwise_differences": {
                    "geometry_minus_anatomical": {
                        "delta_y_mm_first_minus_second": {"mean_absolute": 67.1},
                        "delta_c_cm_first_minus_second": {"mean_absolute": 3.2},
                    }
                }
            },
        }
        result = _waist_context(report)
        self.assertEqual(result["mean_absolute_delta_y_mm"], 67.1)
        self.assertEqual(result["mean_absolute_delta_c_cm"], 3.2)
        self.assertEqual(
            result["role"], "methodological_context_not_accuracy_benchmark"
        )

    def test_waist_context_rejects_unclosed_report(self) -> None:
        with self.assertRaisesRegex(ValueError, "not closed"):
            _waist_context({"status": "pending"})


if __name__ == "__main__":
    unittest.main()
