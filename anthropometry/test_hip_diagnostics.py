#!/usr/bin/env python3
"""Regression tests for Step 2.6C hip diagnostics."""

from __future__ import annotations

import unittest

from hip_diagnostics import lower_bound_sensitivity, near_maximum_plateau


def profile(values: list[float]) -> list[dict[str, object]]:
    return [
        {
            "plane_y_m": index * 0.01,
            "normalized_height": index * 0.01,
            "perimeter_cm": value,
        }
        for index, value in enumerate(values)
    ]


def baseline(values: list[float]) -> dict[str, object]:
    records = profile(values)
    selected_index = max(range(len(values)), key=values.__getitem__)
    return {
        "profile": records,
        "selected": records[selected_index],
    }


class HipDiagnosticsTests(unittest.TestCase):
    def test_plateau_uses_point_one_percent_threshold(self) -> None:
        records = profile([100.0, 99.95, 99.89, 99.70])
        result = near_maximum_plateau(records, 0)
        self.assertEqual(result["plateau_start_index"], 0)
        self.assertEqual(result["plateau_end_index"], 1)
        self.assertAlmostEqual(result["threshold_cm"], 99.9)
        self.assertTrue(result["touches_lower_boundary"])

    def test_plateau_component_is_centered_on_selected_argmax(self) -> None:
        records = profile([99.95, 99.7, 100.0, 99.95])
        result = near_maximum_plateau(records, 2)
        self.assertEqual(result["plateau_start_index"], 2)
        self.assertEqual(result["plateau_end_index"], 3)
        self.assertFalse(result["touches_lower_boundary"])

    def test_case_a_when_larger_joined_layer_fails_stability_gate(self) -> None:
        result = lower_bound_sensitivity(
            baseline([100.0, 99.0]),
            [
                {
                    "plane_y_m": -0.01,
                    "normalized_height": -0.01,
                    "distance_below_lower_mm": 10.0,
                    "perimeter_cm": 101.0,
                    "joined_safe": True,
                    "stable_safe": False,
                }
            ],
        )
        self.assertEqual(result["case"], "A")
        self.assertFalse(result["hidden_stable_maximum"])
        self.assertGreater(
            result["extended_joined_safe_within_20mm"]["delta_c_cm_vs_current"],
            0.0,
        )

    def test_case_b_when_contiguous_stable_extension_has_material_larger_c(self) -> None:
        result = lower_bound_sensitivity(
            baseline([100.0, 99.0]),
            [
                {
                    "plane_y_m": -0.01,
                    "normalized_height": -0.01,
                    "distance_below_lower_mm": 10.0,
                    "perimeter_cm": 100.2,
                    "joined_safe": True,
                    "stable_safe": True,
                }
            ],
        )
        self.assertEqual(result["case"], "B")
        self.assertTrue(result["hidden_stable_maximum"])


if __name__ == "__main__":
    unittest.main()
