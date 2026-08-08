#!/usr/bin/env python3
"""Synthetic regression tests for the topology-constrained hip baseline."""

from __future__ import annotations

import unittest

from hip import (
    classify_pelvic_topology,
    find_first_stable_state,
    find_stable_pelvis_lower_index,
    select_profile_maximum,
)


def contour(contour_id: int, x: float, area: float, perimeter: float) -> dict[str, object]:
    return {
        "id": contour_id,
        "centroid_xz_m": [x, 0.0],
        "area_m2": area,
        "perimeter_m": perimeter,
    }


def pelvic_record(perimeter: float, compactness: float) -> dict[str, object]:
    return {
        "measurement_valid": True,
        "fallback_used": False,
        "centerline_inside": True,
        "perimeter_m": perimeter,
        "compactness": compactness,
    }


class HipTests(unittest.TestCase):
    def test_two_bilateral_dominant_loops_are_split_despite_tiny_third_loop(self) -> None:
        result = classify_pelvic_topology(
            [
                contour(0, -0.1, 0.03, 0.6),
                contour(1, 0.1, 0.03, 0.6),
                contour(2, 0.0, 0.0001, 0.01),
            ],
            0.0,
        )
        self.assertEqual(result["state"], "bilateral_leg_split")
        self.assertEqual(result["dominant_contour_count"], 2)

    def test_one_dominant_loop_is_joined_pelvis(self) -> None:
        result = classify_pelvic_topology(
            [contour(0, 0.0, 0.08, 1.0), contour(1, 0.2, 0.001, 0.1)],
            0.0,
        )
        self.assertEqual(result["state"], "joined_pelvis")

    def test_stable_split_requires_consecutive_layers(self) -> None:
        records = [
            {"topology_state": state}
            for state in (
                "joined_pelvis",
                "bilateral_leg_split",
                "transitional",
                "bilateral_leg_split",
                "bilateral_leg_split",
                "bilateral_leg_split",
            )
        ]
        self.assertEqual(
            find_first_stable_state(records, "bilateral_leg_split", 3), 3
        )

    def test_stability_gate_excludes_merge_spike(self) -> None:
        records = [
            pelvic_record(1.20, 0.60),
            pelvic_record(1.10, 0.75),
            pelvic_record(1.01, 0.92),
            pelvic_record(1.005, 0.93),
            pelvic_record(1.004, 0.94),
            pelvic_record(1.003, 0.95),
        ]
        index = find_stable_pelvis_lower_index(
            records,
            reference_compactness=0.96,
            layers=4,
            compactness_reference_ratio=0.95,
            max_relative_perimeter_change=0.01,
        )
        self.assertEqual(index, 2)

    def test_maximum_uses_perimeter_not_area(self) -> None:
        records = [
            {"perimeter_m": 0.90, "area_m2": 0.20},
            {"perimeter_m": 1.00, "area_m2": 0.10},
            {"perimeter_m": 0.95, "area_m2": 0.30},
        ]
        selected = select_profile_maximum(records)
        self.assertEqual(selected["selected_index"], 1)
        self.assertFalse(selected["boundary_maximum"])

    def test_boundary_maximum_is_flagged_without_replacement(self) -> None:
        records = [{"perimeter_m": 1.1}, {"perimeter_m": 1.0}]
        selected = select_profile_maximum(records)
        self.assertEqual(selected["selected_index"], 0)
        self.assertTrue(selected["boundary_maximum"])


if __name__ == "__main__":
    unittest.main()
