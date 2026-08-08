#!/usr/bin/env python3
"""Unit tests for compact unified-extractor helpers."""

import unittest

from extractor import _circumference_summary


class ExtractorTests(unittest.TestCase):
    def test_compact_summary_omits_profile_and_points(self):
        result = {
            "definition": "demo",
            "status": "baseline",
            "profile": [1, 2, 3],
            "selected": {
                "perimeter_m": 1.0,
                "perimeter_cm": 100.0,
                "plane_y_m": 0.5,
                "normalized_height": 0.6,
                "area_m2": 0.1,
                "centroid_xz_m": [0.0, 0.0],
                "selected_contour_id": 0,
                "selection_mode": "spine_centerline_containment_then_area",
                "fallback_used": False,
                "num_contours": 1,
                "invalid_component_count": 0,
                "remaining_duplicate_segments": 0,
                "ordered_points_m": [[0, 0, 0]],
            },
        }
        summary = _circumference_summary(result, selected=True)
        self.assertNotIn("profile", summary)
        self.assertNotIn("ordered_points_m", summary)
        self.assertEqual(summary["perimeter_cm"], 100.0)


if __name__ == "__main__":
    unittest.main()
