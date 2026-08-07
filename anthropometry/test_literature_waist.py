#!/usr/bin/env python3
"""Synthetic regression tests for ``literature_waist_v1``."""

from __future__ import annotations

import unittest

import numpy as np

from literature_waist import (
    BACK_BELLY_BUTTON_VERTEX_ID,
    BELLY_BUTTON_VERTEX_ID,
    compute_literature_waist_plane,
    measure_literature_waist,
)
from test_waist import synthetic_hourglass_mesh, synthetic_joints


def mesh_with_landmarks(
    *, front_y: float = 0.45, back_y: float = 0.55
) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = synthetic_hourglass_mesh()
    padded = np.zeros((BACK_BELLY_BUTTON_VERTEX_ID + 1, 3), dtype=np.float64)
    padded[: len(vertices)] = vertices
    padded[BELLY_BUTTON_VERTEX_ID] = [0.0, front_y, 0.20]
    padded[BACK_BELLY_BUTTON_VERTEX_ID] = [0.0, back_y, -0.20]
    return padded, faces


class LiteratureWaistTests(unittest.TestCase):
    def test_plane_is_mean_height_of_published_landmarks(self) -> None:
        vertices, _ = mesh_with_landmarks(front_y=0.42, back_y=0.58)
        plane = compute_literature_waist_plane(vertices)
        self.assertAlmostEqual(plane["plane_y_m"], 0.50)
        self.assertAlmostEqual(plane["landmark_absolute_y_mismatch_mm"], 160.0)
        self.assertTrue(plane["horizontal_adaptation"])

    def test_end_to_end_performs_one_direct_slice(self) -> None:
        vertices, faces = mesh_with_landmarks()
        result = measure_literature_waist(vertices, faces, synthetic_joints())
        self.assertEqual(result["definition"], "literature_waist_v1")
        self.assertEqual(result["status"], "baseline")
        self.assertTrue(result["independent_of_circumference_minimum"])
        self.assertEqual(result["search_or_scan"], "none; one direct landmark-defined slice")
        self.assertAlmostEqual(result["plane_definition"]["plane_y_m"], 0.50)
        self.assertFalse(result["fallback_used"])
        self.assertTrue(result["centerline_inside"])
        self.assertGreater(result["perimeter_m"], 0.0)

    def test_plane_does_not_depend_on_body_circumference(self) -> None:
        vertices, _ = mesh_with_landmarks()
        altered = vertices.copy()
        body_vertex_count = len(synthetic_hourglass_mesh()[0])
        altered[:body_vertex_count, [0, 2]] *= 1.7
        first = compute_literature_waist_plane(vertices)
        second = compute_literature_waist_plane(altered)
        self.assertEqual(first["plane_y_m"], second["plane_y_m"])

    def test_missing_published_vertices_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "published SMPL-X waist landmarks"):
            compute_literature_waist_plane(np.zeros((100, 3), dtype=np.float64))


if __name__ == "__main__":
    unittest.main()
