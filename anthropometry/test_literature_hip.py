#!/usr/bin/env python3
"""Synthetic regression tests for ``literature_hip_v1``."""

from __future__ import annotations

import unittest

import numpy as np

from literature_hip import (
    PUBIC_BONE_VERTEX_ID,
    compute_literature_hip_plane,
    measure_literature_hip,
)
from test_waist import synthetic_hourglass_mesh, synthetic_joints


def mesh_with_pubis(*, landmark_y: float = 0.50) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = synthetic_hourglass_mesh()
    padded = np.zeros((PUBIC_BONE_VERTEX_ID + 1, 3), dtype=np.float64)
    padded[: len(vertices)] = vertices
    padded[PUBIC_BONE_VERTEX_ID] = [0.0, landmark_y, 0.20]
    return padded, faces


class LiteratureHipTests(unittest.TestCase):
    def test_plane_uses_published_pubis_vertex_height(self) -> None:
        vertices, _ = mesh_with_pubis(landmark_y=0.47)
        plane = compute_literature_hip_plane(vertices)
        self.assertAlmostEqual(plane["plane_y_m"], 0.47)
        self.assertEqual(plane["landmark"]["vertex_id"], 5949)
        self.assertTrue(plane["horizontal_adaptation"])

    def test_end_to_end_performs_one_direct_slice(self) -> None:
        vertices, faces = mesh_with_pubis()
        result = measure_literature_hip(vertices, faces, synthetic_joints())
        self.assertEqual(result["definition"], "literature_hip_v1")
        self.assertEqual(result["status"], "baseline")
        self.assertTrue(result["independent_of_circumference_maximum"])
        self.assertEqual(
            result["search_or_scan"], "none; one direct landmark-defined slice"
        )
        self.assertEqual(result["topology"]["state"], "joined_pelvis")
        self.assertFalse(result["fallback_used"])
        self.assertTrue(result["centerline_inside"])
        self.assertGreater(result["perimeter_m"], 0.0)

    def test_plane_moves_with_subject_surface_vertex(self) -> None:
        vertices, _ = mesh_with_pubis(landmark_y=0.45)
        altered = vertices.copy()
        altered[PUBIC_BONE_VERTEX_ID, 1] += 0.03
        first = compute_literature_hip_plane(vertices)
        second = compute_literature_hip_plane(altered)
        self.assertAlmostEqual(second["plane_y_m"] - first["plane_y_m"], 0.03)

    def test_missing_published_vertex_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "published SMPL-X hip landmark"):
            compute_literature_hip_plane(np.zeros((100, 3), dtype=np.float64))


if __name__ == "__main__":
    unittest.main()
