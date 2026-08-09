#!/usr/bin/env python3
"""Synthetic regression tests for the generic slicing engine."""

from __future__ import annotations

import unittest

import numpy as np

from slicing import (
    _intersect_mesh_with_horizontal_plane_provenance,
    compute_contour_area,
    compute_contour_perimeter,
    deduplicate_segments,
    intersect_mesh_with_horizontal_plane,
    slice_mesh,
)


def cube(center_x: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, -1, 1],
            [-1, -1, 1],
            [-1, 1, -1],
            [1, 1, -1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=np.float64,
    )
    vertices[:, 0] += center_x
    faces = np.asarray(
        [
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int64,
    )
    return vertices, faces


class SlicingTests(unittest.TestCase):
    def test_cube_midplane_is_one_closed_square(self) -> None:
        vertices, faces = cube()
        result = slice_mesh(vertices, faces, 0.0)
        self.assertEqual(result["diagnostics"]["connectivity"]["invalid_component_count"], 0)
        self.assertEqual(len(result["contours"]), 1)
        contour = result["contours"][0]
        self.assertAlmostEqual(contour["perimeter_m"], 8.0, places=10)
        self.assertAlmostEqual(contour["area_m2"], 4.0, places=10)
        np.testing.assert_allclose(contour["centroid_xz_m"], [0.0, 0.0], atol=1e-12)

    def test_two_disconnected_cubes_return_two_loops(self) -> None:
        first_vertices, first_faces = cube(-3.0)
        second_vertices, second_faces = cube(3.0)
        vertices = np.vstack([first_vertices, second_vertices])
        faces = np.vstack([first_faces, second_faces + len(first_vertices)])
        result = slice_mesh(vertices, faces, 0.0)
        self.assertEqual(len(result["contours"]), 2)
        self.assertTrue(all(item["all_node_degrees_two"] for item in result["contours"]))
        np.testing.assert_allclose(
            sorted(item["centroid_xz_m"][0] for item in result["contours"]),
            [-3.0, 3.0],
            atol=1e-12,
        )

    def test_plane_outside_mesh_returns_no_loop(self) -> None:
        vertices, faces = cube()
        result = slice_mesh(vertices, faces, 2.0)
        self.assertEqual(result["contours"], [])
        self.assertEqual(result["diagnostics"]["intersection"]["raw_segment_count"], 0)

    def test_plane_on_cube_face_uses_boundary_and_ignores_coplanar_faces(self) -> None:
        vertices, faces = cube()
        result = slice_mesh(vertices, faces, -1.0)
        self.assertEqual(result["diagnostics"]["intersection"]["coplanar_face_count"], 2)
        self.assertEqual(len(result["contours"]), 1)
        self.assertAlmostEqual(result["contours"][0]["perimeter_m"], 8.0, places=10)
        self.assertAlmostEqual(result["contours"][0]["area_m2"], 4.0, places=10)

    def test_tiny_topologically_valid_segment_is_preserved(self) -> None:
        vertices, faces = cube()
        result = slice_mesh(
            vertices,
            faces,
            -1.0 + 1e-8,
            eps=1e-12,
            endpoint_tolerance=1e-6,
        )
        connectivity = result["diagnostics"]["connectivity"]
        self.assertEqual(connectivity["endpoint_identity"], "mesh_topology_provenance")
        self.assertFalse(connectivity["coordinate_tolerance_controls_topology"])
        self.assertGreater(
            connectivity["very_short_but_topologically_valid_segment_count"], 0
        )
        self.assertEqual(connectivity["topological_zero_length_segments_removed"], 0)
        self.assertEqual(connectivity["invalid_component_count"], 0)
        self.assertEqual(len(result["contours"]), 1)
        self.assertAlmostEqual(result["contours"][0]["perimeter_m"], 8.0, places=7)

    def test_spatially_close_but_topologically_distinct_loops_do_not_merge(self) -> None:
        first_vertices, first_faces = cube(-1.00000025)
        second_vertices, second_faces = cube(1.00000025)
        vertices = np.vstack([first_vertices, second_vertices])
        faces = np.vstack([first_faces, second_faces + len(first_vertices)])
        result = slice_mesh(vertices, faces, 0.0, endpoint_tolerance=1e-6)
        self.assertEqual(result["diagnostics"]["connectivity"]["invalid_component_count"], 0)
        self.assertEqual(len(result["contours"]), 2)

    def test_adjacent_triangles_share_edge_provenance(self) -> None:
        vertices = np.asarray(
            [[-1, -1, 0], [1, 1, 0], [-1, 1, 0], [1, -1, 0]],
            dtype=np.float64,
        )
        faces = np.asarray([[0, 1, 2], [0, 3, 1]], dtype=np.int64)
        _, endpoint_keys, diagnostics = (
            _intersect_mesh_with_horizontal_plane_provenance(
                vertices, faces, 0.0, eps=1e-12
            )
        )
        self.assertEqual(diagnostics["raw_segment_count"], 2)
        self.assertEqual(
            sum(("edge", 0, 1) in key_pair for key_pair in endpoint_keys), 2
        )

    def test_exact_on_plane_vertices_keep_vertex_identity(self) -> None:
        vertices, faces = cube()
        _, endpoint_keys, diagnostics = (
            _intersect_mesh_with_horizontal_plane_provenance(
                vertices, faces, -1.0
            )
        )
        flattened = {key for pair in endpoint_keys for key in pair}
        self.assertGreater(diagnostics["on_plane_edge_face_count"], 0)
        self.assertIn(("vertex", 0), flattened)

    def test_isolated_coplanar_triangle_is_ignored(self) -> None:
        vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        faces = np.asarray([[0, 1, 2]], dtype=np.int64)
        segments, diagnostics = intersect_mesh_with_horizontal_plane(vertices, faces, 0.0)
        self.assertEqual(len(segments), 0)
        self.assertEqual(diagnostics["coplanar_face_count"], 1)

    def test_duplicate_and_zero_length_segments_are_removed(self) -> None:
        segments = np.asarray(
            [
                [[0, 0, 0], [1, 0, 0]],
                [[1, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [1e-8, 0, 0]],
            ],
            dtype=np.float64,
        )
        unique, diagnostics = deduplicate_segments(segments, tolerance=1e-6)
        self.assertEqual(len(unique), 1)
        self.assertEqual(diagnostics["duplicate_segments_removed"], 1)
        self.assertEqual(diagnostics["zero_length_segments_removed"], 1)
        self.assertEqual(diagnostics["remaining_duplicate_segments"], 0)

    def test_planar_metrics_ignore_y_roundoff(self) -> None:
        points = np.asarray(
            [[-1, 1e-9, -1], [1, -1e-9, -1], [1, 2e-9, 1], [-1, 0, 1]],
            dtype=np.float64,
        )
        self.assertAlmostEqual(compute_contour_perimeter(points), 8.0, places=12)
        self.assertAlmostEqual(compute_contour_area(points), 4.0, places=12)


if __name__ == "__main__":
    unittest.main()
