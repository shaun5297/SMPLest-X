#!/usr/bin/env python3
"""External AGORA regression for the topology-aware slicing repair."""

from __future__ import annotations

import hashlib
import os
import unittest
from pathlib import Path

import numpy as np

from slicing import slice_mesh
from waist import scan_geometry_waist


DEFAULT_FIXTURE = Path(
    "/data0/datasets/AGORA/phase3_fixtures/agora-val-000216_canonical.npz"
)
FIXTURE = Path(os.environ.get("AGORA_000216_CANONICAL_NPZ", DEFAULT_FIXTURE))
FIXTURE_SHA256 = "b4f1cbbc9f6139719a11689942b01bf75e8be109625f0164fd17a6ff31ae1693"
FAILING_PLANE_Y_M = -0.2852943875789642


@unittest.skipUnless(FIXTURE.is_file(), f"licensed external fixture missing: {FIXTURE}")
class Agora000216TopologyRegression(unittest.TestCase):
    def test_short_valid_segment_and_geometry_waist_complete(self) -> None:
        digest = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
        self.assertEqual(digest, FIXTURE_SHA256)
        with np.load(FIXTURE, allow_pickle=False) as data:
            vertices = np.asarray(data["vertices"])
            joints = np.asarray(data["joints"])
            faces = np.asarray(data["faces"])

        sliced = slice_mesh(vertices, faces, FAILING_PLANE_Y_M)
        connectivity = sliced["diagnostics"]["connectivity"]
        self.assertEqual(connectivity["endpoint_identity"], "mesh_topology_provenance")
        self.assertEqual(connectivity["invalid_component_count"], 0)
        self.assertEqual(connectivity["closed_loop_count"], 1)
        self.assertEqual(
            connectivity["very_short_but_topologically_valid_segment_count"], 1
        )
        self.assertAlmostEqual(
            connectivity["minimum_valid_segment_length_m"],
            2.0484591035717338e-7,
            places=15,
        )
        self.assertAlmostEqual(
            sliced["contours"][0]["perimeter_cm"], 89.48490121966213, places=10
        )

        waist = scan_geometry_waist(vertices, faces, joints)
        self.assertEqual(waist["fallback_count"], 0)
        self.assertEqual(waist["selected"]["index"], 43)
        self.assertAlmostEqual(
            waist["selected"]["perimeter_cm"], 86.50406209427047, places=10
        )


if __name__ == "__main__":
    unittest.main()
