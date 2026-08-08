#!/usr/bin/env python3
"""Registry invariants for Step 2.8."""

import unittest

from definitions import DEFINITION_REGISTRY, PRIMARY_TARGETS, get_definition_registry


class DefinitionRegistryTests(unittest.TestCase):
    def test_primary_targets_exist_and_are_not_rejected(self):
        for target in PRIMARY_TARGETS.values():
            self.assertIn(target, DEFINITION_REGISTRY)
            self.assertNotIn("rejected", str(DEFINITION_REGISTRY[target]["status"]))

    def test_geometry_chest_extreme_is_rejected(self):
        self.assertEqual(
            DEFINITION_REGISTRY["geometry_chest_extreme"]["status"],
            "rejected_as_measurement_definition",
        )

    def test_registry_copy_is_defensive(self):
        copy = get_definition_registry()
        copy["raw_height_v0"]["status"] = "changed"
        self.assertEqual(DEFINITION_REGISTRY["raw_height_v0"]["status"], "baseline")


if __name__ == "__main__":
    unittest.main()
