import os
import sys
import tempfile
import unittest

import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from config_parser import parse_stack_file
from grid_builder import build_k_grid


STACK_TEXT = """PERIODIC_BOUNDARY
0,SiN,0.500000,True,False
1,Vacuum,0.100000,False,True
2,SiN,0.500000,True,False
PERIODIC_BOUNDARY
"""


class GridAndGapUnitTests(unittest.TestCase):
    def test_parser_converts_100_nm_gap_to_metres_once(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
            handle.write(STACK_TEXT)
            path = handle.name
        try:
            layers, _ = parse_stack_file(path)
        finally:
            os.unlink(path)
        self.assertAlmostEqual(layers[1]["thickness"], 1.0e-7)

    def test_grid_point_count_is_configurable(self):
        grid, weights = build_k_grid(1.3e14, gap_thickness=1.0e-7, num_points=37)
        self.assertEqual(len(grid), 37)
        self.assertEqual(len(weights), 37)
        self.assertTrue(np.all(np.diff(grid) > 0))
        self.assertAlmostEqual(grid[-1], 1.0e10)

    def test_grid_rejects_fewer_than_two_points(self):
        with self.assertRaises(ValueError):
            build_k_grid(1.3e14, gap_thickness=1.0e-7, num_points=1)


if __name__ == "__main__":
    unittest.main()
