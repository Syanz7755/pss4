import os
import sys
import unittest

import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from fed_engine import classify_prop_evan_modes, compute_prop_evan_boundary
from utils import c


class PropEvanHelperTests(unittest.TestCase):
    def test_vacuum_boundary_is_light_line(self):
        omega = 3.0e14
        self.assertAlmostEqual(compute_prop_evan_boundary(omega), omega / c)

    def test_classifies_equal_boundary_as_evanescent(self):
        omega = 3.0e14
        boundary = compute_prop_evan_boundary(omega)
        k_grid = np.array([0.5 * boundary, boundary, 2.0 * boundary])

        propagating, evanescent, returned_boundary = classify_prop_evan_modes(k_grid, omega)

        self.assertAlmostEqual(returned_boundary, boundary)
        np.testing.assert_array_equal(propagating, np.array([True, False, False]))
        np.testing.assert_array_equal(evanescent, np.array([False, True, True]))


if __name__ == "__main__":
    unittest.main()
