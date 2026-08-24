import importlib.util
import os
from pathlib import Path
import sys
import unittest
import tempfile


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_DIR / "scripts" / "run_continuous_coating_sweep.py"
SPEC = importlib.util.spec_from_file_location("coating_sweep", SCRIPT_PATH)
SWEEP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SWEEP)


class ContinuousCoatingSweepTests(unittest.TestCase):
    def test_default_sweep_has_46_cases(self):
        values = SWEEP.coating_values(0, 45, 1)
        self.assertEqual(len(values), 46)
        self.assertEqual((values[0], values[-1]), (0, 45))

    def test_uncoated_stack_has_three_layers(self):
        layers = SWEEP.stack_layers(1.0, 100, 0)
        self.assertEqual([x[0] for x in layers], ["SiN", "Vacuum", "SiN"])
        self.assertAlmostEqual(sum(x[1] for x in layers), 1.1)
        self.assertAlmostEqual(layers[1][1], 0.1)

    def test_45nm_stack_is_symmetric_with_10nm_gap(self):
        layers = SWEEP.stack_layers(1.0, 100, 45)
        self.assertEqual([x[0] for x in layers],
                         ["SiN", "SiO2", "Vacuum", "SiO2", "SiN"])
        self.assertAlmostEqual(layers[1][1], 0.045)
        self.assertAlmostEqual(layers[2][1], 0.010)
        self.assertAlmostEqual(sum(x[1] for x in layers), 1.1)
        self.assertTrue(layers[0][2] and layers[1][2] and layers[3][2] and layers[4][2])
        self.assertTrue(layers[2][3])

    def test_rejects_closed_vacuum_gap(self):
        with self.assertRaises(ValueError):
            SWEEP.stack_layers(1.0, 100, 50)

    def test_probe_result_path_accepts_shifted_probe_id(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = Path(directory) / "tot_P2.csv"
            expected.touch()
            self.assertEqual(SWEEP.probe_result_path(Path(directory)), expected)


if __name__ == "__main__":
    unittest.main()
