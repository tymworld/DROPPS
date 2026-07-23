import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "dropps"
    / "analysis"
    / "exchange_core.py"
)
SPEC = importlib.util.spec_from_file_location("exchange_core", MODULE_PATH)
exchange_core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exchange_core)


class ExchangeCoreTests(unittest.TestCase):
    def test_periodic_molecular_center_crosses_box_boundary(self):
        center = exchange_core.periodic_weighted_mean_fraction(
            np.asarray([0.98, 0.02]), np.asarray([1.0, 1.0])
        )
        self.assertTrue(np.isclose(center, 0.0) or np.isclose(center, 1.0))

    def test_phase_cutoffs_and_classification(self):
        dense, dilute, offset = exchange_core.phase_cutoffs(
            3.0, 0.5, threshold=0.1, cell_length_nm=20.0
        )
        self.assertAlmostEqual(offset, np.arctanh(0.8) * 0.5)
        states = exchange_core.classify_phase_positions(
            [-dense, 0.0, 0.5 * (dense + dilute), dilute], dense, dilute
        )
        np.testing.assert_array_equal(
            states,
            [
                exchange_core.DENSE,
                exchange_core.DENSE,
                exchange_core.INTERFACE,
                exchange_core.DILUTE,
            ],
        )

    def test_continuous_episodes_use_sampling_midpoints_and_censoring(self):
        times = np.arange(7.0)
        states = [
            exchange_core.DENSE,
            exchange_core.DENSE,
            exchange_core.INTERFACE,
            exchange_core.INTERFACE,
            exchange_core.DILUTE,
            exchange_core.DILUTE,
            exchange_core.DENSE,
        ]
        episodes = exchange_core.extract_phase_episodes(times, states)
        self.assertEqual(len(episodes), 4)
        self.assertEqual(
            [(row["start_time_ns"], row["end_time_ns"]) for row in episodes],
            [(0.0, 1.5), (1.5, 3.5), (3.5, 5.5), (5.5, 6.0)],
        )
        self.assertTrue(episodes[0]["left_censored"])
        self.assertFalse(episodes[0]["right_censored"])
        self.assertFalse(episodes[1]["left_censored"])
        self.assertTrue(episodes[-1]["right_censored"])

    def test_only_confirmed_bulk_to_bulk_passages_are_exchange_events(self):
        times = np.arange(7.0)
        states = [
            exchange_core.DENSE,
            exchange_core.DENSE,
            exchange_core.INTERFACE,
            exchange_core.INTERFACE,
            exchange_core.DILUTE,
            exchange_core.DILUTE,
            exchange_core.DENSE,
        ]
        events = exchange_core.extract_exchange_events(times, states)
        self.assertEqual(len(events), 2)
        first, second = events
        self.assertEqual(first["source_phase"], exchange_core.DENSE)
        self.assertEqual(first["destination_phase"], exchange_core.DILUTE)
        self.assertEqual(first["departure_time_ns"], 1.5)
        self.assertEqual(first["arrival_time_ns"], 3.5)
        self.assertEqual(first["transition_time_ns"], 2.0)
        self.assertTrue(first["waiting_left_censored"])
        self.assertEqual(second["transition_time_ns"], 0.0)
        self.assertEqual(second["waiting_time_since_last_exchange_ns"], 2.0)

        return_states = [
            exchange_core.DENSE,
            exchange_core.INTERFACE,
            exchange_core.DENSE,
            exchange_core.INTERFACE,
            exchange_core.DILUTE,
        ]
        events = exchange_core.extract_exchange_events(np.arange(5.0), return_states)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["departure_time_ns"], 2.5)

    def test_kaplan_meier_accounts_for_right_censoring(self):
        rows = exchange_core.kaplan_meier(
            [1.0, 2.0, 2.0, 3.0], [True, True, False, True]
        )
        self.assertEqual(rows[0]["survival"], 1.0)
        self.assertAlmostEqual(rows[1]["survival"], 0.75)
        self.assertEqual(rows[2]["events"], 1)
        self.assertEqual(rows[2]["censored"], 1)
        self.assertAlmostEqual(rows[2]["survival"], 0.5)
        self.assertEqual(rows[-1]["survival"], 0.0)


if __name__ == "__main__":
    unittest.main()
