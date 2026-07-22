import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


# Loading the NumPy-only core directly keeps this unit test independent of the
# trajectory stack and makes it useful in minimal development environments.
CORE_PATH = (
    Path(__file__).resolve().parents[1]
    / "dropps"
    / "analysis"
    / "contact_ng_core.py"
)
SPEC = importlib.util.spec_from_file_location("dropps_contact_ng_core_test", CORE_PATH)
core = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = core
SPEC.loader.exec_module(core)


class StreamingContactAccumulatorTests(unittest.TestCase):
    def setUp(self):
        self.reference = [[0, 2, 4, 6], [10, 12, 14, 16], [20, 22, 24, 26]]
        self.selection = [[1, 3, 5], [11, 13, 15]]
        self.search_indices = np.unique(
            np.asarray(sum(self.reference + self.selection, []), dtype=np.int64)
        )
        self.layout = core.build_contact_layout(
            self.search_indices, self.reference, self.selection
        )

    def test_streamed_chunks_match_scalar_reference(self):
        rng = np.random.default_rng(20260722)
        all_pairs = np.asarray(
            [
                (i, j)
                for i in range(self.search_indices.size)
                for j in range(i + 1, self.search_indices.size)
            ],
            dtype=np.int64,
        )
        pairs = all_pairs[rng.random(all_pairs.shape[0]) < 0.4]

        accumulator = core.StreamingContactAccumulator(
            self.layout,
            map_categories=core.CONTACT_CATEGORIES,
            time_categories=core.CONTACT_CATEGORIES,
            remove_diagonal=1,
        )
        accumulator.begin_frame()
        for chunk in np.array_split(pairs, 9):
            accumulator.accumulate(chunk[:, 0], chunk[:, 1])
        counts = accumulator.finish_frame()
        maps = accumulator.averaged_maps()

        expected = {
            core.INTER_RS: np.zeros((4, 3)),
            core.INTER_RR: np.zeros((4, 4)),
            core.INTER_SS: np.zeros((3, 3)),
            core.INTRA_R: np.zeros((4, 4)),
            core.INTRA_S: np.zeros((3, 3)),
        }
        ref_lookup = {
            atom: (chain_id, position)
            for chain_id, chain in enumerate(self.reference)
            for position, atom in enumerate(chain)
        }
        sel_lookup = {
            atom: (chain_id, position)
            for chain_id, chain in enumerate(self.selection)
            for position, atom in enumerate(chain)
        }

        for local_a, local_b in pairs:
            atom_a = int(self.search_indices[local_a])
            atom_b = int(self.search_indices[local_b])

            if atom_a in ref_lookup and atom_b in ref_lookup:
                chain_a, pos_a = ref_lookup[atom_a]
                chain_b, pos_b = ref_lookup[atom_b]
                category = core.INTRA_R if chain_a == chain_b else core.INTER_RR
                if category != core.INTRA_R or abs(pos_a - pos_b) > 1:
                    expected[category][pos_a, pos_b] += 1
                    expected[category][pos_b, pos_a] += 1

            if atom_a in sel_lookup and atom_b in sel_lookup:
                chain_a, pos_a = sel_lookup[atom_a]
                chain_b, pos_b = sel_lookup[atom_b]
                category = core.INTRA_S if chain_a == chain_b else core.INTER_SS
                if category != core.INTRA_S or abs(pos_a - pos_b) > 1:
                    expected[category][pos_a, pos_b] += 1
                    expected[category][pos_b, pos_a] += 1

            if atom_a in ref_lookup and atom_b in sel_lookup:
                expected[core.INTER_RS][
                    ref_lookup[atom_a][1], sel_lookup[atom_b][1]
                ] += 1
            if atom_a in sel_lookup and atom_b in ref_lookup:
                expected[core.INTER_RS][
                    ref_lookup[atom_b][1], sel_lookup[atom_a][1]
                ] += 1

        for category, expected_map in expected.items():
            np.testing.assert_array_equal(maps[category], expected_map)
            divisor = 1 if category == core.INTER_RS else 2
            self.assertEqual(counts[category], expected_map.sum() / divisor)

    def test_time_only_output_allocates_no_contact_map(self):
        accumulator = core.StreamingContactAccumulator(
            self.layout,
            map_categories=(),
            time_categories=core.CONTACT_CATEGORIES,
        )
        self.assertEqual(accumulator.maps, {})

    def test_intra_average_is_applied_to_map_and_count(self):
        pairs = np.asarray([[0, 2], [7, 9], [14, 15]], dtype=np.int64)
        accumulator = core.StreamingContactAccumulator(
            self.layout,
            map_categories=(core.INTRA_R,),
            time_categories=(core.INTRA_R,),
            intra_average=True,
            remove_diagonal=-1,
        )
        accumulator.begin_frame()
        accumulator.accumulate(pairs[:, 0], pairs[:, 1])
        counts = accumulator.finish_frame()
        maps = accumulator.averaged_maps()

        self.assertAlmostEqual(counts[core.INTRA_R], 1.0)
        self.assertAlmostEqual(maps[core.INTRA_R].sum(), 2.0)


if __name__ == "__main__":
    unittest.main()
