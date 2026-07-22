import unittest

import numpy as np

from dropps.analysis.contact_ng import (
    _process_fast_neighbor_search_frame,
    _process_neighbor_search_frame,
    getargs_contact_ng,
)
from dropps.analysis.contact_ng_core import (
    CONTACT_CATEGORIES,
    StreamingContactAccumulator,
    build_contact_layout,
)


class ContactNeighborSearchTests(unittest.TestCase):
    def setUp(self):
        self.reference = [
            list(range(0, 4)),
            list(range(4, 8)),
            list(range(8, 12)),
        ]
        self.selection = [list(range(12, 15)), list(range(15, 18))]
        self.search_indices = np.arange(18, dtype=np.int64)
        self.layout = build_contact_layout(
            self.search_indices, self.reference, self.selection
        )
        rng = np.random.default_rng(20260722)
        self.positions = rng.uniform(0.2, 19.8, size=(18, 3)).astype(np.float32)
        self.box = np.asarray([20, 20, 20, 90, 90, 90], dtype=np.float32)
        self.atom_cutoffs = rng.uniform(3.0, 6.0, size=18)

    def _new_accumulator(self, intra_average):
        return StreamingContactAccumulator(
            self.layout,
            map_categories=CONTACT_CATEGORIES,
            time_categories=CONTACT_CATEGORIES,
            intra_average=intra_average,
            remove_diagonal=1,
        )

    def test_fast_is_default_search_mode(self):
        args = getargs_contact_ng(
            ["-s", "topol.tpr", "-f", "traj.xtc", "-cs", "global", "-orr", "rr"]
        )
        self.assertEqual(args.search_mode, "fast")

    def test_fast_and_stream_modes_are_equivalent(self):
        for pbc in (False, True):
            for residue_cutoff in (False, True):
                for intra_average in (False, True):
                    with self.subTest(
                        pbc=pbc,
                        residue_cutoff=residue_cutoff,
                        intra_average=intra_average,
                    ):
                        atom_cutoffs = (
                            self.atom_cutoffs if residue_cutoff else None
                        )
                        cutoff = (
                            float(self.atom_cutoffs.max())
                            if residue_cutoff
                            else 6.0
                        )

                        fast = self._new_accumulator(intra_average)
                        fast_counts = _process_fast_neighbor_search_frame(
                            accumulator=fast,
                            positions=self.positions,
                            box=self.box,
                            cutoff=cutoff,
                            pbc=pbc,
                            pair_chunk_size=2,
                            atom_cutoffs=atom_cutoffs,
                        )

                        stream = self._new_accumulator(intra_average)
                        stream_counts = _process_neighbor_search_frame(
                            accumulator=stream,
                            positions=self.positions,
                            box=self.box,
                            cutoff=cutoff,
                            pbc=pbc,
                            search_batch_size=3,
                            pair_chunk_size=2,
                            atom_cutoffs=atom_cutoffs,
                        )

                        fast_maps = fast.averaged_maps()
                        stream_maps = stream.averaged_maps()
                        for category in CONTACT_CATEGORIES:
                            self.assertAlmostEqual(
                                fast_counts[category], stream_counts[category]
                            )
                            np.testing.assert_array_equal(
                                fast_maps[category],
                                stream_maps[category],
                            )


if __name__ == "__main__":
    unittest.main()
