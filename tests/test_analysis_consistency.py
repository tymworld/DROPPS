import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from openmm.unit import nanometer

from dropps.analysis.assembly import (
    _cluster_members,
    _component_geometry,
    _contact_image_shifts,
    _weighted_cluster_fraction,
)
from dropps.analysis.density_core import DEFAULT_DENSITY_BIN_WIDTH_NM
from dropps.analysis.exchange import getargs_exchange
from dropps.analysis.gyrate import getargs_gyrate
from dropps.analysis.inter_distance import (
    _distance_vectors_nm,
    _interchain_pairs,
)
from dropps.analysis.phase_msd import getargs_phase_msd
from dropps.analysis.rmsd import _reference_frame
from dropps.analysis.time_core import uniform_frame_interval_ns
from dropps.commands.genmesh import (
    _merge_molecule_counts,
    _molecule_center_nm,
)
from dropps.commands.gsd2xtc import _frame_time_ps, _has_valid_box
from dropps.commands.modifyres import find_new_atom_position


class DistanceConsistencyTests(unittest.TestCase):
    def test_inter_distance_uses_nm_and_minimum_image(self):
        direct = _distance_vectors_nm([[0.0, 0.0, 0.0]], [[10.0, 0.0, 0.0]])
        periodic = _distance_vectors_nm(
            [[0.2, 0.0, 0.0]],
            [[9.8, 0.0, 0.0]],
            [10.0, 10.0, 10.0, 90.0, 90.0, 90.0],
        )
        np.testing.assert_allclose(direct, [1.0])
        np.testing.assert_allclose(periodic, [0.04])

    def test_inter_distance_excludes_same_chain_pairs_by_identity(self):
        pairs = _interchain_pairs([10, 11], [20, 21], ["A", "B"], ["A", "B"])
        np.testing.assert_array_equal(pairs, [[10, 21], [11, 20]])


class TimeConsistencyTests(unittest.TestCase):
    def test_uniform_interval_comes_from_real_timestamps(self):
        self.assertAlmostEqual(uniform_frame_interval_ns([0.0, 0.25, 0.5]), 0.25)

    def test_irregular_timestamps_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "irregular"):
            uniform_frame_interval_ns([0.0, 0.25, 0.7])

    def test_rmsd_reference_uses_nearest_actual_timestamp(self):
        class Frames:
            n_frames = 3

            def __getitem__(self, index):
                return SimpleNamespace(time=[0.0, 1000.0, 3000.0][index])

        trajectory = SimpleNamespace(Universe=SimpleNamespace(trajectory=Frames()))
        self.assertEqual(_reference_frame(trajectory, 0, 1.6), 1)


class AssemblyConsistencyTests(unittest.TestCase):
    def test_no_contacts_still_produces_singleton_clusters(self):
        members = _cluster_members(
            np.empty((0, 2), dtype=int),
            np.asarray([0, 1, 2]),
            3,
            1,
        )
        self.assertEqual(members, [[0], [1], [2]])

    def test_composition_is_weighted_by_cluster_size(self):
        fraction = _weighted_cluster_fraction(
            [[0, 1, 2, 3], [4, 5]],
            {0, 4},
            2,
        )
        self.assertAlmostEqual(fraction, 2.0 / 6.0)

    def test_contact_edges_reconstruct_compact_periodic_cluster(self):
        positions = np.asarray([[0.2, 0.0, 0.0], [9.8, 0.0, 0.0]])
        pairs = np.asarray([[0, 1]])
        labels = np.asarray([0, 1])
        box = np.asarray([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
        shifts = _contact_image_shifts(positions, pairs, labels, box)
        groups = [
            SimpleNamespace(positions=positions[[0]], masses=np.ones(1)),
            SimpleNamespace(positions=positions[[1]], masses=np.ones(1)),
        ]
        rg, _, _, _ = _component_geometry(groups, [0, 1], shifts)
        self.assertAlmostEqual(rg, 0.2)


class CommandRegressionTests(unittest.TestCase):
    def test_phase_commands_share_density_center_defaults(self):
        exchange = getargs_exchange(["-s", "run.tpr", "-f", "run.xtc", "-o", "out"])
        phase_msd = getargs_phase_msd(["-s", "run.tpr", "-f", "run.xtc", "-o", "out"])
        for args in (exchange, phase_msd):
            self.assertEqual(args.bin_width, DEFAULT_DENSITY_BIN_WIDTH_NM)
            self.assertEqual(args.dense_phase_threshold, 0.5)
        gyrate = getargs_gyrate(
            ["-s", "run.tpr", "-f", "run.xtc", "-op", "out"]
        )
        self.assertEqual(gyrate.density_bin_width, DEFAULT_DENSITY_BIN_WIDTH_NM)
        self.assertEqual(gyrate.dense_phase_threshold, 0.5)

    def test_genmesh_centers_coordinates_and_merges_nonadjacent_names(self):
        molecule = [
            {"x": 10.0 * nanometer, "y": -2.0 * nanometer, "z": 3.0 * nanometer},
            {"x": 12.0 * nanometer, "y": 4.0 * nanometer, "z": 7.0 * nanometer},
        ]
        np.testing.assert_allclose(_molecule_center_nm(molecule), [11.0, 1.0, 5.0])
        self.assertEqual(
            _merge_molecule_counts(["A", "B", "A"], [2, 3, 4]),
            (["A", "B"], [6, 3]),
        )

    def test_modifyres_ignores_bonded_reference_in_steric_score(self):
        with patch(
            "dropps.commands.modifyres.np.random.uniform",
            side_effect=[np.asarray([0.0, np.pi]), np.asarray([0.0, 0.0])],
        ):
            position = find_new_atom_position(
                [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
                0.1,
                1,
                n_samples=2,
            )
        np.testing.assert_allclose(position, [-0.1, 0.0, 0.0], atol=1.0e-12)

    def test_gsd_time_starts_at_zero_and_box_validation_handles_none(self):
        self.assertEqual(_frame_time_ps(0, 0.1), 0.0)
        self.assertEqual(_frame_time_ps(2, 0.1), 200.0)
        self.assertFalse(_has_valid_box(None))
        self.assertTrue(_has_valid_box([1.0, 2.0, 3.0, 90.0, 90.0, 90.0]))


if __name__ == "__main__":
    unittest.main()
