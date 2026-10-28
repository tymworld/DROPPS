import unittest

import numpy as np

from dropps.analysis.distance_core import (
    ChannelStatistics,
    HistogramAccumulator,
    RunningStatistics,
    build_distance_job,
    build_endpoint_plan,
    iter_channel_descriptors,
    iter_distance_blocks,
    project_endpoint,
)


def _endpoint(indices, label, grouping, position, chains, residues):
    return build_endpoint_plan(
        indices,
        label,
        grouping,
        position,
        chains,
        residues,
    )


class ManyChainDistancePlanningTests(unittest.TestCase):
    def setUp(self):
        self.chain_count = 100
        self.chains = np.repeat(np.arange(self.chain_count), 2)
        self.residues = np.tile([10, 50], self.chain_count)
        self.reference_indices = np.arange(0, self.chain_count * 2, 2)
        self.selection_indices = self.reference_indices + 1
        self.reference = _endpoint(
            self.reference_indices,
            "A",
            "chain",
            "cog",
            self.chains,
            self.residues,
        )
        self.selection = _endpoint(
            self.selection_indices,
            "B",
            "chain",
            "cog",
            self.chains,
            self.residues,
        )

    def test_intra_chain_produces_one_channel_per_chain(self):
        job = build_distance_job(
            "intra",
            self.reference,
            self.selection,
            "intra-chain",
            "pair",
        )
        self.assertEqual(job.entity_pair_count, 100)
        self.assertEqual(job.channel_count, 100)

    def test_inter_chain_produces_directed_off_diagonal_product(self):
        job = build_distance_job(
            "inter",
            self.reference,
            self.selection,
            "inter-chain",
            "pair",
        )
        self.assertEqual(job.entity_pair_count, 100 * 99)
        self.assertEqual(job.channel_count, 100 * 99)

    def test_missing_selection_chain_does_not_shift_correspondence(self):
        selection = _endpoint(
            self.selection_indices[self.chains[self.selection_indices] != 37],
            "B",
            "chain",
            "cog",
            self.chains,
            self.residues,
        )
        job = build_distance_job(
            "intra",
            self.reference,
            selection,
            "intra-chain",
            "pair",
        )
        descriptors = list(iter_channel_descriptors(job))
        self.assertEqual(job.channel_count, 99)
        self.assertFalse(any("chain=37]" in item.selection_entity for item in descriptors))
        self.assertTrue(
            all(
                item.reference_entity.split("chain=")[1].split("]")[0]
                == item.selection_entity.split("chain=")[1].split("]")[0]
                for item in descriptors
            )
        )

    def test_residue_grouping_forms_all_within_chain_residue_pairs(self):
        chains = np.repeat(np.arange(2), 3)
        residues = np.tile([10, 11, 50], 2)
        reference = _endpoint(
            [0, 1, 3, 4], "A-type", "residue", "cog", chains, residues
        )
        selection = _endpoint(
            [2, 5], "B-type", "residue", "cog", chains, residues
        )
        job = build_distance_job(
            "residue-intra", reference, selection, "intra-chain", "pair"
        )
        self.assertEqual(job.channel_count, 4)


class DistanceGeometryTests(unittest.TestCase):
    def test_intra_chain_cog_vectors_are_evaluated_in_blocks(self):
        chains = np.repeat(np.arange(3), 2)
        residues = np.tile([10, 50], 3)
        reference = _endpoint([0, 2, 4], "A", "chain", "cog", chains, residues)
        selection = _endpoint([1, 3, 5], "B", "chain", "cog", chains, residues)
        job = build_distance_job(
            "intra", reference, selection, "intra-chain", "pair"
        )
        positions = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.2, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.2, 0.0, 0.0],
            ]
        )
        masses = np.ones(6)
        ref_points = project_endpoint(reference, positions, masses, pbc=False)
        sel_points = project_endpoint(selection, positions, masses, pbc=False)
        blocks = list(
            iter_distance_blocks(
                job,
                ref_points,
                sel_points,
                pbc=False,
                entity_block_size=2,
            )
        )
        self.assertEqual([block.start for block in blocks], [0, 2])
        np.testing.assert_allclose(
            np.concatenate([block.values for block in blocks]),
            [0.2, 0.2, 0.2],
        )
        np.testing.assert_allclose(
            np.vstack([block.vectors for block in blocks]),
            [[0.2, 0.0, 0.0]] * 3,
        )

    def test_fixed_pair_components_use_selection_minus_reference(self):
        chains = np.asarray([0, 0, 0, 0])
        residues = np.arange(4)
        reference = _endpoint([0, 1], "R", "all", "atom", chains, residues)
        selection = _endpoint([2, 3], "S", "all", "atom", chains, residues)
        job = build_distance_job(
            "fixed", reference, selection, "corresponding", "pair"
        )
        positions = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]
        )
        points_r = project_endpoint(reference, positions, np.ones(4), pbc=False)
        points_s = project_endpoint(selection, positions, np.ones(4), pbc=False)
        block = list(iter_distance_blocks(job, points_r, points_s, pbc=False))[0]
        np.testing.assert_allclose(block.vectors, [[0.5, 0.0, 0.0], [0.0, -1.0, 0.0]])
        np.testing.assert_allclose(block.values, [0.5, 1.0])

    def test_min_and_max_return_the_extremizing_vector(self):
        chains = np.asarray([0, 0, 1, 1])
        residues = np.arange(4)
        reference = _endpoint([0, 1], "R", "all", "atom", chains, residues)
        selection = _endpoint([2, 3], "S", "all", "atom", chains, residues)
        positions = np.asarray(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.5, 0.0, 0.0], [5.0, 0.0, 0.0]]
        )
        points_r = project_endpoint(reference, positions, np.ones(4), pbc=False)
        points_s = project_endpoint(selection, positions, np.ones(4), pbc=False)
        minimum = build_distance_job(
            "minimum", reference, selection, "corresponding", "min"
        )
        maximum = build_distance_job(
            "maximum", reference, selection, "corresponding", "max"
        )
        min_block = list(
            iter_distance_blocks(minimum, points_r, points_s, pbc=False)
        )[0]
        max_block = list(
            iter_distance_blocks(maximum, points_r, points_s, pbc=False)
        )[0]
        np.testing.assert_allclose(min_block.values, [0.5])
        np.testing.assert_allclose(min_block.vectors, [[0.5, 0.0, 0.0]])
        np.testing.assert_allclose(max_block.values, [5.0])
        np.testing.assert_allclose(max_block.vectors, [[5.0, 0.0, 0.0]])

    def test_minimum_image_is_applied_to_scalar_and_component(self):
        chains = np.asarray([0, 1])
        residues = np.asarray([1, 1])
        reference = _endpoint([0], "R", "all", "atom", chains, residues)
        selection = _endpoint([1], "S", "all", "atom", chains, residues)
        job = build_distance_job(
            "pbc", reference, selection, "corresponding", "pair"
        )
        positions = np.asarray([[0.2, 0.0, 0.0], [9.8, 0.0, 0.0]])
        box = np.asarray([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
        points_r = project_endpoint(reference, positions, np.ones(2), box, True)
        points_s = project_endpoint(selection, positions, np.ones(2), box, True)
        block = list(iter_distance_blocks(job, points_r, points_s, box, True))[0]
        np.testing.assert_allclose(block.values, [0.4], atol=1.0e-12)
        np.testing.assert_allclose(block.vectors, [[-0.4, 0.0, 0.0]], atol=1.0e-12)

    def test_com_and_cog_use_expected_weights(self):
        chains = np.asarray([0, 0])
        residues = np.asarray([1, 1])
        positions = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        cog = _endpoint([0, 1], "G", "all", "cog", chains, residues)
        com = _endpoint([0, 1], "G", "all", "com", chains, residues)
        np.testing.assert_allclose(
            project_endpoint(cog, positions, [1.0, 3.0], pbc=False)[0],
            [[1.0, 0.0, 0.0]],
        )
        np.testing.assert_allclose(
            project_endpoint(com, positions, [1.0, 3.0], pbc=False)[0],
            [[1.5, 0.0, 0.0]],
        )

    def test_unique_interchain_pairs_require_identical_endpoints(self):
        chains = np.arange(4)
        residues = np.zeros(4, dtype=int)
        endpoint = _endpoint(range(4), "A", "chain", "cog", chains, residues)
        job = build_distance_job(
            "unique",
            endpoint,
            endpoint,
            "inter-chain",
            "pair",
            unique_entity_pairs=True,
        )
        self.assertEqual(job.channel_count, 6)

    def test_same_atom_exclusion_removes_zero_self_candidates(self):
        chains = np.asarray([0, 0])
        residues = np.asarray([1, 2])
        endpoint = _endpoint([0, 1], "R", "all", "atom", chains, residues)
        job = build_distance_job(
            "nonself",
            endpoint,
            endpoint,
            "corresponding",
            "min",
            exclusion="same-atom",
        )
        positions = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        points = project_endpoint(endpoint, positions, np.ones(2), pbc=False)
        block = list(iter_distance_blocks(job, points, points, pbc=False))[0]
        np.testing.assert_allclose(block.values, [1.0])
        np.testing.assert_allclose(block.vectors, [[-1.0, 0.0, 0.0]])


class DistanceAccumulatorTests(unittest.TestCase):
    def test_running_and_channel_statistics_match_numpy(self):
        values = np.asarray([[1.0, 2.0], [3.0, 6.0], [5.0, 4.0]])
        pooled = RunningStatistics()
        channel = ChannelStatistics(2)
        for frame in values:
            pooled.update(frame)
            channel.update(0, frame)
        self.assertEqual(pooled.count, values.size)
        self.assertAlmostEqual(pooled.mean, float(values.mean()))
        self.assertAlmostEqual(pooled.std, float(values.std()))
        np.testing.assert_allclose(channel.mean, values.mean(axis=0))
        np.testing.assert_allclose(channel.std, values.std(axis=0))

    def test_histogram_normalizations(self):
        histogram = HistogramAccumulator([0.0, 1.0, 2.0], 1)
        histogram.update_series(0, [0.2, 0.8, 1.2])
        np.testing.assert_array_equal(histogram.normalized("count"), [[2.0, 1.0]])
        np.testing.assert_allclose(
            histogram.normalized("probability"), [[2.0 / 3.0, 1.0 / 3.0]]
        )
        np.testing.assert_allclose(
            histogram.normalized("density"), [[2.0 / 3.0, 1.0 / 3.0]]
        )


if __name__ == "__main__":
    unittest.main()
