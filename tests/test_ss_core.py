import unittest

import numpy as np

from dropps.analysis.ss_core import (
    STATE_HELIX,
    STATE_MIXED,
    STATE_STRAND,
    apply_strand_pairing,
    calculate_backbone_geometry,
    classify_secondary_structure,
    consensus_states,
    detect_interchain_strand_pairing,
)


def _helix_coordinates(length=12, handedness=1.0):
    rotation = handedness * np.deg2rad(100.0)
    rise = 0.15
    bond = 0.38
    radius = np.sqrt((bond**2 - rise**2) / (2.0 - 2.0 * np.cos(rotation)))
    index = np.arange(length, dtype=np.float64)
    return np.column_stack(
        (
            radius * np.cos(index * rotation),
            radius * np.sin(index * rotation),
            rise * index,
        )
    )


def _strand_coordinates(length=10):
    angle = np.deg2rad(28.0)
    directions = np.column_stack(
        (
            np.full(length - 1, np.cos(angle)),
            np.sin(angle) * np.where(np.arange(length - 1) % 2 == 0, 1.0, -1.0),
            np.zeros(length - 1),
        )
    )
    return np.vstack((np.zeros(3), np.cumsum(0.38 * directions, axis=0)))


class SecondaryStructureGeometryTests(unittest.TestCase):
    def test_ideal_right_handed_helix_is_assigned_h(self):
        coordinates = _helix_coordinates()

        geometry = calculate_backbone_geometry(coordinates)
        result = classify_secondary_structure(coordinates, method="psea")

        self.assertAlmostEqual(np.nanmedian(geometry.angle_degrees), 90.53, places=1)
        self.assertAlmostEqual(np.nanmedian(geometry.dihedral_degrees), 50.39, places=1)
        np.testing.assert_array_equal(result.states, STATE_HELIX)

    def test_mirror_helix_is_rejected_by_chirality_prior(self):
        coordinates = _helix_coordinates(handedness=-1.0)

        result = classify_secondary_structure(coordinates, method="psea")

        self.assertFalse(np.any(result.helix))

    def test_planar_zigzag_is_assigned_e(self):
        coordinates = _strand_coordinates()

        geometry = calculate_backbone_geometry(coordinates)
        result = classify_secondary_structure(coordinates, method="psea")

        self.assertAlmostEqual(np.nanmedian(geometry.angle_degrees), 124.0, places=1)
        np.testing.assert_array_equal(result.states, STATE_STRAND)

    def test_bond_normalization_preserves_scaled_geometry(self):
        coordinates = _helix_coordinates()

        reference = classify_secondary_structure(coordinates, method="psea")
        scaled = classify_secondary_structure(1.7 * coordinates, method="psea")

        np.testing.assert_array_equal(scaled.states, reference.states)


class StrandPairingTests(unittest.TestCase):
    def test_parallel_registered_strands_are_retained(self):
        strand = _strand_coordinates(8)
        coordinates = np.stack((strand, strand + [0.0, 0.0, 0.5]))
        local = np.ones((2, 8), dtype=bool)
        pairs = np.asarray([[0, index, 1, index] for index in range(8)])
        vectors = np.tile([0.0, 0.0, 0.5], (8, 1))

        paired, records = detect_interchain_strand_pairing(
            coordinates,
            local,
            pairs,
            vectors,
        )

        self.assertTrue(np.all(paired))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].orientation, "parallel")

    def test_antiparallel_registered_strands_are_retained(self):
        strand = _strand_coordinates(8)
        coordinates = np.stack((strand, strand[::-1] + [0.0, 0.0, 0.5]))
        local = np.ones((2, 8), dtype=bool)
        pairs = np.asarray([[0, index, 1, 7 - index] for index in range(8)])
        vectors = np.tile([0.0, 0.0, 0.5], (8, 1))

        paired, records = detect_interchain_strand_pairing(
            coordinates,
            local,
            pairs,
            vectors,
        )

        self.assertTrue(np.all(paired))
        self.assertEqual(records[0].orientation, "antiparallel")

    def test_isolated_contact_does_not_confirm_a_strand(self):
        strand = _strand_coordinates(8)
        coordinates = np.stack((strand, strand + [0.0, 0.0, 0.5]))
        local = np.ones((2, 8), dtype=bool)

        paired, records = detect_interchain_strand_pairing(
            coordinates,
            local,
            [[0, 3, 1, 3]],
            [[0.0, 0.0, 0.5]],
        )

        self.assertFalse(np.any(paired))
        self.assertEqual(records, [])
        states = np.full((2, 8), STATE_STRAND, dtype=np.uint8)
        self.assertFalse(np.any(apply_strand_pairing(states, paired) == STATE_STRAND))


class ConsensusTests(unittest.TestCase):
    def test_subthreshold_plurality_is_marked_mixed(self):
        fractions = np.asarray(
            [[0.4, 0.3, 0.3], [0.1, 0.7, 0.2], [0.0, 0.5, 0.5]]
        )

        states = consensus_states(fractions, threshold=0.5)

        np.testing.assert_array_equal(
            states,
            [STATE_MIXED, STATE_HELIX, STATE_MIXED],
        )


if __name__ == "__main__":
    unittest.main()
