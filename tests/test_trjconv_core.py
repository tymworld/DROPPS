import unittest

import numpy as np

from dropps.commands.trjconv_core import (
    FitState,
    NoJumpState,
    build_whole_plan,
    center_coordinates,
    center_dense_phase,
    fit_coordinates,
    make_whole,
    select_time_indices,
    wrap_atoms,
    wrap_groups,
)


BOX = np.diag([10.0, 10.0, 10.0])


class TimeSelectionTests(unittest.TestCase):
    def test_equal_bounds_select_nearest_frame_and_prefer_earlier_tie(self):
        result = select_time_indices(
            [0.0, 1000.0, 2000.0],
            start_time=1.5,
            end_time=1.5,
            time_unit="ns",
        )

        np.testing.assert_array_equal(result.indices, [1])
        np.testing.assert_allclose(result.times_ps, [1000.0])
        self.assertTrue(result.single_time)

    def test_range_uses_real_times_and_nearest_delta_targets(self):
        result = select_time_indices(
            [0.0, 900.0, 2100.0, 3000.0, 4100.0],
            start_time=0.0,
            end_time=4.1,
            delta_time=2.0,
            time_unit="ns",
        )

        np.testing.assert_array_equal(result.indices, [0, 2, 4])
        np.testing.assert_allclose(result.times_ps, [0.0, 2100.0, 4100.0])

    def test_time_validation_rejects_bad_ranges_and_delta(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            select_time_indices([0.0, 1.0], start_time=-1.0, time_unit="ps")
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            select_time_indices([0.0, 1.0], delta_time=0.0, time_unit="ps")
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            select_time_indices(
                [0.0, 1.0],
                start_time=0.5,
                end_time=0.5,
                delta_time=0.1,
                time_unit="ps",
            )

    def test_float32_timestamp_roundoff_at_endpoint_is_accepted(self):
        result = select_time_indices(
            np.asarray([0.0, 1000.0], dtype=np.float32).astype(np.float64)
            + [0.0, -5.0e-5],
            start_time=1.0,
            end_time=1.0,
            time_unit="ns",
        )

        np.testing.assert_array_equal(result.indices, [1])


class PeriodicTransformTests(unittest.TestCase):
    def test_make_whole_reconstructs_a_bond_across_the_boundary(self):
        positions = np.asarray([[9.5, 1.0, 1.0], [0.5, 1.0, 1.0]])
        plan = build_whole_plan([[0, 1]], [[0, 1]], atom_count=2)

        result = make_whole(positions, BOX, plan)

        np.testing.assert_allclose(result[1] - result[0], [1.0, 0.0, 0.0])

    def test_atom_and_molecule_wrapping_have_distinct_semantics(self):
        positions = np.asarray([[9.5, 1.0, 1.0], [10.5, 1.0, 1.0]])

        atom_result = wrap_atoms(positions, BOX)
        molecule_result = wrap_groups(positions, BOX, [[0, 1]], [1.0, 1.0])

        np.testing.assert_allclose(atom_result[:, 0], [9.5, 0.5])
        np.testing.assert_allclose(molecule_result[:, 0], [-0.5, 0.5])
        np.testing.assert_allclose(
            molecule_result[1] - molecule_result[0], [1.0, 0.0, 0.0]
        )

    def test_nojump_tracks_continuous_motion_across_the_box(self):
        state = NoJumpState([[8.0, 0.0, 0.0]], BOX)

        first = state.apply([[9.0, 0.0, 0.0]], BOX)
        second = state.apply([[0.5, 0.0, 0.0]], BOX)

        np.testing.assert_allclose(first[0, 0], 9.0)
        np.testing.assert_allclose(second[0, 0], 10.5)

    def test_centering_can_be_restricted_to_one_axis(self):
        result = center_coordinates(
            [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]],
            BOX,
            [0, 1],
            "z",
        )

        np.testing.assert_allclose(result.mean(axis=0), [2.0, 3.0, 5.0])

    def test_dense_slab_crossing_boundary_is_centered(self):
        positions = np.asarray(
            [
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [9.8, 0.0, 0.0],
                [9.9, 0.0, 0.0],
                [5.0, 0.0, 0.0],
            ]
        )

        centered = center_dense_phase(
            positions,
            BOX,
            [0, 1, 2, 3, 4],
            np.ones(5),
            axis="x",
            threshold=0.6,
            bin_width=1.0,
        )
        wrapped = wrap_atoms(centered, BOX)

        dense_x = wrapped[:4, 0]
        self.assertTrue(np.all((dense_x > 4.0) & (dense_x < 6.0)))


class FitTransformTests(unittest.TestCase):
    def test_full_translation_aligns_all_three_axes(self):
        result = fit_coordinates(
            [[4.0, 5.0, 6.0]],
            [0],
            [[1.0, 2.0, 3.0]],
            "translation",
        )

        np.testing.assert_allclose(result, [[1.0, 2.0, 3.0]])

    def test_full_rotation_and_translation_fit_is_applied_to_all_atoms(self):
        reference = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        mobile = np.asarray(
            [[3.0, 4.0, 0.0], [3.0, 5.0, 0.0], [2.0, 4.0, 0.0]]
        )

        result = fit_coordinates(mobile, [0, 1], reference, "rot+trans")

        np.testing.assert_allclose(result[:2], reference, atol=1.0e-12)
        np.testing.assert_allclose(result[2], [0.0, 1.0, 0.0], atol=1.0e-12)

    def test_xy_translation_does_not_change_z(self):
        result = fit_coordinates(
            [[4.0, 5.0, 7.0]],
            [0],
            [[1.0, 2.0, 0.0]],
            "transxy",
        )

        np.testing.assert_allclose(result, [[1.0, 2.0, 7.0]])

    def test_xy_rotation_and_translation_leave_z_unchanged(self):
        reference = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        mobile = np.asarray(
            [[2.0, 3.0, 10.0], [2.0, 4.0, 11.0], [1.0, 3.0, 12.0]]
        )

        result = fit_coordinates(
            mobile,
            [0, 1],
            reference,
            "rotxy+transxy",
        )

        np.testing.assert_allclose(result[:2, :2], reference[:, :2], atol=1.0e-12)
        np.testing.assert_allclose(result[:, 2], mobile[:, 2])

    def test_progressive_fit_updates_its_reference(self):
        state = FitState(
            [0, 1],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            "progressive",
        )
        first = state.apply([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        second = state.apply([[4.0, 0.0, 0.0], [5.0, 0.0, 0.0]])

        np.testing.assert_allclose(first, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        np.testing.assert_allclose(second, first)


if __name__ == "__main__":
    unittest.main()
