import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from dropps.analysis.densmap import (
    _group_output_paths,
    _quantity_configuration,
    densmap,
    getargs_densmap,
    write_density_map_dat,
)
from dropps.analysis.density_core import DALTON_PER_NM3_TO_MG_PER_ML
from dropps.analysis.densmap_core import DEFAULT_DENSMAP_BIN_WIDTH_NM


class _Group:
    def __init__(self, indices, masses, charges):
        self.indices = np.asarray(indices, dtype=int)
        self.masses = np.asarray(masses, dtype=float)
        self.charges = np.asarray(charges, dtype=float)

    def __len__(self):
        return len(self.indices)


class _TrajectoryFrames:
    def __init__(self, frames):
        self.frames = frames

    def __iter__(self):
        return iter(self.frames)

    def __getitem__(self, index):
        return self.frames[index]


class _FakeTrajectory:
    def __init__(self, positions_by_frame=None, masses=None, charges=None):
        if positions_by_frame is None:
            positions_by_frame = [
                np.asarray(
                    [
                        [5.0, 5.0, 10.0],
                        [15.0, 5.0, 10.0],
                        [5.0, 15.0, 10.0],
                        [15.0, 15.0, 10.0],
                    ]
                )
            ]
        frames = [
            SimpleNamespace(
                time=float(frame_index),
                positions=np.asarray(positions, dtype=float),
                triclinic_dimensions=np.diag([20.0, 20.0, 20.0]),
            )
            for frame_index, positions in enumerate(positions_by_frame)
        ]
        particle_count = len(frames[0].positions)
        if masses is None:
            masses = [0.0] * particle_count
        if charges is None:
            charges = [0.0] * particle_count
        self.Universe = SimpleNamespace(trajectory=_TrajectoryFrames(frames))
        self.index = SimpleNamespace(print_all=lambda: None)
        self.group = _Group(range(particle_count), masses, charges)

    def getSelection(self, selection, help_text):
        return self.group, "System"


class DensmapArgumentTests(unittest.TestCase):
    def test_gromacs_compatible_planar_options_are_available(self):
        args = getargs_densmap(
            [
                "-s",
                "run.tpr",
                "-f",
                "run.xtc",
                "-od",
                "densmap.dat",
                "-aver",
                "y",
                "-tp",
                "charge",
                "-unit",
                "nm-2",
                "-n1",
                "20",
                "-n2",
                "30",
                "-sel",
                "1",
                "2",
            ]
        )

        self.assertEqual(args.output, "densmap.dat")
        self.assertEqual(args.average_axis, "y")
        self.assertEqual(args.type, "charge")
        self.assertEqual(args.normalization, "nm-2")
        self.assertEqual((args.bins_first, args.bins_second), (20, 30))
        self.assertEqual(args.selection, [1, 2])
        self.assertEqual(args.center_mode, "none")

    def test_default_grid_spacing_matches_gromacs_densmap(self):
        args = getargs_densmap(
            ["-s", "run.tpr", "-f", "run.xtc", "-o", "map.dat"]
        )

        self.assertEqual(args.bin_width, DEFAULT_DENSMAP_BIN_WIDTH_NM)

    def test_multiple_groups_receive_stable_distinct_paths(self):
        paths = _group_output_paths("map.dat", ["Protein A", "Protein/A"], ".dat")

        self.assertEqual(paths, [Path("map.1-Protein-A.dat"), Path("map.2-Protein-A.dat")])

    def test_mass_number_and_charge_weights_have_explicit_units(self):
        group = _Group([0, 1], [10.0, 20.0], [1.0, -1.0])

        mass, mass_conversion, mass_unit = _quantity_configuration(
            group, "mass", "volume"
        )
        number, number_conversion, number_unit = _quantity_configuration(
            group, "number", "volume"
        )
        charge, charge_conversion, charge_unit = _quantity_configuration(
            group, "charge", "volume"
        )

        np.testing.assert_array_equal(mass, [10.0, 20.0])
        self.assertEqual(mass_conversion, DALTON_PER_NM3_TO_MG_PER_ML)
        self.assertEqual(mass_unit, "mg/mL")
        np.testing.assert_array_equal(number, [1.0, 1.0])
        self.assertEqual((number_conversion, number_unit), (1.0, "nm^-3"))
        np.testing.assert_array_equal(charge, [1.0, -1.0])
        self.assertEqual((charge_conversion, charge_unit), (1.0, "e/nm^3"))


class DensmapOutputTests(unittest.TestCase):
    def test_dat_matrix_keeps_coordinates_and_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "map.dat"
            write_density_map_dat(
                path,
                [0.5, 1.5],
                [0.25, 0.75],
                [[1.0, 2.0], [3.0, 4.0]],
                {"unit": "nm^-3", "average_axis": "z"},
            )
            text = path.read_text(encoding="utf-8")
            matrix = np.loadtxt(path)

        self.assertIn("# unit=nm^-3", text)
        np.testing.assert_allclose(
            matrix,
            [[0.0, 0.25, 0.75], [0.5, 1.0, 2.0], [1.5, 3.0, 4.0]],
        )

    def test_command_normalizes_each_planar_cell_by_its_3d_volume(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "map.dat"
            args = getargs_densmap(
                [
                    "-s",
                    "run.tpr",
                    "-f",
                    "run.xtc",
                    "-o",
                    str(output),
                    "-sel",
                    "0",
                    "-tp",
                    "number",
                    "-aver",
                    "z",
                    "-n1",
                    "2",
                    "-n2",
                    "2",
                ]
            )
            with patch(
                "dropps.analysis.densmap.trajectory_class",
                return_value=_FakeTrajectory(),
            ):
                densmap(args)
            matrix = np.loadtxt(output)

        np.testing.assert_allclose(matrix[1:, 1:], np.full((2, 2), 0.5))

    def test_frame_centering_aligns_a_droplet_that_crosses_pbc(self):
        positions_by_frame = [
            [[19.0, 19.0, 10.0], [1.0, 1.0, 10.0]],
            [[4.0, 4.0, 10.0], [6.0, 6.0, 10.0]],
        ]
        fake = _FakeTrajectory(positions_by_frame, masses=[1.0, 1.0])

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "centered.dat"
            args = getargs_densmap(
                [
                    "-s",
                    "run.tpr",
                    "-f",
                    "run.xtc",
                    "-o",
                    str(output),
                    "-sel",
                    "0",
                    "-tp",
                    "number",
                    "-unit",
                    "count",
                    "-n1",
                    "2",
                    "-n2",
                    "2",
                    "--center-mode",
                    "frame",
                ]
            )
            with patch(
                "dropps.analysis.densmap.trajectory_class",
                return_value=fake,
            ):
                densmap(args)
            density_map = np.loadtxt(output)[1:, 1:]

        np.testing.assert_allclose(density_map, [[1.0, 0.0], [0.0, 1.0]])


if __name__ == "__main__":
    unittest.main()
