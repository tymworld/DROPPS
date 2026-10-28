import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from dropps.analysis.sasa import (
    _box_geometry_nm,
    _periodic_neighbor_vectors,
    _resolve_selections,
    getargs_sasa,
    sasa,
)
from dropps.analysis.sasa_core import fibonacci_sphere_points, shrake_rupley_areas
from dropps.fileio.xvg_reader import read_xvg


class _FakeGroup:
    def __init__(self, universe, indices):
        self.universe = universe
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return self.indices.size

    @property
    def positions(self):
        return self.universe.trajectory.current.positions[self.indices]


class _FakeReader:
    def __init__(self, frames):
        self.frames = frames
        self.current = frames[0]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        self.current = self.frames[index]
        return self.current


class _FakeUniverse:
    def __init__(self, frames, atom_count):
        self.trajectory = _FakeReader(frames)
        self.atoms = _FakeGroup(self, np.arange(atom_count))


class _FakeIndex:
    def print_all(self):
        return None


class _FakeTrajectory:
    def __init__(self, frames):
        self.Universe = _FakeUniverse(frames, 2)
        atoms = [SimpleNamespace(abbr="A"), SimpleNamespace(abbr="B")]
        self.tpr = SimpleNamespace(
            itp_list=[
                SimpleNamespace(
                    atoms=atoms,
                    type2sigma={"A": 0.60, "B": 0.60},
                )
            ]
        )
        self.index = _FakeIndex()

    def num_atoms(self):
        return 2

    def time2frame(self, start_time, end_time, delta_time):
        return 0, len(self.Universe.trajectory) - 1, 1

    def getSelection(self, specification, help="analyzing group"):
        if specification in {"0", "group 0"}:
            return self.Universe.atoms, "SystemFromIndex"
        if specification in {"1", "group 1"}:
            return _FakeGroup(self.Universe, [0]), "FirstBead"
        raise ValueError(f"Unknown fake selection {specification}")


class SASACommandTests(unittest.TestCase):
    def test_parser_accepts_gromacs_style_surface_and_multiple_outputs(self):
        args = getargs_sasa(
            [
                "-s",
                "run.tpr",
                "-f",
                "run.xtc",
                "-o",
                "sasa.xvg",
                "-surface",
                "0",
                "-output",
                "1",
                "2",
                "-output",
                "res ARG | res LYS",
            ]
        )
        self.assertEqual(args.surface, "0")
        self.assertEqual(
            args.output_selections,
            [["1", "2"], ["res ARG | res LYS"]],
        )

    def test_noninteractive_default_surface_uses_all_atoms_not_index_group_zero(self):
        frame = SimpleNamespace(
            positions=np.zeros((2, 3)),
            dimensions=np.asarray([100.0, 100.0, 100.0, 90.0, 90.0, 90.0]),
            time=0.0,
        )
        trajectory = _FakeTrajectory([frame])
        args = SimpleNamespace(surface=None, output_selections=None)
        surface, name, outputs = _resolve_selections(
            trajectory,
            args,
            interactive=False,
        )
        np.testing.assert_array_equal(surface.indices, [0, 1])
        self.assertEqual(name, "System")
        self.assertEqual(outputs, [])

    def test_output_must_be_subset_of_surface(self):
        frame = SimpleNamespace(
            positions=np.zeros((2, 3)),
            dimensions=np.asarray([100.0, 100.0, 100.0, 90.0, 90.0, 90.0]),
            time=0.0,
        )
        trajectory = _FakeTrajectory([frame])
        args = SimpleNamespace(surface="1", output_selections=[["0"]])
        with self.assertRaisesRegex(ValueError, "not a subset"):
            _resolve_selections(trajectory, args, interactive=False)

    def test_output_rejects_duplicate_atoms(self):
        frame = SimpleNamespace(
            positions=np.zeros((2, 3)),
            dimensions=np.asarray([100.0, 100.0, 100.0, 90.0, 90.0, 90.0]),
            time=0.0,
        )
        trajectory = _FakeTrajectory([frame])
        duplicate_group = _FakeGroup(trajectory.Universe, [0, 0])
        with patch.object(
            trajectory,
            "getSelection",
            return_value=(duplicate_group, "Duplicate"),
        ):
            args = SimpleNamespace(surface="all", output_selections=[["duplicate"]])
            with self.assertRaisesRegex(ValueError, "duplicate atoms"):
                _resolve_selections(trajectory, args, interactive=False)

    def test_triclinic_periodic_search_reduces_large_lattice_shifts(self):
        dimensions_angstrom = np.asarray(
            [100.0, 110.0, 120.0, 80.0, 75.0, 70.0],
            dtype=np.float64,
        )
        _, cell_nm = _box_geometry_nm(dimensions_angstrom)
        near = np.asarray(
            [[0.10, 0.10, 0.10], [0.13, 0.10, 0.10]],
            dtype=np.float64,
        ) @ cell_nm
        far = near.copy()
        far[0] += 100.0 * cell_nm[0] - 80.0 * cell_nm[2]
        far[1] += -120.0 * cell_nm[1] + 65.0 * cell_nm[2]
        expanded_radii = np.asarray([0.30, 0.30], dtype=np.float64)

        near_neighbors = _periodic_neighbor_vectors(
            near,
            expanded_radii,
            dimensions_angstrom,
        )
        far_neighbors = _periodic_neighbor_vectors(
            far,
            expanded_radii,
            dimensions_angstrom,
        )
        points = fibonacci_sphere_points(960)
        near_areas = shrake_rupley_areas(
            expanded_radii,
            *near_neighbors,
            points,
        )
        far_areas = shrake_rupley_areas(
            expanded_radii,
            *far_neighbors,
            points,
        )
        np.testing.assert_allclose(far_areas, near_areas, atol=1e-12)

    def test_command_writes_surface_and_subset_series_for_far_coordinates(self):
        dimensions = np.asarray([100.0, 100.0, 100.0, 90.0, 90.0, 90.0])
        frames = [
            SimpleNamespace(
                positions=np.asarray([[10001.0, 0.0, 0.0], [-9995.0, 0.0, 0.0]]),
                dimensions=dimensions,
                time=0.0,
            ),
            SimpleNamespace(
                positions=np.asarray([[20001.0, 0.0, 0.0], [-19995.0, 0.0, 0.0]]),
                dimensions=dimensions,
                time=1000.0,
            ),
        ]
        trajectory = _FakeTrajectory(frames)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "sasa"
            args = getargs_sasa(
                [
                    "-s",
                    "run.tpr",
                    "-f",
                    "run.xtc",
                    "-o",
                    str(output),
                    "-surface",
                    "all",
                    "-output",
                    "1",
                    "--sphere-points",
                    "960",
                ]
            )
            with patch("dropps.analysis.sasa.trajectory_class", return_value=trajectory):
                sasa(args)

            output_path = output.with_suffix(".xvg")
            data = np.atleast_2d(read_xvg(output_path))
            self.assertEqual(data.shape, (2, 3))
            np.testing.assert_allclose(data[:, 0], [0.0, 1.0])
            np.testing.assert_allclose(data[0, 1:], data[1, 1:], atol=1e-12)
            self.assertAlmostEqual(data[0, 2] * 2.0, data[0, 1], places=9)
            text = output_path.read_text(encoding="utf-8")
            self.assertIn('@ s0 legend "Surface:System"', text)
            self.assertIn('@ s1 legend "FirstBead"', text)
            self.assertIn("pbc=minimum-image-per-frame", text)


if __name__ == "__main__":
    unittest.main()
