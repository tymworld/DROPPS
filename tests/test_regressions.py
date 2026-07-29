import ast
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from dropps.analysis.density import _frame_windows
from dropps.fileio.itp_reader import ITPTopology, read_itp, write_itp
from dropps.fileio.pdb_reader import write_pdb
from dropps.fileio.xpm_reader import write_xpm
from dropps.fileio.xvg_reader import read_xvg
from dropps.share.build_system import _build_exclusion_pairs


class TopologyRegressionTests(unittest.TestCase):
    def test_write_itp_preserves_nrexcl(self):
        topology = ITPTopology(
            {
                "function_type_LJ": "Ashbaugh-Hatch",
                "function_type_Coulomb": "Debye-Huckel",
                "relative_permittivity_mode": "constant",
                "relative_permittivity": 80.0,
                "relative_permittivity_coeffs": None,
                "moleculetype": ["MOL", "2"],
                "atomtypes": [
                    {
                        "abbr": "A",
                        "name": "ALA",
                        "sigma": 0.5,
                        "lambda": 0.5,
                        "T0": 0.0,
                        "T1": 0.0,
                        "T2": 0.0,
                    }
                ],
                "epsilon": 1.0,
                "atoms": [
                    {
                        "abbr": "A",
                        "name": "ALA",
                        "residue": "ALA",
                        "residueid": 1,
                        "mass": 71.0,
                        "charge": 0.0,
                    }
                ],
            }
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "molecule.itp"
            write_itp(output, topology)
            reloaded = read_itp(output)

        self.assertEqual(reloaded.nrexcl, 2)

    def test_exclusions_are_per_molecule_and_unique(self):
        bond = lambda a1, a2: SimpleNamespace(a1=a1, a2=a2)
        nrexcl_one = SimpleNamespace(
            molecule_name="ONE",
            nrexcl=1,
            bonds=[bond(0, 1), bond(1, 2)],
        )
        nrexcl_two = SimpleNamespace(
            molecule_name="TWO",
            nrexcl=2,
            bonds=[bond(0, 1), bond(1, 2), bond(2, 0)],
        )

        exclusions = _build_exclusion_pairs(
            [nrexcl_one, nrexcl_two], [0, 3]
        )

        self.assertEqual(
            exclusions,
            [[0, 1], [1, 2], [3, 4], [3, 5], [4, 5]],
        )


class FileIoRegressionTests(unittest.TestCase):
    def test_read_xvg_accepts_gromacs_comments(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "data.xvg"
            path.write_text(
                "@ title \"test\"\n# comment\n0 1\n2 3\n",
                encoding="utf-8",
            )
            data = read_xvg(path)

        np.testing.assert_array_equal(data, [[0.0, 1.0], [2.0, 3.0]])

    def test_read_xvg_raises_a_clear_error(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "broken.xvg"
            path.write_text("not numeric\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Cannot load XVG"):
                read_xvg(path)

    def test_write_xpm_supports_continuous_values(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "image.xpm"
            write_xpm(np.array([[0.0, 0.5], [2.0, 10.0]]), path, "Map")
            lines = path.read_text(encoding="utf-8").splitlines()

        quoted_lines = [line for line in lines if line.startswith('"')]
        self.assertEqual(quoted_lines[0], '"2 2 64 1",')
        self.assertEqual(len(quoted_lines), 1 + 64 + 2)

    def test_write_pdb_accepts_blank_chain_id(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "blank-chain.pdb"
            write_pdb(
                path,
                [1.0, 1.0, 1.0],
                ["ATOM"],
                [1],
                ["A"],
                ["MOL"],
                [""],
                [1],
                [0.0],
                [0.0],
                [0.0],
                [1.0],
                [0.0],
                ["C"],
                [1],
            )
            atom_line = path.read_text(encoding="utf-8").splitlines()[1]

        self.assertEqual(atom_line[21], " ")


class BoundaryAndCompatibilityRegressionTests(unittest.TestCase):
    def test_single_frame_density_window_is_not_dropped(self):
        self.assertEqual(_frame_windows(5, 5, 10), [(5, 6)])

    def test_density_windows_include_the_last_frame(self):
        self.assertEqual(
            _frame_windows(0, 4, 2), [(0, 2), (2, 4), (4, 5)]
        )

    def test_all_python_sources_parse_as_python_310(self):
        repository = Path(__file__).resolve().parents[1]
        for source_path in repository.rglob("*.py"):
            if source_path.name.startswith("._"):
                continue
            with self.subTest(source=str(source_path.relative_to(repository))):
                source = source_path.read_text(encoding="utf-8")
                ast.parse(source, filename=str(source_path), feature_version=(3, 10))


if __name__ == "__main__":
    unittest.main()
