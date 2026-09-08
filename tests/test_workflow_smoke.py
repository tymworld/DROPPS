from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from dropps.dps import main
from dropps.fileio.pdb_reader import read_pdb
from dropps.fileio.tpr_reader import read_tpr
from openmm.unit import nanometer


ROOT = Path(__file__).resolve().parents[1]
SMOKE_MDP = ROOT / "tests" / "data" / "smoke.mdp"


def run_quiet(arguments):
    output = io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
        code = main(["dps", *map(str, arguments)])
    if code != 0:
        raise AssertionError(
            f"dps {' '.join(map(str, arguments))} returned {code}:\n{output.getvalue()}"
        )
    return output.getvalue()


class WorkflowSmokeTests(unittest.TestCase):
    def test_all_atom_ca_mapping_is_centered_inside_the_output_box(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            all_atom = work / "all-atom.pdb"
            all_atom.write_text(
                "MODEL        1\n"
                "ATOM      1  CA  ALA A   1      10.000  20.000  30.000  1.00  0.00           C\n"
                "ATOM      2  CA  GLY A   2      12.000  22.000  31.000  1.00  0.00           C\n"
                "ATOM      3  CA  ALA A   3      14.000  24.000  32.000  1.00  0.00           C\n"
                "ENDMDL\n"
                "MODEL        2\n"
                "ATOM      4  CA  ALA A   1     999.000 999.000 999.000  1.00  0.00           C\n"
                "ENDMDL\n",
                encoding="utf-8",
            )
            mapped = work / "mapped.pdb"
            run_quiet(
                [
                    "pdb2dps",
                    "-s",
                    "AGA",
                    "-f",
                    all_atom,
                    "-ff",
                    "HPS",
                    "-oc",
                    mapped,
                ]
            )
            with contextlib.redirect_stdout(io.StringIO()):
                atoms, box = read_pdb(mapped)
            lengths = box.value_in_unit(nanometer)
            coordinates = [
                [atom[axis].value_in_unit(nanometer) for axis in ("x", "y", "z")]
                for atom in atoms
            ]
            self.assertEqual(len(atoms), 3)
            for coordinate in coordinates:
                self.assertTrue(
                    all(
                        0 < value < length for value, length in zip(coordinate, lengths)
                    )
                )
            self.assertAlmostEqual(coordinates[2][0] - coordinates[0][0], 0.4)

    def test_sequence_to_analysis_workflow_is_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            first_pdb = work / "chain-a.pdb"
            first_itp = work / "chain-a.itp"
            second_pdb = work / "chain-b.pdb"
            second_itp = work / "chain-b.itp"

            common = [
                "pdb2dps",
                "-s",
                "AGAG",
                "-ff",
                "HPS",
                "--seed",
                "2026",
                "-on",
                "PEP",
            ]
            run_quiet([*common, "-oc", first_pdb, "-op", first_itp])
            run_quiet([*common, "-oc", second_pdb, "-op", second_itp])
            self.assertEqual(first_pdb.read_bytes(), second_pdb.read_bytes())
            self.assertEqual(first_itp.read_bytes(), second_itp.read_bytes())

            system_pdb = work / "system.pdb"
            system_top = work / "system.top"
            run_quiet(
                [
                    "genmesh",
                    "-f",
                    first_pdb,
                    "-p",
                    first_itp,
                    "-n",
                    "1",
                    "-mesh",
                    "1",
                    "1",
                    "1",
                    "-mx",
                    "4",
                    "-my",
                    "4",
                    "-mz",
                    "4",
                    "-oc",
                    system_pdb,
                    "-op",
                    system_top,
                ]
            )

            tpr_path = work / "run.tpr"
            run_quiet(
                [
                    "grompp",
                    "-f",
                    system_pdb,
                    "-p",
                    system_top,
                    "-m",
                    SMOKE_MDP,
                    "-o",
                    tpr_path,
                ]
            )
            tpr = read_tpr(tpr_path)
            self.assertEqual(tpr.format_version, 2)
            self.assertEqual(tpr.metadata["integrator_semantics"], "langevin-middle")
            self.assertEqual(tpr.mdtopology.getNumAtoms(), 4)
            self.assertTrue(tpr.metadata["sources"])

            run_prefix = work / "short"
            run_quiet(
                [
                    "mdrun",
                    "-s",
                    tpr_path,
                    "-o",
                    run_prefix,
                    "--platform",
                    "Reference",
                    "-cpt",
                    "0",
                ]
            )
            for suffix in (".xtc", ".edr", ".log", ".chk", ".state.xml", ".pdb"):
                self.assertTrue(Path(f"{run_prefix}{suffix}").is_file(), suffix)

            run_quiet(["check", "-s", tpr_path, "-f", f"{run_prefix}.xtc"])
            density = work / "density.xvg"
            run_quiet(
                [
                    "density",
                    "-s",
                    tpr_path,
                    "-f",
                    f"{run_prefix}.xtc",
                    "-o",
                    density,
                    "-selfit",
                    "0",
                    "-sel",
                    "0",
                    "--center-mode",
                    "none",
                ]
            )
            self.assertIn("@", density.read_text(encoding="utf-8"))

            msd = work / "msd.xvg"
            run_quiet(
                [
                    "msd",
                    "-s",
                    tpr_path,
                    "-f",
                    f"{run_prefix}.xtc",
                    "-sel",
                    "0",
                    "-o",
                    msd,
                ]
            )
            self.assertIn("Mean-square displacement", msd.read_text(encoding="utf-8"))

            steep_mdp = work / "steep.mdp"
            steep_mdp.write_text(
                SMOKE_MDP.read_text(encoding="utf-8")
                .replace("integrator = Langevin", "integrator = steep")
                .replace("gen-vel = True", "gen-vel = False"),
                encoding="utf-8",
            )
            steep_tpr = work / "steep.tpr"
            run_quiet(
                [
                    "grompp",
                    "-f",
                    system_pdb,
                    "-p",
                    system_top,
                    "-m",
                    steep_mdp,
                    "-o",
                    steep_tpr,
                ]
            )
            self.assertEqual(
                read_tpr(steep_tpr).metadata["integrator_semantics"],
                "steep-minimization",
            )
            minimized = work / "minimized"
            run_quiet(
                [
                    "mdrun",
                    "-s",
                    steep_tpr,
                    "-o",
                    minimized,
                    "--platform",
                    "Reference",
                    "-cpt",
                    "0",
                ]
            )
            for suffix in (".chk", ".state.xml", ".pdb"):
                self.assertTrue(Path(f"{minimized}{suffix}").is_file(), suffix)
            self.assertFalse(Path(f"{minimized}.xtc").exists())

    def test_grompp_rejects_cutoff_at_half_the_shortest_box(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            pdb = work / "chain.pdb"
            itp = work / "chain.itp"
            run_quiet(
                [
                    "pdb2dps",
                    "-s",
                    "AGA",
                    "-ff",
                    "HPS",
                    "-oc",
                    pdb,
                    "-op",
                    itp,
                ]
            )
            system_pdb = work / "system.pdb"
            system_top = work / "system.top"
            run_quiet(
                [
                    "genmesh",
                    "-f",
                    pdb,
                    "-p",
                    itp,
                    "-n",
                    "1",
                    "-mesh",
                    "1",
                    "1",
                    "1",
                    "-mx",
                    "2",
                    "-my",
                    "2",
                    "-mz",
                    "2",
                    "-oc",
                    system_pdb,
                    "-op",
                    system_top,
                ]
            )

            with contextlib.redirect_stdout(io.StringIO()):
                _, box = read_pdb(system_pdb)
            half_shortest = min(box.value_in_unit(nanometer)) / 2
            invalid_mdp = work / "cutoff-equality.mdp"
            invalid_mdp.write_text(
                SMOKE_MDP.read_text(encoding="utf-8")
                .replace("cutoff-lj = 1.0", f"cutoff-lj = {half_shortest}")
                .replace("cutoff-coul = 1.0", f"cutoff-coul = {half_shortest}"),
                encoding="utf-8",
            )

            output = io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                code = main(
                    [
                        "dps",
                        "grompp",
                        "-f",
                        str(system_pdb),
                        "-p",
                        str(system_top),
                        "-m",
                        str(invalid_mdp),
                        "-o",
                        str(work / "invalid.tpr"),
                    ]
                )
            self.assertEqual(code, 1)
            self.assertIn("half the shortest box length", output.getvalue())


if __name__ == "__main__":
    unittest.main()
