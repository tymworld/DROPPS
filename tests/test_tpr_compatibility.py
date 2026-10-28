import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from openmm import unit

from dropps.commands.convert_tpr import convert_tpr
from dropps.fileio import itp_reader
from dropps.fileio.itp_reader import Angle, Atomtype_AH_DH, Bond
from dropps.fileio.tpr_reader import (
    TPRFormatError,
    is_portable_tpr,
    read_tpr,
    write_tpr,
)
from tests.test_mdrun_resume import _runtime


def _runtime_with_current_atomtypes(nsteps=2):
    runtime = _runtime(nsteps=nsteps, checkpoint_interval=0)
    runtime["ITP_list"] = [
        SimpleNamespace(
            molecule_name="MOL",
            nrexcl=1,
            function_type_LJ="Ashbaugh-Hatch",
            function_type_Coulomb="Debye-Huckel",
            relative_permittivity_mode="constant",
            relative_permittivity=80.0,
            relative_permittivity_coeffs=None,
            simulation_settings=[],
            typelist=["H"],
            atomtypes=[Atomtype_AH_DH("H", "helium", 0.4, 0.5, 0.0, 0.0, 0.0)],
            atoms=[],
            bonds=[
                Bond(
                    0,
                    1,
                    0.4 * unit.nanometer,
                    10
                    * unit.kilojoule_per_mole
                    / unit.nanometer**2,
                )
            ],
            angles=[
                Angle(
                    0,
                    1,
                    2,
                    120 * unit.degree,
                    5 * unit.kilojoule_per_mole / unit.radian**2,
                )
            ],
            type2sigma={"H": 0.4},
            type2mylambda={"H": 0.5},
            epsilon=1.0,
        )
    ]
    return runtime


class TPRCompatibilityTests(unittest.TestCase):
    def test_portable_tpr_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "run.tpr"
            source = Path(temporary_directory) / "input.mdp"
            source.write_text("integrator = Langevin\n", encoding="utf-8")

            manifest = write_tpr(
                path,
                _runtime_with_current_atomtypes(),
                sources={"parameters": source},
            )
            content = read_tpr(path)

            self.assertTrue(is_portable_tpr(path))
            self.assertEqual(content.format_version, 2)
            self.assertEqual(content.run_id, manifest["run_id"])
            self.assertEqual(content.mdsystem.getNumParticles(), 10)
            self.assertEqual(content.mdtopology.getNumAtoms(), 10)
            np.testing.assert_allclose(
                content.positions.value_in_unit(unit.nanometer),
                _runtime_with_current_atomtypes()["positions"].value_in_unit(
                    unit.nanometer
                ),
            )
            self.assertEqual(content.itp_list[0].atomtypes[0].abbr, "H")

    def test_current_legacy_pickle_tpr_is_supported_and_convertible(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "legacy.tpr"
            destination = Path(temporary_directory) / "portable.tpr"
            with source.open("wb") as stream:
                pickle.dump(_runtime_with_current_atomtypes(), stream)

            legacy = read_tpr(source)
            self.assertEqual(legacy.format_version, 1)
            self.assertEqual(legacy.itp_list[0].atomtypes[0].abbr, "H")
            self.assertEqual(
                legacy.itp_list[0].bonds[0].k.value_in_unit(
                    unit.kilojoule_per_mole / unit.nanometer**2
                ),
                10,
            )

            convert_tpr(
                SimpleNamespace(run_input=str(source), output=str(destination))
            )
            self.assertEqual(read_tpr(destination).format_version, 2)

    def test_pre_split_atomtype_class_is_intentionally_unsupported(self):
        old_atomtype = type("Atomtype", (), {})
        old_atomtype.__module__ = itp_reader.__name__
        setattr(itp_reader, "Atomtype", old_atomtype)
        try:
            with tempfile.TemporaryDirectory() as temporary_directory:
                path = Path(temporary_directory) / "too-old.tpr"
                with path.open("wb") as stream:
                    pickle.dump({"ITP_list": [old_atomtype()]}, stream)
                with self.assertRaisesRegex(
                    TPRFormatError,
                    "predates the Atomtype_AH_DH/Atomtype_WF_DH format",
                ):
                    read_tpr(path)
        finally:
            delattr(itp_reader, "Atomtype")

    def test_system_identity_ignores_continuation_length(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            first = Path(temporary_directory) / "first.tpr"
            second = Path(temporary_directory) / "second.tpr"
            first_manifest = write_tpr(first, _runtime_with_current_atomtypes(2))
            second_manifest = write_tpr(second, _runtime_with_current_atomtypes(4))
            self.assertNotEqual(first_manifest["run_id"], second_manifest["run_id"])
            self.assertEqual(
                first_manifest["system_id"], second_manifest["system_id"]
            )


if __name__ == "__main__":
    unittest.main()
