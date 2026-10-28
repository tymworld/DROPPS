import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from dropps.commands.report_methods import (
    build_methods_report,
    render_latex,
    render_text,
    report_methods,
)


class _Residue:
    def __init__(self, name):
        self.name = name


class _Chain:
    def __init__(self, residue_names):
        self._residues = [_Residue(name) for name in residue_names]

    def residues(self):
        return iter(self._residues)


class _Topology:
    def __init__(self):
        self._chains = [_Chain(["ALA", "GLY"]), _Chain(["ALA", "GLY"])]

    def chains(self):
        return iter(self._chains)

    def bonds(self):
        return iter(((0, 1), (2, 3)))

    def getNumAtoms(self):
        return 4


def _content():
    parameters = {
        "integrator": "Langevin",
        "dt": 0.01,
        "nsteps": 100000,
        "production_temperature": 300.0,
        "initial_temperature": 150.0,
        "friction": 1.0,
        "pcoulp": False,
        "bondtype": "bond",
        "comm_mode": "Linear",
        "minimize": True,
        "vdwtype": "pLJ",
        "cutoff_scheme_lj": "static",
        "cutoff_lj": 1.5,
        "cutoff_lj_multi": 3.0,
        "shift_lj": True,
        "coulombtype": "yukawa",
        "cutoff_coul": 1.5,
        "salt_conc": 0.1,
        "shift_coul": True,
        "nst_xout": 1000,
        "nst_energy": 10000,
        "energy_grps": "potentialEnergy,temperature",
        "nst_stress": 0,
    }
    return SimpleNamespace(
        parameters=parameters,
        metadata={
            "format_version": 2,
            "dropps_version": "0.5.0b4",
            "openmm_version": "8.5.2",
            "python_version": "3.13.7",
            "integrator_semantics": "langevin-middle",
            "run_id": "run_id_with_underscore",
            "system_id": "system-id",
            "sources": {"sources/parameters/md.mdp": {}},
        },
        mdtopology=_Topology(),
        mdsystem=SimpleNamespace(getNumConstraints=lambda: 0),
        itp_list=[
            SimpleNamespace(
                molecule_name="PROT",
                function_type_LJ="Ashbaugh-Hatch",
                function_type_Coulomb="Debye-Huckel",
            )
        ],
        format_version=2,
        run_id="run_id_with_underscore",
        system_id="system-id",
    )


class ReportMethodsTests(unittest.TestCase):
    def test_minimization_report_does_not_claim_production_dynamics(self):
        content = _content()
        content.parameters.update(integrator="steep", max_step=100, forcetol=10.0)
        text = render_text(build_methods_report(content, "minimize.tpr"))
        self.assertIn("L-BFGS", text)
        self.assertIn("No production dynamics were requested", text)
        self.assertIn("10 kJ mol^-1 nm^-1", text)
        self.assertNotIn("Production duration", text)
        self.assertNotIn("Dynamics used", text)

    def test_text_and_latex_reports_include_methods_and_provenance(self):
        report = build_methods_report(_content(), "run_input.tpr")
        text = render_text(report)
        latex = render_latex(report)

        self.assertIn("Coarse-grained beads", text)
        self.assertIn("Production duration", text)
        self.assertIn("1 ns", text)
        self.assertIn("Ashbaugh-Hatch", text)
        self.assertIn("Methods summary", text)
        self.assertIn(r"run\_id\_with\_underscore", latex)
        self.assertIn(r"\section*{DROPPS simulation methods}", latex)

    def test_command_writes_independent_text_and_latex_outputs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            run_input = directory / "run.tpr"
            run_input.write_bytes(b"placeholder")
            args = SimpleNamespace(
                run_input=str(run_input),
                output=str(directory / "methods"),
                latex_output=str(directory / "methods-latex"),
            )
            with patch(
                "dropps.commands.report_methods.read_tpr",
                return_value=_content(),
            ):
                report_methods(args)

            self.assertIn(
                "DROPPS simulation methods report",
                (directory / "methods.out").read_text(encoding="utf-8"),
            )
            self.assertIn(
                r"\section*{DROPPS simulation methods}",
                (directory / "methods-latex.tex").read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
