import tempfile
import unittest
from pathlib import Path

from dropps.share.parameters import (
    getparameter,
    parameters_dict_template,
    validate_parameters,
)


class EnergyParameterTests(unittest.TestCase):
    def test_complete_parameter_validation_rejects_bad_enums_and_values(self):
        parameters = dict(parameters_dict_template)
        parameters["coulombtype"] = "Yukawa"
        with self.assertRaisesRegex(ValueError, "coulombtype"):
            validate_parameters(parameters)

        parameters = dict(parameters_dict_template)
        parameters["dt"] = 0.0
        with self.assertRaisesRegex(ValueError, "dt"):
            validate_parameters(parameters)

        parameters = dict(parameters_dict_template)
        parameters["tau_P"] = 2.5
        with self.assertRaisesRegex(ValueError, "tau-P"):
            validate_parameters(parameters)

    def test_packaged_template_defines_edr_controls_and_fixed_logs(self):
        template_path = (
            Path(__file__).resolve().parents[1]
            / "dropps"
            / "share"
            / "templates"
            / "md.mdp"
        )
        template = template_path.read_text(encoding="utf-8")
        self.assertIn("nst-energy", template)
        self.assertIn("energy-grps", template)
        self.assertIn("seed", template)
        self.assertIn("keep-warming-trajectory", template)
        self.assertNotIn("nst-cp =", template)
        self.assertNotIn("screenlog-grps =", template)
        self.assertNotIn("filelog-grps =", template)

        parameters = getparameter(template_path)
        self.assertEqual(parameters["integrator"], "Langevin")
        self.assertEqual(parameters["cutoff_scheme_lj"], "static")
        self.assertEqual(parameters["nst_energy"], 10000)
        self.assertEqual(parameters["nst_stress"], 0)
        self.assertFalse(parameters["keep_warming_trajectory"])

    def test_warming_trajectory_boolean_is_parsed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "warming.mdp"
            path.write_text(
                "keep-warming-trajectory = True\n",
                encoding="utf-8",
            )
            parameters = getparameter(path)
            self.assertTrue(parameters["keep_warming_trajectory"])

    def test_mdp_reference_is_linked_from_readme(self):
        repository = Path(__file__).resolve().parents[1]
        reference = repository / "docs" / "mdp-reference.md"
        readme = (repository / "README.md").read_text(encoding="utf-8")
        self.assertTrue(reference.is_file())
        self.assertIn("docs/mdp-reference.md", readme)

    def test_legacy_filelog_parameters_map_to_energy_controls(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "legacy.mdp"
            path.write_text(
                "nst-filelog = 25\nfilelog-grps = step,potentialEnergy,temperature\n",
                encoding="utf-8",
            )
            parameters = getparameter(path)
            self.assertEqual(parameters["nst_energy"], 25)
            self.assertEqual(
                parameters["energy_grps"],
                "step,potentialEnergy,temperature",
            )


if __name__ == "__main__":
    unittest.main()
