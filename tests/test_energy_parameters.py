import tempfile
import unittest
from pathlib import Path

from dropps.share.parameters import getparameter


class EnergyParameterTests(unittest.TestCase):
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
        self.assertNotIn("nst-cp =", template)
        self.assertNotIn("screenlog-grps =", template)
        self.assertNotIn("filelog-grps =", template)

        parameters = getparameter(template_path)
        self.assertEqual(parameters["integrator"], "Langevin")
        self.assertEqual(parameters["cutoff_scheme_lj"], "static")
        self.assertEqual(parameters["nst_energy"], 10000)
        self.assertEqual(parameters["nst_stress"], 0)

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
