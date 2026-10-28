import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from dropps.commands.mdrun import getargs_mdrun
from dropps.share.mdrun_config import (
    next_part_number,
    numbered_checkpoint_path,
    resolve_output_paths,
    resolve_runtime_options,
)


def _args(**overrides):
    values = {
        "output_prefix": "run",
        "trajectory_output": None,
        "energy_output": None,
        "log_output": None,
        "checkpoint_output": None,
        "final_structure": None,
        "stress_output": None,
        "nsteps": None,
        "seed": None,
        "checkpoint_minutes": 5.0,
        "max_hours": -1.0,
        "append": True,
        "number_checkpoints": False,
        "platform": "auto",
        "device_index": None,
        "precision": None,
        "threads": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class MdrunParserTests(unittest.TestCase):
    def test_gromacs_style_prefix_and_runtime_defaults(self):
        args = getargs_mdrun(["-s", "run.tpr", "-deffnm", "run"])

        self.assertEqual(args.output_prefix, "run")
        self.assertEqual(args.checkpoint_minutes, 5.0)
        self.assertEqual(args.max_hours, -1.0)
        self.assertTrue(args.append)
        self.assertEqual(args.platform, "auto")

    def test_operational_overrides_are_parsed(self):
        args = getargs_mdrun(
            [
                "-s",
                "run.tpr",
                "-o",
                "run",
                "-x",
                "traj/custom.xtc",
                "-e",
                "energy.csv",
                "-g",
                "progress.csv",
                "-cpo",
                "state.chk",
                "-c",
                "last.pdb",
                "--stress-output",
                "tensor.xvg",
                "--platform",
                "opencl",
                "-gpu_id",
                "1",
                "--precision",
                "double",
                "-nsteps",
                "200",
                "--seed",
                "42",
                "-cpt",
                "12.5",
                "-cpnum",
                "-maxh",
                "23",
                "-noappend",
            ]
        )

        self.assertEqual(args.platform, "OpenCL")
        self.assertEqual(args.device_index, "1")
        self.assertEqual(args.precision, "double")
        self.assertEqual(args.nsteps, 200)
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.checkpoint_minutes, 12.5)
        self.assertTrue(args.number_checkpoints)
        self.assertEqual(args.max_hours, 23.0)
        self.assertFalse(args.append)


class MdrunConfigurationTests(unittest.TestCase):
    def test_individual_output_paths_override_prefix_defaults(self):
        paths = resolve_output_paths(
            _args(
                trajectory_output="custom.xtc",
                checkpoint_output="custom.chk",
                final_structure="custom.pdb",
                stress_output="custom.stress.xvg",
            ),
            {"stress_output": "from-mdp.xvg"},
        )

        self.assertEqual(paths.log, "run.log")
        self.assertEqual(paths.energy, "run.edr")
        self.assertEqual(paths.trajectory, "custom.xtc")
        self.assertEqual(paths.warming_trajectory, "run.warming.xtc")
        self.assertEqual(paths.checkpoint, "custom.chk")
        self.assertEqual(paths.final_structure, "custom.pdb")
        self.assertEqual(paths.stress, "custom.stress.xvg")

    def test_cli_runtime_values_override_tpr(self):
        options = resolve_runtime_options(
            _args(
                nsteps=25,
                seed=9,
                checkpoint_minutes=7.5,
                max_hours=4,
                platform="CPU",
                threads=6,
            ),
            {"nsteps": 100, "seed": 3},
        )

        self.assertEqual(options.production_steps, 25)
        self.assertEqual(options.random_seed, 9)
        self.assertEqual(options.checkpoint_interval_seconds, 450.0)
        self.assertEqual(options.max_run_seconds, 4 * 3600 * 0.99)
        self.assertEqual(options.platform, "CPU")
        self.assertEqual(options.threads, 6)

    def test_tpr_supplies_steps_and_seed_without_cli_overrides(self):
        options = resolve_runtime_options(
            _args(), {"nsteps": 100, "seed": 3}
        )

        self.assertEqual(options.production_steps, 100)
        self.assertEqual(options.random_seed, 3)
        self.assertIsNone(options.max_run_seconds)

    def test_cpu_threads_and_gpu_properties_cannot_be_mixed(self):
        with self.assertRaisesRegex(ValueError, "nt cannot be combined"):
            resolve_runtime_options(
                _args(threads=2, precision="double"),
                {"nsteps": 1, "seed": 1},
            )

    def test_non_appending_continuations_use_next_free_part(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            prefix = str(Path(temporary_directory) / "run")
            paths = resolve_output_paths(
                _args(output_prefix=prefix), {"stress_output": "auto"}
            )
            part_two = paths.with_part_number(2)
            Path(part_two.trajectory).touch()

            self.assertEqual(next_part_number(paths), 3)
            part_three = paths.with_part_number(3)
            self.assertTrue(part_three.log.endswith("run.part0003.log"))
            self.assertTrue(
                part_three.stress.endswith("run.part0003.stress.xvg")
            )
            self.assertEqual(part_three.checkpoint, prefix + ".chk")

    def test_numbered_checkpoint_contains_step(self):
        self.assertEqual(
            numbered_checkpoint_path("run.chk", 123),
            "run.step000000000123.chk",
        )


if __name__ == "__main__":
    unittest.main()
