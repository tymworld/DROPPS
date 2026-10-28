import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import openmm
import openmm.app
from openmm import unit

from dropps.commands.mdrun import (
    _create_simulation,
    _platform_properties,
    _run_production,
    _run_temperature_ramp,
    _save_runtime_checkpoint,
    _set_simulation_temperature,
    _temperature_ramp_schedule,
)
from dropps.share.mdrun_config import RuntimeOptions


def _options(**overrides):
    values = {
        "production_steps": 2000,
        "random_seed": 1,
        "checkpoint_minutes": 0,
        "max_hours": None,
        "append": True,
        "number_checkpoints": False,
        "platform": "Reference",
        "device_index": None,
        "precision": None,
        "threads": None,
    }
    values.update(overrides)
    return RuntimeOptions(**values)


class _FakeSimulation:
    def __init__(self):
        self.currentStep = 0
        self.step_calls = []
        self.integrator = SimpleNamespace(
            setTemperature=lambda _temperature: None
        )
        self.context = SimpleNamespace(
            getParameters=lambda: {},
            setParameter=lambda _name, _value: None,
        )
        self._dropps_restart_metadata = {}

    def step(self, count):
        self.step_calls.append(count)
        self.currentStep += count


class _FakePlatform:
    def __init__(self, name, properties):
        self._name = name
        self._properties = properties

    def getName(self):
        return self._name

    def getPropertyNames(self):
        return self._properties


class MdrunWallClockTests(unittest.TestCase):
    def test_interrupted_temperature_ramp_records_exact_resume_progress(self):
        simulation = _FakeSimulation()
        schedule = [(151.0, 2000), (152.0, 2000)]
        stopped = _run_temperature_ramp(
            simulation,
            schedule,
            "run.chk",
            _options(checkpoint_minutes=0, max_hours=1.0 / 3600.0),
            clock=iter([0.0, 1.0]).__next__,
            started_at=0.0,
        )
        self.assertTrue(stopped)
        self.assertEqual(simulation.currentStep, 1000)
        self.assertEqual(
            simulation._dropps_restart_metadata["warming_segment_index"], 0
        )
        self.assertEqual(
            simulation._dropps_restart_metadata["warming_segment_step"], 1000
        )

        stopped = _run_temperature_ramp(
            simulation,
            schedule,
            "run.chk",
            _options(checkpoint_minutes=0),
            start_segment=0,
            completed_segment_steps=1000,
            clock=lambda: 0.0,
            started_at=0.0,
        )
        self.assertFalse(stopped)
        self.assertEqual(simulation.currentStep, 4000)
        self.assertEqual(
            simulation._dropps_restart_metadata["warming_segment_index"], 2
        )

    def test_temperature_ramp_has_no_duplicate_initial_stage(self):
        schedule = _temperature_ramp_schedule(298.25, 300.0, 500.0, 0.01)
        self.assertEqual(len(schedule), 2)
        np.testing.assert_allclose([target for target, _ in schedule], [299.125, 300.0])
        self.assertEqual([steps for _, steps in schedule], [175, 175])

    def test_temperature_ramp_supports_cooling_and_fractional_temperatures(self):
        schedule = _temperature_ramp_schedule(300.5, 299.0, 1000.0, 0.01)
        np.testing.assert_allclose([target for target, _ in schedule], [299.75, 299.0])
        self.assertEqual([steps for _, steps in schedule], [75, 75])

    @patch("dropps.commands.mdrun._save_runtime_checkpoint")
    def test_periodic_checkpoint_uses_wall_clock_minutes(self, save_checkpoint):
        simulation = _FakeSimulation()
        clock_values = iter([0.0, 0.0, 301.0, 301.0, 302.0])

        remaining, stopped = _run_production(
            simulation,
            remaining_steps=2000,
            checkpoint_path="run.chk",
            options=_options(checkpoint_minutes=5),
            clock=lambda: next(clock_values),
        )

        self.assertEqual(remaining, 0)
        self.assertFalse(stopped)
        save_checkpoint.assert_called_once_with(
            simulation, "run.chk", False
        )

    @patch("dropps.commands.mdrun._save_runtime_checkpoint")
    def test_maxh_stops_after_a_bounded_step_chunk(self, save_checkpoint):
        simulation = _FakeSimulation()
        clock_values = iter([0.0, 0.0, 1.0])

        remaining, stopped = _run_production(
            simulation,
            remaining_steps=2000,
            checkpoint_path="run.chk",
            options=_options(max_hours=1.0 / 3600.0),
            clock=lambda: next(clock_values),
        )

        self.assertEqual(simulation.step_calls, [1000])
        self.assertEqual(remaining, 1000)
        self.assertTrue(stopped)
        save_checkpoint.assert_not_called()

    @patch("dropps.commands.mdrun.save_portable_restart_safely")
    @patch("dropps.commands.mdrun.save_checkpoint_safely")
    def test_cpnum_keeps_latest_and_numbered_checkpoint(
        self, save_checkpoint, save_portable_restart
    ):
        simulation = _FakeSimulation()
        simulation.currentStep = 17
        save_portable_restart.side_effect = lambda simulation, path, metadata=None: (
            path.replace(".chk", ".state.xml"),
            path.replace(".chk", ".restart.json"),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = str(Path(temporary_directory) / "run.chk")
            _save_runtime_checkpoint(
                simulation,
                checkpoint,
                number_checkpoints=True,
            )

        self.assertEqual(save_checkpoint.call_count, 2)
        self.assertEqual(save_portable_restart.call_count, 2)
        self.assertEqual(save_checkpoint.call_args_list[0].args[1], checkpoint)
        self.assertTrue(
            save_checkpoint.call_args_list[1].args[1].endswith(
                "run.step000000000017.chk"
            )
        )


class MdrunPlatformTests(unittest.TestCase):
    def test_explicit_reference_platform_and_seed_reach_context(self):
        system = openmm.System()
        system.addParticle(1.0)
        topology = openmm.app.Topology()
        chain = topology.addChain()
        residue = topology.addResidue("MOL", chain)
        topology.addAtom(
            "A", openmm.app.Element.getBySymbol("He"), residue
        )

        simulation = _create_simulation(
            topology,
            system,
            _options(platform="Reference", random_seed=17),
            300 * unit.kelvin,
            1 / unit.picosecond,
            0.001 * unit.picosecond,
        )
        try:
            self.assertEqual(
                simulation.context.getPlatform().getName(), "Reference"
            )
            self.assertEqual(simulation.integrator.getRandomNumberSeed(), 17)
            self.assertIsInstance(
                simulation.integrator, openmm.LangevinMiddleIntegrator
            )
        finally:
            del simulation

    def test_temperature_updates_integrator_barostat_and_forcefield(self):
        system = openmm.System()
        system.addParticle(10.0)
        system.addParticle(10.0)
        force = openmm.CustomNonbondedForce("temperature/r")
        force.addGlobalParameter("temperature", 150.0)
        force.addParticle([])
        force.addParticle([])
        force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        force.setCutoffDistance(1.0 * unit.nanometer)
        system.addForce(force)
        system.addForce(
            openmm.MonteCarloBarostat(
                1.0 * unit.bar, 150.0 * unit.kelvin, 5
            )
        )
        system.setDefaultPeriodicBoxVectors(
            openmm.Vec3(3, 0, 0),
            openmm.Vec3(0, 3, 0),
            openmm.Vec3(0, 0, 3),
        )
        topology = openmm.app.Topology()
        chain = topology.addChain()
        residue = topology.addResidue("MOL", chain)
        for name in ("A", "B"):
            topology.addAtom(
                name, openmm.app.Element.getBySymbol("He"), residue
            )
        topology.setPeriodicBoxVectors(
            [
                openmm.Vec3(3, 0, 0),
                openmm.Vec3(0, 3, 0),
                openmm.Vec3(0, 0, 3),
            ]
        )
        simulation = _create_simulation(
            topology,
            system,
            _options(platform="Reference"),
            150.0 * unit.kelvin,
            1.0 / unit.picosecond,
            0.001 * unit.picosecond,
        )
        try:
            simulation.context.setPositions(
                [[0, 0, 0], [0.5, 0, 0]] * unit.nanometer
            )
            _set_simulation_temperature(simulation, 275.5)
            self.assertAlmostEqual(
                simulation.integrator.getTemperature().value_in_unit(unit.kelvin),
                275.5,
            )
            self.assertAlmostEqual(
                simulation.context.getParameter("temperature"), 275.5
            )
            self.assertAlmostEqual(
                simulation.context.getParameter(
                    openmm.MonteCarloBarostat.Temperature()
                ),
                275.5,
            )
        finally:
            del simulation

    def test_opencl_device_and_precision_properties(self):
        properties = _platform_properties(
            _FakePlatform("OpenCL", ["DeviceIndex", "Precision"]),
            _options(
                platform="OpenCL", device_index="0,1", precision="double"
            ),
        )

        self.assertEqual(
            properties, {"DeviceIndex": "0,1", "Precision": "double"}
        )

    def test_gpu_precision_defaults_to_mixed(self):
        properties = _platform_properties(
            _FakePlatform("CUDA", ["DeviceIndex", "Precision"]),
            _options(platform="CUDA"),
        )

        self.assertEqual(properties, {"Precision": "mixed"})

    def test_cpu_threads_property(self):
        properties = _platform_properties(
            _FakePlatform("CPU", ["Threads"]),
            _options(platform="CPU", threads=8),
        )

        self.assertEqual(properties, {"Threads": "8"})


if __name__ == "__main__":
    unittest.main()
