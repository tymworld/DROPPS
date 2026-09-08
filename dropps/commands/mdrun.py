# mdrun tool in CGPS package by Yiming Tang @ Fudan
# Development started on June 6 2025

import json
import math
import os
import sys
import time
from copy import deepcopy

import numpy as np

from datetime import datetime

import openmm
import openmm.app
from openmm.unit import (
    bar,
    kelvin,
    kilojoule_per_mole,
    nanometer,
    nanosecond,
    picosecond,
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dropps.share.argument_parser import ArgumentParser

from dropps.fileio.pdb_reader import write_pdb
from dropps.fileio.tpr_reader import read_tpr
from dropps.share.command_class import single_command
from dropps.share.restart import (
    RestartableXTCReporter,
    load_restart_metadata,
    portable_state_path,
    save_checkpoint_safely,
    save_portable_restart_safely,
    truncate_delimited_report,
    truncate_xtc,
    validate_native_restart,
    validate_portable_restart,
)
from dropps.share.energy_reporter import (
    EnergyReporter,
    PRESSURE_FIELDS,
    normalize_field_names,
)
from dropps.share.progress_reporter import ProgressReporter
from dropps.share.stress_tensor_reporter import StressTensorReporter
from dropps.share.mdrun_config import (
    RuntimeOptions,
    next_part_number,
    numbered_checkpoint_path,
    resolve_output_paths,
    resolve_runtime_options,
)
from dropps.share.parameters import parameters_dict_template, validate_parameters


_PLATFORM_PREFERENCE = ("CUDA", "OpenCL", "CPU", "Reference")
_GPU_PLATFORMS = frozenset({"CUDA", "OpenCL"})
_WALLCLOCK_POLL_STEPS = 1000
_INTEGRATOR_SEMANTICS = "langevin-middle"


def _platform_candidates(options):
    available = {
        openmm.Platform.getPlatform(index).getName(): openmm.Platform.getPlatform(index)
        for index in range(openmm.Platform.getNumPlatforms())
    }
    print("## Available OpenMM platforms:", ", ".join(available))

    if options.platform != "auto":
        if options.platform not in available:
            raise RuntimeError(
                f"Requested OpenMM platform {options.platform} is not available. "
                f"Available platforms: {', '.join(available)}."
            )
        names = [options.platform]
    elif options.threads is not None:
        names = ["CPU"] if "CPU" in available else []
    elif options.device_index is not None or options.precision is not None:
        names = [name for name in ("CUDA", "OpenCL") if name in available]
    else:
        names = [name for name in _PLATFORM_PREFERENCE if name in available]

    if not names:
        raise RuntimeError(
            "No available OpenMM platform can satisfy the requested runtime options."
        )
    return [available[name] for name in names]


def _platform_properties(platform, options):
    platform_name = platform.getName()
    property_names = set(platform.getPropertyNames())
    properties = {}

    if options.device_index is not None:
        if platform_name not in _GPU_PLATFORMS or "DeviceIndex" not in property_names:
            raise ValueError(
                f"OpenMM platform {platform_name} does not support gpu_id."
            )
        properties["DeviceIndex"] = options.device_index

    if platform_name in _GPU_PLATFORMS:
        precision = options.precision or "mixed"
        if "Precision" not in property_names:
            if options.precision is not None:
                raise ValueError(
                    f"OpenMM platform {platform_name} does not support precision."
                )
        else:
            properties["Precision"] = precision
    elif options.precision is not None:
        raise ValueError(f"OpenMM platform {platform_name} does not support precision.")

    if options.threads is not None:
        if platform_name != "CPU" or "Threads" not in property_names:
            raise ValueError(f"OpenMM platform {platform_name} does not support nt.")
        properties["Threads"] = str(options.threads)

    return properties


def _create_simulation(
    topology,
    system,
    options,
    temperature,
    friction,
    timestep,
):
    errors = []
    for platform in _platform_candidates(options):
        try:
            properties = _platform_properties(platform, options)
            integrator = openmm.LangevinMiddleIntegrator(
                temperature, friction, timestep
            )
            integrator.setRandomNumberSeed(options.random_seed)
            simulation = openmm.app.Simulation(
                topology,
                system,
                integrator,
                platform,
                properties,
            )
        except Exception as exc:
            if options.platform != "auto":
                raise RuntimeError(
                    f"Could not initialize OpenMM platform {platform.getName()}: {exc}"
                ) from exc
            errors.append(f"{platform.getName()}: {exc}")
            continue

        print(f"## Using OpenMM platform {platform.getName()}.")
        if properties:
            print("## Platform properties:", json.dumps(properties, sort_keys=True))
        return simulation

    detail = "; ".join(errors) or "no platform candidates were available"
    raise RuntimeError(f"Could not initialize an OpenMM simulation: {detail}")


def _set_simulation_temperature(simulation, temperature_K):
    """Synchronize every temperature-dependent component in one Context."""

    temperature_K = float(temperature_K)
    simulation.integrator.setTemperature(temperature_K * kelvin)
    context_parameters = simulation.context.getParameters()
    if "temperature" in context_parameters:
        simulation.context.setParameter("temperature", temperature_K)
    barostat_parameter = openmm.MonteCarloBarostat.Temperature()
    if barostat_parameter in context_parameters:
        simulation.context.setParameter(barostat_parameter, temperature_K)


def _temperature_ramp_schedule(
    initial_temperature_K,
    final_temperature_K,
    warming_speed_K_per_ns,
    timestep_ps,
):
    """Return equal-duration, at-most-1 K temperature-ramp segments."""

    initial = float(initial_temperature_K)
    final = float(final_temperature_K)
    speed = float(warming_speed_K_per_ns)
    timestep = float(timestep_ps)
    delta = final - initial
    if delta == 0.0:
        return []
    if speed <= 0.0 or timestep <= 0.0:
        raise ValueError("Temperature-ramp speed and timestep must be positive.")

    segment_count = int(math.ceil(abs(delta)))
    requested_total_steps = abs(delta) * 1000.0 / (speed * timestep)
    steps_per_segment = max(1, int(round(requested_total_steps / segment_count)))
    return [
        (
            initial + delta * segment_index / segment_count,
            steps_per_segment,
        )
        for segment_index in range(1, segment_count + 1)
    ]


def _run_temperature_ramp(
    simulation,
    schedule,
    checkpoint_path,
    options,
    *,
    warming_trajectory_path=None,
    start_segment=0,
    completed_segment_steps=0,
    clock=None,
    started_at=None,
):
    """Run a restartable temperature ramp with wall-clock safeguards."""

    if clock is None:
        clock = time.monotonic
    if started_at is None:
        started_at = clock()
    checkpoint_interval = options.checkpoint_interval_seconds
    next_checkpoint_at = (
        None if checkpoint_interval is None else started_at + checkpoint_interval
    )
    schedule_data = [[float(target), int(steps)] for target, steps in schedule]

    warming_trajectory = None
    if warming_trajectory_path is not None and schedule:
        interval = int(schedule[0][1])
        append = (
            start_segment > 0
            and os.path.isfile(warming_trajectory_path)
            and os.path.getsize(warming_trajectory_path) > 0
        )
        warming_trajectory = openmm.app.XTCFile(
            warming_trajectory_path,
            simulation.topology,
            simulation.integrator.getStepSize(),
            firstStep=interval,
            interval=interval,
            append=append,
        )

    metadata = simulation._dropps_restart_metadata
    metadata.update(
        {
            "stage": "warming",
            "warming_schedule": schedule_data,
            "warming_segment_index": int(start_segment),
            "warming_segment_step": int(completed_segment_steps),
        }
    )

    for segment_index in range(start_segment, len(schedule)):
        target_temperature, segment_steps = schedule[segment_index]
        segment_progress = (
            completed_segment_steps if segment_index == start_segment else 0
        )
        if not 0 <= segment_progress <= segment_steps:
            raise ValueError("Restart manifest contains invalid warming progress.")
        _set_simulation_temperature(simulation, target_temperature)
        metadata["warming_segment_index"] = segment_index
        metadata["warming_segment_step"] = segment_progress

        remaining = segment_steps - segment_progress
        while remaining > 0:
            now = clock()
            if (
                options.max_run_seconds is not None
                and now - started_at >= options.max_run_seconds
            ):
                return True

            chunk_steps = min(remaining, _WALLCLOCK_POLL_STEPS)
            simulation.step(chunk_steps)
            remaining -= chunk_steps
            segment_progress += chunk_steps
            metadata["warming_segment_step"] = segment_progress
            now = clock()

            if next_checkpoint_at is not None and now >= next_checkpoint_at:
                _save_runtime_checkpoint(
                    simulation,
                    checkpoint_path,
                    options.number_checkpoints,
                    reason="warming checkpoint",
                )
                while next_checkpoint_at <= now:
                    next_checkpoint_at += checkpoint_interval

            if (
                options.max_run_seconds is not None
                and now - started_at >= options.max_run_seconds
                and remaining > 0
            ):
                return True

        metadata["warming_segment_index"] = segment_index + 1
        metadata["warming_segment_step"] = 0
        if warming_trajectory is not None:
            state = simulation.context.getState(
                getPositions=True,
                enforcePeriodicBox=True,
            )
            warming_trajectory.writeModel(
                state.getPositions(),
                periodicBoxVectors=state.getPeriodicBoxVectors(),
            )
        if segment_index == len(schedule) - 1 or (segment_index + 1) % 10 == 0:
            print(
                f"## Temperature ramp reached {target_temperature:.6g} K "
                f"at {datetime.now().strftime('%Y-%m-%d, %H:%M:%S')}."
            )

    return False


def _write_final_structure(simulation, pdb_raw, final_file_path):
    state = simulation.context.getState(getPositions=True)
    positions = state.getPositions(asNumpy=True).value_in_unit(nanometer)
    box_vectors = state.getPeriodicBoxVectors()
    box_vectors_no_unit = tuple(
        vector.value_in_unit(nanometer) for vector in box_vectors
    )
    box_vec = np.asarray(
        [
            box_vectors_no_unit[0][0],
            box_vectors_no_unit[1][1],
            box_vectors_no_unit[2][2],
        ],
        dtype=float,
    )
    wrapped_positions = positions % box_vec
    write_pdb(
        final_file_path,
        box_vec,
        pdb_raw.record_names,
        pdb_raw.serial_numbers,
        pdb_raw.atom_names,
        pdb_raw.residue_names,
        pdb_raw.chain_IDs,
        pdb_raw.residue_sequence_numbers,
        wrapped_positions[:, 0],
        wrapped_positions[:, 1],
        wrapped_positions[:, 2],
        pdb_raw.occupancys,
        pdb_raw.bfactors,
        pdb_raw.elements,
        pdb_raw.molecule_length_list,
    )


def _save_runtime_checkpoint(
    simulation,
    checkpoint_path,
    number_checkpoints=False,
    reason="periodic checkpoint",
):
    restart_metadata = getattr(simulation, "_dropps_restart_metadata", None)

    def save_pair(path):
        save_checkpoint_safely(simulation, path)
        return save_portable_restart_safely(simulation, path, metadata=restart_metadata)

    state_path, _ = save_pair(checkpoint_path)
    numbered_path = None
    if number_checkpoints:
        numbered_path = numbered_checkpoint_path(
            checkpoint_path, simulation.currentStep
        )
        save_pair(numbered_path)
    message = (
        f"## Saved {reason} at step {simulation.currentStep} to "
        f"{checkpoint_path} and portable state {state_path}"
    )
    if numbered_path is not None:
        message += f" and {numbered_path}"
    print(message + ".")


def _run_production(
    simulation,
    remaining_steps,
    checkpoint_path,
    options: RuntimeOptions,
    clock=None,
    started_at=None,
):
    """Run production in bounded chunks for wall-clock control."""

    if clock is None:
        clock = time.monotonic
    if started_at is None:
        started_at = clock()
    checkpoint_interval = options.checkpoint_interval_seconds
    next_checkpoint_at = (
        None if checkpoint_interval is None else started_at + checkpoint_interval
    )
    stopped_for_maxh = False

    while remaining_steps > 0:
        now = clock()
        max_run_seconds = options.max_run_seconds
        if max_run_seconds is not None and now - started_at >= max_run_seconds:
            stopped_for_maxh = True
            break

        chunk_steps = min(remaining_steps, _WALLCLOCK_POLL_STEPS)
        simulation.step(chunk_steps)
        remaining_steps -= chunk_steps
        now = clock()

        if (
            next_checkpoint_at is not None
            and now >= next_checkpoint_at
            and remaining_steps > 0
        ):
            _save_runtime_checkpoint(
                simulation,
                checkpoint_path,
                options.number_checkpoints,
            )
            while next_checkpoint_at <= now:
                next_checkpoint_at += checkpoint_interval

        if (
            max_run_seconds is not None
            and now - started_at >= max_run_seconds
            and remaining_steps > 0
        ):
            stopped_for_maxh = True
            break

    return remaining_steps, stopped_for_maxh


def mdrun(args):
    run_started_at = time.monotonic()

    tpr = read_tpr(args.run_input)
    parameters = deepcopy(parameters_dict_template)
    parameters.update(tpr.parameters)
    validate_parameters(parameters)
    minimization_only = parameters["integrator"] == "steep"
    mdsystem = tpr.mdsystem
    mdtopology = tpr.mdtopology
    positions = tpr.positions
    pdb_raw = tpr.pdb_raw

    print(
        f"## Loaded DROPPS TPR format v{tpr.format_version} (run ID {tpr.run_id[:12]})."
    )
    source_semantics = tpr.metadata.get("integrator_semantics")
    current_semantics = (
        "steep-minimization" if minimization_only else _INTEGRATOR_SEMANTICS
    )
    if source_semantics and source_semantics != current_semantics:
        print(
            "## WARNING: This TPR was generated with integrator semantics "
            f"{source_semantics!r}, while OpenMM {openmm.__version__} uses "
            f"{current_semantics!r}. A continued trajectory will not be "
            "identical to an OpenMM 8.1 trajectory."
        )

    runtime_options = resolve_runtime_options(args, parameters)
    output_paths = resolve_output_paths(args, parameters)

    nst_energy = parameters.get("nst_energy", parameters.get("nst_filelog", 0))
    energy_fields = parameters.get(
        "energy_grps",
        parameters.get("filelog_grps", "potentialEnergy,kineticEnergy"),
    )
    if nst_energy > 0:
        energy_fields = normalize_field_names(energy_fields)
        if (
            PRESSURE_FIELDS.intersection(energy_fields)
            and mdsystem.getNumConstraints() > 0
        ):
            raise ValueError(
                "Pressure terms in energy-grps cannot be used with constrained "
                "bonds because the constraint virial is unavailable. Use "
                "bondtype = bond."
            )

    checkpoint_input = args.checkpoint
    if checkpoint_input == "auto":
        checkpoint_input = output_paths.checkpoint
        if not os.path.isfile(checkpoint_input):
            state_candidate = portable_state_path(checkpoint_input)
            if os.path.isfile(state_candidate):
                checkpoint_input = state_candidate
    is_resume = checkpoint_input is not None
    if minimization_only and is_resume:
        raise ValueError(
            "integrator = steep performs a standalone minimization and cannot "
            "resume a dynamics checkpoint."
        )
    if is_resume and not os.path.isfile(checkpoint_input):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_input}")
    if not is_resume and not minimization_only and not parameters["gen_vel"]:
        raise ValueError(
            "gen-vel = False is invalid for a fresh DROPPS run because TPR "
            "files do not contain velocities. Enable gen-vel or resume from "
            "a checkpoint/portable State."
        )
    restart_manifest = load_restart_metadata(checkpoint_input) if is_resume else None
    resume_warming = bool(
        restart_manifest and restart_manifest.get("stage") == "warming"
    )
    production_resume = is_resume and not resume_warming

    if is_resume and not runtime_options.append:
        part_number = next_part_number(output_paths)
        output_paths = output_paths.with_part_number(part_number)
        print(
            f"## RESTART: Append is disabled; continuation outputs use "
            f"part {part_number:04d}."
        )

    log_file_path = output_paths.log
    energy_file_path = output_paths.energy
    traj_file_path = output_paths.trajectory
    warming_traj_file_path = output_paths.warming_trajectory
    checkpoint_file_path = output_paths.checkpoint
    final_file_path = output_paths.final_structure
    stress_file_path = output_paths.stress

    # We first get structure and topology

    print("## Following parameters are read and phrased from the parameter file.")
    print(json.dumps(parameters, indent=4))
    print(
        "## Runtime overrides: "
        + json.dumps(
            {
                "nsteps": runtime_options.production_steps,
                "seed": runtime_options.random_seed,
                "cpt_minutes": runtime_options.checkpoint_minutes,
                "maxh": runtime_options.max_hours,
                "append": runtime_options.append,
                "cpnum": runtime_options.number_checkpoints,
                "platform": runtime_options.platform,
                "device_index": runtime_options.device_index,
                "precision": runtime_options.precision,
                "threads": runtime_options.threads,
            },
            sort_keys=True,
        )
    )

    # We now set up the integrator

    temperature_initial = parameters["initial_temperature"] * kelvin
    temperature_final = parameters["production_temperature"] * kelvin
    starting_temperature = temperature_final if is_resume else temperature_initial
    friction = parameters["friction"] / picosecond
    timestep = parameters["dt"] * picosecond

    print(
        f"## Initializing Langevin-middle integrator with temperature: "
        f"{starting_temperature}, "
        f"friction: {friction}, time step: {timestep}, and random seed: "
        f"{runtime_options.random_seed}."
    )

    # We now set translational velocity remover
    if parameters["comm_mode"] == "Linear":
        mdsystem.addForce(openmm.CMMotionRemover(parameters["nstcomm"]))
        print(
            f"## Center of mass translational velocity will be removed every {parameters['nstcomm']} steps."
        )
    else:
        print("## WARNING: Center of mass translational velocity will not be removed.")

    if parameters["pcoulp"] is True and not minimization_only:
        pressure = parameters["ref_P"] * bar
        tau_pressure = int(parameters["tau_P"])

        pressure_coupling = openmm.MonteCarloBarostat(
            pressure, starting_temperature, tau_pressure
        )
        pressure_coupling.setRandomNumberSeed(runtime_options.random_seed)
        mdsystem.addForce(pressure_coupling)

        # mdsystem.getForce(1).setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)

        # if parameters["coulombtype"] == "yukawa":
        #    mdsystem.getForce(2).setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)

        print(
            f"## Pressure coupling at {pressure} will be performed per {tau_pressure} steps."
        )

    else:
        print("## Simulation will be performed without pressure coupling.")

    for i, force in enumerate(mdsystem.getForces()):
        if isinstance(force, openmm.MonteCarloBarostat):
            print(f"## Barostat found at index {i}")

    simulation = _create_simulation(
        mdtopology,
        mdsystem,
        runtime_options,
        starting_temperature,
        friction,
        timestep,
    )
    simulation.context.setPositions(positions)
    if not is_resume:
        _set_simulation_temperature(simulation, parameters["initial_temperature"])
    print("## Initial positions for the system succussfully passed to simulation.")

    simulation._dropps_restart_metadata = {
        "tpr_run_id": tpr.run_id,
        "tpr_system_id": tpr.system_id,
        "integrator_semantics": current_semantics,
        "random_seed": runtime_options.random_seed,
        "stage": "setup",
    }
    if restart_manifest is not None:
        for key in (
            "stage",
            "warming_schedule",
            "warming_segment_index",
            "warming_segment_step",
        ):
            if key in restart_manifest:
                simulation._dropps_restart_metadata[key] = restart_manifest[key]

    if is_resume:
        portable_resume = str(checkpoint_input).endswith(".state.xml")
        if portable_resume:
            validate_portable_restart(
                checkpoint_input, expected_system_id=tpr.system_id
            )
            simulation.loadState(checkpoint_input)
            print(f"## Loaded portable State from {checkpoint_input}.")
        else:
            try:
                validate_native_restart(
                    checkpoint_input,
                    expected_system_id=tpr.system_id,
                    manifest=restart_manifest,
                )
                simulation.loadCheckpoint(checkpoint_input)
            except Exception as checkpoint_error:
                state_candidate = portable_state_path(checkpoint_input)
                if not os.path.isfile(state_candidate):
                    raise RuntimeError(
                        f"Could not load native checkpoint {checkpoint_input}, "
                        "and no paired portable State is available. Root cause: "
                        f"{checkpoint_error}"
                    ) from checkpoint_error
                validate_portable_restart(
                    state_candidate, expected_system_id=tpr.system_id
                )
                simulation.loadState(state_candidate)
                portable_resume = True
                print(
                    f"## WARNING: Native checkpoint {checkpoint_input} could "
                    f"not be loaded ({checkpoint_error}). Loaded portable "
                    f"State {state_candidate} instead. Random-number generator "
                    "state was not restored, so this is a new trajectory segment."
                )
            else:
                print(f"## Loaded native checkpoint from {checkpoint_input}.")

    platform = simulation.context.getPlatform()
    print("## Simulation is running on platform:", platform.getName())

    # List all available platform properties
    property_names = platform.getPropertyNames()
    print("Available platform properties:")
    for name in property_names:
        value = platform.getPropertyValue(simulation.context, name)
        print(f"  {name} = {value}")

    if production_resume:
        print("## RESTART: Continuing production without minimization or warming.")
    elif resume_warming:
        print("## RESTART: Continuing the interrupted temperature ramp.")

    if (parameters["minimize"] is True or minimization_only) and not is_resume:
        print("## ENERGY MINIMIZE: Starting energy minimization.")

        minimization_step = parameters["max_step"]
        minimization_force_tol = parameters["forcetol"] * kilojoule_per_mole / nanometer

        print(
            f"## ENERGY MINIMIZE: The system will be ralexed until all force smaller than {minimization_force_tol} or reaching {minimization_step} steps."
        )

        simulation.minimizeEnergy(minimization_force_tol, minimization_step)

        # Get the current state including energy and forces
        state = simulation.context.getState(getEnergy=True, getForces=True)
        potential_energy = state.getPotentialEnergy()
        forces = state.getForces(asNumpy=True)
        # Compute force magnitudes for each particle
        force_magnitudes = np.linalg.norm(forces, axis=1)
        max_force = np.max(force_magnitudes) * kilojoule_per_mole / nanometer

        # Compute RMS force
        rms_force = (
            np.sqrt(np.mean(force_magnitudes**2)) * kilojoule_per_mole / nanometer
        )

        # Print results
        print(
            f"## ENERGY MINIMIZE: Potential Energy after minimization: {potential_energy}"
        )
        print(
            f"## ENERGY MINIMIZE: Maximum force magnitude after minimization: {max_force}"
        )
        print(
            f"## ENERGY MINIMIZE: RMS force magnitude after minimization: {rms_force}"
        )
        print("## ENERGY MINIMIZE: Ending energy minimization.")

    if minimization_only:
        simulation._dropps_restart_metadata["stage"] = "minimization-complete"
        _write_final_structure(simulation, pdb_raw, final_file_path)
        _save_runtime_checkpoint(
            simulation,
            checkpoint_file_path,
            runtime_options.number_checkpoints,
            reason="minimization checkpoint",
        )
        print(
            f"## Steep minimization complete; final structure written to "
            f"{final_file_path}."
        )
        return

    print(
        f"## The mdsystem contains {simulation.system.getNumForces()} types of forces. They are listed as follows:"
    )
    force_names = [
        str(simulation.system.getForce(i)).split("'")[1].split(" ")[0]
        for i in range(simulation.system.getNumForces())
    ]
    print(f"       {', '.join(force_names)}")

    production_nstep = runtime_options.production_steps

    if parameters["gen_vel"] is True and not is_resume:
        simulation.context.setVelocitiesToTemperature(
            temperature_initial, runtime_options.random_seed
        )
        print(
            "## Initializing all velocities according to a Boltzmann "
            f"distribution at {temperature_initial} with seed "
            f"{runtime_options.random_seed}."
        )

    # Temperature ramp (heating or cooling).
    ramp_schedule = []
    ramp_start_segment = 0
    ramp_segment_progress = 0
    if resume_warming:
        raw_schedule = restart_manifest.get("warming_schedule")
        if not raw_schedule:
            raise ValueError(
                "Warming restart manifest does not contain a temperature schedule."
            )
        ramp_schedule = [(float(target), int(steps)) for target, steps in raw_schedule]
        ramp_start_segment = int(restart_manifest.get("warming_segment_index", 0))
        ramp_segment_progress = int(restart_manifest.get("warming_segment_step", 0))
    elif not is_resume:
        ramp_schedule = _temperature_ramp_schedule(
            parameters["initial_temperature"],
            parameters["production_temperature"],
            parameters["warming_speed"],
            parameters["dt"],
        )

    if ramp_schedule:
        ramp_steps = sum(steps for _target, steps in ramp_schedule)
        ramp_delta = ramp_schedule[-1][0] - parameters["initial_temperature"]
        ramp_duration_ns = ramp_steps * parameters["dt"] / 1000.0
        actual_speed = abs(ramp_delta) / ramp_duration_ns
        print(
            f"## Temperature ramp: {parameters['initial_temperature']:g} -> "
            f"{parameters['production_temperature']:g} K in {ramp_steps} "
            f"steps ({actual_speed:g} K/ns actual discretized rate)."
        )
        warming_trajectory_path = None
        if parameters.get("keep_warming_trajectory", False):
            warming_trajectory_path = warming_traj_file_path
            print(
                "## WARMING TRAJECTORY: Will write one frame per ramp "
                f"segment to {warming_traj_file_path}."
            )
        try:
            stopped_during_warming = _run_temperature_ramp(
                simulation,
                ramp_schedule,
                checkpoint_file_path,
                runtime_options,
                warming_trajectory_path=warming_trajectory_path,
                start_segment=ramp_start_segment,
                completed_segment_steps=ramp_segment_progress,
                started_at=run_started_at,
            )
        except KeyboardInterrupt:
            _save_runtime_checkpoint(
                simulation,
                checkpoint_file_path,
                runtime_options.number_checkpoints,
                reason="warming interrupt checkpoint",
            )
            raise
        if stopped_during_warming:
            _save_runtime_checkpoint(
                simulation,
                checkpoint_file_path,
                runtime_options.number_checkpoints,
                reason="warming maxh checkpoint",
            )
            _write_final_structure(simulation, pdb_raw, final_file_path)
            print(
                "## Temperature ramp stopped safely for maxh; rerun with "
                "-cpi to continue the saved ramp."
            )
            return
        print("## Temperature ramp completed.")
        production_resume = False

    _set_simulation_temperature(simulation, parameters["production_temperature"])
    if not production_resume:
        # Checkpoints and production outputs use a production-local step/time
        # axis, independent of any warming steps performed above.
        simulation.currentStep = 0
        simulation.context.setTime(0 * picosecond)
        simulation._dropps_restart_metadata.update(
            {
                "stage": "production",
                "warming_segment_index": len(ramp_schedule),
                "warming_segment_step": 0,
            }
        )

    checkpoint_step = simulation.currentStep
    production_step_offset = 0
    if production_resume and checkpoint_step > production_nstep:
        # Checkpoints created by older DROPPS releases used a single step axis
        # for warming and production. Recognize completed-warming checkpoints
        # so existing long simulations can still be continued.
        legacy_warming_steps = 0
        if temperature_initial < temperature_final:
            warming_speed = parameters["warming_speed"] * kelvin / nanosecond
            step_per_kelvin = int(1 * kelvin / warming_speed / timestep)
            temperature_count = len(
                range(
                    int(temperature_initial / kelvin),
                    int(temperature_final / kelvin) + 1,
                )
            )
            legacy_warming_steps = step_per_kelvin * temperature_count
        if (
            legacy_warming_steps > 0
            and legacy_warming_steps <= checkpoint_step
            and checkpoint_step <= legacy_warming_steps + production_nstep
        ):
            production_step_offset = legacy_warming_steps
            print(
                "## RESTART: Detected a legacy checkpoint whose step count "
                f"includes {legacy_warming_steps} warming steps."
            )
        else:
            raise ValueError(
                f"Checkpoint step {checkpoint_step} is beyond the configured "
                f"production target ({production_nstep}) and is not a "
                "recognized legacy checkpoint."
            )

    completed_steps = checkpoint_step - production_step_offset
    target_context_step = production_step_offset + production_nstep
    remaining_steps = target_context_step - checkpoint_step

    trajectory_append = False
    file_log_append = False
    energy_append = False
    stress_append = False
    if production_resume and runtime_options.append:
        trajectory_append = (
            os.path.isfile(traj_file_path) and os.path.getsize(traj_file_path) > 0
        )
        file_log_append = (
            os.path.isfile(log_file_path) and os.path.getsize(log_file_path) > 0
        )
        energy_append = (
            os.path.isfile(energy_file_path) and os.path.getsize(energy_file_path) > 0
        )
        stress_append = (
            os.path.isfile(stress_file_path) and os.path.getsize(stress_file_path) > 0
        )
        if trajectory_append and truncate_xtc(traj_file_path, checkpoint_step):
            print(
                f"## RESTART: Truncated {traj_file_path} to checkpoint step "
                f"{checkpoint_step}."
            )
            trajectory_append = (
                os.path.isfile(traj_file_path) and os.path.getsize(traj_file_path) > 0
            )
        if file_log_append and truncate_delimited_report(
            log_file_path, checkpoint_step, separator=","
        ):
            print(
                f"## RESTART: Truncated {log_file_path} to checkpoint step "
                f"{checkpoint_step}."
            )
        if energy_append and truncate_delimited_report(
            energy_file_path, checkpoint_step, separator=","
        ):
            print(
                f"## RESTART: Truncated {energy_file_path} to checkpoint step "
                f"{checkpoint_step}."
            )
        if stress_append and truncate_delimited_report(
            stress_file_path, checkpoint_step
        ):
            print(
                f"## RESTART: Truncated {stress_file_path} to checkpoint step "
                f"{checkpoint_step}."
            )

    if production_resume:
        print(
            f"## Resume mdrun.py from production step {completed_steps}; "
            f"{remaining_steps} steps remain"
        )
    else:
        print("## Start of mdrun.py")

    reporter_screen = None
    if parameters["nst_screenlog"] > 0:
        print(
            "## REPORTER: Will report fixed progress fields step, time, "
            f"progress, speed, elapsedTime, and remainingTime to screen every "
            f"{parameters['nst_screenlog']} steps."
        )
        reporter_screen = ProgressReporter(
            sys.stdout,
            report_interval=parameters["nst_screenlog"],
            total_steps=target_context_step,
            separator="\t",
        )
        simulation.reporters.append(reporter_screen)

    reporter_file = None
    if parameters["nst_filelog"] > 0:
        print(
            "## REPORTER: Will report fixed progress fields step, time, "
            f"progress, speed, elapsedTime, and remainingTime to file "
            f"{log_file_path} every {parameters['nst_filelog']} steps."
        )
        reporter_file = ProgressReporter(
            log_file_path,
            report_interval=parameters["nst_filelog"],
            total_steps=target_context_step,
            append=file_log_append,
            separator=",",
        )
        simulation.reporters.append(reporter_file)

    reporter_energy = None
    if nst_energy > 0:
        stress_device = parameters.get("stress_device", "auto")
        if stress_device.lower() == "auto":
            stress_device = None
        stress_threads = parameters.get("stress_threads", 0)
        if stress_threads <= 0:
            stress_threads = None
        reporter_energy = EnergyReporter(
            energy_file_path,
            report_interval=nst_energy,
            fields=energy_fields,
            append=energy_append,
            temperature=parameters["production_temperature"],
            pressure_coupling=parameters["pcoulp"],
            stress_strain=parameters.get("stress_strain", 0.0003),
            stress_platform=parameters.get("stress_platform", "auto"),
            stress_precision=parameters.get("stress_precision", "double"),
            stress_device=stress_device,
            stress_threads=stress_threads,
        )
        simulation.reporters.append(reporter_energy)
        print(
            f"## REPORTER: Will report {', '.join(reporter_energy.fields)} to "
            f"{energy_file_path} every {nst_energy} production steps."
        )

    reporter_xtc = None
    if parameters["nst_xout"] > 0:
        reporter_xtc = RestartableXTCReporter(
            traj_file_path,
            report_interval=parameters["nst_xout"],
            append=trajectory_append,
        )
        simulation.reporters.append(reporter_xtc)
        print(
            f"## REPORTER: Will report trajectory to file {traj_file_path} every {parameters['nst_xout']} steps."
        )

    reporter_stress = None
    nst_stress = parameters.get("nst_stress", 0)
    if nst_stress > 0:
        if reporter_energy is not None and PRESSURE_FIELDS.intersection(
            reporter_energy.fields
        ):
            print(
                "## WARNING: Both EDR pressure terms and the legacy stress "
                "reporter are enabled; pressure tensors will be calculated twice."
            )
        if simulation.system.getNumConstraints() > 0:
            raise ValueError(
                "nst-stress cannot be used with constrained bonds because the "
                "constraint virial is unavailable. Use bondtype = bond."
            )
        if parameters["pcoulp"] is True:
            print(
                "## WARNING: Pressure coupling is enabled. The tensor can be "
                "recorded, but Green-Kubo viscosity must be calculated from an "
                "NVT production trajectory (pcoulp = False)."
            )
        stress_device = parameters.get("stress_device", "auto")
        if stress_device.lower() == "auto":
            stress_device = None
        stress_threads = parameters.get("stress_threads", 0)
        if stress_threads <= 0:
            stress_threads = None
        reporter_stress = StressTensorReporter(
            stress_file_path,
            report_interval=nst_stress,
            strain=parameters.get("stress_strain", 0.0003),
            temperature=parameters["production_temperature"],
            pressure_coupling=parameters["pcoulp"],
            platform=parameters.get("stress_platform", "auto"),
            precision=parameters.get("stress_precision", "double"),
            device=stress_device,
            threads=stress_threads,
            append=stress_append,
        )
        simulation.reporters.append(reporter_stress)
        print(
            f"## REPORTER: Will report the pressure tensor to {stress_file_path} "
            f"every {nst_stress} production steps."
        )

    if runtime_options.checkpoint_interval_seconds is None:
        print(
            "## CHECKPOINT: Periodic checkpointing is disabled; a checkpoint "
            "will still be saved on exit or interruption."
        )
    else:
        print(
            f"## CHECKPOINT: Will save {checkpoint_file_path} every "
            f"{runtime_options.checkpoint_minutes:g} wall-clock minutes, "
            "together with a portable State XML."
        )

    print(
        f"## The mdsystem contains {len(simulation.reporters)} types of reporters. They are listed as follows:"
    )
    reporter_names = [type(reporter).__name__ for reporter in simulation.reporters]
    print(f"       {', '.join(reporter_names)}")

    if reporter_xtc is not None and not trajectory_append:
        if simulation.currentStep % parameters["nst_xout"] == 0:
            reporter_xtc.report(
                simulation,
                simulation.context.getState(getPositions=True),
            )

    print(
        f"## Production target is {production_nstep} steps corresponding to "
        f"{parameters['dt'] / 1000 * nanosecond * production_nstep}."
    )
    print(
        f"## Production initialized at step {completed_steps}; "
        f"running {remaining_steps} remaining steps."
    )
    _set_simulation_temperature(simulation, parameters["production_temperature"])
    simulation._dropps_restart_metadata["stage"] = "production"

    stopped_for_maxh = False
    try:
        remaining_after_run, stopped_for_maxh = _run_production(
            simulation,
            remaining_steps,
            checkpoint_file_path,
            runtime_options,
            started_at=run_started_at,
        )
    except KeyboardInterrupt:
        _save_runtime_checkpoint(
            simulation,
            checkpoint_file_path,
            runtime_options.number_checkpoints,
            reason="interrupt checkpoint",
        )
        raise
    else:
        checkpoint_reason = (
            "maxh checkpoint" if stopped_for_maxh else "final checkpoint"
        )
        _save_runtime_checkpoint(
            simulation,
            checkpoint_file_path,
            runtime_options.number_checkpoints,
            reason=checkpoint_reason,
        )
    finally:
        if reporter_file is not None:
            reporter_file.close()
        if reporter_energy is not None:
            reporter_energy.close()
        if reporter_stress is not None:
            reporter_stress.close()
    now = datetime.now()  # current date and time
    date_time = now.strftime("%Y-%m-%d, %H:%M:%S")
    if stopped_for_maxh:
        print(
            f"## Production stopped safely for maxh at step "
            f"{simulation.currentStep}; {remaining_after_run} steps remain "
            f"({date_time})."
        )
    else:
        print(f"## Production run finalized at {date_time}.")

    _write_final_structure(simulation, pdb_raw, final_file_path)


prog = "mdrun"
desc = "Run or resume a molecular-dynamics simulation from a DROPPS run-input file."


def _nonnegative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise ValueError("must be zero or greater")
    return parsed


def _nonnegative_float(value):
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError("must be a finite number that is zero or greater")
    return parsed


def _max_hours(value):
    parsed = float(value)
    if not math.isfinite(parsed) or (parsed < 0 and parsed != -1):
        raise ValueError("must be -1 or a finite number that is zero or greater")
    return parsed


def _platform_name(value):
    names = {
        "auto": "auto",
        "cuda": "CUDA",
        "opencl": "OpenCL",
        "cpu": "CPU",
        "reference": "Reference",
    }
    try:
        return names[value.lower()]
    except KeyError as exc:
        raise ValueError("unknown OpenMM platform") from exc


def getargs_mdrun(argv):
    parser = ArgumentParser(prog=prog, description=desc)

    parser.add_argument(
        "-s",
        "--run-input",
        type=str,
        required=True,
        help=(
            "Input DROPPS run file (.tpr) containing the system and "
            "simulation settings."
        ),
    )
    parser.add_argument(
        "-cpi",
        "--checkpoint",
        type=str,
        nargs="?",
        const="auto",
        required=False,
        help=(
            "Resume from a native checkpoint (.chk) or portable State "
            "(.state.xml). If FILE is omitted, use the resolved checkpoint "
            "path and fall back to its paired State XML."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-prefix",
        "-deffnm",
        type=str,
        required=True,
        help=(
            "Default output prefix for .log, .edr, production .xtc, optional "
            ".warming.xtc, .chk, final .pdb, and optional .stress.xvg files."
        ),
    )
    parser.add_argument(
        "-x",
        "--trajectory-output",
        help="Override the compressed trajectory output path.",
    )
    parser.add_argument(
        "-e",
        "--energy-output",
        help="Override the DROPPS EDR output path.",
    )
    parser.add_argument(
        "-g",
        "--log-output",
        help="Override the fixed-schema progress-log output path.",
    )
    parser.add_argument(
        "-cpo",
        "--checkpoint-output",
        help="Override the latest checkpoint output path.",
    )
    parser.add_argument(
        "-c",
        "--final-structure",
        section="output",
        help="Override the final PDB structure output path.",
    )
    parser.add_argument(
        "--stress-output",
        help="Override the standalone pressure-tensor output path.",
    )

    parser.add_argument(
        "--platform",
        type=_platform_name,
        choices=["auto", "CUDA", "OpenCL", "CPU", "Reference"],
        default="auto",
        help="OpenMM compute platform.",
    )
    parser.add_argument(
        "-gpu_id",
        "-gpu-id",
        "--device-index",
        help="OpenMM CUDA/OpenCL device index or comma-separated indices.",
    )
    parser.add_argument(
        "--precision",
        type=str.lower,
        choices=["single", "mixed", "double"],
        help="CUDA/OpenCL arithmetic precision; GPU default is mixed.",
    )
    parser.add_argument(
        "-nt",
        "--threads",
        type=_nonnegative_int,
        default=0,
        help="CPU platform thread count; 0 uses the OpenMM default.",
    )
    parser.add_argument(
        "-nsteps",
        "--nsteps",
        type=_nonnegative_int,
        help="Override the TPR production target in steps for this run.",
    )
    parser.add_argument(
        "--seed",
        type=_nonnegative_int,
        help="Override the TPR random seed for a new run.",
    )
    parser.add_argument(
        "-cpt",
        "--checkpoint-minutes",
        type=_nonnegative_float,
        default=5.0,
        help=(
            "Wall-clock minutes between periodic checkpoints; 0 disables "
            "periodic saves, but a final or interrupt checkpoint is still saved."
        ),
    )
    parser.add_argument(
        "-cpnum",
        "--number-checkpoints",
        action="store_true",
        default=False,
        help="Keep step-numbered checkpoint copies in addition to the latest file.",
    )
    parser.add_argument(
        "-maxh",
        "--max-hours",
        type=_max_hours,
        default=-1.0,
        help=(
            "Stop production safely after 0.99 times this many hours; -1 "
            "disables the limit."
        ),
    )
    append_group = parser.add_mutually_exclusive_group(section="parameters")
    append_group.add_argument(
        "-append",
        "--append",
        dest="append",
        action="store_true",
        default=True,
        help=(
            "Append restart outputs after rolling them back to the checkpoint. "
            "Default: append mode."
        ),
    )
    append_group.add_argument(
        "-noappend",
        "--no-append",
        dest="append",
        action="store_false",
        help=(
            "Write restart outputs to the next .partNNNN files. Default: append mode."
        ),
    )

    args = parser.parse_args(argv)
    return args


mdrun_commands = single_command("mdrun", getargs_mdrun, mdrun, desc)
