# mdrun tool in CGPS package by Yiming Tang @ Fudan
# Development started on June 6 2025

import os
import sys
import json
import pickle

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
from dropps.share.command_class import single_command
from dropps.share.restart import (
    RestartableXTCReporter,
    save_checkpoint_safely,
    truncate_delimited_report,
    truncate_xtc,
)
from dropps.share.stress_tensor_reporter import StressTensorReporter

def mdrun(args):

    with open(args.run_input, "rb") as f:
        loaded_params = pickle.load(f)

    parameters = loaded_params["parameters"]
    mdsystem = loaded_params["mdsystem"]
    mdtopology = loaded_params["mdtopology"]
    positions = loaded_params["positions"]
    pdb_raw = loaded_params["pdb_raw"]

    log_file_path = args.output_prefix + ".log"
    traj_file_path = args.output_prefix + ".xtc"
    checkpoint_file_path = args.output_prefix + ".chk"
    final_file_path = args.output_prefix + ".pdb"

    checkpoint_input = args.checkpoint
    if checkpoint_input == "auto":
        checkpoint_input = checkpoint_file_path
    is_resume = checkpoint_input is not None
    if is_resume and not os.path.isfile(checkpoint_input):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_input}")

    def log(line):
        print(line)
        log_file = open(log_file_path, 'a+')
        log_file.write("%s  %s\n" 
                    % (datetime.now().strftime("%Y %b.%d %H:%M"), line))
        log_file.close()

    def notimelog(line):
        print(line)
        log_file = open(log_file_path, 'a+')
        log_file.write("%s\n" % line)
        log_file.close()

    # We first get structure and topology

    print("## Following parameters are read and phrased from the parameter file.")
    print(json.dumps(parameters, indent=4))

    # We now set up the integrator

    if parameters["integrator"] != "Langevin":
        print(f"ERROR: Interator {parameters['integrator']} is not supported")
        quit()

    temperature = parameters["production_temperature"] * kelvin
    friction = parameters["friction"] / picosecond
    timestep = parameters["dt"] * picosecond

    integrator = openmm.LangevinIntegrator(temperature, friction, timestep)
    print(f"## Initializing Langevin integrator with temperature: {temperature}, friction: {friction}, time step: {timestep}.")

    # We check and select platforms.
    # Ordered list of preferred platforms
    preferred_platforms = ['CUDA', 'OpenCL', 'CPU']

    # Get available platforms
    available_platforms = [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]
    print("## Available platforms on this machine: ", available_platforms)

    # Initialize platform and properties
    platform = None
    _properties = {}

    # Try to select the best available platform
    for name in preferred_platforms:
        if name in available_platforms:
            platform = openmm.Platform.getPlatformByName(name)
            print(f"## Using {name} platform.")
            if name == 'CUDA':
                _properties = {'DeviceIndex': '0', 'Precision': 'mixed'}
            elif name == 'OpenCL':
                _properties = {'OpenCLDeviceIndex': '0', 'OpenCLPrecision': 'mixed'}
            else:
                _properties = {}  # CPU typically needs no special properties
            break

    # Safety check
    if platform is None:
        raise RuntimeError("No suitable OpenMM platform found.")

    # We now set translational velocity remover
    if parameters["comm_mode"] == "Linear":
        mdsystem.addForce(openmm.CMMotionRemover(parameters["nstcomm"]))
        print(f"## Center of mass translational velocity will be removed every {parameters['nstcomm']} steps.")
    else:
        print("## WARNING: Center of mass translational velocity will not be removed.")


    if parameters["pcoulp"] is True:

        temperature = parameters["production_temperature"] * kelvin
        pressure = parameters["ref_P"] * bar 
        tau_pressure = parameters["tau_P"]
        
        pressure_coupling = openmm.MonteCarloBarostat(pressure, temperature, tau_pressure)
        mdsystem.addForce(pressure_coupling)

    #mdsystem.getForce(1).setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)

    #if parameters["coulombtype"] == "yukawa":
    #    mdsystem.getForce(2).setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)

        print(f"## Pressure coupling at {pressure} will be performed per {tau_pressure} steps.")

    else:
        print("## Simulation will be performed without pressure coupling.")

    for i, force in enumerate(mdsystem.getForces()):
        if isinstance(force, openmm.MonteCarloBarostat):
            print(f"## Barostat found at index {i}")

    simulation = openmm.app.Simulation(mdtopology, mdsystem, integrator)
    simulation.context.setPositions(positions)
    print("## Initial positions for the system succussfully passed to simulation.")

    if is_resume:
        simulation.loadCheckpoint(checkpoint_input)
        print(f"## Loaded checkpoint from {checkpoint_input}.")

    platform = simulation.context.getPlatform()
    print("## Simulation is running on platform:", platform.getName())

    # List all available platform properties
    property_names = platform.getPropertyNames()
    print("Available platform properties:")
    for name in property_names:
        value = platform.getPropertyValue(simulation.context, name)
        print(f"  {name} = {value}")

    if is_resume:
        print("## WARNING: You have specified a checkpoint file.")
        print("## Thus, the simulation will be performed without energy minimization and warming.")

    if parameters["minimize"] is True and not is_resume:
        print("## ENERGY MINIMIZE: Starting energy minimization.")

        minimization_step = parameters["max_step"]
        minimization_force_tol = parameters["forcetol"] * kilojoule_per_mole / nanometer

        print(f"## ENERGY MINIMIZE: The system will be ralexed until all force smaller than {minimization_force_tol} or reaching {minimization_step} steps.")
        
        simulation.minimizeEnergy(minimization_force_tol, minimization_step)
        simulation.minimizeEnergy()

        # Get the current state including energy and forces
        state = simulation.context.getState(getEnergy=True, getForces=True)
        potential_energy = state.getPotentialEnergy()
        forces = state.getForces(asNumpy=True)
        # Compute force magnitudes for each particle
        force_magnitudes = np.linalg.norm(forces, axis=1)
        max_force = np.max(force_magnitudes) * kilojoule_per_mole / nanometer

        # Compute RMS force
        rms_force = np.sqrt(np.mean(force_magnitudes**2)) * kilojoule_per_mole / nanometer

        # Print results
        print(f"## ENERGY MINIMIZE: Potential Energy after minimization: {potential_energy}")
        print(f"## ENERGY MINIMIZE: Maximum force magnitude after minimization: {max_force}")
        print(f"## ENERGY MINIMIZE: RMS force magnitude after minimization: {rms_force}")
        print("## ENERGY MINIMIZE: Ending energy minimization.")


    print(f"## The mdsystem contains {simulation.system.getNumForces()} types of forces. They are listed as follows:")
    force_names = [str(simulation.system.getForce(i)).split("'")[1].split(' ')[0]
        for i in range(simulation.system.getNumForces())]
    print(f"       {', '.join(force_names)}")

    production_nstep = parameters["nsteps"]

    temperature_initial = parameters["initial_temperature"] * kelvin
    temperature_final = parameters["production_temperature"] * kelvin

    if parameters["gen_vel"] is True and not is_resume:
        simulation.context.setVelocitiesToTemperature(temperature_initial)
        print(f"## Initializing all velocity according to a Boltzmann distribution at {temperature_initial}.")

    # Warming
    if temperature_initial < temperature_final and not is_resume:
        warming_speed = parameters["warming_speed"] * kelvin / nanosecond
        step_per_kelvin = int(1 * kelvin / warming_speed / timestep)
        print(f"## Will perform warming at a speed of {warming_speed} with {step_per_kelvin} steps per K.")

        for temperature in range(int(temperature_initial / kelvin), int(temperature_final / kelvin) + 1):
            simulation.integrator.setTemperature(temperature * kelvin)
            simulation.step(step_per_kelvin)
            if int(temperature - temperature_initial / kelvin) % 10 == 0:

                now = datetime.now() # current date and time
                date_time = now.strftime("%Y-%m-%d, %H:%M:%S")

                print(f"## Temperature has been rised to {temperature} K at {date_time}.")

        print("## Warming completed.")

    if not is_resume:
        # Checkpoints and production outputs use a production-local step/time
        # axis, independent of any warming steps performed above.
        simulation.currentStep = 0
        simulation.context.setTime(0 * picosecond)

    checkpoint_step = simulation.currentStep
    production_step_offset = 0
    if is_resume and checkpoint_step > production_nstep:
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

    stress_file_path = parameters.get("stress_output", "auto")
    if stress_file_path.lower() == "auto":
        stress_file_path = args.output_prefix + ".stress.xvg"

    trajectory_append = False
    file_log_append = False
    stress_append = False
    if is_resume:
        trajectory_append = (
            os.path.isfile(traj_file_path) and os.path.getsize(traj_file_path) > 0
        )
        file_log_append = (
            os.path.isfile(log_file_path) and os.path.getsize(log_file_path) > 0
        )
        stress_append = (
            os.path.isfile(stress_file_path)
            and os.path.getsize(stress_file_path) > 0
        )
        if trajectory_append and truncate_xtc(traj_file_path, checkpoint_step):
            print(
                f"## RESTART: Truncated {traj_file_path} to checkpoint step "
                f"{checkpoint_step}."
            )
            trajectory_append = (
                os.path.isfile(traj_file_path)
                and os.path.getsize(traj_file_path) > 0
            )
        if file_log_append and truncate_delimited_report(
            log_file_path, checkpoint_step, separator=","
        ):
            print(
                f"## RESTART: Truncated {log_file_path} to checkpoint step "
                f"{checkpoint_step}."
            )
        if stress_append and truncate_delimited_report(
            stress_file_path, checkpoint_step
        ):
            print(
                f"## RESTART: Truncated {stress_file_path} to checkpoint step "
                f"{checkpoint_step}."
            )

    if is_resume:
        log(
            f"## Resume mdrun.py from production step {completed_steps}; "
            f"{remaining_steps} steps remain"
        )
    else:
        log("## Start of mdrun.py")

    valid_keys = [
        "step", "time", "potentialEnergy", "kineticEnergy", "totalEnergy",
        "temperature", "volume", "density", "progress", "remainingTime",
        "speed", "elapsedTime"
    ]

    if parameters["nst_screenlog"] > 0:
        fields_screenlog = parameters["screenlog_grps"].strip().split(',')
        kwargs = {key: (key in fields_screenlog) for key in valid_keys}
        print(f"## REPORTER: Will report {', '.join([key for key in kwargs.keys() if kwargs[key]])} to screen every {parameters['nst_screenlog']} steps.")
        reporter_screen = openmm.app.StateDataReporter(
            sys.stdout,
            reportInterval=parameters["nst_screenlog"],
            **kwargs,
            totalSteps=target_context_step,
            separator='\t',
        )
        simulation.reporters.append(reporter_screen)

    if parameters["nst_filelog"] > 0:
        fields_filelog = parameters["filelog_grps"].strip().split(',')
        kwargs = {key: (key in fields_filelog) for key in valid_keys}
        print(f"## REPORTER: Will report {', '.join([key for key in kwargs.keys() if kwargs[key]])} to file {log_file_path} every {parameters['nst_filelog']} steps.")
        reporter_file = openmm.app.StateDataReporter(
            log_file_path,
            reportInterval=parameters["nst_filelog"],
            **kwargs,
            totalSteps=target_context_step,
            append=file_log_append,
        )
        simulation.reporters.append(reporter_file)

    reporter_xtc = None
    if parameters["nst_xout"] > 0:
        reporter_xtc = RestartableXTCReporter(
            traj_file_path,
            report_interval=parameters["nst_xout"],
            append=trajectory_append,
        )
        simulation.reporters.append(reporter_xtc)
        print(f"## REPORTER: Will report trajectory to file {traj_file_path} every {parameters['nst_xout']} steps.")

    reporter_stress = None
    nst_stress = parameters.get("nst_stress", 0)
    if nst_stress > 0:
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

    # Keep this reporter last so a checkpoint never gets ahead of trajectory,
    # log, or stress output scheduled for the same step.
    if parameters["nst_cp"] > 0:
        reporter_checkpoint = openmm.app.CheckpointReporter(
            checkpoint_file_path, reportInterval=parameters["nst_cp"]
        )
        simulation.reporters.append(reporter_checkpoint)
        print(f"## REPORTER: Will flush checkpoint to file {checkpoint_file_path} every {parameters['nst_cp']} steps.")

    print(f"## The mdsystem contains {len(simulation.reporters)} types of reporters. They are listed as follows:")
    reporter_names = [
        type(reporter).__name__ for reporter in simulation.reporters
    ]
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
    simulation.integrator.setTemperature(temperature_final)

    try:
        simulation.step(remaining_steps)
    except KeyboardInterrupt:
        if parameters["nst_cp"] > 0:
            save_checkpoint_safely(simulation, checkpoint_file_path)
            print(
                f"## Interrupted at step {simulation.currentStep}; saved "
                f"checkpoint to {checkpoint_file_path}."
            )
        raise
    else:
        if parameters["nst_cp"] > 0:
            save_checkpoint_safely(simulation, checkpoint_file_path)
            print(
                f"## Saved final checkpoint at step {simulation.currentStep} "
                f"to {checkpoint_file_path}."
            )
    finally:
        if reporter_stress is not None:
            reporter_stress.close()
    now = datetime.now() # current date and time
    date_time = now.strftime("%Y-%m-%d, %H:%M:%S")
    print(f"## Production run finilized at {date_time}.")

    # Get positions in nanometers without units
    positions = simulation.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(nanometer)

    # Get box vectors and strip units
    box_vectors = simulation.context.getState().getPeriodicBoxVectors()
    box_vectors_no_unit = tuple(v.value_in_unit(nanometer) for v in box_vectors)
    box_vec = [box_vectors_no_unit[0][0], box_vectors_no_unit[1][1], box_vectors_no_unit[2][2]]

    # Wrap positions into box
    wrapped_positions = positions % box_vec  # modulo operation wraps coordinates

    x_nms = [position[0] for position in wrapped_positions]
    y_nms = [position[1] for position in wrapped_positions]
    z_nms = [position[2] for position in wrapped_positions]

    # Set the unitless box vectors on the topology
    #topology = simulation.topology
    #topology.setPeriodicBoxVectors(box_vectors_no_unit)

    # Write the PDB file
    #with open(final_file_path, "w") as f:
    #    openmm.app.PDBFile.writeFile(topology, positions * 10, f, keepIds=True)

    PDB = pdb_raw
    write_pdb(final_file_path, box_vec,
                PDB.record_names, PDB.serial_numbers, PDB.atom_names, PDB.residue_names,
                PDB.chain_IDs, PDB.residue_sequence_numbers, x_nms, y_nms, z_nms,
                PDB.occupancys, PDB.bfactors, PDB.elements, PDB.molecule_length_list)

prog = "mdrun"
desc = "Run or resume a molecular-dynamics simulation from a DROPPS run-input file."

def getargs_mdrun(argv):

    parser = ArgumentParser(prog=prog, description=desc)

    parser.add_argument('-s', '--run-input', type=str, required=True, 
                        help="Input DROPPS run file (.tpr) containing the system and simulation settings.")
    parser.add_argument(
        '-cpi', '--checkpoint',
        type=str,
        nargs='?',
        const='auto',
        required=False,
        help=(
            "Resume from an input checkpoint file (.chk). If FILE is omitted, use "
            "<output-prefix>.chk."
        ),
    )
    parser.add_argument('-o', '--output-prefix', type=str, required=True,
                        help="Output prefix for .log, .xtc, .chk, final .pdb, and optional .stress.xvg files.")

    args = parser.parse_args(argv)
    return args

mdrun_commands = single_command("mdrun", getargs_mdrun, mdrun, desc)
