#!/usr/bin/env python3

# mdrun tool in CGPS package by Yiming Tang @ Fudan
# Development started on June 6 2025

import os
import sys
import json

import numpy as np

from datetime import datetime

import openmm
import openmm.app
from openmm.unit import nanometer, kilojoule_per_mole, moles, liter, picosecond, kelvin, bar, nanosecond

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from argparse import ArgumentParser
from dropps.share.forcefield import forcefield_list
from dropps.share.parameters import getparameter
from datetime import datetime
from math import ceil,floor

from dropps.share.build_system import build_system

from dropps.fileio.pdb_reader import read_pdb, write_pdb, phrase_pdb_atoms

prog = "mdrun"
desc = '''This program initates and runs an molecular dynamics simulation.'''

parser = ArgumentParser(prog=prog, description=desc)

parser.add_argument('-f', '--structure', type=str, required=True, 
                    help="GSD file containing the initial configuration of the system.")
parser.add_argument('-p', '--topology', type=str, required=True, 
                    help="GSD file containing the initial configuration of the system.")
parser.add_argument('-m', '--parameter', type=str, required=True,
                    help="Parameter file (mdp) for this simulation.")
#parser.add_argument('-ff', '--forcefield', choices=forcefield_list, help="Forcefield selection",
#                    default='HPS')
parser.add_argument('-o', '--output-prefix', type=str, required=True,
                    help="Prefix of all output files.")

args = parser.parse_args()

log_file_path = args.output_prefix + ".log"
traj_file_path = args.output_prefix + ".xtc"
checkpoint_file_path = args.output_prefix + ".chk"
final_file_path = args.output_prefix + ".pdb"

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

log("## Start of mdrun.py")

# We first get structure and topology

parameters = getparameter(args.parameter)
print("## Following parameters are read and phrased from the parameter file.")
print(json.dumps(parameters, indent=4))

mdsystem, mdtopology, positions = build_system(args.structure, args.topology, getparameter(args.parameter))
# We now set up the integrator

if parameters["integrator"] != "Langevin":
    print(f"ERROR: Interator {parameters["integrator"]} is not supported")
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
properties = {}

# Try to select the best available platform
for name in preferred_platforms:
    if name in available_platforms:
        platform = openmm.Platform.getPlatformByName(name)
        print(f"## Using {name} platform.")
        if name == 'CUDA':
            properties = {'DeviceIndex': '0', 'Precision': 'mixed'}
        elif name == 'OpenCL':
            properties = {'OpenCLDeviceIndex': '0', 'OpenCLPrecision': 'mixed'}
        else:
            properties = {}  # CPU typically needs no special properties
        break

# Safety check
if platform is None:
    raise RuntimeError("No suitable OpenMM platform found.")

# We now set translational velocity remover
if parameters["comm_mode"] == "Linear":
    mdsystem.addForce(openmm.CMMotionRemover(parameters["nstcomm"]))
    print(f"## Center of mass translational velocity will be removed every {parameters["nstcomm"]} steps.")
else:
    print(f"## WARNING: Center of mass translational velocity will not be removed.")


if parameters["pcoulp"] is True:

    temperature = parameters["production_temperature"] * kelvin
    pressure = parameters["ref_P"] * bar 
    tau_pressure = parameters["tau_P"]
    
    pressure_coupling = openmm.MonteCarloBarostat(pressure, temperature, tau_pressure)
    mdsystem.addForce(pressure_coupling)

    mdsystem.getForce(1).setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)

    if parameters["coulombtype"] == "yukawa":
        mdsystem.getForce(2).setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)


    print(f"## Pressure coupling at {pressure} will be performed per {tau_pressure} steps.")

else:
    print(f"## Simulation will be performed without pressure coupling.")

for i, force in enumerate(mdsystem.getForces()):
    if isinstance(force, openmm.MonteCarloBarostat):
        print(f"## Barostat found at index {i}")

simulation = openmm.app.Simulation(mdtopology, mdsystem, integrator)
simulation.context.setPositions(positions)
print(f"## Initial positions for the system succussfully passed to simulation.")

platform = simulation.context.getPlatform()
print("## Simulation is running on platform:", platform.getName())

# List all available platform properties
property_names = platform.getPropertyNames()
print("Available platform properties:")
for name in property_names:
    value = platform.getPropertyValue(simulation.context, name)
    print(f"  {name} = {value}")


if parameters["minimize"] is True:
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
    print(f"## ENERGY MINIMIZE: Ending energy minimization.")


print(f"## The mdsystem contains {simulation.system.getNumForces()} types of forces. They are listed as follows:")
force_names = [str(simulation.system.getForce(i)).split("'")[1].split(' ')[0]
    for i in range(simulation.system.getNumForces())]
print(f"       {', '.join(force_names)}")

valid_keys = [
    "step", "time", "potentialEnergy", "kineticEnergy", "totalEnergy", "temperature",
    "volume", "density", "progress", "remainingTime", "speed", "elapsedTime"
]

production_nstep = parameters["nsteps"]
warming_nstep = (parameters["production_temperature"] - parameters["initial_temperature"]) * kelvin \
    / (parameters["warming_speed"] * kelvin / nanosecond) / timestep

if parameters["nst_screenlog"] > 0:

    fields_screenlog = parameters["screenlog_grps"].strip().split(',')
    kwargs = {key: (key in fields_screenlog) for key in valid_keys}
    print(f"## REPORTER: Will report {', '.join([key for key in kwargs.keys() if kwargs[key]])} to screen every {parameters["nst_screenlog"]} steps.")
    reporter_screen = openmm.app.StateDataReporter(sys.stdout, reportInterval = parameters["nst_screenlog"], 
                                                   **kwargs, totalSteps=(production_nstep + warming_nstep),
                                                   separator='\t')
    simulation.reporters.append(reporter_screen)

if parameters["nst_filelog"] > 0:

    fields_filelog = parameters["filelog_grps"].strip().split(',')
    kwargs = {key: (key in fields_filelog) for key in valid_keys}
    print(f"## REPORTER: Will report {', '.join([key for key in kwargs.keys() if kwargs[key]])} to file {log_file_path} every {parameters["nst_filelog"]} steps.")
    reporter_file = openmm.app.StateDataReporter(log_file_path, reportInterval = parameters["nst_filelog"], 
                                                 **kwargs, totalSteps=(production_nstep + warming_nstep))
    simulation.reporters.append(reporter_file)

if parameters["nst_xout"] > 0:

    reporter_xtc = openmm.app.XTCReporter(traj_file_path, reportInterval = parameters["nst_xout"])
    simulation.reporters.append(reporter_xtc)
    print(f"## REPORTER: Will report trajectory to file {traj_file_path} every {parameters["nst_xout"]} steps.")

if parameters["nst_cp"] > 0:

    reporter_checkpoint = openmm.app.CheckpointReporter(checkpoint_file_path, reportInterval=parameters["nst_cp"])
    simulation.reporters.append(reporter_checkpoint)
    print(f"## REPORTER: Will flush checkpoint to file {checkpoint_file_path} every {parameters["nst_cp"]} steps.")

print(f"## The mdsystem contains {len(simulation.reporters)} types of reporters. They are listed as follows:")

reporter_names = [str(simulation.reporters[i]).split(" ")[0].split('.')[-1]
    for i in range(len(simulation.reporters))]
print(f"       {', '.join(reporter_names)}")

temperature_initial = parameters["initial_temperature"] * kelvin
temperature_final = parameters["production_temperature"] * kelvin

if parameters["gen_vel"] is True:
    simulation.context.setVelocitiesToTemperature(temperature_initial)
    print(f"## Initializing all velocity according to a Boltzmann distribution at {temperature_initial}.")

# Warming
if temperature_initial < temperature_final:
    warming_speed = parameters["warming_speed"] * kelvin / nanosecond
    step_per_kelvin = int(1 * kelvin / warming_speed / timestep)
    print(f"## Will perform warming at a speed of {warming_speed} with {step_per_kelvin} steps per K.")

    for temperature in range(int(temperature_initial / kelvin), int(temperature_final / kelvin) + 1):
        simulation.integrator.setTemperature(temperature * kelvin)
        simulation.step(step_per_kelvin)
        if int(temperature - temperature_initial / kelvin) % 10 == 0:

            now = datetime.now() # current date and time
            year = now.strftime("%Y")
            month = now.strftime("%m")
            day = now.strftime("%d")
            time = now.strftime("%H:%M:%S")
            date_time = now.strftime("%Y-%m-%d, %H:%M:%S")

            print(f"## Temperature has been rised to {temperature} K at {date_time}.")

    print(f"## Warming completed.")
    
# Production run
print(f"## Production initilized for {parameters["nsteps"]} steps corresponding to {parameters["dt"] / 1000 * nanosecond * parameters["nsteps"]}.")
simulation.integrator.setTemperature(temperature_final * kelvin)
simulation.step(parameters["nsteps"])
now = datetime.now() # current date and time
year = now.strftime("%Y")
month = now.strftime("%m")
day = now.strftime("%d")
time = now.strftime("%H:%M:%S")
date_time = now.strftime("%Y-%m-%d, %H:%M:%S")
print(f"## Production run finilized at {date_time}.")

# Get positions in nanometers without units
positions = simulation.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(nanometer)

x_nms = [position[0] for position in positions]
y_nms = [position[1] for position in positions]
z_nms = [position[2] for position in positions]

# Get box vectors and strip units
box_vectors = simulation.context.getState().getPeriodicBoxVectors()
box_vectors_no_unit = tuple(v.value_in_unit(nanometer) for v in box_vectors)
box_vec = [box_vectors_no_unit[0][0], box_vectors_no_unit[1][1], box_vectors_no_unit[2][2]]

# Set the unitless box vectors on the topology
topology = simulation.topology
topology.setPeriodicBoxVectors(box_vectors_no_unit)

# Write the PDB file
#with open(final_file_path, "w") as f:
#    openmm.app.PDBFile.writeFile(topology, positions * 10, f, keepIds=True)

atoms, box = read_pdb(args.structure)
PDB = phrase_pdb_atoms(atoms, box)
write_pdb(final_file_path, box_vec,
            PDB.record_names, PDB.serial_numbers, PDB.atom_names, PDB.residue_names,
            PDB.chain_IDs, PDB.residue_sequence_numbers, x_nms, y_nms, z_nms,
            PDB.occupancys, PDB.bfactors, PDB.elements, PDB.molecule_length_list)