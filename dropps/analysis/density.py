# density tool in DROPPS package by Yiming Tang @ Fudan
# Development started on July 17 2025

from argparse import ArgumentParser
from dropps.share.trajectory import trajectory_class

from openmm.unit import nanosecond, picosecond
from MDAnalysis.analysis import lineardensity as lin
import MDAnalysis as mda
import numpy as np
import tqdm
from math import floor

from dropps.fileio.xvg_reader import write_xvg

prog = "density"
desc = '''This program calculate the density profile of a simulation box along one axis.'''


def shift_density_center(density_profiles, dense_phase_threshold, density_profile_for_center):
    """
    Centers the dense phase in the z direction of a slab-like simulation box,
    considering the periodic boundary condition.

    Parameters:
        density_profile (numpy.ndarray): 1D array representing the density profile along the z-axis.

    Returns:
        numpy.ndarray: Shifted density profile with the dense phase centered.
    """

    # Start of shift determination

    density_profile = density_profile_for_center

    #z_length = len(density_profile)
    max_density = np.max(density_profile)
    threshold_density = dense_phase_threshold * max_density

    dense_bins = density_profile > threshold_density

    # Find the continuous dense phase regions considering periodic boundary conditions
    extended_dense_bins = np.concatenate([dense_bins, dense_bins])  # Extend for periodicity
    dense_region_indices = np.where(extended_dense_bins)[0]

    # Find the largest continuous region in the dense bins
    dense_phases = []
    current_phase = [dense_region_indices[0]]
    for idx in dense_region_indices[1:]:
        if idx == current_phase[-1] + 1:
            current_phase.append(idx)
        else:
            dense_phases.append(current_phase)
            current_phase = [idx]
    dense_phases.append(current_phase)
 
    # Map back to the original bins and choose the largest region
    num_bins = len(density_profile)
    largest_dense_phase = max(dense_phases, key=len)
    dense_phase_center = floor(np.mean(largest_dense_phase) % num_bins)
    
    shift = floor(num_bins / 2 - dense_phase_center)

    # End of shift determination

    shifted_density_profiles = [np.roll(density_profile_single, shift) for density_profile_single in density_profiles]

    return shifted_density_profiles

def getargs_density(argv):

    parser = ArgumentParser(prog=prog, description=desc)

    parser.add_argument('-s', '--run-input', type=str, required=True, 
                        help="TPR file containing all information for a simulation run.")
    
    parser.add_argument('-f', '--input', type=str, required=True, 
                    help="XTC file which is taken as input trajectory.")
    
    parser.add_argument('-n', '--index', type=str, required=False,
                        help="Index file containing non-default groups.")

    parser.add_argument('-o', '--output', type=str, required=True, 
                        help="XVG file to write density profile.")

    parser.add_argument('-x', '--axis', type=str, choices=['x', 'y', 'z'], default='z',
                        help="Axis along which the density profile will be calculated. Default: z")
    
    parser.add_argument('-tp', '--type', type=str, choices=["mass", "charge"], default="mass",
                        help="Calculate mass density or charge density.")

    parser.add_argument('-b', '--start-time', type=int,
                        help="Time (ns) of the first frame to calculate.")

    parser.add_argument('-e', '--end-time', type=int,
                        help="Time (ns) of the last frame to calculate.")

    parser.add_argument('-dt', '--delta-time', type=int, default=1,
                        help="Intervals (ns) between two attempts recentering density profiles.")
    
    parser.add_argument('-selfit', '--selection-fit', type=int,
                        help="Group of atoms on which the dense phase is determined.")
    
    parser.add_argument('-sel', '--selection-calculate', type=int, nargs='+',
                        help="Group of atoms on which the density profile will be printed")

    parser.add_argument('-nc', '--no-center', default=False, action='store_true',
                        help="Do not center density profile for each time window. Default is center.")

    parser.add_argument('-t', '--dense-phase-threshold', default=0.5, type=float,
                        help="Threshold for dense phase which is a multiplier of the highest density.")

    args = parser.parse_args(argv)
    return args

def density(args):

    cal_charge = True if args.type == "charge" else False
    if cal_charge:
        print(f"## Depend on user input, this program will calculate charge density instead of mass.")

    output_file_name = args.output if args.output.endswith(".xvg") else args.output + ".xvg"

    # We now create output file.
    try:
        ofile = open(output_file_name, 'w')
        print("## Open text file %s for output." % output_file_name)
    except:
        print("## An exception occurred when trying to open text file %s for output." % output_file_name)
        quit()

    # load trajectory into memory

    try:
        trajectory = trajectory_class(args.run_input, args.index, args.input)
    except:
        print("## An exception occurred when trying to open trajectory file %s." % args.input)
        quit()

    # We treat time for analysis and generate frame for analysis

    start_time = args.start_time * nanosecond if args.start_time is not None else trajectory.time_init()
    end_time = args.end_time * nanosecond if args.end_time is not None else trajectory.time_end()
    center_interval_time = args.delta_time * nanosecond

    if start_time < trajectory.time_init() or end_time > trajectory.time_end():
        print(f"ERROR: Trajectory containing {trajectory.time_init()} - {trajectory.time_end()} "
              + f"while you demanding {start_time} - {end_time}.")
        print(f"YOU MUST BE KIDDING ME.")
        quit()

    start_frame = int((start_time - trajectory.time_init()).value_in_unit(picosecond) / trajectory.time_step().value_in_unit(picosecond))
    end_frame = int((end_time - trajectory.time_init()).value_in_unit(picosecond) / trajectory.time_step().value_in_unit(picosecond))
    center_interval_frame = int(center_interval_time.value_in_unit(picosecond) / trajectory.time_step().value_in_unit(picosecond))

    print(f"## Will use time {start_time} to {end_time} with recentering interval of {center_interval_time}.")
    print(f"## Will use frame {start_frame} to {end_frame} with recentering interval of {center_interval_frame}.")
    
    print(f"## Will calculate density profile along the {args.axis} axis.")

    if args.no_center:
        print(f"## Will not center the density profile.")
    else:
        print(f"## Will center dense which is defined as blocks with density higher than {args.dense_phase_threshold} * highest-density.")

    # Build a frame list
    frames = range(start_frame, end_frame, center_interval_frame)

    # We now generate atom groups for dense phase determination and density calculations

    if args.selection_fit is None or args.selection_calculate is None:
        trajectory.index.print_all()

    if args.selection_fit is not None:
        print(f"## Will use group {args.selection_fit} for dense phase determination.")
        fit_group, fit_group_name = trajectory.getSelection(f"group {args.selection_fit}")
    else:
        fit_group, fit_group_name = trajectory.getSelection_interactive("group for dense phase determination")
    
    if args.selection_calculate is not None:
        print(f"## Will use groups {','.join([f"{i}" for i in args.selection_calculate])} for density calculations.")
        density_groups = [trajectory.getSelection(f"group {gid}")[0] for gid in args.selection_calculate]
        density_groups_names = [f"group{i}" for i in args.selection_calculate]
    else:
        density_groups, density_groups_names = trajectory.getSelection_interactive_multiple("groups for density calculation")
    
    # Initialize the main density profile array
    density_profiles = None
    period_number = len(frames) - 1

    #all_groups = [fit_group]
    #all_groups.extend(density_groups)

    #for sel in all_groups:
    #    assert isinstance(sel, mda.core.groups.AtomGroup), "Selection must be an AtomGroup"

    for index in tqdm.tqdm(range(period_number), desc='## Calculating'):
        start_frame = frames[index]
        end_frame = frames[index + 1]

        # Load selected frames into MDAnalysis

        mass_density_for_center_analyzer = lin.LinearDensity(fit_group, grouping='atoms', binsize=0.5).run(start=start_frame, stop=end_frame)

        density_profiles_temp = [lin.LinearDensity(selection, grouping='atoms', binsize=0.5).run(start=start_frame, stop=end_frame)
                                for selection in density_groups]
        
        axis = getattr(args, "axis", "z")  # Default to "z" if not set
        component = getattr(mass_density_for_center_analyzer.results, axis)
        field = "charge_density" if cal_charge else "mass_density"

        raw_densities = [
            np.array(density.results[axis][field]) * (1e3 if not cal_charge else 1)
            for density in density_profiles_temp
        ]
        mass_density_for_center = np.array(getattr(component, "mass_density"))
        x = (mass_density_for_center_analyzer.results[axis].hist_bin_edges[1:] + mass_density_for_center_analyzer.results[axis].hist_bin_edges[:-1]) / 2


        # Center density profile if specified
        if not args.no_center:
            output_densities = np.array(shift_density_center(raw_densities, args.dense_phase_threshold, mass_density_for_center))
        else:
            output_densities = np.array(raw_densities)
        
        # Accumulate the density profiles
        if density_profiles is None:
            density_profiles = output_densities
        else:
            density_profiles += output_densities
    
    # Average density profile across frames

    density_profiles /= period_number
    density_profiles = density_profiles

    # We now create output file.

    try:

        xlabel = f"{axis} Axis (nm)"
        ylabel = f"Charge density (e/mol/mL)" if cal_charge else f"Mass density (mg/mL)"
        title = f"Charge density" if cal_charge else f"Mass density"
        legends = density_groups_names

        write_xvg(output_file_name, x / 10.0, density_profiles, title=title, xlabel=xlabel, ylabel=ylabel, legends=legends)

    except:
        print("## An exception occurred when trying to open text file %s for output." % output_file_name)
        quit()

    print(f"## Density profile written to {output_file_name}.")   

from dropps.share.command_class import single_command
density_commands = single_command("density", getargs_density, density, desc)
