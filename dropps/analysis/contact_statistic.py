# contact statistic toolin DROPPS package by Yiming Tang @ Fudan
# Development started on Feb 2 2026

from dropps.share.argument_parser import ArgumentParser, RawDescriptionHelpFormatter
from dropps.share.command_class import single_command
import numpy as np

from dropps.share.trajectory import trajectory_class
from dropps.fileio.filename_control import validate_extension

prog = "cstat"
desc = "Calculate residue-level statistics from a precomputed contact map."

known_grouping_schemes = {
    "HPST": {
        "Aromatic": ("A", ["H", "F", "W", "Y"]),
        "Hydrophobic": ("H", ["A", "I", "L", "M", "V"]),
        "Other": ("O", ["C", "G", "P"]),
        "Charged": ("C", ["K", "R", "D", "E"]),
        "Polar": ("P", ["N", "Q", "S", "T"]),
    },
    "AHCP": {
        "Aromatic": ("A", ["H", "F", "W", "Y"]),
        "Hydrophobic": ("H", ["A", "I", "L", "M", "V", "P"]),
        "Charged": ("C", ["K", "R", "D", "E"]),
        "Polar": ("P", ["N", "Q", "S", "T", "C", "G"]),
    },
    "HCP": {
        "Hydrophobic": ("H", ["H", "F", "W", "Y", "A", "I", "L", "M", "V", "P"]),
        "Charged": ("C", ["K", "R", "D", "E"]),
        "Polar": ("P", ["N", "Q", "S", "T", "C", "G"]),
    },
    "HPST_SC": {
        "Aromatic": ("A", ["H", "F", "W", "Y"]),
        "Hydrophobic": ("H", ["A", "I", "L", "M", "V"]),
        "Other": ("O", ["C", "G", "P"]),
        "Charged+": ("C+", ["K", "R"]),
        "Charged-": ("C-", ["D", "E"]),
        "Polar": ("P", ["N", "Q", "S", "T"]),
    },
    "AHCP_SC": {
        "Aromatic": ("A", ["H", "F", "W", "Y"]),
        "Hydrophobic": ("H", ["A", "I", "L", "M", "V", "P"]),
        "Charged+": ("C+", ["K", "R"]),
        "Charged-": ("C-", ["D", "E"]),
        "Polar": ("P", ["N", "Q", "S", "T", "C", "G"]),
    },
    "HCP_SC": {
        "Hydrophobic": ("H", ["H", "F", "W", "Y", "A", "I", "L", "M", "V", "P"]),
        "Charged+": ("C+", ["K", "R"]),
        "Charged-": ("C-", ["D", "E"]),
        "Polar": ("P", ["N", "Q", "S", "T", "C", "G"]),
    },
}

epilog_lines = []
for scheme_name, scheme in known_grouping_schemes.items():
    scheme_details = ", ".join(
        [
            f"{subk} ({','.join(residues)})"
            for subk, (abbreviation, residues) in scheme.items()
        ]
    )
    epilog_lines.append(f"  {scheme_name}: {scheme_details}")
epilog = "Known grouping schemes:\n" + "\n".join(epilog_lines)

full_desc = desc + "\n\n" + epilog


def getargs_contact_statistic(argv):
    parser = ArgumentParser(
        prog=prog, description=full_desc, formatter_class=RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-m",
        "--map-input",
        type=str,
        required=True,
        help="Input contact-map matrix file (.dat).",
    )

    parser.add_argument(
        "-s",
        "--run-input",
        type=str,
        required=True,
        help="Input DROPPS run file (.tpr) containing the system topology.",
    )

    parser.add_argument(
        "-ref",
        "--reference-group",
        type=int,
        help="Reference index group corresponding to the contact-map x axis.",
    )

    parser.add_argument(
        "-sel",
        "--selection-group",
        type=int,
        help="Selection index group corresponding to the contact-map y axis.",
    )

    parser.add_argument(
        "-n",
        "--index",
        type=str,
        required=False,
        help="Optional index file (.ndx) defining additional atom groups.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output Excel workbook (.xlsx); the extension is added if omitted.",
    )

    parser.add_argument(
        "-gr",
        "--group-residue",
        action="store_true",
        default=False,
        help="Aggregate residues into residue-type groups.",
    )

    parser.add_argument(
        "-gs",
        "--group-scheme",
        type=str,
        choices=list(known_grouping_schemes.keys()),
        default=None,
        help="Predefined residue-grouping scheme.",
    )

    parser.add_argument(
        "-gf",
        "--group-file",
        type=str,
        default=None,
        help="Optional custom grouping file with GROUP_NAME ABBREVIATION RESIDUE... on each line.",
    )

    parser.add_argument(
        "-ag",
        "--aggregate",
        type=str,
        choices=["sum", "average"],
        default="sum",
        help="Aggregation method for grouped contact statistics.",
    )

    args = parser.parse_args(argv)

    return args


def contact_statistic(args):
    # load contact map

    try:
        with open(args.map_input) as f:
            first_line = f.readline()
        ncol = len(first_line.split())
        contact_map = np.fromfile(args.map_input, sep=" ").reshape(-1, ncol)
        print(
            f"## Loaded contact map from {args.map_input} with shape {contact_map.shape}."
        )

    except Exception as e:
        raise RuntimeError(f"Failed to load contact map file {args.map_input}: {e}")

    # load trajectory into memory

    try:
        trajectory = trajectory_class(args.run_input, args.index)

    except Exception as e:
        raise RuntimeError(
            f"Failed to load trajectory from {args.run_input} and/or index {args.index}: {e}"
        )

    # Read chains and determine residue grouping

    # We now generate atom groups for calculations

    if args.reference_group is None or args.selection_group is None:
        trajectory.index.print_all()

    if args.reference_group is not None:
        print(f"## Will use group {args.reference_group} as reference group.")
        reference_group, reference_group_name = trajectory.getSelection(
            f"group {args.reference_group}"
        )
    else:
        reference_group, reference_group_name = trajectory.getSelection_interactive(
            f"reference group (chain length: {contact_map.shape[0]})"
        )

    if args.selection_group is not None:
        print(f"## Will use group {args.selection_group} as selection group.")
        selection_group, selection_group_name = trajectory.getSelection(
            f"group {args.selection_group}"
        )
    else:
        selection_group, selection_group_name = trajectory.getSelection_interactive(
            f"selection group (chain length: {contact_map.shape[1]})"
        )

    ref_length = len(reference_group.indices)
    sel_length = len(selection_group.indices)

    if ref_length == 0:
        print("ERROR: Reference group has zero atoms.")
        quit()
    if sel_length == 0:
        print("ERROR: Selection group has zero atoms.")
        quit()

    # We now split the indices and check chains

    reference_chains = trajectory.index.splitch_indices(reference_group.indices)
    chain_length_list_reference = [len(chain) for chain in reference_chains]
    if len(set(chain_length_list_reference)) != 1:
        print("ERROR: Chains in reference groups are not same in length.")
        quit()
    chain_length_reference = chain_length_list_reference[0]
    print(
        f"## Processed {len(reference_chains)} chains of length {chain_length_reference} in reference group."
    )

    selection_chains = trajectory.index.splitch_indices(selection_group.indices)
    chain_length_list_selection = [len(chain) for chain in selection_chains]
    if len(set(chain_length_list_selection)) != 1:
        print("ERROR: Chains in selection groups are not same in length.")
        quit()
    chain_length_selection = chain_length_list_selection[0]
    print(
        f"## Processed {len(selection_chains)} chains of length {chain_length_selection} in selection group."
    )

    # We now check the contact map dimensions

    seq_reference = [
        name.replace("_N", "").replace("_C", "")
        for name in trajectory.Universe.atoms[reference_chains[0]].names
    ]
    seq_selection = [
        name.replace("_N", "").replace("_C", "")
        for name in trajectory.Universe.atoms[selection_chains[0]].names
    ]

    if len(seq_reference) != contact_map.shape[0]:
        print(
            f"ERROR: Contact map row number {contact_map.shape[0]} does not match reference chain length {len(seq_reference)}."
        )
        quit()
    if len(seq_selection) != contact_map.shape[1]:
        print(
            f"ERROR: Contact map column number {contact_map.shape[1]} does not match selection chain length {len(seq_selection)}."
        )
        quit()

    # We now determine residue grouping if needed

    if args.group_residue:
        if args.group_scheme is not None:
            grouping_scheme = known_grouping_schemes[args.group_scheme]
            print(
                f"## Using known grouping scheme {args.group_scheme} for residue grouping."
            )
        elif args.group_file is not None:
            grouping_scheme = {}
            try:
                with open(args.group_file) as f:
                    for line in f:
                        parts = line.strip().split()
                        group_name = parts[0]
                        abbreviation = parts[1]
                        residues = parts[2:]
                        grouping_scheme[group_name] = (abbreviation, residues)
                print(
                    f"## Loaded custom grouping scheme from {args.group_file} for residue grouping."
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load grouping scheme file {args.group_file}: {e}"
                )
        else:
            print(
                "ERROR: Grouping scheme or grouping file must be provided when --group-residue is set."
            )
            quit()

        print("## Grouping scheme details:")
        for group_name, (abbreviation, residues) in grouping_scheme.items():
            print(f"   Group {group_name} ({abbreviation}): {', '.join(residues)}")

        all_assigned = True

        for res in set(seq_reference).union(set(seq_selection)):
            found = any(res in v[1] for v in grouping_scheme.values())
            if not found:
                all_assigned = False
                # If residue not found in any group, let user determine their grouping
                assigned = False
                while not assigned:
                    print(f"> Residue {res} not found in any grouping scheme.")
                    abbreviation = input(
                        f"  Please enter abbreviation for residue {res}: "
                    ).strip()
                    if abbreviation not in [v[0] for v in grouping_scheme.values()]:
                        print(
                            f"ERROR: Abbreviation {abbreviation} not recognized in existing grouping scheme."
                        )
                        continue
                    group_name = [
                        k for k, v in grouping_scheme.items() if v[0] == abbreviation
                    ][0]
                    grouping_scheme[group_name][1].append(res)
                    assigned = True

        if not all_assigned:
            print("## Final grouping scheme after user assignment:")
            for group_name, (abbreviation, residues) in grouping_scheme.items():
                print(f"   Group {group_name} ({abbreviation}): {', '.join(residues)}")

    else:
        # Treat each residue as a separate group
        grouping_scheme = {}
        unique_residues = set(seq_reference).union(set(seq_selection))
        for res in unique_residues:
            grouping_scheme[res] = (res, [res])
        print("## Final grouping scheme:")
        for group_name, (abbreviation, residues) in grouping_scheme.items():
            print(f"   Group {group_name} ({abbreviation}): {', '.join(residues)}")

    # We generate an index to group mapping for reference and selection chains
    def generate_residue_to_group_map(seq, grouping_scheme):
        residue_to_group = {}
        for group_name, (abbreviation, residues) in grouping_scheme.items():
            for res in residues:
                residue_to_group[res] = (group_name, abbreviation)
        seq_group_map = [residue_to_group[res] for res in seq]
        return seq_group_map

    reference_residue_to_group = generate_residue_to_group_map(
        seq_reference, grouping_scheme
    )
    selection_residue_to_group = generate_residue_to_group_map(
        seq_selection, grouping_scheme
    )

    # We now generate contact numbers between each type of residues
    contact_counts = {}
    contact_pair_counts = {}
    for i in range(contact_map.shape[0]):
        ref_group_name, ref_abbr = reference_residue_to_group[i]
        for j in range(contact_map.shape[1]):
            sel_group_name, sel_abbr = selection_residue_to_group[j]
            key = (ref_abbr, sel_abbr)
            if key not in contact_counts:
                contact_counts[key] = 0
                contact_pair_counts[key] = 0
            contact_counts[key] += contact_map[i, j]
            contact_pair_counts[key] += 1

    if args.aggregate == "average":
        for key in contact_counts:
            contact_counts[key] /= contact_pair_counts[key]
        print("## Aggregation method: average.")
    else:
        print("## Aggregation method: sum.")

    # We generate contact matrix
    residue_type_abbr = sorted(set([v[0] for v in grouping_scheme.values()]))
    contact_matrix = np.zeros(
        (len(residue_type_abbr), len(residue_type_abbr)), dtype=float
    )
    abbr_to_index = {abbr: idx for idx, abbr in enumerate(residue_type_abbr)}
    for (ref_abbr, sel_abbr), count in contact_counts.items():
        i = abbr_to_index[ref_abbr]
        j = abbr_to_index[sel_abbr]
        contact_matrix[i, j] = count

    # We now write output to excel file with index
    try:
        import pandas as pd

        output_path = validate_extension(args.output, "xlsx")
        df = pd.DataFrame(
            contact_matrix,
            index=[
                f"{reference_group_name}_{restype}" for restype in residue_type_abbr
            ],
            columns=[
                f"{selection_group_name}_{restype}" for restype in residue_type_abbr
            ],
        )
        df.to_excel(output_path, float_format="%.6f")
        print(f"## Contact statistic written to {output_path}.")
    except Exception as e:
        output_path = validate_extension(args.output, "xlsx")
        raise RuntimeError(f"Failed to write output to {output_path}: {e}")


contact_statistic_commands = single_command(
    "cstat", getargs_contact_statistic, contact_statistic, desc
)
