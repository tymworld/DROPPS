"""Shared command-line arguments for residue-contact analyses."""


def add_contact_arguments(parser):
    parser.add_argument(
        "-s",
        "--run-input",
        required=True,
        help="Input DROPPS run file (.tpr) containing the system and simulation settings.",
    )
    parser.add_argument(
        "-f",
        "--input",
        required=True,
        help="Input trajectory file (.xtc).",
    )
    parser.add_argument(
        "-n",
        "--index",
        help="Optional index file (.ndx) defining additional atom groups.",
    )
    parser.add_argument(
        "-ref",
        "--reference-group",
        help="Reference index group forming the x axis of each contact map.",
    )
    parser.add_argument(
        "-sel",
        "--selection-group",
        help="Optional selection index group forming the y axis of cross-group maps.",
    )
    parser.add_argument(
        "-cs",
        "--cutoff-scheme",
        choices=("global", "residue"),
        required=True,
        help="Contact-cutoff scheme: one global distance or residue-specific distances.",
    )
    parser.add_argument(
        "-c",
        "--cutoff",
        type=float,
        default=0.7,
        help="Global contact cutoff, in nm.",
    )
    parser.add_argument(
        "-cm",
        "--cutoff-multiplier",
        type=float,
        default=1.2,
        help="Multiplier applied to residue sigma values in the residue cutoff scheme.",
    )
    parser.add_argument(
        "-rd",
        "--remove-diagonal",
        type=int,
        default=2,
        help="Number of main and neighboring diagonals removed from intra-chain maps.",
    )
    parser.add_argument(
        "-b",
        "--start-time",
        type=int,
        help="First trajectory time to analyze, in ns.",
    )
    parser.add_argument(
        "-e",
        "--end-time",
        type=int,
        help="Last trajectory time to analyze, in ns.",
    )
    parser.add_argument(
        "-dt",
        "--delta-time",
        type=int,
        help="Approximate interval between analyzed frames, in ns.",
    )
    parser.add_argument(
        "-pbc",
        "--treat-pbc",
        action="store_true",
        default=False,
        help="Apply periodic-boundary distances during contact calculation.",
    )

    map_outputs = (
        (
            "-ors",
            "--output-inter-reference-selection",
            "inter-chain reference-selection",
        ),
        (
            "-orr",
            "--output-inter-reference-reference",
            "inter-chain reference-reference",
        ),
        (
            "-oss",
            "--output-inter-selection-selection",
            "inter-chain selection-selection",
        ),
        ("-or", "--output-intra-reference", "intra-chain reference"),
        ("-os", "--output-intra-selection", "intra-chain selection"),
    )
    for short_option, long_option, label in map_outputs:
        parser.add_argument(
            short_option,
            long_option,
            help=(
                f"Output {label} contact map; the --output-type extension is "
                "added if omitted."
            ),
        )

    time_outputs = (
        (
            "-otrs",
            "--output-time-inter-reference-selection",
            "inter-chain reference-selection",
        ),
        (
            "-otrr",
            "--output-time-inter-reference-reference",
            "inter-chain reference-reference",
        ),
        (
            "-otss",
            "--output-time-inter-selection-selection",
            "inter-chain selection-selection",
        ),
        ("-otr", "--output-time-intra-reference", "intra-chain reference"),
        ("-ots", "--output-time-intra-selection", "intra-chain selection"),
    )
    for short_option, long_option, label in time_outputs:
        parser.add_argument(
            short_option,
            long_option,
            help=(
                f"Output {label} contact-number time series (.xvg); the "
                "extension is added if omitted."
            ),
        )

    parser.add_argument(
        "-intraavg",
        "--intra-average",
        action="store_true",
        default=False,
        help="Average intra-chain contact maps instead of summing them.",
    )
    parser.add_argument(
        "-otype",
        "--output-type",
        choices=("dat", "xpm", "xlsx"),
        default="dat",
        help="File format for contact-map outputs.",
    )
