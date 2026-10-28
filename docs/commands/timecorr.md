# dps timecorr

[All commands](../command-reference.md) · [User guide](../user-guide.md)

Calculate residue-contact correlation functions and lifetime matrices.

DROPPS 1.0.0. This page is generated from the executable CLI; defaults, choices, required inputs, and output options below match the installed command.

## Usage and options

```text
usage: dps timecorr [-h] [-db] -s FILE -f FILE [-n FILE] [-r REFERENCE_GROUP]
                    [-sg SELECTION_GROUP] -cs {global,residue} [-c CUTOFF]
                    [-cm CUTOFF_MULTIPLIER] [-rd REMOVE_DIAGONAL]
                    [-t THRESHOLD] [-ml MAX_LAG] [-b START_TIME] [-e END_TIME]
                    [-dt DELTA_TIME] [-p] [-v] [-oz {dat,xlsx}] [-oa FILE]
                    [-ob FILE] [-oc FILE] [-oi FILE] [-oj FILE] [-oe FILE]
                    [-of FILE] [-og FILE] [-oh FILE] [-ol FILE]
                    [-oy {dat,xpm,xlsx}]

Calculate residue-contact correlation functions and lifetime matrices.

Input:
  -s FILE, --run-input FILE
                        Input DROPPS run file (.tpr) containing the system and
                        simulation settings. (required)
  -f FILE, --input FILE
                        Input trajectory file (.xtc). (required)
  -n FILE, --index FILE
                        Optional index file (.ndx) defining additional atom
                        groups. (default: None)

Output:
  -oz {dat,xlsx}, --average-output-type {dat,xlsx}
                        File format for non-verbose averaged correlation
                        matrices. (default: dat)
  -oa FILE, --output-corr-inter-reference-selection FILE
                        Output inter-chain reference-selection correlations;
                        verbose mode writes .npz, otherwise --average-output-
                        type is used. (default: None)
  -ob FILE, --output-corr-inter-reference-reference FILE
                        Output inter-chain reference-reference correlations;
                        verbose mode writes .npz, otherwise --average-output-
                        type is used. (default: None)
  -oc FILE, --output-corr-inter-selection-selection FILE
                        Output inter-chain selection-selection correlations;
                        verbose mode writes .npz, otherwise --average-output-
                        type is used. (default: None)
  -oi FILE, --output-corr-intra-reference FILE
                        Output intra-chain reference correlations; verbose
                        mode writes .npz, otherwise --average-output-type is
                        used. (default: None)
  -oj FILE, --output-corr-intra-selection FILE
                        Output intra-chain selection correlations; verbose
                        mode writes .npz, otherwise --average-output-type is
                        used. (default: None)
  -oe FILE, --output-lifetime-inter-reference-selection FILE
                        Output inter-chain reference-selection lifetime matrix
                        in verbose mode. (default: None)
  -of FILE, --output-lifetime-inter-reference-reference FILE
                        Output inter-chain reference-reference lifetime matrix
                        in verbose mode. (default: None)
  -og FILE, --output-lifetime-inter-selection-selection FILE
                        Output inter-chain selection-selection lifetime matrix
                        in verbose mode. (default: None)
  -oh FILE, --output-lifetime-intra-reference FILE
                        Output intra-chain reference lifetime matrix in
                        verbose mode. (default: None)
  -ol FILE, --output-lifetime-intra-selection FILE
                        Output intra-chain selection lifetime matrix in
                        verbose mode. (default: None)
  -oy {dat,xpm,xlsx}, --output-type {dat,xpm,xlsx}
                        File format for verbose lifetime-matrix outputs.
                        (default: dat)

Parameters:
  -h, --help            Show this help message and exit.
  -db, --debug          Show runtime context, parsed arguments, timing, and
                        tracebacks. (default: False)
  -r REFERENCE_GROUP, --reference-group REFERENCE_GROUP
                        Reference index group forming the x axis of each
                        residue-pair matrix. (default: None)
  -sg SELECTION_GROUP, --selection-group SELECTION_GROUP
                        Optional selection index group forming the y axis of
                        cross-group matrices. (default: None)
  -cs {global,residue}, --cutoff-scheme {global,residue}
                        Contact-cutoff scheme: one global distance or residue-
                        specific distances. (required)
  -c CUTOFF, --cutoff CUTOFF
                        Global contact cutoff, in nm. (default: 0.7)
  -cm CUTOFF_MULTIPLIER, --cutoff-multiplier CUTOFF_MULTIPLIER
                        Multiplier applied to residue sigma values in the
                        residue cutoff scheme. (default: 1.2)
  -rd REMOVE_DIAGONAL, --remove-diagonal REMOVE_DIAGONAL
                        Number of main and neighboring diagonals removed from
                        intra-chain matrices. (default: 2)
  -t THRESHOLD, --threshold THRESHOLD
                        Normalized-correlation threshold used to estimate
                        contact lifetimes. (default: 0.5)
  -ml MAX_LAG, --max-lag MAX_LAG
                        Maximum correlation lag, in ns; if omitted, use all
                        analyzed frames. (default: None)
  -b START_TIME, --start-time START_TIME
                        First trajectory time to analyze, in ns. (default:
                        None)
  -e END_TIME, --end-time END_TIME
                        Last trajectory time to analyze, in ns. (default:
                        None)
  -dt DELTA_TIME, --delta-time DELTA_TIME
                        Approximate interval between analyzed frames, in ns.
                        (default: None)
  -p, --treat-pbc       Apply periodic-boundary distances during contact
                        calculation. (default: False)
  -v, --verbose         Write lag-resolved correlation tensors and lifetime
                        matrices instead of only lag-averaged matrices.
                        (default: False)

At least one correlation output (-oa, -ob, -oc, -oi, or -oj) is required in non-verbose mode.

Notes:
  Calculate contact time correlations/lifetimes from a uniformly sampled trajectory with the chosen reference/selection groups and contact definition.

  Specify the required cutoff scheme and correlation output. The maximum lag and saved-frame interval determine the time range that can be resolved.

Examples:
  dps timecorr -s run.tpr -f run.xtc --reference-group 0 --selection-group 1 --cutoff-scheme global --output-corr-inter-reference-selection contact-correlation.dat
```
