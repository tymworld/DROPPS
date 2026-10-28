# dps gyrate

[All commands](../command-reference.md) · [User guide](../user-guide.md)

Calculate chain radii of gyration, including optional dense/dilute resolution.

DROPPS 1.0.0. This page is generated from the executable CLI; defaults, choices, required inputs, and output options below match the installed command.

## Usage and options

```text
usage: dps gyrate [-h] [-db] -s FILE -f FILE [-n FILE]
                  [-sc SELECTION_CALCULATE [SELECTION_CALCULATE ...]]
                  [-b START_TIME] [-e END_TIME] [-dt DELTA_TIME] [-p]
                  [-ov FILE] [-oa FILE] [-oh FILE] [-op FILE] [-bw BIN_WIDTH]
                  [-r REFERENCE_GROUP] [-a {x,y,z}] [-cm {frame,block,none}]
                  [--phases {dense,interface,dilute} [{dense,interface,dilute} ...]]
                  [--density-bin-width DENSITY_BIN_WIDTH]
                  [--dense-phase-threshold DENSE_PHASE_THRESHOLD]
                  [--interface-threshold INTERFACE_THRESHOLD] [-bl BLOCKS]
                  [--no-phase-observations]

Calculate chain radii of gyration, including optional dense/dilute resolution.

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
  -ov FILE, --output-verbose FILE
                        Output per-selected-group radius-of-gyration time
                        series (.xvg); the extension is added if omitted.
                        (default: None)
  -oa FILE, --output-average FILE
                        Output the mean radius of gyration across selected
                        groups (.xvg); the extension is added if omitted.
                        (default: None)
  -oh FILE, --output-histogram FILE
                        Output the pooled selected-group/frame radius-of-
                        gyration distribution (.xvg); the extension is added
                        if omitted. (default: None)
  -op FILE, --output-phase FILE
                        Output phase-resolved mean Rg file (.xvg); companion
                        observation, summary, histogram, and reference-profile
                        files are also written. (default: None)

Parameters:
  -h, --help            Show this help message and exit.
  -db, --debug          Show runtime context, parsed arguments, timing, and
                        tracebacks. (default: False)
  -sc SELECTION_CALCULATE [SELECTION_CALCULATE ...], --selection-calculate SELECTION_CALCULATE [SELECTION_CALCULATE ...]
                        Index groups whose radii of gyration are calculated;
                        if omitted, prompt interactively. Phase-resolved mode
                        splits each group into molecules. (default: None)
  -b START_TIME, --start-time START_TIME
                        First trajectory time to analyze, in ns. (default:
                        None)
  -e END_TIME, --end-time END_TIME
                        Last trajectory time to analyze, in ns. (default:
                        None)
  -dt DELTA_TIME, --delta-time DELTA_TIME
                        Approximate interval between analyzed frames, in ns.
                        (default: None)
  -p, --treat-pbc       Unwrap molecules across periodic boundaries before
                        legacy group analysis. (default: False)
  -bw BIN_WIDTH, --bin-width BIN_WIDTH
                        Histogram bin width, in nm. (default: 0.1)
  -r REFERENCE_GROUP, --reference-group REFERENCE_GROUP
                        Index group used to locate and fit the dense slab in
                        phase-resolved mode. (default: None)
  -a {x,y,z}, --axis {x,y,z}
                        Interface-normal axis in phase-resolved mode.
                        (default: z)
  -cm {frame,block,none}, --center-mode {frame,block,none}
                        Slab centering used for phase classification: frame,
                        contiguous block, or none. (default: frame)
  --phases {dense,interface,dilute} [{dense,interface,dilute} ...]
                        Phase regions included in phase-resolved means,
                        summaries, and PDFs. (default: ('dense', 'dilute'))
  --density-bin-width DENSITY_BIN_WIDTH
                        Requested reference density-bin width in phase-
                        resolved mode, in nm. (default: 0.05)
  --dense-phase-threshold DENSE_PHASE_THRESHOLD
                        Fraction of maximum reference density used to locate
                        the dense slab. (default: 0.5)
  --interface-threshold INTERFACE_THRESHOLD
                        Normalized dilute-side threshold defining the
                        interface state; 0.1 uses the fitted 10--90% region.
                        (default: 0.1)
  -bl BLOCKS, --blocks BLOCKS
                        Number of fit/statistics blocks and block-centering
                        groups. (default: 5)
  --no-phase-observations
                        Skip the potentially large per-frame, per-molecule
                        phase CSV. (default: False)

At least one output option (-ov, -oa, -oh, or -op) is required.

Notes:
  Measure mass-weighted radius of gyration in nm for selected groups, with time series, distributions, and per-chain details controlled by output options.

  --output-phase adds per-molecule dense/interface/dilute Rg, block summaries, occupancy observations, and slab-fit profiles. Use polystat for an explicit many-copy size/shape cohort analysis.

Examples:
  dps gyrate -s run.tpr -f run.xtc --output-average rg-average.xvg
  dps gyrate -s slab.tpr -f slab.xtc -n groups.ndx --output-phase phase-rg.xvg --reference-group 0 --selection-calculate 1 --axis z
```
