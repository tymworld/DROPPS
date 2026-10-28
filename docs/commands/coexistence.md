# dps coexistence

[All commands](../command-reference.md) · [User guide](../user-guide.md)

Analyze dense/dilute coexistence and interface width in a planar slab trajectory.

DROPPS 1.0.0. This page is generated from the executable CLI; defaults, choices, required inputs, and output options below match the installed command.

## Usage and options

```text
usage: dps coexistence [-h] [-db] -s FILE -f FILE [-n FILE] -o FILE [-ou FILE]
                       [-r REFERENCE_GROUP]
                       [-sg SELECTION_GROUPS [SELECTION_GROUPS ...]]
                       [-a {x,y,z}] [-cm {frame,block,none}]
                       [-tp {mass,number}] [-bw BIN_WIDTH]
                       [-t DENSE_PHASE_THRESHOLD] [-b START_TIME]
                       [-e END_TIME] [-dt DELTA_TIME] [-bl BLOCKS]
                       [--plateau-buffer PLATEAU_BUFFER]

Analyze dense/dilute coexistence and interface width in a planar slab trajectory.

Input:
  -s FILE, --run-input FILE
                        Input DROPPS run file (.tpr) containing the system and
                        simulation settings. (required)
  -f FILE, --input FILE
                        Input planar-slab trajectory file (.xtc). (required)
  -n FILE, --index FILE
                        Optional index file (.ndx) defining additional atom
                        groups. (default: None)

Output:
  -o FILE, --output FILE
                        Output density-profile file (.xvg); the extension is
                        added if omitted. (required)
  -ou FILE, --summary-output FILE
                        Optional output summary file (.csv); if omitted, use
                        the profile path with a .csv suffix. (default: None)

Parameters:
  -h, --help            Show this help message and exit.
  -db, --debug          Show runtime context, parsed arguments, timing, and
                        tracebacks. (default: False)
  -r REFERENCE_GROUP, --reference-group REFERENCE_GROUP
                        Index group used to locate and fit the dense slab.
                        (default: None)
  -sg SELECTION_GROUPS [SELECTION_GROUPS ...], --selection-groups SELECTION_GROUPS [SELECTION_GROUPS ...]
                        Additional index groups for dense/dilute plateau
                        densities. (default: None)
  -a {x,y,z}, --axis {x,y,z}
                        Interface-normal axis. (default: z)
  -cm {frame,block,none}, --center-mode {frame,block,none}
                        Slab centering strategy: frame aligns every analyzed
                        frame; block uses one center per contiguous --blocks
                        segment; none applies no shift and expects an already
                        box-centered slab. (default: frame)
  -tp {mass,number}, --type {mass,number}
                        Mass density (mg/mL) or selected-particle number
                        density (nm^-3). (default: mass)
  -bw BIN_WIDTH, --bin-width BIN_WIDTH
                        Requested density-bin width, in nm. The default
                        matches the density command.
  -t DENSE_PHASE_THRESHOLD, --dense-phase-threshold DENSE_PHASE_THRESHOLD
                        Dense-phase threshold as a fraction of the maximum
                        reference density; shared with the density command.
                        (default: 0.5)
  -b START_TIME, --start-time START_TIME
                        First trajectory time to analyze, in ns. (default:
                        None)
  -e END_TIME, --end-time END_TIME
                        Last trajectory time to analyze, in ns. (default:
                        None)
  -dt DELTA_TIME, --delta-time DELTA_TIME
                        Approximate interval between analyzed frames, in ns.
                        (default: None)
  -bl BLOCKS, --blocks BLOCKS
                        Number of contiguous blocks used for standard errors.
                        (default: 5)
  --plateau-buffer PLATEAU_BUFFER
                        Distance excluded on each side of a fitted interface,
                        in units of the tanh interface parameter. (default:
                        2.0)

Notes:
  Fit a centered planar-slab mass-density profile with a double-tanh interface model to obtain dense/dilute densities and interface widths.

  Use a reference group that identifies the slab, inspect the profile and fit, and choose contiguous blocks for uncertainty estimation. This geometry assumes a planar slab, not a spherical droplet.

Examples:
  dps coexistence -s run.tpr -f run.xtc -o coexistence.xvg --axis z
```
