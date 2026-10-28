# dps density

[All commands](../command-reference.md) · [User guide](../user-guide.md)

Calculate a one-dimensional mass- or charge-density profile.

DROPPS 1.0.0. This page is generated from the executable CLI; defaults, choices, required inputs, and output options below match the installed command.

## Usage and options

```text
usage: dps density [-h] [-db] -s FILE -f FILE [-n FILE] -o FILE [-a {x,y,z}]
                   [-tp {mass,charge}] [-b START_TIME] [-e END_TIME]
                   [-dt DELTA_TIME] [-bw BIN_WIDTH]
                   [--selection-fit SELECTION_FIT]
                   [-sc SELECTION_CALCULATE [SELECTION_CALCULATE ...]]
                   [-cm {frame,block,none}] [-bl BLOCKS] [-nc]
                   [-t DENSE_PHASE_THRESHOLD]

Calculate a one-dimensional mass- or charge-density profile.

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
  -o FILE, --output FILE
                        Output density-profile file (.xvg); the extension is
                        added if omitted. (required)

Parameters:
  -h, --help            Show this help message and exit.
  -db, --debug          Show runtime context, parsed arguments, timing, and
                        tracebacks. (default: False)
  -a {x,y,z}, --axis {x,y,z}
                        Axis along which to calculate the density profile.
                        (default: z)
  -tp {mass,charge}, --type {mass,charge}
                        Density quantity to calculate. (default: mass)
  -b START_TIME, --start-time START_TIME
                        First trajectory time to analyze, in ns. (default:
                        None)
  -e END_TIME, --end-time END_TIME
                        Last trajectory time to analyze, in ns. (default:
                        None)
  -dt DELTA_TIME, --delta-time DELTA_TIME
                        Approximate interval between analyzed frames, in ns.
                        (default: None)
  -bw BIN_WIDTH, --bin-width BIN_WIDTH
                        Requested density-bin width, in nm. (default: 0.05)
  --selection-fit SELECTION_FIT
                        Index group used to locate the dense phase; if
                        omitted, prompt interactively. (default: None)
  -sc SELECTION_CALCULATE [SELECTION_CALCULATE ...], --selection-calculate SELECTION_CALCULATE [SELECTION_CALCULATE ...]
                        Index groups whose density profiles are written; if
                        omitted, prompt interactively. (default: None)
  -cm {frame,block,none}, --center-mode {frame,block,none}
                        Slab centering strategy shared with coexistence: frame
                        aligns each analyzed frame; block uses one center per
                        contiguous --blocks segment; none applies no shift.
                        (default: frame)
  -bl BLOCKS, --blocks BLOCKS
                        Number of contiguous groups used by --center-mode
                        block. (default: 5)
  -nc, --no-center      Compatibility alias for --center-mode none. (default:
                        False)
  -t DENSE_PHASE_THRESHOLD, --dense-phase-threshold DENSE_PHASE_THRESHOLD
                        Dense-phase threshold as a fraction of the maximum
                        reference density. (default: 0.5)

Notes:
  Calculate one-dimensional number, mass, or charge profiles along the selected axis. Supply --selection-fit and --selection-calculate to avoid interactive selection.

  Per-frame box lengths and volumes determine density normalization. Frame/block/none centering controls alignment of a drifting slab; the output header specifies coordinates and density units.

Examples:
  dps density -s run.tpr -f run.xtc -o density.xvg -a z
```
