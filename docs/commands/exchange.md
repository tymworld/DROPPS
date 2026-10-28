# dps exchange

[All commands](../command-reference.md) · [User guide](../user-guide.md)

Analyze molecular exchange and phase residence times across a planar slab interface.

DROPPS 1.0.0. This page is generated from the executable CLI; defaults, choices, required inputs, and output options below match the installed command.

## Usage and options

```text
usage: dps exchange [-h] [-db] -s FILE -f FILE [-n FILE] -o PREFIX
                    [-r REFERENCE_GROUP]
                    [-sg SELECTION_GROUPS [SELECTION_GROUPS ...]] [-a {x,y,z}]
                    [-cm {frame,block,none}] [-bw BIN_WIDTH]
                    [--dense-phase-threshold DENSE_PHASE_THRESHOLD]
                    [--interface-threshold INTERFACE_THRESHOLD]
                    [-b START_TIME] [-e END_TIME] [-dt DELTA_TIME]
                    [-bl BLOCKS] [--no-states]

Analyze molecular exchange and phase residence times across a planar slab interface.

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
  -o PREFIX, --output-prefix PREFIX
                        Output prefix for .profile.xvg, .events.csv,
                        .residence.csv, .survival.csv, .summary.csv, and
                        optional .states.csv files. (required)

Parameters:
  -h, --help            Show this help message and exit.
  -db, --debug          Show runtime context, parsed arguments, timing, and
                        tracebacks. (default: False)
  -r REFERENCE_GROUP, --reference-group REFERENCE_GROUP
                        Index group used to locate and fit the dense slab.
                        (default: None)
  -sg SELECTION_GROUPS [SELECTION_GROUPS ...], --selection-groups SELECTION_GROUPS [SELECTION_GROUPS ...]
                        Index groups whose molecules are tracked; if omitted,
                        use the reference group. (default: None)
  -a {x,y,z}, --axis {x,y,z}
                        Interface-normal axis. (default: z)
  -cm {frame,block,none}, --center-mode {frame,block,none}
                        Slab centering strategy shared with coexistence: frame
                        aligns every analyzed frame; block uses one center per
                        contiguous --blocks segment; none expects an already
                        centered slab. (default: frame)
  -bw BIN_WIDTH, --bin-width BIN_WIDTH
                        Requested reference density-bin width, in nm; shared
                        with density. (default: 0.05)
  --dense-phase-threshold DENSE_PHASE_THRESHOLD
                        Dense-phase threshold as a fraction of the maximum
                        reference density; shared with density and
                        coexistence. (default: 0.5)
  --interface-threshold INTERFACE_THRESHOLD
                        Normalized dilute-side threshold defining the
                        interface state; 0.1 uses the fitted 10--90% region.
                        Must be between 0 and 0.5. (default: 0.1)
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
                        Number of fit/statistics blocks and block-centering
                        groups. (default: 5)
  --no-states           Skip the potentially large per-frame .states.csv
                        output file. (default: False)

Notes:
  Classify molecular centers as dense/interface/dilute using fitted slab boundaries, then calculate phase transitions, survival, and residence statistics.

  The output prefix produces the population/transition and residence data plus the reference profile. Time windows are in ns; trajectories need at least two frames and adequate temporal resolution for crossings.

Examples:
  dps exchange -s run.tpr -f run.xtc -o exchange --axis z
```
