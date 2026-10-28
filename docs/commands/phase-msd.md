# dps phase-msd

[All commands](../command-reference.md) · [User guide](../user-guide.md)

Calculate phase-conditioned molecular MSD with phase-translation removal in a planar slab.

DROPPS 1.0.0. This page is generated from the executable CLI; defaults, choices, required inputs, and output options below match the installed command.

## Usage and options

```text
usage: dps phase-msd [-h] [-db] -s FILE -f FILE [-n FILE] -o FILE
                     [-r REFERENCE_GROUP]
                     [-sg SELECTION_GROUPS [SELECTION_GROUPS ...]]
                     [-a {x,y,z}] [-cm {frame,block,none}]
                     [-t {xyz,xy,yz,xz,x,y,z}]
                     [--phases {dense,interface,dilute} [{dense,interface,dilute} ...]]
                     [--conditioning {continuous,intermittent,both}]
                     [-ml MAX_LAG] [--min-particles MIN_PARTICLES]
                     [-bw BIN_WIDTH]
                     [--dense-phase-threshold DENSE_PHASE_THRESHOLD]
                     [--interface-threshold INTERFACE_THRESHOLD]
                     [-b START_TIME] [-e END_TIME] [-dt DELTA_TIME]
                     [-bl BLOCKS]

Calculate phase-conditioned molecular MSD with phase-translation removal in a planar slab.

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
                        Output MSD file (.xvg); matching .csv and .profile.xvg
                        files are also written. (required)

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
                        Centering used for phase classification: frame,
                        contiguous block, or none. Dynamics are always
                        calculated from unwrapped raw molecular positions.
                        (default: frame)
  -t {xyz,xy,yz,xz,x,y,z}, --msd-type {xyz,xy,yz,xz,x,y,z}
                        Cartesian dimensions included in the MSD. (default:
                        xyz)
  --phases {dense,interface,dilute} [{dense,interface,dilute} ...]
                        Phase regions to analyze. (default: ('dense',
                        'dilute'))
  --conditioning {continuous,intermittent,both}
                        continuous requires phase membership throughout each
                        interval; intermittent requires it only at both
                        endpoints. (default: both)
  -ml MAX_LAG, --max-lag MAX_LAG
                        Maximum lag time, in ns; if omitted, use half the
                        analyzed trajectory. (default: None)
  --min-particles MIN_PARTICLES
                        Minimum eligible molecules at one time origin for
                        phase-translation removal. (default: 2)
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
                        Number of contiguous time-origin blocks for standard
                        errors. (default: 5)

Notes:
  Calculate molecular displacement by slab phase and residence-conditioning rule. Select which molecule groups to track and how to remove phase translation.

  Output includes lag-time MSD curves in nm^2, detailed statistics, and the fitted reference density profile. Uniformly spaced frames are required for lag-based analysis.

Examples:
  dps phase-msd -s run.tpr -f run.xtc -o phase-msd.xvg --axis z
```
