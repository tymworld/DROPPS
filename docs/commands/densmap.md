# dps densmap

[All commands](../command-reference.md) · [User guide](../user-guide.md)

Calculate planar two-dimensional mass, number, or charge density maps.

DROPPS 1.0.0. This page is generated from the executable CLI; defaults, choices, required inputs, and output options below match the installed command.

## Usage and options

```text
usage: dps densmap [-h] [-db] -s FILE -f FILE [-n FILE] -o FILE [-ox FILE]
                   [-aa {x,y,z}] [-t {mass,number,charge}]
                   [-nm {volume,area,count,nm-3,nm-2}] [-bw BIN_WIDTH]
                   [-n1 BINS_FIRST] [-n2 BINS_SECOND]
                   [--average-min AVERAGE_MIN] [--average-max AVERAGE_MAX]
                   [-b START_TIME] [-e END_TIME] [-dt DELTA_TIME]
                   [-se SELECTION [SELECTION ...]]
                   [--selection-fit SELECTION_FIT] [-cm {frame,block,none}]
                   [-bl BLOCKS]

Calculate planar two-dimensional mass, number, or charge density maps.

Input:
  -s FILE, --run-input FILE
                        Input DROPPS run file (.tpr) containing topology
                        masses and charges. (required)
  -f FILE, --input FILE
                        Input trajectory file (.xtc). (required)
  -n FILE, --index FILE
                        Optional index file (.ndx) defining additional atom
                        groups. (default: None)

Output:
  -o FILE, --output FILE
                        Output GROMACS-style text matrix (.dat). With multiple
                        -se groups, the group number and name are appended to
                        this path. (required)
  -ox FILE, --xpm-output FILE
                        Optional grayscale XPM map. With multiple -se groups,
                        the group number and name are appended to this path.
                        (default: None)

Parameters:
  -h, --help            Show this help message and exit.
  -db, --debug          Show runtime context, parsed arguments, timing, and
                        tracebacks. (default: False)
  -aa {x,y,z}, --average-axis {x,y,z}
                        Direction to average over; z produces an x-y map.
                        (default: z)
  -t {mass,number,charge}, --type {mass,number,charge}
                        Particle property accumulated in each grid cell.
                        (default: number)
  -nm {volume,area,count,nm-3,nm-2}, --normalization {volume,area,count,nm-3,nm-2}
                        volume/nm-3 gives a true 3D density, area/nm-2 gives a
                        projected surface density, and count gives mean weight
                        per cell. (default: volume)
  -bw BIN_WIDTH, --bin-width BIN_WIDTH
                        Requested grid spacing in both retained directions, in
                        nm. (default: 0.02)
  -n1 BINS_FIRST, --bins-first BINS_FIRST
                        Grid cells along the first retained axis; zero derives
                        it from -bw. (default: 0)
  -n2 BINS_SECOND, --bins-second BINS_SECOND
                        Grid cells along the second retained axis; zero
                        derives it from -bw. (default: 0)
  --average-min AVERAGE_MIN
                        Optional lower coordinate included along the averaging
                        axis, in nm. (default: None)
  --average-max AVERAGE_MAX
                        Optional upper coordinate excluded along the averaging
                        axis, in nm. (default: None)
  -b START_TIME, --start-time START_TIME
                        First trajectory time to analyze, in ns. (default:
                        None)
  -e END_TIME, --end-time END_TIME
                        Last trajectory time to analyze, in ns. (default:
                        None)
  -dt DELTA_TIME, --delta-time DELTA_TIME
                        Approximate interval between analyzed frames, in ns.
                        (default: None)
  -se SELECTION [SELECTION ...], --selection SELECTION [SELECTION ...]
                        Index groups to map; if omitted, prompt for one or
                        more groups interactively. (default: None)
  --selection-fit SELECTION_FIT
                        Index group used for optional planar centering. If
                        omitted, the first mapped group is used. (default:
                        None)
  -cm {frame,block,none}, --center-mode {frame,block,none}
                        Periodic planar alignment: frame follows a compact
                        droplet each frame, block uses one center per
                        contiguous block, and none keeps the trajectory
                        coordinates. (default: none)
  -bl BLOCKS, --blocks BLOCKS
                        Number of contiguous groups used by --center-mode
                        block. (default: 5)

Notes:
  Average along one axis and histogram the remaining plane into a two-dimensional density map. Choose number, mass, or charge and one or more selection groups.

  The text outputs include grid coordinates, dimensions, units, and normalization. PNG output requires the plot extra. Centering can track a moving droplet before averaging.

Examples:
  dps densmap -s run.tpr -f run.xtc -o densmap.dat --selection 0 --average-axis z --type number
  dps densmap -s run.tpr -f run.xtc -o droplet.dat --selection 1 2 --type mass --center-mode frame --selection-fit 1
```
