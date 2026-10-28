# dps msd

[All commands](../command-reference.md) · [User guide](../user-guide.md)

Calculate the mean-square displacement of a selected atom group.

DROPPS 1.0.0. This page is generated from the executable CLI; defaults, choices, required inputs, and output options below match the installed command.

## Usage and options

```text
usage: dps msd [-h] [-db] -s FILE -f FILE [-n FILE] [-se SELECTION]
               [-t {xyz,xy,yz,xz,x,y,z}] [-b START_TIME] [-e END_TIME]
               [-dt DELTA_TIME] -o FILE

Calculate the mean-square displacement of a selected atom group.

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
                        Output mean-square-displacement time series (.xvg);
                        the extension is added if omitted. (required)

Parameters:
  -h, --help            Show this help message and exit.
  -db, --debug          Show runtime context, parsed arguments, timing, and
                        tracebacks. (default: False)
  -se SELECTION, --selection SELECTION
                        Index group whose mean-square displacement is
                        calculated; if omitted, prompt interactively.
                        (default: None)
  -t {xyz,xy,yz,xz,x,y,z}, --msd-type {xyz,xy,yz,xz,x,y,z}
                        Cartesian dimensions included in the MSD. (default:
                        xyz)
  -b START_TIME, --start-time START_TIME
                        First trajectory time to analyze, in ns. (default:
                        None)
  -e END_TIME, --end-time END_TIME
                        Last trajectory time to analyze, in ns. (default:
                        None)
  -dt DELTA_TIME, --delta-time DELTA_TIME
                        Approximate interval between analyzed frames, in ns.
                        (default: None)

Notes:
  Compute the time-origin-averaged atomic mean-square displacement in the requested Cartesian directions. Output is lag time in ns versus MSD in nm^2.

  All saved frames are unwrapped before applying time-window/stride selection. Frames must be uniformly sampled, and motion between saved frames must be small enough to determine the periodic image.

  This is atomic MSD without center-of-mass drift subtraction. Use phase-msd for molecular motion conditioned on slab phase.

Examples:
  dps msd -s run.tpr -f run.xtc -o msd.xvg -t xyz
```
