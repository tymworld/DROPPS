# dps check

[All commands](../command-reference.md) · [User guide](../user-guide.md)

Inspect DROPPS run, trajectory, and index files before analysis.

DROPPS 1.0.0. This page is generated from the executable CLI; defaults, choices, required inputs, and output options below match the installed command.

## Usage and options

```text
usage: dps check [-h] [-db] -s FILE [-f FILE] [-n FILE]

Inspect DROPPS run, trajectory, and index files before analysis.

Input:
  -s FILE, --run-input FILE
                        Input DROPPS run file (.tpr) containing the system and
                        simulation settings. (required)
  -f FILE, --trajectory FILE
                        Optional input trajectory file (.xtc). (default: None)
  -n FILE, --index FILE
                        Optional index file (.ndx) defining additional atom
                        groups. (default: None)

Parameters:
  -h, --help            Show this help message and exit.
  -db, --debug          Show runtime context, parsed arguments, timing, and
                        tracebacks. (default: False)

Notes:
  Inspect TPR, XTC, and optional NDX to confirm atom counts, topology, frame times, box geometry, and available groups before analysis.

  TPR and EDR are DROPPS formats; files produced by GROMACS with the same extensions are not interchangeable.

Examples:
  dps check -s run.tpr -f run.xtc -n index.ndx
```
