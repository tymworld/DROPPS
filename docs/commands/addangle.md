# dps addangle

[All commands](../command-reference.md) · [User guide](../user-guide.md)

Add angle restraints to an ITP topology.

DROPPS 1.0.0. This page is generated from the executable CLI; defaults, choices, required inputs, and output options below match the installed command.

## Usage and options

```text
usage: dps addangle [-h] [-db] -p FILE -op FILE -a FILE

Add angle restraints to an ITP topology.

Input:
  -p FILE, --input-topology FILE
                        Input topology file (.itp). (required)
  -a FILE, --angle-list FILE
                        Input text file with one 'BEAD_ID ANGLE_DEG
                        FORCE_CONSTANT' record per line; bead IDs are 1-based.
                        (required)

Output:
  -op FILE, --output-topology FILE
                        Output topology file (.itp) with angle restraints; the
                        extension is added if omitted. (required)

Parameters:
  -h, --help            Show this help message and exit.
  -db, --debug          Show runtime context, parsed arguments, timing, and
                        tracebacks. (default: False)

Notes:
  Each angle-list row is BEAD_ID ANGLE_DEG FORCE_CONSTANT. BEAD_ID is the 1-based central bead of a consecutive three-bead angle.

  Angles are supplied in degrees; force constants use kJ mol^-1 rad^-2. This command edits the ITP; 'angle' measures angles in a trajectory.

Examples:
  dps addangle -p chain.itp -op chain-angles.itp -a angles.dat
```
