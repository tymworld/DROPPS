# dps grompp

[All commands](../command-reference.md) · [User guide](../user-guide.md)

Build a DROPPS run-input file from structure, topology, and simulation parameters.

DROPPS 1.0.0. This page is generated from the executable CLI; defaults, choices, required inputs, and output options below match the installed command.

## Usage and options

```text
usage: dps grompp [-h] [-db] -f FILE -p FILE -m FILE -o FILE

Build a DROPPS run-input file from structure, topology, and simulation parameters.

Input:
  -f FILE, --structure FILE
                        Input structure file (.pdb) containing the initial
                        coordinates and box. (required)
  -p FILE, --topology FILE
                        Input system topology file (.top) referencing the
                        required ITP files. (required)
  -m FILE, --parameter FILE
                        Input molecular-dynamics parameter file (.mdp).
                        (required)

Output:
  -o FILE, --output FILE
                        Output DROPPS run-input file (.tpr); the extension is
                        added if omitted. (required)

Parameters:
  -h, --help            Show this help message and exit.
  -db, --debug          Show runtime context, parsed arguments, timing, and
                        tracebacks. (default: False)

Notes:
  Compile a matching coarse-grained PDB, system TOP, referenced ITP files, and MDP settings into a checksummed portable TPR v2.

  TOP sections are [ itp files ] and [ system ]; the latter lists MOLECULE_NAME COUNT in coordinate order. ITP paths are resolved relative to the TOP.

  Box lengths and cutoffs use nm. Each nonbonded cutoff must be strictly less than half the shortest box length. Use the bundled md.mdp template and MDP reference for supported settings.

Examples:
  dps grompp -f system.pdb -p system.top -m md.mdp -o run.tpr
```
