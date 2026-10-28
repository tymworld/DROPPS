# dps pdb2bond

[All commands](../command-reference.md) · [User guide](../user-guide.md)

Add PDB CONECT records from a DROPPS run file or system topology.

DROPPS 1.0.0. This page is generated from the executable CLI; defaults, choices, required inputs, and output options below match the installed command.

## Usage and options

```text
usage: dps pdb2bond [-h] [-db] (-s FILE | -p FILE) -f FILE -o FILE
                    [--preserve-atom-types]

Add PDB CONECT records from a DROPPS run file or system topology.

Input:
  -s FILE, --run-input FILE
                        Input DROPPS run file (.tpr) containing system bonds.
                        (required: choose one option from this group)
  -p FILE, --topology FILE
                        Input system topology file (.top) containing system
                        bonds. (required: choose one option from this group)
  -f FILE, --input FILE
                        Input structure file (.pdb). (required)

Output:
  -o FILE, --output FILE
                        Output structure file (.pdb) containing CONECT
                        records. (required)

Parameters:
  -h, --help            Show this help message and exit.
  -db, --debug          Show runtime context, parsed arguments, timing, and
                        tracebacks. (default: False)
  --preserve-atom-types
                        Preserve input element fields instead of writing every
                        bead as carbon. (default: False)

Notes:
  Add PDB CONECT records from either a TPR or TOP/ITP topology. The input PDB bead order must match the supplied topology.

  Bond records improve structure display in molecular viewers. They do not change coordinates or simulation forces.

Examples:
  dps pdb2bond -s run.tpr -f frame.pdb -o bonded.pdb
```
