# dps genelastic

[All commands](../command-reference.md) · [User guide](../user-guide.md)

Add a distance-based elastic network to an ITP topology.

DROPPS 1.0.0. This page is generated from the executable CLI; defaults, choices, required inputs, and output options below match the installed command.

## Usage and options

```text
usage: dps genelastic [-h] [-db] -f FILE -p FILE -o FILE -r FILE
                      [-ef ELASTIC_FORCE_CONSTANT] [-el ELASTIC_LOWER]
                      [-eu ELASTIC_UPPER]

Add a distance-based elastic network to an ITP topology.

Input:
  -f FILE, --structure FILE
                        Input reference structure file (.pdb). (required)
  -p FILE, --topology FILE
                        Input topology file (.itp). (required)
  -r FILE, --elastic-residues FILE
                        Input text file listing one elastic-network bead group
                        per line, using 1-based bead indices. (required)

Output:
  -o FILE, --output FILE
                        Output topology file (.itp); the extension is added if
                        omitted. (required)

Parameters:
  -h, --help            Show this help message and exit.
  -db, --debug          Show runtime context, parsed arguments, timing, and
                        tracebacks. (default: False)
  -ef ELASTIC_FORCE_CONSTANT, --elastic-force-constant ELASTIC_FORCE_CONSTANT
                        Elastic-bond force constant, in kJ mol^-1 nm^-2.
                        (default: 5000)
  -el ELASTIC_LOWER, --elastic-lower ELASTIC_LOWER
                        Lower distance cutoff for elastic bonds, in nm.
                        (default: 0.5)
  -eu ELASTIC_UPPER, --elastic-upper ELASTIC_UPPER
                        Upper distance cutoff for elastic bonds, in nm.
                        (default: 0.9)

Notes:
  The elastic-residue file contains one whitespace-separated group of 1-based bead indices per line. Bonds are added between eligible bead pairs within each group.

  Distances and cutoffs are in nm; the elastic force constant is in kJ mol^-1 nm^-2. The output is an ITP with the added bonds; retain the matching PDB.

Examples:
  dps genelastic -f chain.pdb -p chain.itp -o elastic.itp -r elastic-groups.dat
```
