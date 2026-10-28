# dps modifyres

[All commands](../command-reference.md) · [User guide](../user-guide.md)

Apply residue modifications consistently to an ITP topology and PDB structure.

DROPPS 1.0.0. This page is generated from the executable CLI; defaults, choices, required inputs, and output options below match the installed command.

## Usage and options

```text
usage: dps modifyres [-h] [-db] -p FILE -f FILE -op FILE -o FILE
                     [-ff {CALVADOS2,HPS,HPSRNA,HPST,MPiPi,MPiPi_PTM}] -m
                     MODIFICATIONS [MODIFICATIONS ...]

Apply residue modifications consistently to an ITP topology and PDB structure.

Input:
  -p FILE, --input-topology FILE
                        Input topology file (.itp). (required)
  -f FILE, --input-structure FILE
                        Input structure file (.pdb). (required)

Output:
  -op FILE, --output-topology FILE
                        Output modified topology file (.itp); the extension is
                        added if omitted. (required)
  -o FILE, --output-structure FILE
                        Output modified structure file (.pdb); the extension
                        is added if omitted. (required)

Parameters:
  -h, --help            Show this help message and exit.
  -db, --debug          Show runtime context, parsed arguments, timing, and
                        tracebacks. (default: False)
  -ff {CALVADOS2,HPS,HPSRNA,HPST,MPiPi,MPiPi_PTM}, --forcefield {CALVADOS2,HPS,HPSRNA,HPST,MPiPi,MPiPi_PTM}
                        Force field to use; if omitted, prompt interactively.
                        (default: None)
  -m MODIFICATIONS [MODIFICATIONS ...], --modifications MODIFICATIONS [MODIFICATIONS ...]
                        Residue modifications in ORIGINAL+NUMBER+MODIFIED
                        form, for example S129SMP. (required)

Notes:
  Read a matched PDB/ITP pair and write both modified files. Select a force field that defines the requested modification, for example S129SMP.

  The current modification implementation addresses 1-based bead positions and cannot modify terminal residues. Regenerate the system TOP/TPR after changing topology.

Examples:
  dps modifyres -p chain.itp -f chain.pdb -op modified.itp -o modified.pdb -m S129SMP
```
