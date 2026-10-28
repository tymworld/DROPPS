# dps genmesh

[All commands](../command-reference.md) · [User guide](../user-guide.md)

Pack one or more molecule types onto a three-dimensional simulation-box mesh.

DROPPS 1.0.0. This page is generated from the executable CLI; defaults, choices, required inputs, and output options below match the installed command.

## Usage and options

```text
usage: dps genmesh [-h] [-db] -f FILE [FILE ...] -p FILE [FILE ...] -n NUMBER
                   [NUMBER ...] [-g GAP] [--mesh MESH MESH MESH]
                   [-bt {xy,cubic,anosotropy}] [-mx MINIMUM_X] [-my MINIMUM_Y]
                   [-mz MINIMUM_Z] [-s] [--non-cubic-molecule] [-o FILE]
                   [-op FILE] [--seed SEED]

Pack one or more molecule types onto a three-dimensional simulation-box mesh.

Input:
  -f FILE [FILE ...], --structure FILE [FILE ...]
                        Input structure files (.pdb), one per molecule type.
                        (required)
  -p FILE [FILE ...], --topology FILE [FILE ...]
                        Input topology files (.itp), in the same order as
                        --structure. (required)

Output:
  -o FILE, --output-conformation FILE
                        Output packed structure file (.pdb); the extension is
                        added if omitted. (default: system.pdb)
  -op FILE, --output-topology FILE
                        Output system topology file (.top); the extension is
                        added if omitted. (default: system.top)

Parameters:
  -h, --help            Show this help message and exit.
  -db, --debug          Show runtime context, parsed arguments, timing, and
                        tracebacks. (default: False)
  -n NUMBER [NUMBER ...], --number NUMBER [NUMBER ...]
                        Molecule counts, one integer for each input structure.
                        (required)
  -g GAP, --gap GAP     Minimum gap between neighboring molecules, in nm.
                        (default: 1)
  --mesh MESH MESH MESH
                        Mesh dimensions NX NY NZ; if omitted, choose a cubic
                        mesh automatically. (default: None)
  -bt {xy,cubic,anosotropy}, --box-type {xy,cubic,anosotropy}
                        Box geometry; anosotropy is a compatibility alias for
                        anisotropy. (default: anosotropy)
  -mx MINIMUM_X, --minimum-x MINIMUM_X
                        Minimum box length along x, in nm; zero selects the
                        smallest fitted length. (default: 0)
  -my MINIMUM_Y, --minimum-y MINIMUM_Y
                        Minimum box length along y, in nm; zero selects the
                        smallest fitted length. (default: 0)
  -mz MINIMUM_Z, --minimum-z MINIMUM_Z
                        Minimum box length along z, in nm; zero selects the
                        smallest fitted length. (default: 0)
  -s, --shuffle         Shuffle molecule types before inserting them on the
                        mesh. (default: False)
  --non-cubic-molecule  Use separate molecular extents along x, y, and z
                        instead of one cubic extent. (default: False)
  --seed SEED           Random seed used when --shuffle is enabled. (default:
                        1215)

Notes:
  Supply equally long --structure, --topology, and --number lists. Each PDB must match its ITP bead count and molecule definition.

  --mesh sets NX NY NZ slots, --gap sets the intermolecular gap in nm, and --box-type chooses anisotropy, xy-equal, or cubic box lengths. --shuffle uses --seed.

  The output PDB and TOP describe the same molecule ordering. ITP paths in the TOP are relative to the TOP directory; keep those files together when moving a system.

Examples:
  dps genmesh -f chain.pdb -p chain.itp -n 100 -o system.pdb -op system.top
```
