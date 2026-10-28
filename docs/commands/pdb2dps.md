# dps pdb2dps

[All commands](../command-reference.md) · [User guide](../user-guide.md)

Generate coarse-grained PDB conformations and an ITP topology from a protein sequence.

DROPPS 1.0.0. This page is generated from the executable CLI; defaults, choices, required inputs, and output options below match the installed command.

## Usage and options

```text
usage: dps pdb2dps [-h] [-db] -s SEQUENCE [-f FILE] [-ri RESIDUE_INDEX]
                   [-pt POST_TRANSLATIONAL_MODIFICATION [POST_TRANSLATIONAL_MODIFICATION ...]]
                   [-r RADIUS] [-n NUMBER] [-e DEGREE_EXTEND]
                   [-ff {CALVADOS2,HPS,HPSRNA,HPST,MPiPi,MPiPi_PTM}] [-o FILE]
                   [-op FILE] [-on NAME] [-cn] [-cc] [--seed SEED]
                   [--max-attempts MAX_ATTEMPTS]

Generate coarse-grained PDB conformations and an ITP topology from a protein sequence.

Input:
  -s SEQUENCE, --sequence SEQUENCE
                        One-letter protein sequence. (required)
  -f FILE, --input-pdb FILE
                        Optional all-atom PDB file (.pdb) used to initialize
                        C-alpha coordinates. (default: None)

Output:
  -o FILE, --output-conformation FILE
                        Output PDB path or prefix; multiple conformations use
                        numbered .pdb files. (default: None)
  -op FILE, --output-topology FILE
                        Output topology file (.itp); the extension is added if
                        omitted. (default: None)
  -on NAME, --output-name NAME
                        Molecule name written to the ITP topology. (default:
                        MOL)

Parameters:
  -h, --help            Show this help message and exit.
  -db, --debug          Show runtime context, parsed arguments, timing, and
                        tracebacks. (default: False)
  -ri RESIDUE_INDEX, --residue-index RESIDUE_INDEX
                        Residue number assigned to the first bead. (default:
                        1)
  -pt POST_TRANSLATIONAL_MODIFICATION [POST_TRANSLATIONAL_MODIFICATION ...], --post-translational-modification POST_TRANSLATIONAL_MODIFICATION [POST_TRANSLATIONAL_MODIFICATION ...]
                        Post-translational modifications in
                        ORIGINAL+NUMBER+MODIFIED form, for example S129SMP.
                        (default: None)
  -r RADIUS, --radius RADIUS
                        Confinement radius for generated chain coordinates, in
                        nm. (default: 2.0)
  -n NUMBER, --number NUMBER
                        Number of conformations to generate. (default: 1)
  -e DEGREE_EXTEND, --degree-extend DEGREE_EXTEND
                        Chain-extension fraction between 0 and 1. (default:
                        0.5)
  -ff {CALVADOS2,HPS,HPSRNA,HPST,MPiPi,MPiPi_PTM}, --forcefield {CALVADOS2,HPS,HPSRNA,HPST,MPiPi,MPiPi_PTM}
                        Force field to use; if omitted, prompt interactively.
                        (default: None)
  -cn, --charged-ntd    Add a positive charge patch to the N terminus.
                        (default: False)
  -cc, --charged-ctd    Add a negative charge patch to the C terminus.
                        (default: False)
  --seed SEED           Random seed for reproducible conformation generation.
                        (default: 1215)
  --max-attempts MAX_ATTEMPTS
                        Maximum whole-chain construction attempts before
                        failing. (default: 10000)

Notes:
  Provide a one-letter sequence and at least one of --output-conformation (PDB) or --output-topology (ITP). Set --forcefield to avoid the interactive prompt.

  An optional all-atom PDB supplies the first model's C-alpha positions; its C-alpha count must match the sequence. Otherwise --seed controls random chain generation and --max-attempts bounds retries.

  --radius is a construction confinement radius in nm, not a measured radius of gyration. Modification numbers refer to the residue numbering set by --residue-index.

Examples:
  dps pdb2dps -s MSEQNNTEMTFQIQRIYTKDISFEAPNAPHVFQKDW -o chain.pdb -op chain.itp
```
