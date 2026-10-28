# dps ss

[All commands](../command-reference.md) · [User guide](../user-guide.md)

Infer H/E/C states from residue-level coarse-grained backbone geometry.

DROPPS 1.0.0. This page is generated from the executable CLI; defaults, choices, required inputs, and output options below match the installed command.

## Usage and options

```text
usage: dps ss [-h] [-db] -s FILE -f FILE [-n FILE] -o FILE [-om FILE]
              [-on FILE] [-oo FILE] [-os FILE] [-op FILE] [-og PREFIX]
              [-se GROUP [GROUP ...] | -ex EXPR] [--method {angle,psea}]
              [--normalize-bonds | --no-normalize-bonds]
              [--min-helix-length MIN_HELIX_LENGTH]
              [--min-strand-length MIN_STRAND_LENGTH]
              [--helix-angle CENTER HALF_WIDTH]
              [--strand-angle CENTER HALF_WIDTH]
              [--helix-dihedral CENTER HALF_WIDTH]
              [--strand-dihedral CENTER HALF_WIDTH]
              [--strand-pairing {off,interchain}]
              [--strand-contact-scheme {global,residue}]
              [--strand-contact-cutoff STRAND_CONTACT_CUTOFF]
              [--strand-contact-multiplier STRAND_CONTACT_MULTIPLIER]
              [--strand-axis-angle STRAND_AXIS_ANGLE]
              [--strand-lateral-angle STRAND_LATERAL_ANGLE]
              [--strand-min-pairs STRAND_MIN_PAIRS]
              [--strand-registry-slack STRAND_REGISTRY_SLACK]
              [--consensus-threshold CONSENSUS_THRESHOLD]
              [-p | --pbc | --no-pbc] [-b START_TIME] [-e END_TIME]
              [-dt DELTA_TIME]
              [--plot-formats {png,pdf,svg} [{png,pdf,svg} ...]]
              [--plot-dpi PLOT_DPI]

Infer H/E/C states from residue-level coarse-grained backbone geometry.

Input:
  -s FILE, --run-input FILE
                        Input DROPPS run file (.tpr) containing topology and
                        coordinates. (required)
  -f FILE, --input FILE
                        Input trajectory file (.xtc or .pdb). (required)
  -n FILE, --index FILE
                        Optional index file (.ndx) defining backbone
                        selections. (default: None)
  -se GROUP [GROUP ...], --selection-groups GROUP [GROUP ...]
                        Index groups containing one backbone bead per selected
                        residue. (default: None)
  -ex EXPR, --select EXPR, --selection-expression EXPR
                        DROPPS selection expression; repeat for independent
                        selections. (default: None)

Output:
  -o FILE, --output FILE
                        Long-form per-frame, per-residue state fractions
                        (.csv). (required)
  -om FILE, --map-output FILE
                        GROMACS-compatible discrete state map (.xpm).
                        (default: None)
  -on FILE, --number-output FILE
                        Mean C/H/E residue counts per molecule copy versus
                        time (.xvg). (default: None)
  -oo FILE, --occupancy-output FILE
                        Per-residue C/H/E occupancies over frames and copies
                        (.xvg). (default: None)
  -os FILE, --strand-stat-output FILE
                        Per-residue local, paired, and final E occupancies
                        (.xvg). (default: None)
  -op FILE, --strand-pair-output FILE
                        Accepted inter-chain strand pairings for every frame
                        (.csv). (default: None)
  -og PREFIX, --plot-prefix PREFIX
                        Also render map, count, and occupancy plots with this
                        prefix. (default: None)

Parameters:
  -h, --help            Show this help message and exit.
  -db, --debug          Show runtime context, parsed arguments, timing, and
                        tracebacks. (default: False)
  --method {angle,psea}
                        Local geometry classifier; psea also uses torsions and
                        span distances. (default: psea)
  --normalize-bonds, --no-normalize-bonds
                        Scale span distances to a 0.38 nm reference backbone
                        bond. (default: True)
  --min-helix-length MIN_HELIX_LENGTH
                        Minimum number of consecutive residues in an H
                        segment. (default: 4)
  --min-strand-length MIN_STRAND_LENGTH
                        Minimum number of consecutive residues in an E
                        segment. (default: 3)
  --helix-angle CENTER HALF_WIDTH
                        Virtual-angle center and half-width for H, in degrees.
                        (default: (89.0, 12.0))
  --strand-angle CENTER HALF_WIDTH
                        Virtual-angle center and half-width for E, in degrees.
                        (default: (124.0, 14.0))
  --helix-dihedral CENTER HALF_WIDTH
                        Circular pseudo-dihedral center and half-width for H,
                        in degrees. (default: (50.0, 20.0))
  --strand-dihedral CENTER HALF_WIDTH
                        Circular pseudo-dihedral center and half-width for E,
                        in degrees. (default: (-170.0, 45.0))
  --strand-pairing {off,interchain}
                        Require registered inter-chain contacts before
                        retaining local E. (default: off)
  --strand-contact-scheme {global,residue}
                        Contact cutoff used only for optional strand pairing.
                        (default: global)
  --strand-contact-cutoff STRAND_CONTACT_CUTOFF
                        Global strand-pair contact cutoff in nm. (default:
                        0.7)
  --strand-contact-multiplier STRAND_CONTACT_MULTIPLIER
                        Multiplier applied to bead sigma for residue-
                        responsive contacts. (default: 1.2)
  --strand-axis-angle STRAND_AXIS_ANGLE
                        Maximum parallel or antiparallel strand-axis mismatch
                        in degrees. (default: 30.0)
  --strand-lateral-angle STRAND_LATERAL_ANGLE
                        Maximum contact-vector deviation from the lateral
                        plane in degrees. (default: 35.0)
  --strand-min-pairs STRAND_MIN_PAIRS
                        Minimum registered residue contacts needed for a
                        strand pair. (default: 2)
  --strand-registry-slack STRAND_REGISTRY_SLACK
                        Allowed residue-index deviation within one strand
                        registry. (default: 1)
  --consensus-threshold CONSENSUS_THRESHOLD
                        Minimum copy fraction for a categorical state in an
                        ensemble map. (default: 0.5)
  -p, --pbc, --no-pbc   Unwrap bonded molecules and use minimum-image strand
                        contacts. (default: True)
  -b START_TIME, --start-time START_TIME
                        First trajectory time to analyze, in ns. (default:
                        None)
  -e END_TIME, --end-time END_TIME
                        Last trajectory time to analyze, in ns. (default:
                        None)
  -dt DELTA_TIME, --delta-time DELTA_TIME
                        Approximate interval between analyzed frames, in ns.
                        (default: None)
  --plot-formats {png,pdf,svg} [{png,pdf,svg} ...]
                        File formats written when --plot-prefix is set.
                        (default: ('png',))
  --plot-dpi PLOT_DPI   Raster resolution for direct plots. (default: 300)

This is a geometry-based C-alpha/bead-trace inference, not a hydrogen-bond DSSP assignment. Select exactly one ordered backbone bead per residue across one or more molecule copies.

Notes:
  Assign coarse-grained backbone geometry states and optional strand pairings. States describe the bead geometry; they are not all-atom hydrogen-bond assignments.

  The main CSV contains state observations; optional XPM, state counts, occupancies, and pairing tables describe residue/time patterns. Use compatible sequential backbone selections.

Examples:
  dps ss -s run.tpr -f run.xtc -o ss-state.csv --select "mol PROT" --map-output ss-map.xpm --number-output ss-count.xvg --occupancy-output ss-occupancy.xvg
  dps ss -s run.tpr -f run.xtc -o ss-paired.csv --select "mol PROT" --strand-pairing interchain --strand-pair-output strand-pairs.csv
```
