# Geometry-based secondary structure (`dps ss`)

`dps ss` estimates helix (`H`), beta-strand (`E`), and coil (`C`) states from
an ordered trace containing one coarse-grained backbone bead per residue. It is
intended for DROPPS residue-level models and is not a hydrogen-bond DSSP
implementation.

## Local assignment

The default `--method psea` is P-SEA-inspired. It evaluates virtual bond
angles, signed four-bead pseudo-dihedrals, and normalized distances between
beads separated by two, three, and four residues. The initial windows are:

| State | Angle | Pseudo-dihedral | d2 | d3 | d4 |
| --- | --- | --- | --- | --- | --- | --- |
| H | 89 ± 12° | 50 ± 20° | 0.55 ± 0.05 nm | 0.53 ± 0.05 nm | 0.64 ± 0.06 nm |
| E | 124 ± 14° | −170 ± 45° | 0.67 ± 0.06 nm | 0.99 ± 0.09 nm | 1.24 ± 0.11 nm |

These windows come from Labesse *et al.*, *Bioinformatics* **13**, 291–295
(1997), DOI: `10.1093/bioinformatics/13.3.291`. DROPPS uses a CG-adapted
segment and pairing procedure, so this mode is not a byte-for-byte
reimplementation of the original P-SEA algorithm.

Distance windows are normalized to a 0.38 nm backbone bond by default, which
can be disabled with `--no-normalize-bonds`. A coherent positive torsion is
still required for a distance-rescued helix so that a mirror-image helix is not
classified as `H`. The default minimum runs are four residues for `H` and three
for `E`. `--method angle` is a more permissive angle-only alternative.

## Optional inter-chain beta pairing

With `--strand-pairing interchain`, a residue remains `E` only when its local
strand segment participates in a registered inter-chain pairing. Pair searches
are performed among equivalent molecule copies within each automatically
constructed cohort; heterotypic pairing between different molecule types is
not part of this first implementation. A candidate pair must pass all of the
following:

- bead contact, using either a global cutoff (0.7 nm by default) or the
  residue-responsive DROPPS sigma cutoff;
- parallel or antiparallel segment axes within 30°;
- a contact vector lying within 35° of the plane normal to both strand axes;
- at least two non-conflicting contacts following one residue registry, with a
  default registry slack of one residue.

The main output reports both `e_local_fraction` and `e_final_fraction`; when
pairing is active it also reports `e_paired_fraction`. This makes false-negative
pairing filters visible instead of silently relabeling every local strand.

## Outputs

```bash
dps ss -s run.tpr -f run.xtc --select "mol PROT" \
  -o ss-state.csv -map ss-map.xpm -num ss-count.xvg \
  -occ ss-occupancy.xvg
```

- `-o`: long-form frame × residue state fractions and consensus state.
- `-map`: GROMACS-compatible enhanced discrete XPM, with time on x and residue
  ID on y. `H` is blue, `E` red, `C` white, and mixed-copy consensus gray.
- `-num`: mean number of `C`, `H`, and `E` residues per molecule copy over
  time.
- `-occ`: per-residue state occupancy pooled across frames and molecule copies.
- `--strand-stat-output`: local, paired, and final beta occupancies.
- `--strand-pair-output`: accepted parallel/antiparallel pair records.
- `--plot-prefix`: direct map, count, and occupancy plots. Install the optional
  `dropps[plot]` dependencies and choose formats with `--plot-formats`.

When a selection contains several equivalent molecule copies, continuous
statistics are averaged over copies. The categorical XPM uses the most common
state only if its fraction reaches `--consensus-threshold` (0.5 by default),
otherwise it displays the mixed state `M`.

The four angle and pseudo-dihedral windows can be calibrated from the CLI with
`--helix-angle`, `--strand-angle`, `--helix-dihedral`, and
`--strand-dihedral`; each accepts `CENTER HALF_WIDTH` in degrees.

## Interpretation and calibration

The thresholds above are an initial transferable prior, not a universal force
field definition. They should be calibrated against structures or trajectories
with trusted secondary-structure annotations for the specific DROPPS mapping.
In particular, a geometry-only method cannot reconstruct backbone hydrogen
bonds, distinguish every beta-like extended coil, or infer DSSP subclasses such
as turns and bends with the same physical meaning as atomistic DSSP.
