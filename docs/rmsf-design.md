# PBC-aware multi-molecule RMSF design

## Goal

Add a `dps rmsf` analysis based on the scientifically useful behavior of
`gmx rmsf`, while making molecule-copy systems a first-class DROPPS use case.
The command calculates positional population fluctuations after optional
least-squares fitting:

\[
U_i = \left\langle
(x_i-\langle x_i\rangle)(x_i-\langle x_i\rangle)^T
\right\rangle,
\qquad
\mathrm{RMSF}_i = \sqrt{\operatorname{tr}(U_i)}.
\]

`gmx rmsf` treats the selected group as one fit object. DROPPS simulations
often contain tens or hundreds of independent copies of one coarse-grained
molecule. Fitting all copies together would retain their diffusion and
relative rearrangement, obscuring molecular flexibility. `dps rmsf` therefore
uses independent per-molecule fitting by default, while retaining a
whole-selection mode for GROMACS-style system fitting.

## Command interface

```text
dps rmsf -s run.tpr -f run.xtc -o rmsf.xvg \
  --select "(mol PROT) & (resid 20-80)"

dps rmsf -s run.tpr -f run.xtc -o rmsf-residue.xvg \
  -n index.ndx -sel 3 -res \
  -od reference-deviation.xvg -ox average.pdb -oq reference.pdb
```

### Inputs and frame selection

- `-s/--run-input` supplies topology, masses, bonds, and reference-compatible
  atom identity.
- `-f/--input` supplies the XTC or PDB trajectory.
- `-n/--index` optionally adds index groups.
- `-sel/--selection-groups` accepts one or more group numbers.
- `--select/--selection-expression` can be repeated. Each expression selects
  the same molecular region across any number of molecule copies.
- Without either selection form, the normal DROPPS interactive multiple-group
  selector is used.
- `-b`, `-e`, and `-dt` select analyzed frames in ns.
- `-ref/--reference-time` chooses the nearest saved reference frame. The first
  analyzed frame is the default.

### Fitting

`--fit-mode` accepts:

- `molecule` (default): fit every selected molecular region independently to
  its own coordinates in the reference frame;
- `selection`: fit all atoms in one selection together, corresponding to the
  one-group fit model of `gmx rmsf`; or
- `none`: retain physical sub-box translation and rotation.

`--fit-weighting mass` is the default and matches the mass-weighted
least-squares fit in `gmx rmsf`. `uniform` is available for coarse-grained
models where equal bead weighting is preferable.

Every represented complete molecule is reconstructed along its topology bond
graph before selected coordinates are extracted. The current molecule is then
mapped to the periodic image nearest its own reference center. This image
normalization prevents simple box crossings from producing box-length RMSF
artifacts. In `none` mode it does not provide unlimited no-jump diffusion
tracking; use a no-jump trajectory when long-time translational RMSF is the
intended observable.

### Atom and residue output

Atomic output is the default. `-res` is a compatibility spelling for
`--grouping residue`. Residue covariance is calculated by averaging atomic
covariance tensors first and taking the square root afterward:

\[
U_R = \frac{\sum_{i\in R} w_i U_i}{\sum_{i\in R}w_i},
\qquad
\mathrm{RMSF}_R = \sqrt{\operatorname{tr}(U_R)}.
\]

This reproduces the mass-weighted residue convention in `gmx rmsf`, rather
than taking an arithmetic mean of already square-rooted atomic RMSF values.
`--group-weighting uniform` changes only this within-residue average.

Equivalent sites are identified in topology order within each molecule type.
All copies of one molecule type in a selection must expose the same ordered
atom or residue signature. Missing or inconsistent sites are rejected instead
of silently shifting profile rows.

## Outputs

### RMSF profile and detailed table

`-o rmsf.xvg` writes one `xydy` data set for each selection/molecule-type
profile:

- `x` is the one-based atom position inside the complete molecule, or the
  topology residue ID with `-res`;
- `y` is the mean of independently calculated molecule-copy RMSF values; and
- `dy` is their population standard deviation.

The companion `rmsf.csv` keeps every molecule/site value and includes chain
ID, molecule type, atom/residue identity, RMSF, positional deviation from the
reference, isotropic B factor, all six unique covariance components, and full
analysis metadata. `--summary-only` suppresses this table.

The mean/std summary is intentionally not a pooled trajectory/molecule RMSF.
Pooling would include static conformational differences between copies. Users
can construct that observable explicitly from the aligned average coordinates
and detailed covariance data if required.

### GROMACS-inspired optional outputs

- `-od/--output-deviation` writes the profile of
  `sqrt(<|x(t)-x_ref|^2>)` after the same PBC and fit operations.
- `-ox/--output-average` writes average aligned coordinates to PDB.
- `-oq/--output-reference` writes selected reference coordinates to PDB.
- Both PDB outputs store isotropic B factors calculated as
  `B = 8*pi^2/3 * RMSF^2`, with the nm-to-Angstrom conversion applied.

PDB outputs require exactly one top-level selection so each output atom has
one unambiguous coordinate and B factor. A selection may still span any number
of molecule copies.

## Streaming and numerical stability

The trajectory loop retains only per-site state:

- frame count;
- online mean coordinates;
- Welford second central-moment tensors; and
- summed squared displacement from the reference.

Memory is therefore `O(n_selected_atoms)` and does not grow with the number of
frames. Population covariance uses a denominator of `N`, matching the
trajectory-average convention in GROMACS. One analyzed frame is valid and
produces zero RMSF.

## Validation and current limitations

- Empty/duplicate selections, invalid masses, invalid periodic boxes,
  non-equivalent molecule-copy sites, and disconnected multi-atom topology
  molecules produce explicit errors.
- Atom correspondence is topology-based; symmetry-aware atom permutation is
  outside the command's scope.
- The first release writes full anisotropic covariance components to CSV but
  does not emit PDB `ANISOU` records, compare experimental anisotropic tensors,
  or write principal-axis logs. Those outputs can be added without changing
  the accumulator or command semantics.
- Reference coordinates currently come from a saved trajectory frame rather
  than directly from the TPR structure. This is consistent with `dps rmsd`
  and makes the exact reference timestamp explicit in every output.
