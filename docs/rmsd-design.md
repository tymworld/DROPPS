# Configurable molecular/selection RMSD design

## Goal

Add a `dps rmsd` trajectory analysis that measures conformational RMSD for
systems containing many copies of one or more molecules. A user selects an atom
region once; DROPPS automatically partitions that selection by topology
molecule and reconstructs every complete molecule across periodic boundaries.
Fitting and RMSD output granularity are configured independently.

The topology molecule boundary is the existing DROPPS chain/ITP-instance ID.
The selected atoms retain their topology order and are never permuted.

## Command interface

```text
dps rmsd -s run.tpr -f run.xtc -o rmsd.xvg -sel GROUP [GROUP ...]
dps rmsd -s run.tpr -f run.xtc -o rmsd.xvg \
  --select "(mol PROT) & (resid 20-80)"
```

- `-sel/--selection-groups` accepts one or more index groups. Each group may
  contain atoms from any number of molecules.
- `--select/--selection-expression` may be repeated and accepts the normal
  DROPPS selection language directly, avoiding a mandatory `.ndx` round trip.
- If neither form is supplied, the existing interactive selector is used.
- `-b`, `-e`, and `-dt` select analyzed frames.
- `-ref/--reference-time` chooses the nearest saved reference frame. By default,
  the first analyzed frame is the reference.
- Fitting and RMSD use equal atom weights by default. `--mass-weighted` applies
  selected-atom masses to centering, rotation fitting, and RMSD.
- `--fit-mode` accepts `none`, `molecule` (default), or `selection`. `molecule`
  performs an independent Kabsch fit for every molecule; `selection` performs
  one Kabsch fit using every atom in that selection.
- `--output-mode` accepts `molecule` (default) or `selection`. Molecule mode
  reports one RMSD per represented molecule. Selection mode pools the weighted
  squared residuals of all selected atoms into one RMSD per selection.
- `--summary-only` suppresses the detailed CSV when its size would be
  impractical or only the XVG time series is required.

Each molecule is compared with its own coordinates in the reference frame.
This measures conformational change without assuming that separately
initialized molecule copies have identical starting conformations.

## Geometry and PBC algorithm

1. Resolve every selection once and split its atoms by topology molecule ID.
2. Form the union of all represented molecules and selected atoms, so complete
   molecules are unwrapped only once per trajectory frame even when several
   selection regions overlap.
3. Build a static breadth-first traversal of each complete molecule's topology
   bond graph. Disconnected multibead molecules are rejected because their
   relative periodic images cannot be determined unambiguously.
4. For each frame, reconstruct complete molecules along bonded
   minimum-image vectors using that frame's orthorhombic or triclinic box.
   Only after reconstruction are the selected atoms extracted.
5. Map each current molecule to the periodic image nearest its own reference
   center. This removes only lattice-image ambiguity, including in `none` fit
   mode; physical sub-box translation and rotation are retained.
6. Apply the requested fit. Molecules with the same selected-atom count are
   batched for per-molecule Kabsch fits. Whole-selection fitting flattens the
   molecule regions and solves one proper-rotation Kabsch fit. Reflection is
   explicitly disallowed.
7. Calculate RMSD at the requested output scope:

   \[
   \mathrm{RMSD} =
   \sqrt{\frac{\sum_i w_i\lVert R(x_i-\bar{x})-(y_i-\bar{y})\rVert^2}
   {\sum_i w_i}}
   \]

   where `x` is the current selection and `y` is its reference. With molecule
   output the sum covers one molecule; with selection output it covers every
   atom represented by that selection.

All six fit/output combinations are valid. In particular, molecule fitting plus
selection output pools the fitted atomic residuals rather than arithmetically
averaging molecular RMSDs. Selection fitting plus molecule output applies the
single global fit first and then partitions its residuals by molecule.

The frame loop streams results and does not retain an
`n_frames × n_molecules` array in memory.

## Outputs

For `-o rmsd.xvg`:

- With molecule output, `rmsd.xvg` contains the per-frame molecular mean and
  population standard deviation for every selection, and `rmsd.csv` contains
  the long-form per-molecule values.
- With selection output, `rmsd.xvg` contains one series per selection and
  `rmsd.csv` contains one row per selection and frame.

Both files record enough metadata to identify the reference and fitting
convention.

## Validation and limitations

- Empty selections, missing/invalid boxes, duplicate selected atoms, invalid
  masses, and disconnected bonded molecules produce explicit errors.
- One selected atom is accepted. Its RMSD is necessarily zero with molecule
  fitting, but can retain displacement with no fit or whole-selection fitting.
- Atom correspondence follows topology indices. Symmetry-aware atom
  permutation and comparison to an external reference structure are outside
  this first implementation.
- Tests cover rigid translation/rotation invariance, deformation, weighted
  fits, reflection handling, batched molecules, triclinic PBC reconstruction,
  command parsing, and output-path behavior.
