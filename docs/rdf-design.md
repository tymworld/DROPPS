# Radial distribution function analysis design

## Goal

Add a public trajectory-analysis command:

```text
dps rdf
```

The command calculates radial distribution functions from one reference
selection to one or more target selections.  Its scientific definitions follow
the useful behavior of
[GROMACS `gmx rdf`](https://manual.gromacs.org/current/onlinehelp/gmx-rdf.html):

- one reference position set and multiple target position sets;
- atom, center-of-geometry (COG), and center-of-mass (COM) positions;
- three-dimensional and projected x-y distributions;
- `rdf`, radial number-density, and unnormalized pair-count densities;
- topology exclusions, a lower distance cutoff, and a maximum radius;
- surface-relative distributions; and
- an optional cumulative coordination-number output.

The interface is adapted to existing DROPPS conventions rather than copying the
GROMACS selection language.  Atom selection, entity grouping, and position
construction remain independent choices, as in `dps distance`.  The first
implementation uses static selections and streams histograms without retaining
an all-pairs or all-frames distance matrix.

## User model

An RDF job contains one reference endpoint and one or more target endpoints:

```text
reference atom selection -> grouping -> atom/COG/COM positions --+
                                                               +-> pair search
target atom selection    -> grouping -> atom/COG/COM positions --+      |
                                                                         v
                                     exclusions -> radial bins -> normalization
```

The same reference definition is used for every target curve.  Each target
selection produces one series in `-o`, and, when requested, one cumulative
series in `-cn`.

For the standard mode, every reference position is compared with every target
position.  `--surface` changes only the reference interpretation: for each
target position and each reference molecule/residue, only the distance to the
closest selected reference atom contributes.

## Command interface

```text
dps rdf -s run.tpr -f run.xtc [-n index.ndx] \
  -ref REF -sel SEL [-sel SEL ...] -o rdf.xvg \
  [-cn rdf-cn.xvg] [--norm {rdf,number-density,none}] \
  [--bin-width 0.002] [--cut 0] [--rmax R]
```

The complete option groups are:

```text
Input:
  -s, --run-input FILE              DROPPS run input with topology
  -f, --input FILE                  .xtc or .pdb trajectory
  -n, --index FILE                  optional .ndx groups

Output:
  -o, --output FILE                 RDF curves (.xvg), required
  -cn, --output-cumulative FILE     cumulative coordination curves (.xvg)

Selections and positions:
  -ref, --reference SELECTION       exactly one reference selection
  -sel, --selection SELECTION       repeat for one or more target curves
  --position {atom,cog,com}         common position type, default atom
  --ref-position {atom,cog,com}     reference-side override
  --sel-position {atom,cog,com}     target-side override
  --grouping {all,chain,residue,atom}
  --ref-grouping {all,chain,residue,atom}
  --sel-grouping {all,chain,residue,atom}

Geometry and sampling:
  --geometry {3d,xy}                default 3d
  --pbc / --no-pbc                  default --pbc
  -b, --start-time FLOAT            ns
  -e, --end-time FLOAT              ns
  -dt, --delta-time FLOAT           ns
  --bin-width FLOAT                 nm, default 0.002
  --cut FLOAT                       nm, default 0
  --rmax FLOAT                      nm; safe automatic value if omitted

Pair policy:
  --exclude {none,topology,same-molecule}   default none
  --surface {none,chain,residue}            default none
  --norm {rdf,number-density,none}          default rdf
```

`number_density` is accepted as a compatibility spelling of
`number-density`, and `molecule` is accepted as a compatibility spelling of
`chain`.  Help and output metadata use DROPPS's public spellings.

### Selection behavior

- `-ref` accepts one DROPPS selection expression or a bare index-group number.
- `-sel` is repeatable; command-line order determines XVG series order.
- If both are omitted, interactive mode asks for one reference followed by one
  or more targets.  Supplying only one side is an error so a batch command
  cannot unexpectedly pause for input.
- Empty selections and selections containing duplicate atom indices are
  rejected.  Repeated atoms across the reference and a target are allowed and
  handled by the distance cutoff/exclusion policy.
- Labels are derived from the original selection text.  Duplicate labels gain
  stable `#2`, `#3`, ... suffixes.

Selections are static: the atom membership cannot change from frame to frame.
Coordinate-dependent/dynamic selections are outside the first implementation.

### Position construction

Grouping partitions the selected atoms into entities:

- `all`: one entity containing the full selection;
- `chain`: one entity per DROPPS topology molecule/chain instance;
- `residue`: one entity per `(chain ID, residue ID)` pair; and
- `atom`: one singleton entity per atom.

The position type is then applied to every entity:

- `atom`: retain all selected atoms as individual positions;
- `cog`: emit one equal-weight center per entity; and
- `com`: emit one mass-weighted center per entity.

Grouping does not collapse atom positions: `--position atom --grouping all`
still emits every selected atom.  For `cog` and `com`, the default grouping
`all` emits one center for the complete selection.  The GROMACS concepts
`mol_com`, `mol_cog`, `res_com`, and `res_cog` therefore map to:

```text
--position com --grouping chain
--position cog --grouping chain
--position com --grouping residue
--position cog --grouping residue
```

Side-specific options override their common option.  All target selections
share the target position/grouping configuration; heterogeneous target
position definitions should be separate command invocations in the first
version.

Examples:

```text
# Atom-atom RDF from protein beads to solvent beads
dps rdf -s run.tpr -f run.xtc -ref "mol PROT" -sel "mol SOL" \
  -o protein-solvent.xvg

# Intermolecular protein COM-COM RDF for many copies
dps rdf -s run.tpr -f run.xtc -ref "mol PROT" -sel "mol PROT" \
  --position com --grouping chain --exclude same-molecule \
  -o protein-com.xvg -cn protein-com-cn.xvg

# Residue COG distribution around one protein COM
dps rdf -s run.tpr -f run.xtc -ref 3 -sel "res LYS" \
  --ref-position com --ref-grouping all \
  --sel-position cog --sel-grouping residue -o lys.xvg

# In-plane RDF for a membrane
dps rdf -s run.tpr -f run.xtc -ref "res POPC" -sel "res CHOL" \
  --position cog --grouping residue --geometry xy -o membrane-rdf.xvg
```

## Pair semantics and exclusions

In standard mode, a frame contributes one directed candidate for every
reference/target position pair.  When reference and target are the same
selection, both `(i,j)` and `(j,i)` are present.  After the radial grid has
resolved its actual outer edge, a pair contributes only when

```text
cut < distance < actual-rmax
```

The strict lower comparison means the default `--cut 0` removes self-pairs and
all other exactly coincident positions, matching `gmx rdf`.  It does not remove
nearby bonded pairs.

`--exclude` is applied before radial binning:

- `none`: apply no provenance-based exclusion beyond `--cut`;
- `topology`: remove the 1-2/1-3 non-bonded exclusions encoded by each ITP
  molecule's `nrexcl`; valid only when both endpoints emit atom positions; and
- `same-molecule`: remove a pair when the two emitted positions have any
  underlying topology molecule/chain ID in common.

Topology exclusion uses the same canonical exclusion builder as simulation
setup; it must be moved from `build_system.py` to a lightweight shared module
rather than reimplemented.  `topology` is incompatible with `--surface`, just
as in GROMACS, because a surface distance is the result of a nearest-atom
reduction rather than one identifiable pair.

Exclusions do **not** change any RDF normalization factor.  For example, a
same-selection atom RDF is still normalized with `N_ref * N_sel`, not
`N_ref * (N_sel - 1)`, and a same-molecule exclusion does not replace the
target density with an available intermolecular density.  Consequently, a
small finite same-selection system can approach `(N-1)/N` rather than exactly
one.  This is deliberate GROMACS-compatible behavior and is recorded in the
XVG metadata.

## Surface-relative mode

```text
--surface {chain,residue}
```

The reference must emit atoms.  It is partitioned by topology chain or by
`(chain ID, residue ID)`.  For every target position, the minimum distance to
the selected atoms in each reference entity is calculated, and those minimum
distances are binned.  Thus a target position can contribute once per
reference entity, rather than once per reference atom.

Surface bins have irregular, conformation-dependent volumes.  Therefore:

- `--surface` requires `--norm none` (implicitly selected when `--norm` was not
  supplied, but an explicit incompatible normalization is an error);
- `--exclude topology` is invalid;
- reference COG/COM and `--ref-grouping` are invalid because surface grouping
  defines the reference entities; and
- `-cn` remains valid and reports the mean number of target positions within a
  surface distance.

The term "surface" follows GROMACS usage: this is a closest-selected-atom
distance to each molecule/residue, not a solvent-accessible molecular surface.

## Geometry and PBC

Coordinates enter from MDAnalysis in Angstrom and are converted once to nm at
the numerical-kernel boundary.  Time input/output uses ns, consistent with
current DROPPS analyses.

With `--geometry 3d`, distances use all Cartesian components and shell measures
are spherical volumes.  With `--geometry xy`, the z component is ignored,
periodic minimum images are taken in the lattice spanned by the first two box
vectors, and shell measures are annular areas.  The density is correspondingly
in `nm^-3` or `nm^-2`.

PBC is enabled by default.  Orthorhombic and triclinic boxes use a true
minimum-image implementation.  To guarantee that each base position pair has
one unique image in the histogram, `rmax` may not exceed the inscribed radius
of the periodic Wigner-Seitz cell (or its x-y analogue) in any analyzed frame.

The command performs a cheap box pre-pass over the selected frames:

- if `--rmax` is omitted, it uses `0.99 * min(safe cutoff over frames)`;
- an explicit `--rmax` above any frame's safe cutoff is rejected with the
  offending frame/time and allowed value;
- the requested safe limit is snapped downward to the largest complete radial
  bin edge and becomes `actual-rmax`; and
- the requested/automatic limit, actual bin edge, and limiting frame are
  printed and stored in metadata.

This is intentionally safer for NPT trajectories than deriving the cutoff only
from the first frame.  With `--no-pbc`, no uniqueness constraint is applied;
the default radius is three times the largest box edge observed in the selected
frames, matching the practical GROMACS fallback.  `rdf` normalization without
PBC is allowed but warns that finite-box boundary effects generally prevent a
large-r plateau at one.

COG/COM positions under PBC are constructed after complete topology molecules
are reconstructed along bonded minimum-image vectors, reusing the RMSD/distance
unwrapping plan.  A center entity spanning several molecules uses the existing
deterministic first-molecule image anchor.  A compactness warning is emitted
when its span approaches the safe half-box extent.  Disconnected multibead
molecules needed for center construction, invalid COM masses, missing boxes,
and non-finite coordinates are explicit errors.  Atom-atom RDFs do not require
bonded reconstruction.

## Bin grid and normalization

### GROMACS-compatible radial grid

The RDF grid follows `gmx rdf`, which reports radii

\[
r_i = i\,\Delta r, \qquad i=0,1,\ldots
\]

and treats the first bin as a half-width bin.  Its boundaries are

\[
b_0=0, \qquad b_{i+1}=(i+\tfrac12)\Delta r.
\]

For 3D, the shell volume is

\[
\Delta V_i=\frac{4\pi}{3}\left(b_{i+1}^3-b_i^3\right),
\]

and for x-y geometry the annular area is

\[
\Delta A_i=\pi\left(b_{i+1}^2-b_i^2\right).
\]

This unusual first half-bin is specified explicitly so validation against
GROMACS does not suffer a half-bin shift.  For `n` emitted bins, the actual
exclusive outer edge is \(b_n=(n-\tfrac12)\Delta r\).  The requested/automatic
maximum radius is snapped downward to the largest such edge not exceeding the
safe limit; a value smaller than the first half-bin is rejected.  `--cut`
filters distances before the histogram and does not shift the grid origin.

### Accumulated quantities

For target series `s`, let:

- `H[s,i]` be the mean raw pair count in bin `i` over analyzed frames;
- `N_ref` be the mean number of reference positions, or reference surface
  entities in surface mode;
- `rho_s = mean(N_s(frame) / measure(frame))`, where `measure` is box volume
  in 3D or projected box area in x-y geometry; and
- `Delta_i` be `Delta V_i` or `Delta A_i`.

Static selections make the position counts constant, but the average form is
retained to define NPT behavior and future dynamic selections.

The output series is:

\[
g_s(r_i)=
\begin{cases}
H_{s,i}/(N_{ref}\,\Delta_i\,\rho_s), & \texttt{rdf},\\
H_{s,i}/(N_{ref}\,\Delta_i), & \texttt{number-density},\\
H_{s,i}/(N_{ref}\,\Delta r), & \texttt{none}.
\end{cases}
\]

The `rdf` formula deliberately uses the average target density, rather than
averaging separately normalized frame RDFs, matching GROMACS's normalization.
Units are dimensionless, `nm^-3`/`nm^-2`, and `nm^-1`, respectively.

The cumulative coordination number is calculated directly from the same raw
shell counts and is independent of `--norm`.  It is reported at each shell's
outer edge \(R_i=b_{i+1}\):

\[
C_s(R_i)=\frac{1}{N_{ref}}
          \sum_{j=0}^{i}H_{s,j}.
\]

It therefore represents the mean number of admitted target positions within
`R_i` per reference position/entity.  `--cut` and exclusions also remove pairs
from the cumulative result.  Using the shell edge rather than the shell center
keeps the coordinate faithful to the phrase "within a distance"; the values
remain directly comparable with GROMACS after accounting for its cumulative
plot's half-bin x-coordinate convention.

## Outputs

`-o/--output` writes one XVG series per target selection:

```text
radius_nm  target_0  target_1 ...
```

The y-axis label depends on normalization and geometry:

- `Radial distribution g(r)` for `rdf`;
- `Number density (nm^-3)` or `(nm^-2)` for `number-density`; and
- `Pair count density (nm^-1)` for `none`.

`-cn/--output-cumulative` uses the RDF bin outer edges and the same legends,
with y-axis label `Cumulative number`.

Both files contain comment metadata for:

- original reference/target expressions and stable labels;
- grouping and position type for each side;
- geometry, PBC/image convention, and surface mode;
- exclusion policy and the unchanged-denominator convention;
- requested and actual time range, selected frame count, and frame times;
- bin width, lower cutoff, requested/actual maximum radius, and limiting box
  frame;
- mean reference count, target counts, mean target densities, and mean box
  measure; and
- DROPPS version and output-column order.

Output paths gain `.xvg` when omitted, must be distinct, and are opened only
after all selections and arguments have passed validation.

## Implementation structure

```text
dropps/analysis/rdf.py              CLI, selection resolution, trajectory passes,
                                    metadata, and XVG writers
dropps/analysis/rdf_core.py         grids, shell measures, pair filtering,
                                    histogram accumulation, normalization
dropps/share/topology_exclusions.py shared ITP nrexcl exclusion construction
tests/test_rdf.py                   command and output integration tests
tests/test_rdf_core.py              dependency-light numerical tests
```

Registration additionally updates `dropps/analysis/__init__.py`,
`dropps/share/all_commands.py`, the command-help tests, and the generated README
command table.

The entity/position planner should be shared with `distance_core.py` rather
than independently redefining grouping, COM validation, provenance, or
PBC-aware center construction.  A small common `position_core.py` extraction is
preferred if importing distance-specific job types would couple the commands.

Suggested immutable core types are:

- `PositionEndpointPlan`: selected atoms, entity partitions, position type,
  provenance, masses, and label;
- `RdfPlan`: reference endpoint, target endpoints, surface/exclusion policy,
  geometry, grid, and exact position counts;
- `RadialGrid`: reported radii, bin edges, shell measures, cutoff, and maximum
  radius; and
- `RdfAccumulator`: integer/float64 raw counts, reference factors, target
  density factors, and final normalization.

The core accepts NumPy coordinates and six-value MDAnalysis box dimensions.  It
must not depend on CLI parsing, progress bars, trajectory objects, or file I/O.

## Performance and memory

The box pre-pass stores only one safe-cutoff scalar per limiting frame, not
coordinates.  The main trajectory pass constructs all endpoint positions once
per frame and updates fixed-size histograms.

Standard pair searches use a cutoff-aware cell/neighborhood search and bounded
target tiles.  They must not materialize an unrestricted
`N_ref x N_target x 3` displacement array.  Complexity is approximately
`O(N + P)` per frame for `P` pairs within `rmax`, with memory
`O(N + bins * targets + tile_pairs)`.

Surface mode reduces minimum distances per `(target position, reference
entity)` in bounded tiles.  It may be more expensive than the standard RDF, but
still must not retain results across frames.

Before the main pass, the command prints:

- reference and target atom/entity/position counts;
- number of radial bins and actual maximum radius;
- selected frame count;
- exclusion/surface policy; and
- a conservative pair-search workload estimate.

## Validation plan

Unit tests for `rdf_core.py` cover:

- exact GROMACS-compatible radii, half-width first bin, 3D shell volumes, and
  x-y annular areas;
- known pair distances on bin boundaries and strict
  `cut < r < actual-rmax` rules;
- all three normalization formulas and units;
- NPT normalization using `mean(N_target / volume)`;
- cumulative numbers derived from raw counts, independent of normalization;
- same-selection self removal with an unchanged `N * N` denominator;
- topology and same-molecule exclusions without denominator correction;
- orthorhombic and triclinic minimum images in 3D and x-y geometry;
- safe-radius calculation and explicit oversized-radius rejection;
- atom, COG, and COM endpoints, including bonded boundary reconstruction;
- surface reductions for multiple molecules and repeated residue IDs; and
- tiled/cell-list histograms equal to brute-force NumPy references.

Command/integration tests cover:

- public command registration, help sections, defaults, and compatibility
  spellings;
- one reference with multiple targets and stable legend order;
- interactive-vs-batch selection validation;
- `-b`, `-e`, and `-dt` selection from actual timestamps;
- output extension, collision checks, metadata, and units;
- automatic NPT-safe `rmax` and informative explicit-radius failures;
- disconnected-center and invalid-mass errors; and
- bounded-memory behavior on a many-chain synthetic system.

Numerical validation includes small equivalent DROPPS/GROMACS trajectories for
atom RDF, molecule COM RDF, x-y RDF, topology exclusions, and cumulative
numbers.  RDF curves are compared on the GROMACS radial grid; cumulative values
are compared after converting the GROMACS x coordinates to bin outer edges,
with tolerances stated in the tests.  A separate uniformly distributed periodic
ideal-gas fixture checks that a distinct-selection RDF approaches one away from
finite-sample noise.

## First implementation boundary

The first release includes static atom selections, all position/grouping
combinations, 3D/x-y geometry, all three normalizations, cumulative output,
PBC/triclinic boxes, topology/same-molecule exclusions, and chain/residue
surface mode.

Outside this change are dynamic coordinate-dependent selections, arbitrary
projection axes, angular/orientational distribution functions, anisotropic
`g(r,z)` maps, per-frame RDF output, block uncertainty estimates, partial
structure factors, potential-of-mean-force postprocessing, and automatic
finite-size normalization corrections.  Those are separate observables or
postprocessing operations and should not complicate the base `dps rdf`
contract.

## References

- [GROMACS 2026.3 `gmx rdf` manual](https://manual.gromacs.org/current/onlinehelp/gmx-rdf.html)
- [GROMACS `rdf.cpp` implementation](https://github.com/gromacs/gromacs/blob/main/src/gromacs/trajectoryanalysis/modules/rdf.cpp)
