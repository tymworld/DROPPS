# Many-protein polymer statistics design

## Goal

Add a `dps polystat` trajectory analysis inspired by `gmx polystat`, but make
systems containing tens to thousands of protein copies the primary use case.
A user selects one protein region across all copies. DROPPS splits that
selection by topology molecule, reconstructs every complete molecule across
periodic boundaries, and reports compact ensemble statistics without requiring
one index group or one output column per protein.

The first version covers:

- end-to-end distance;
- radius and principal components of gyration;
- rotationally invariant molecular shape;
- mean-square internal distance as a function of sequence separation; and
- backbone tangent correlation and a descriptive 1/e persistence length.

No calculation forms pairs between different proteins. The base calculation is
linear in the number of selected particles, and all trajectory-sized results
are written or accumulated as a stream.

## Relationship to `gmx polystat`

`gmx polystat` splits one index group into molecules and writes the root mean
square end-to-end distance, radius of gyration derived from the molecule-
averaged gyration tensor, tensor principal components, optional molecular
principal components, mean-square internal distances, and an optional
persistence length.

`dps polystat` retains directly comparable RMS end-to-end and total gyration
values, but changes two defaults for protein ensembles:

1. Principal values are diagonalized for each protein before averaging. An
   average tensor in laboratory coordinates becomes nearly spherical when
   many anisotropic proteins have random orientations, so its eigenvalues do
   not describe the typical protein shape.
2. Tangent correlations use every backbone lag. The GROMACS even-lag rule is
   appropriate for an all-trans polymer whose consecutive bonds alternate,
   but is not the right default for a residue-level protein backbone.

The mapping is:

| GROMACS quantity or option | DROPPS design |
| --- | --- |
| Split one group into molecules | Split once by DROPPS topology molecule/chain ID |
| End-to-end value in `-o` | `ree_rms_nm`, with arithmetic mean and population standard deviation added |
| Total gyration value in `-o` | `rg_rms_nm`, with arithmetic mean and population standard deviation added |
| Eigenvalues of the averaged tensor | Not a default output because it is orientation dependent |
| `-pc` molecular eigenvalues | Default rotation-invariant principal gyration radii |
| `-v` averaged-tensor eigenvectors | Deferred; useful only for explicitly aligned systems |
| `-mw` | `--weight {mass,uniform}`, default `mass` |
| `-i` | Counted, cohort-aware internal-distance table |
| `-p` even-lag 1/e crossing in bonds | All-lag 1/e crossing in both backbone steps and nm |

An optional laboratory-frame tensor/eigenvector output can be added later for
aligned fibres or interfaces. It must be named explicitly and must not replace
the molecular principal values.

## Command interface

```text
dps polystat -s run.tpr -f run.xtc -o polystat.xvg \
  --select "mol FUS"

dps polystat -s run.tpr -f run.xtc -n index.ndx \
  -sel 1 -o polystat.xvg \
  -i internal-distance.csv \
  -p persistence.xvg \
  --tangent-output tangent-correlation.csv
```

### Inputs and selection

```text
-s/--run-input RUN.tpr
-f/--input TRAJECTORY.xtc
-n/--index INDEX.ndx

-sel/--selection-groups GROUP [GROUP ...]
--select EXPR                         # repeatable

-b/--start-time NS
-e/--end-time NS
-dt/--delta-time NS
--pbc / --no-pbc                     # default: --pbc
```

The two selection forms are mutually exclusive. Each index group or repeated
expression is one analysis job and may contain particles from any number of
protein copies or molecule types. If neither form is supplied, the existing
interactive selector is used.

Selected particles are ordered by topology index within each molecule. The
first and last selected particles define the end-to-end vector. Internal-
distance and tangent calculations additionally require every adjacent pair in
that order to be connected by a topology bond. Extra bonds, such as a
crosslink, do not change the selected path. A non-path selection can still be
used with `--metrics rg shape`.

The normal DROPPS coarse-grained protein representation has one selected bead
per residue, so one molecule-type selection normally supplies both the mass
distribution and the ordered backbone. A future separate gyration/backbone
selection is preferable to guessing a path for a multi-bead residue model.

### Calculations

```text
--metrics {ree,rg,shape} [{ree,rg,shape} ...]
                                      # default: ree rg shape
--weight {mass,uniform}               # default: mass; gyration only
--max-lag N                           # default: all available lags
--lag-step N                          # default: 1
--allow-heterogeneous-cohorts
```

`-i` enables mean-square internal distances and `-p` enables persistence
length. `--tangent-output` writes the correlation curve used by `-p` and also
enables that calculation. `--lag-step 2` provides the GROMACS sampling
convention when it is scientifically appropriate. Lag zero is always retained
internally for tangent correlation even when the requested step is larger.

### Outputs

```text
-o/--output polystat.xvg              # required compact time series
-i/--internal-distance-output FILE.csv
-p/--persistence-output FILE.xvg
--tangent-output FILE.csv
--molecule-output FILE.csv            # optional long-form details
--allow-large-output
```

The primary output is always ensemble-sized: its number of columns depends on
the number of selections and molecule cohorts, not on the number of protein
copies. Per-molecule output is opt-in because its row count is
`n_frames * n_selected_molecules`.

Before opening `--molecule-output`, the command estimates rows and bytes. It
rejects more than 100 million rows or an estimated size above 1 GiB unless
`--allow-large-output` is present. The warning reports both counts so a user can
choose the compact output deliberately.

## Molecule cohorts

A selection can span several protein types, and different protein types can
have different lengths. Values are never silently pooled across those types.
Each analysis job is partitioned into cohorts using:

```text
(selection label, molecule name, complete topology signature,
 selected local-index signature)
```

This gives every batch a fixed selected length, mass vector, and path. Cohorts
are labelled with the molecule name; a stable suffix disambiguates distinct
signatures with the same name. The complete topology signature covers local
particle count, particle identities/masses, and the local bond-edge list; it
does not depend on global atom or molecule IDs.

All copies of one DROPPS molecule type are expected to have the same selected
local-index signature. A mismatch for the same molecule name is an error by
default because it usually means a malformed index group or a missing residue.
An advanced `--allow-heterogeneous-cohorts` option can admit and label those
cohorts separately. Different molecule names are separated automatically and
do not require that option.

The command reports, for every cohort:

- the number of represented copies and the total number of that molecule type
  in the topology;
- selected particle count and residue range;
- whether an ordered bonded path was validated; and
- whether all copies have identical weights and topology signatures.

Matching is always by topology molecule ID, never by the incidental order of
selection fragments.

## Geometry definitions

For molecule `m`, selected coordinates are `r_mi` and the gyration weights are
`w_mi`. Coordinates below are complete-molecule, PBC-reconstructed coordinates
in nm.

### End-to-end distance

```text
R_ee,m = |r_m,L-1 - r_m,0|
```

For each frame and cohort, DROPPS writes:

```text
ree_rms_nm  = sqrt(mean_m(R_ee,m^2))
ree_mean_nm = mean_m(R_ee,m)
ree_std_nm  = sqrt(mean_m((R_ee,m - ree_mean_nm)^2))
```

`ree_rms_nm` is the directly comparable `gmx polystat` quantity. The standard
deviation is the population spread across protein copies in that frame, not a
standard error.

### Gyration tensor and shape

For each molecule:

```text
r_cm,m = sum_i(w_mi r_mi) / sum_i(w_mi)

S_m = sum_i[w_mi (r_mi-r_cm,m)(r_mi-r_cm,m)^T] / sum_i(w_mi)
```

Let `lambda_1 >= lambda_2 >= lambda_3 >= 0` be the eigenvalues of `S_m`.
Numerically negative eigenvalues within a scale-aware round-off tolerance are
clamped to zero; a more negative value is an error.

```text
Rg_m^2 = trace(S_m) = lambda_1 + lambda_2 + lambda_3

kappa_m^2 = (3/2) * sum_j((lambda_j - mean(lambda))^2)
                    / (sum_j(lambda_j))^2
```

`kappa^2` lies in `[0, 1]`, from spherical to rod-like. It is `NaN` for a
zero-size selection.

The per-frame output contains:

```text
rg_rms_nm   = sqrt(mean_m(Rg_m^2))
rg_mean_nm  = mean_m(Rg_m)
rg_std_nm   = population_std_m(Rg_m)

g1_rms_nm   = sqrt(mean_m(lambda_1))
g2_rms_nm   = sqrt(mean_m(lambda_2))
g3_rms_nm   = sqrt(mean_m(lambda_3))
kappa2_mean = mean_m(kappa_m^2)
```

`rg_rms_nm` equals the square root of the trace of the molecule-averaged
gyration tensor and therefore matches the total GROMACS value. The three
`g*_rms_nm` values instead match the intent of GROMACS `-pc`: diagonalize each
molecule first, sort, average the eigenvalues, and then take square roots.
They remain meaningful when proteins rotate independently.

Masses affect only the center and gyration tensor. End-to-end distance,
internal distance, and tangent correlation are geometric quantities and are
not mass weighted.

### Mean-square internal distance

For a cohort with path length `L`, lag `s` is a sequence/path separation:

```text
D(s) = sum_f sum_m sum_i |r_m,i+s - r_m,i|^2
       / sum_f sum_m (L-s)
```

The long-form `-i` table contains one row per cohort and requested lag:

```text
selection, cohort, molecule_name, copies, selected_particles,
lag_steps, mean_contour_nm, observations,
mean_r2_nm2, rms_distance_nm, r2_per_step_nm2
```

`observations` is explicit so truncated selections and future missing-data
support cannot bias a curve silently. `r2_per_step_nm2 = D(s)/s` preserves the
traditional GROMACS plot, while `mean_r2_nm2` and its square root avoid hiding
the normalization. `mean_contour_nm` is accumulated from the actual intervening
bond lengths and supports comparisons between models with different bead
spacing.

### Tangent correlation and 1/e length

For consecutive backbone points, define unit tangents:

```text
t_mi = (r_m,i+1-r_m,i) / |r_m,i+1-r_m,i|

C(s) = mean_f,m,i(t_mi dot t_m,i+s),  C(0) = 1
```

Zero-length backbone bonds are invalid. The optional tangent CSV contains
`lag_steps`, mean contour separation, observation count, mean cosine, and the
number of frames contributing to each cohort/lag.

For every frame and cohort, persistence output uses the first crossing of
`C(s) = exp(-1)`. Positive bracketing values are interpolated linearly in
`log(C)`. If the lower bracket is non-positive, the result is `NaN` rather than
applying a logarithm to an invalid value. A curve that does not cross within
the available/requested lags is right-censored and also written as `NaN`.

The output series are:

```text
lp_1e_steps
lp_1e_nm
```

If the crossing lies a fraction `alpha` between lags `s0` and `s1`, the step
value is `s0 + alpha*(s1-s0)`. The nm value applies the same interpolation to
the measured mean contour separations at those two lags. For internal-distance
pairs, contour separation is the sum of intervening bond lengths. For tangent
pairs, it is the path distance between the two bond midpoints. This avoids
assuming one constant protein-backbone bond length.

The final terminal summary reports valid, non-positive-bracket, and censored
frame counts. For finite or intrinsically disordered proteins this first-
crossing number is a descriptive correlation-decay length; the command must
not present it as proof that a worm-like-chain exponential model is valid.

## PBC algorithm

PBC handling follows the existing DROPPS RMSD and distance analyses:

1. Resolve selections once and split by topology molecule ID.
2. Form the union of all represented complete molecules and selected atoms.
3. Build one static breadth-first traversal from the topology bond graph for
   each complete molecule.
4. In every frame, reconstruct bonded minimum-image vectors using the current
   orthorhombic or triclinic box.
5. Extract each selected path only after the complete molecule is whole.

A complete molecule is reconstructed at most once per frame even when several
metrics use it. Missing/invalid boxes and disconnected multibead molecules
produce explicit diagnostics. `--no-pbc` bypasses reconstruction for
trajectories already made whole by the user.

No protein center is moved into a common image or aligned before the
calculation. All reported default quantities are translation invariant, and
the principal gyration values are rotation invariant.

## Streaming and performance

### Base statistics

Equivalent molecules are represented as coordinate blocks with shape
`(block_molecules, selected_particles, 3)`. For each block:

- endpoint distances are one vectorized subtraction;
- centers and tensors use `einsum`-style reductions;
- all `3 x 3` tensors use batched symmetric eigendecomposition; and
- frame means, second moments, minima, and maxima are merged with numerically
  stable online accumulators.

Temporary blocks target a fixed memory budget. The base computational cost is
`O(n_frames * n_selected_particles)` plus PBC bond traversal, and memory is
independent of frame count.

### Dense lag curves

A direct all-lag implementation is quadratic in protein length. For a dense
set of lags, internal distances use the identity

```text
sum_i |r_i+s-r_i|^2
 = sum_i (|r_i+s|^2 + |r_i|^2) - 2 sum_i (r_i dot r_i+s)
```

Coordinate autocorrelations are evaluated with a batched real FFT, while the
squared-norm terms use prefix sums. Tangent correlations use the same FFT
autocorrelation approach. Coordinates are first translated near their own
center to reduce cancellation, and round-off-sized negative squared distances
are clamped to zero.

Dense lag cost is therefore approximately
`O(n_frames * n_molecules * L log L)`. For a sparse requested lag set, direct
vectorized differences/dot products can be cheaper; the numerical core chooses
the method from `L` and the number of requested lags. Both paths merge only
`sum`, `count`, and contour-length accumulators, so memory is
`O(n_cohorts * max_lag)` rather than trajectory-sized.

### Detailed output

`--molecule-output` is written one frame at a time and never materializes an
`n_frames x n_molecules` matrix. Rows contain:

```text
frame, time_ns, selection, cohort, molecule_id, chain_id, molecule_name,
selected_particles, ree_nm, rg_nm, lambda1_nm2, lambda2_nm2, lambda3_nm2,
kappa2, weighting, pbc
```

Only columns for requested metrics are present. Per-molecule persistence is not
written: a short single-protein tangent curve is generally too noisy, while the
per-frame cohort curve is the defined estimator.

## Output metadata and terminal summary

Every file records:

- command and DROPPS version;
- source run input and trajectory;
- exact selection text or index-group identity;
- cohort signature and copy count;
- time window and actual selected frames;
- PBC mode, gyration weighting, lag range, and lag step; and
- the formulas/aggregation scope needed to distinguish RMS, arithmetic mean,
  and population standard deviation.

After the stream completes, online accumulators print one summary per cohort:
pooled molecule-frame mean, population standard deviation, RMS value, valid
observation count, and persistence censoring counts. Frames are correlated
trajectory observations; these values are not labelled as independent-sample
uncertainties.

## Validation and failure modes

- Empty selections and duplicate selected indices are errors.
- `ree` requires at least two selected particles per molecule; tangent
  correlation and persistence require at least three.
- Internal-distance and persistence selections must follow consecutive
  topology bonds in topology order. Selections that require branch/path
  discovery are unsupported for path metrics but remain valid for gyration;
  extra bonds between nonconsecutive selected particles are ignored.
- Mass weighting rejects missing, non-finite, negative, or zero-total masses.
- Cohorts with the same molecule name but inconsistent local selection
  signatures fail unless heterogeneous cohorts were explicitly allowed.
- A requested `--max-lag` larger than a cohort supports is clipped per cohort
  and recorded, not padded with fictitious zero observations.
- Invalid PBC boxes and disconnected complete molecules fail before partial
  output is presented as complete.
- A one-particle gyration selection has zero `Rg`; its shape anisotropy is
  undefined (`NaN`).
- Eigenvector sign and degeneracy are intentionally absent from default
  outputs.

## Test plan

The NumPy-only core is tested independently of trajectory I/O:

- straight chains with known `R_ee`, `Rg`, principal values, `D(s)=b^2 s^2`,
  and `C(s)=1`;
- independently rotated anisotropic proteins, verifying invariant molecular
  principal values and exposing the isotropization of a lab-frame tensor;
- mass-weighted and uniform tensors;
- exact frame RMS, arithmetic mean, and population standard deviation across
  unequal molecule conformations;
- synthetic correlations bracketing `1/e`, non-positive brackets, and censored
  short chains;
- dense FFT and sparse direct lag kernels against an explicit pair loop;
- molecule-block-size invariance and online-statistic merge invariance; and
- mixed-length/type cohort separation without cross-type pooling.

Integration tests cover:

- parser defaults and GROMACS-compatible `--lag-step 2`;
- automatic splitting of 100 and 1000 protein copies from one selection;
- no `O(n_molecules^2)` channel or allocation growth;
- bonded reconstruction across orthorhombic and triclinic boundaries;
- multiple molecule types and malformed partial-copy selections;
- time selection and metadata;
- streaming XVG/CSV output and detailed-output guards; and
- numerical comparison of `ree_rms_nm`, `rg_rms_nm`, and molecular principal
  radii with `gmx polystat` on the same homogeneous test trajectory.

## Implementation layout

```text
dropps/analysis/polystat_core.py  # pure NumPy geometry, lag kernels, accumulators
dropps/analysis/polystat.py       # CLI, topology planning, PBC, streaming output
tests/test_polystat_core.py
tests/test_polystat.py
```

The command reuses `trajectory_class.time2frame`, DROPPS selection parsing, and
`image_interaction_core.build_unwrap_plan`. It is registered as a normal
analysis command. The existing `dps gyrate` remains available during the first
release; deprecation should be considered only after `polystat` has equivalent
single-selection ergonomics and documented migration examples.

## First-release boundary

The first release includes the compact time series, optional molecule CSV,
internal-distance table, tangent curve, and 1/e persistence output described
above. The following are deliberately deferred:

- laboratory-frame averaged tensor eigenvectors;
- orientational alignment before tensor averaging;
- fitted Flory exponents or worm-like-chain models;
- confidence intervals that account for temporal autocorrelation;
- dynamic selections or missing particles within a trajectory; and
- branched-polymer path discovery.

These features require separate scientific choices and should not complicate
the fast, unambiguous many-protein default.

## Reference

The compatibility definitions above follow the official
[GROMACS `gmx polystat` documentation](https://manual.gromacs.org/2024.5/onlinehelp/gmx-polystat.html)
and the
[`gmx_polystat.cpp` implementation](https://gitlab.com/gromacs/gromacs/-/blob/release-2026/src/gromacs/gmxana/gmx_polystat.cpp)
in the GROMACS source tree.
