# Unified distance analysis design

## Goal

Replace the chain-specific `idist` and `odist` implementations with one public
trajectory-analysis command:

```text
dps distance
```

The command must support:

- fixed corresponding atom/position pairs;
- COG and COM positions derived from selected groups;
- minimum and maximum distances between groups of positions;
- Cartesian displacement components;
- individual and averaged time series;
- per-distance statistics; and
- distance histograms.

Chain identity is selection metadata, not an implicit part of the distance
definition. Intra-chain, inter-chain, residue-wise, and molecule-wise analyses
are therefore configurations of the same engine.

Systems with tens to thousands of copies of one protein are a first-class use
case. A user selects a protein type and residue region once; the planner splits
the selection by topology chain and generates intra-chain or inter-chain entity
pairs without requiring one index group per chain.

This follows the useful separation in GROMACS between position construction,
pairing/reduction, and output projection, but keeps fixed-pair and group-pair
calculations in one DROPPS command instead of exposing separate `distance` and
`pairdist` commands.

## User model

Every calculation is a `DistanceJob`:

```text
reference selection -> reference entities --+
                                           +-> chain scope -> entity pairs
selection           -> selection entities --+                    |
                                                                  v
                                                   positions -> operator -> channels
```

The five independent choices are:

1. **Selection**: which atoms belong to each side.
2. **Grouping**: whether a selection is one entity or is split by chain,
   residue, or atom.
3. **Scope**: pair corresponding entities or generate intra-chain,
   inter-chain, or all entity pairs.
4. **Position type**: retain the atoms as positions or collapse each entity to
   its COG/COM.
5. **Operator**: zip corresponding positions (`pair`) or reduce the Cartesian
   distances within every entity pair (`min` or `max`).

Outputs consume the resulting channels without changing their definition.

## Command interface

```text
dps distance -s run.tpr -f run.xtc \
  --mode {pair,min,max} \
  [--scope {corresponding,intra-chain,inter-chain,all}] \
  -ref REF [-ref REF ...] -sel SEL [-sel SEL ...] \
  [-oall distances.xvg] [-oav average.xvg] \
  [-oxyz components.xvg] [-ostat statistics.csv] \
  [-oh histogram.xvg]
```

### Selections and jobs

- `-ref/--reference` and `-sel/--selection` are repeatable DROPPS selection
  expressions. A bare integer continues to mean an index-group number.
- One reference and N selections, or N references and one selection, creates N
  jobs by broadcasting the singleton side.
- N references and N selections creates N jobs by zipping them in command-line
  order.
- Any other reference/selection count is rejected.
- `--label` may be repeated once per resulting job. Without it, a unique label
  is derived from the two selection expressions.
- If neither side is supplied, interactive selection asks for one reference
  followed by one or more selections. Supplying just one side is an error.

Selection order is semantically important for `pair` mode. A simple `.ndx`
group reference preserves file order. A compound boolean expression has no
user-defined order and resolves in topology-index order. The current selection
resolver must be adjusted because it converts group members through a `set`.
Duplicate indices remain invalid; repeated one-atom reference selections can be
used when the same atom participates in several explicitly defined jobs.

### Position construction

```text
--position       {atom,cog,com}       # common default: atom
--ref-position   {atom,cog,com}       # optional reference-side override
--sel-position   {atom,cog,com}       # optional selection-side override

--grouping       {all,chain,residue,atom}  # default: all, or chain for chain scope
--ref-grouping   {all,chain,residue,atom}  # optional reference-side override
--sel-grouping   {all,chain,residue,atom}  # optional selection-side override
```

Grouping is applied first:

- `all`: one entity containing the complete resolved selection;
- `chain`: one entity per DROPPS topology molecule/chain instance;
- `residue`: one entity per `(molecule ID, residue ID)` pair; and
- `atom`: one singleton entity per selected atom.

Position type is then applied to every entity:

- `atom`: retain all entity atoms as ordered positions;
- `cog`: emit one equal-weight coordinate center; and
- `com`: emit one mass-weighted coordinate center.

This makes the common cases direct:

```text
# Fixed atom pairs: ref[0]-sel[0], ref[1]-sel[1], ...
dps distance ... --mode pair -ref 3 -sel 4 -oall fixed.xvg

# COM of group 3 to COG of group 4
dps distance ... --mode pair -ref 3 -sel 4 \
  --ref-position com --sel-position cog -oall centers.xvg

# One COM-to-COM distance for every corresponding chain
dps distance ... --mode pair -ref "group Protein" -sel "group Ligand" \
  --grouping chain --position com -oall chain-centers.xvg

# Global closest atom-atom distance between two groups
dps distance ... --mode min -ref 3 -sel 4 -oall closest.xvg

# Farthest atom-atom distance for every chain pair
dps distance ... --mode max -ref 3 -sel 4 \
  --grouping chain --scope all -oall chain-max.xvg

# Every atom-atom distance (min over singleton groups is the sole distance)
dps distance ... --mode min -ref 3 -sel 4 \
  --grouping atom --scope all -oall pairwise.xvg
```

The common options set both sides; a side-specific option takes precedence.
`molecule` is accepted as a hidden compatibility spelling of `chain`, but help
and metadata use the term `chain` that DROPPS users already see.

### Entity scope and chain relationships

```text
--scope {corresponding,intra-chain,inter-chain,all}
--unique-entity-pairs
```

Scope acts on entities before their atom positions are compared:

- `corresponding` (default) pairs reference and selection entities in stable
  order. It is the normal choice for explicit fixed pairs and for two `all`
  entities.
- `intra-chain` joins reference and selection entities with the same topology
  chain ID.
- `inter-chain` forms the Cartesian entity product and retains pairs with
  different topology chain IDs.
- `all` forms the complete Cartesian entity product without a chain filter.

For `intra-chain` and `inter-chain`, grouping defaults to `chain`. An explicit
grouping is accepted only when every produced entity belongs to exactly one
chain; this prevents a multi-chain COG from being mislabeled as an intra-chain
position. Matching is by topology chain ID, never by incidental list position.
If a selected residue is missing from a chain, that chain is reported and is
absent from the result instead of being paired with the next chain.

`--unique-entity-pairs` removes the mirrored half of a Cartesian entity product.
It is accepted only when the normalized reference and selection definitions
(selection, grouping, and position type) are identical; otherwise the two
directions need not represent the same observable and the option is rejected.

Entity scope and distance mode are independent. Scope says *which chains are
compared*; mode says *how the atoms or centers inside each selected entity pair
are reduced*.

### Many-chain example

Assume the topology contains 100 copies of molecule type `PROT`, and residue
10 is A while residue 50 is B. The user selects each residue once across the
protein type:

```text
REF="(mol PROT) & (resid 10)"
SEL="(mol PROT) & (resid 50)"
```

For an all-atom residue center, calculate all 100 intra-chain A-B distances:

```text
dps distance -s run.tpr -f run.xtc \
  -ref "$REF" -sel "$SEL" \
  --scope intra-chain --position cog --mode pair \
  -ostat intra.csv --statistics-scope both -oh intra-hist.xvg
```

The planner creates `A(chain i) -> B(chain i)` for every chain containing both
selections: 100 channels when all copies are complete.

For all inter-chain distances:

```text
dps distance -s run.tpr -f run.xtc \
  -ref "$REF" -sel "$SEL" \
  --scope inter-chain --position cog --mode pair \
  -ostat inter.csv --statistics-scope both -oh inter-hist.xvg
```

This creates the role-directed pairs `A(chain i) -> B(chain j), i != j`: for
100 complete chains there are `100 * 99 = 9900` channels. They are intentionally
directed because, for different residues A and B, `A_i-B_j` and `A_j-B_i` are
different observables. When both endpoint definitions are identical,
`--unique-entity-pairs` retains only one of `(i,j)` and `(j,i)`, producing
`N*(N-1)/2` inter-chain channels.

If each residue is represented by one CG bead, `--position atom` gives the same
one-point result and avoids center construction. To calculate the closest atoms
between the two residues instead, retain atom positions and use `--mode min`.
No per-chain `.ndx` groups are needed in any case.

If A and B mean residue *types* with multiple occurrences per chain rather than
two residue indices, select them with expressions such as `res ALA` and add
`--grouping residue`. Each residue becomes one entity keyed by
`(chain ID, residue ID)`; intra-chain scope then forms A-residue/B-residue pairs
only within the same chain, while inter-chain scope forms them only across
different chains.

### Operators

`--mode` is required so an accidental pairing/reduction choice cannot silently
produce a plausible-looking result.

#### `pair`

Within every entity pair admitted by `--scope`, emitted positions are zipped in
stable order and their position counts must match. For `corresponding` scope,
the two entity collections must also have matching entity counts.

An endpoint containing exactly one entity that emits exactly one position may
broadcast across the entities on the other side, provided each of those
entities also emits one position. This permits one group COM/COG to be measured
against each molecular COM/COG. No other implicit broadcasting is allowed.

#### `min` and `max`

For every entity pair admitted by `--scope`, the operator chooses the minimum or
maximum norm over all position pairs inside that entity pair. Output ordering
follows GROMACS `pairdist`: the reference entity changes fastest, then the
selection entity, then the top-level job.

With `atom` grouping each entity pair contains only one candidate, so `min` and
`max` with `--scope all` both yield the complete pairwise distance set. There is
no additional public `all-pairs` mode.

### Exclusions

```text
--exclude {none,same-atom,same-molecule}   # default: none
```

Exclusion acts on candidate position pairs:

- `none` never infers an intra/inter-chain relationship;
- `same-atom` removes candidates whose underlying atom provenance overlaps;
  and
- `same-molecule` removes candidates whose underlying topology molecule sets
  overlap.

In `pair` mode, positions are zipped first and an excluded matched pair is
removed without re-pairing the remaining positions. In `min`/`max` mode,
candidates are filtered before reduction. An entity pair with no remaining
candidate is omitted and recorded in output metadata. A whole job with no
channels is an error. This supports generic intermolecular calculations without
restoring `odist`'s hard-coded chain rule.

## Geometry and PBC semantics

Coordinates are calculated internally in Angstrom because MDAnalysis supplies
them that way, and are converted exactly once to nm at the kernel boundary.
Time is reported in ns, matching current DROPPS analysis commands.

Periodic distance calculation is enabled by default:

```text
--pbc / --no-pbc       # default: --pbc
```

For a reference point `r` and selection point `s`, the displacement is defined
as

\[
\Delta r = \operatorname{minimum\_image}(s-r)
\]

and the scalar distance is `norm(delta_r)`. Components are the Cartesian
`dx`, `dy`, and `dz` of this same vector in nm. Orthorhombic and triclinic boxes
use the same minimum-image implementation. In `max` mode the maximum is taken
over minimum-image distances, not over arbitrary periodic replicas.

For `min`/`max`, the component vector belongs to the candidate that realizes
the selected scalar extremum. Exact ties are broken by the stable topology and
selection order so results are reproducible.

COG/COM construction under PBC requires an explicit image convention:

1. Partition an entity by topology molecule.
2. Reconstruct each represented complete molecule along its topology bond graph
   using minimum-image bond vectors. The existing RMSD unwrapping plan can be
   reused.
3. For a multi-molecule entity, shift each reconstructed molecular chunk so its
   selected-atom COG lies in the nearest image of the first chunk's COG.
4. Calculate COG or COM from those compact-image coordinates.
5. Apply a final minimum image to the vector between the two emitted centers.

The anchor convention is deterministic and wrapping-invariant for compact
groups. A multi-molecule group whose extent approaches half a box is physically
image-ambiguous; the command records `center_image=first-molecule-anchor` in
metadata and prints a warning when a compact-image span reaches half the
shortest box height. Users analyzing an intentionally extended group should use
a suitably unwrapped trajectory with `--no-pbc`.

`com` requires finite, non-negative atom masses and positive total mass for
every entity. Missing/invalid boxes, disconnected multibead molecules needed
for a center, empty selections, and non-finite coordinates are explicit errors.
Plain atom-atom distances do not require bonded molecular reconstruction.

## Outputs

At least one output must be requested. All output files include the selections,
job labels, grouping, position types, operator, exclusions, PBC convention,
time window, and stable channel labels in metadata.

### Individual time series: `-oall/--output-all`

One XVG series per distance channel:

```text
time_ns  distance_0_nm  distance_1_nm ...
```

### Averaged time series: `-oav/--output-average`

Two XVG series per top-level job: the arithmetic mean and population standard
deviation across that job's chain/entity channels in the current frame. This is
the ensemble evolution over chains, not a different distance definition, and
does not require writing individual channel time series.

### Components: `-oxyz/--output-components`

Three XVG series per channel, ordered `dx`, `dy`, `dz`. Components are emitted
for `pair`, `min`, and `max`; for reductions they follow the extremizing pair.

### Statistics: `-ostat/--output-statistics`

```text
--statistics-scope {pooled,channel,both}   # default: pooled
```

The CSV schema is:

```text
scope, job, mode, entity_scope, reference_position, selection_position,
exclusion, pbc, channel, reference_entity, selection_entity,
samples, mean_nm, std_nm, min_nm, max_nm
```

- `pooled` writes one row per job over every `(frame, entity-pair)` sample. For
  the 100-chain inter-chain example, this summarizes all `9900 * n_frames`
  values.
- `channel` writes one row per entity pair over trajectory frames: 100 rows for
  the intra-chain example or 9900 rows for the directed inter-chain example.
- `both` writes both levels.

`std_nm` is a population standard deviation (`ddof=0`). Welford accumulation
avoids retaining the trajectory-sized result matrix. Pooled statistics require
O(jobs) state; channel statistics require O(channels) state.

### Histogram: `-oh/--output-histogram`

```text
--bin-width FLOAT                  # nm, default 0.01
--hist-min FLOAT                   # optional inclusive lower edge
--hist-max FLOAT                   # optional exclusive upper edge
--hist-scope {job,channel}         # default: job
--hist-normalization {probability,count,density}  # default: probability
```

Job scope pools all channels and frames within each job, matching the useful
GROMACS behavior of one distribution per selection. Channel scope produces one
distribution for every channel. All series use common bin edges. Without an
explicit range, the first pass obtains the global finite min/max and a second
trajectory pass fills bins; memory remains independent of frame count.

## Implementation structure

```text
dropps/analysis/distance.py       CLI, selection resolution, frame loop, writers
dropps/analysis/distance_core.py  plans, position construction, PBC kernels,
                                  reducers, streaming statistics/histograms
```

Core types should be immutable dataclasses:

- `EndpointPlan`: selection atoms, ordered entity partitions, position type,
  weights, provenance, and labels;
- `DistanceJobPlan`: two endpoints, entity scope, operator, exclusion, lazy
  entity-pair mapping, exact channel count, and labels;
- `FrameResultBlock`: a bounded channel slice with scalar values `(B,)` and
  displacement vectors `(B, 3)`; and
- `RunningStatistics` / `HistogramAccumulator`: streaming output state.

The numerical core accepts NumPy arrays and box dimensions and has no trajectory,
CLI, progress-bar, or file-writing dependencies. Pair mode is vectorized O(N).
Inter-chain entity products and min/max atom products are both traversed in
bounded tiles instead of constructing unbounded `N x M` channel or
`N x M x 3` vector arrays. Future cutoff/grid acceleration can be added behind
the same reducer API.

All endpoint positions are constructed once per frame before entity-pair tiles
are visited. Complete topology chains required by overlapping COG/COM endpoints
are unioned into one unwrap plan, so a chain is reconstructed at most once per
frame rather than once for every one of its inter-chain partners.

The frame loop streams XVG fields and aggregation blocks directly. Pooled
statistics, pooled histograms, and per-frame ensemble summaries therefore do
not materialize all inter-chain channels. Per-channel statistics use
O(channels) state only when explicitly requested. Automatic histograms use a
second pass rather than an `n_frames x n_channels` allocation.

Before reading frames, the planner prints reference/selection entity counts,
the exact channel count, computational pair count, and estimated detailed-output
width. `-oall`, `-oxyz`, channel-scoped statistics, and channel-scoped
histograms above 100,000 channels require `--allow-large-output`; pooled
statistics and histograms remain available without that override. The command
also warns when the estimated detailed time-series size exceeds 1 GiB. This
guards files with millions of columns while preserving large-chain ensemble
analysis.

## Migration

- Register and document only `distance` as the supported public interface.
- Keep `idist` and `odist` for one compatibility release as deprecated argument
  adapters into the shared planner/kernel; they must contain no distance
  calculation code.
- The `idist` adapter converts each legacy two-atoms-per-chain group into
  chain-grouped fixed endpoint pairs.
- The `odist` adapter builds atom-grouped pairwise jobs with
  `scope=inter-chain`.
- Legacy adapters preserve their old no-PBC default and output layout where
  possible. The new `distance` command uses PBC by default.
- Remove both aliases in the next breaking release after the migration warning.

## Validation plan

Unit tests for `distance_core.py`:

- ordered fixed pairs, mismatched pair validation, singleton broadcasting;
- atom/COG/COM positions and invalid masses;
- all/chain/residue/atom grouping with repeated residue IDs in molecules;
- corresponding, intra-chain, inter-chain, all, and unique entity-pair scopes;
- missing selections in some chains without ordinal mispairing;
- minimum and maximum reducers against brute-force NumPy results;
- same-atom and same-molecule exclusions;
- scalar/component consistency and deterministic extrema ties;
- orthorhombic and triclinic minimum images;
- bonded molecule reconstruction and multi-molecule anchored centers;
- Welford statistics and all histogram normalizations; and
- tiled results identical to untiled small reference calculations.

Command/integration tests:

- CLI examples and required mode/output validation;
- reference broadcasting and reference/selection zipping;
- selection-order preservation for simple index groups;
- frame selection through `-b`, `-e`, and `-dt`;
- stable channel labels and metadata in all outputs;
- output streaming on a multi-frame trajectory; and
- 100-chain count checks: 100 intra-chain and 9900 directed inter-chain
  channels;
- pooled results from tiled blocks equal explicit all-channel references; and
- numerical equivalence of legacy `idist`/`odist` adapters on representative
  inputs.

## First implementation boundary

The first version uses static DROPPS selections and topology-defined molecule
and residue membership. Dynamic selections, nearest-neighbor identity tracking
between frames, pair lifetime analysis, contact counts, distance matrices, and
cutoff/grid acceleration are outside this change. The planned data model leaves
room for them without adding more distance commands.
