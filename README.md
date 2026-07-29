# DROPPS

Distributed Rapid Operation Platform for Phase-separation Simulations.

## Overview

DROPPS is a command-line toolkit for coarse-grained biomolecular simulation workflows, including:

- System modeling and preprocessing
- Simulation packaging and running
- Trajectory conversion and analysis

CLI entrypoint:

```bash
dps
```

## Website

For tutorials, updates, and online resources, visit:

- https://dropps.online

## Installation

Requirements:

- Python `>=3.10`
- OpenMM `8.1.*`
- MDAnalysis `>=2.7.0`

Install from source:

```bash
pip install dist/dropps-0.4.1-py3-none-any.whl
```

or install dependencies manually:

```bash
pip install -r requirements.txt
pip install -e .
```

## Basic Usage

Show command list:

```bash
dps help commands
```

Show help for one command:

```bash
dps help <command>
# or
dps <command> -h
```

## Commands

<!-- BEGIN AUTO COMMANDS -->
_This section is auto-generated from source. Do not edit manually._

### Modeling

- `pdb2dps`: Generate coarse-grained PDB conformations and an ITP topology from a protein sequence.
  Required args: `-s/--sequence`
- `genelastic`: Add a distance-based elastic network to an ITP topology.
  Required args: `-f/--structure`, `-p/--topology`, `-o/--output`, `-er/--elastic-residues`
- `genmesh`: Pack one or more molecule types onto a three-dimensional simulation-box mesh.
  Required args: `-f/--structure`, `-p/--topology`, `-n/--number`
- `addangle`: Add angle restraints to an ITP topology.
  Required args: `-ip/--input-topology`, `-op/--output-topology`, `-al/--angle-list`
- `editconf`: Resize a PDB simulation box and optionally unwrap coordinates along selected axes.
  Required args: `-f/--structure`, `-o/--output`
- `grompp`: Build a DROPPS run-input file from structure, topology, and simulation parameters.
  Required args: `-f/--structure`, `-p/--topology`, `-m/--parameter`, `-o/--output`
- `modifyres`: Apply residue modifications consistently to an ITP topology and PDB structure.
  Required args: `-ip/--input-topology`, `-if/--input-structure`, `-op/--output-topology`, `-of/--output-structure`, `-m/--modifications`
- `trjconv`: Select, transform, and convert a DROPPS trajectory.
  Required args: `-s/--run-input`, `-f/--input`, `-o/--output`
- `gsd2xtc`: Convert a GSD trajectory to XTC format.
  Required args: `-f/--input`, `-o/--output`

### Simulation

- `mdrun`: Run or resume a molecular-dynamics simulation from a DROPPS run-input file.
  Required args: `-s/--run-input`, `-o/--output-prefix`
- `rerun`: Re-evaluate configurational observables from an XTC trajectory.
  Required args: `-s/--run-input`, `-f/--input`, `-o/--output`

### Analysis

- `make_ndx`: Create or extend an atom-group index file interactively.
  Required args: `-s/--run-input`, `-o/--output`
- `extract`: Extract one trajectory frame as a PDB structure.
  Required args: `-s/--run-input`, `-f/--input`, `-o/--output`, one of `-t/--time` or `-fm/--frame`
- `check`: Inspect DROPPS run, trajectory, and index files before analysis.
  Required args: `-s/--run-input`
- `energy`: Interactively select observables from a DROPPS EDR and export XVG.
  Required args: `-f/--input`
- `density`: Calculate a one-dimensional mass- or charge-density profile.
  Required args: `-s/--run-input`, `-f/--input`, `-o/--output`
- `coexistence`: Analyze dense/dilute coexistence and interface width in a planar slab trajectory.
  Required args: `-s/--run-input`, `-f/--input`, `-o/--output`
- `exchange`: Analyze molecular exchange and phase residence times across a planar slab interface.
  Required args: `-s/--run-input`, `-f/--input`, `-o/--output-prefix`
- `phase-msd`: Calculate phase-conditioned molecular MSD with phase-translation removal in a planar slab.
  Required args: `-s/--run-input`, `-f/--input`, `-o/--output`
- `gyrate`: Calculate chain radii of gyration and optional distributions.
  Required args: `-s/--run-input`, `-f/--input`
- `angle`: Calculate backbone-angle profiles and time series for equivalent protein chains.
  Required args: `-s/--run-input`, `-f/--input`
- `idist`: Calculate intra-chain distance profiles for one or more bead pairs.
  Required args: `-s/--run-input`, `-f/--input`
- `odist`: Calculate distances between corresponding beads in two chain groups.
  Required args: `-s/--run-input`, `-f/--input`
- `contact`: Calculate residue-level contact maps and contact-number time series.
  Required args: `-s/--run-input`, `-f/--input`, `-cs/--cutoff-scheme`
- `contact.ng`: Calculate residue-level contact maps with memory-efficient aggregation.
  Required args: `-s/--run-input`, `-f/--input`, `-cs/--cutoff-scheme`
- `timecorr`: Calculate residue-contact correlation functions and lifetime matrices.
  Required args: `-s/--run-input`, `-f/--input`, `-cs/--cutoff-scheme`
- `cstat`: Calculate residue-level statistics from a precomputed contact map.
  Required args: `-m/--map-input`, `-s/--run-input`, `-o/--output`
- `msd`: Calculate the mean-square displacement of a selected atom group.
  Required args: `-s/--run-input`, `-f/--input`, `-o/--output`
- `assembly`: Analyze the formation, size, composition, and shape of molecular assemblies.
  Required args: `-s/--run-input`, `-f/--input`
- `pdb2bond`: Add PDB CONECT records from a DROPPS run file or system topology.
  Required args: one of `-s/--run-input` or `-p/--topology`, `-f/--input`, `-o/--output`
- `surftension`: Calculate planar-interface surface tension from a DROPPS TPR and XTC trajectory.
  Required args: `-s/--run-input`, `-f/--input`, `-o/--output`
- `viscosity-gk`: Calculate zero-shear viscosity from a DROPPS runtime pressure tensor.
  Required args: `-f/--input`, `-o/--output`

<!-- END AUTO COMMANDS -->





## Typical Workflow

1. Build molecules and system.
2. Prepare simulation runtime files.
3. Run simulation.
4. Analyze trajectories.

Example:

```bash
# 1) Build single-chain model
dps pdb2dps -s "MSEQNNTEMTFQIQRIYTKDISFEAPNAPHVFQKDW" -oc chain.pdb -op chain.top

# 2) Build multi-molecule system
dps genmesh -f chain.pdb -p chain.itp -n 100 -oc system.pdb -op system.top

# 3) Prepare runtime input
dps grompp -f system.pdb -p system.top -m nvt.mdp -o run.tpr

# 4) Run
dps mdrun -s run.tpr -o run

# 5) Analyze
dps density -s run.tpr -f run.xtc -o density.xvg
```

### Configure `mdrun`

Scientific settings such as the integration time step, temperature, pressure,
and output sampling intervals remain in the TPR generated by `grompp`.
Operational settings can be selected when `mdrun` starts. For example:

```bash
dps mdrun -s run.tpr -deffnm run \
  --platform CUDA -gpu_id 0 --precision mixed \
  -cpt 10 -maxh 23.5
```

`-deffnm` is an alias of the existing `-o/--output-prefix`; the meaning of
`-o` has not changed. The platform can be `auto`, `CUDA`, `OpenCL`, `CPU`, or
`Reference`. Use `-nt` to select CPU-platform threads. GPU device and precision
can be set with `-gpu_id` and `--precision`.

Individual output paths can override the prefix defaults:

```bash
dps mdrun -s run.tpr -o run \
  -x trajectory.xtc -e thermo.edr -g progress.log \
  -cpo state.chk -c final.pdb --stress-output pressure.xvg
```

`-nsteps` temporarily overrides the production target stored in the TPR, while
`--seed` overrides its random seed for a new run. Neither option modifies the
TPR. The resolved seed controls the Langevin integrator, velocity generation,
and Monte Carlo barostat.

Periodic checkpoints use wall-clock time. `-cpt` is measured in minutes and
defaults to 5; `-cpt 0` disables periodic saves. A final checkpoint is still
written after successful completion, a `-maxh` stop, or Ctrl-C. `-maxh` stops
after approximately 99% of the requested hours so a batch job has time to
write its checkpoint. Add `-cpnum` to retain step-numbered copies alongside
the latest checkpoint file.

### Resume an interrupted simulation

`mdrun` writes `<output-prefix>.chk` every five wall-clock minutes by default.
To continue the same run, keep the original run input and output prefix and use:

```bash
dps mdrun -s run.tpr -o run -cpi
```

With no path after `-cpi`, DROPPS loads `run.chk`. A checkpoint at another
location can be selected explicitly:

```bash
dps mdrun -s run.tpr -o run -cpi /path/to/run.chk
```

On restart, `nsteps` remains the target total number of production steps, not
the number of extra steps to add. DROPPS skips minimization and warming, runs
only the remaining production steps, and appends to the existing XTC, fixed
progress log, CSV-format EDR, and stress-tensor outputs. If any output is newer
than the checkpoint, it is first
rolled back to the checkpoint step so the resumed files do not contain
duplicate or out-of-order samples. A final checkpoint is also written after a
successful run or a Ctrl-C interruption.

To keep the original outputs unchanged, use `-noappend`. Continuation outputs
are then written as `run.part0002.xtc`, `run.part0002.edr`, and so on, while the
latest checkpoint remains `run.chk`:

```bash
dps mdrun -s run.tpr -o run -cpi -noappend
```

OpenMM binary checkpoints require the same system definition, OpenMM version,
compute platform, and compatible hardware. Keep the original TPR for an ordinary
continuation. When intentionally extending a run, regenerate a compatible TPR
whose simulation definition changes only by increasing `nsteps`.

### Progress log and energy output

The complete parameter reference, including units, inactive compatibility
keys, NVT/NPT examples, and pressure-output costs, is available in
[`docs/mdp-reference.md`](docs/mdp-reference.md).

`mdrun` writes runtime progress and thermodynamic state through separate data
streams. `<output-prefix>.log` has a fixed CSV schema containing step, time,
progress, speed, elapsed wall time, and estimated remaining time. Its file and
screen intervals are controlled by `nst-filelog` and `nst-screenlog`.

`<output-prefix>.edr` is a versioned, self-describing CSV table. It is not a
GROMACS binary EDR. Configure it in the MDP file, for example:

```ini
nst-energy  = 10000
energy-grps = potentialEnergy,kineticEnergy,totalEnergy,temperature,boxX,boxY,boxZ,volume,density
```

Set `nst-energy = 0` to disable it. Available terms include energies,
temperature, box-vector lengths, volume, density, isotropic pressure, and the
six independent pressure-tensor components. Pressure terms invoke the costly
finite-strain pressure calculation and retain its orthorhombic-box and
unconstrained-bond requirements.

List stored terms without exporting them:

```bash
dps energy -f run.edr --list
```

With no `--terms`, `dps energy` prints the numbered terms and interactively
accepts numbers, names, unique name prefixes, or `all`; finish with an empty
line or `0`:

```bash
dps energy -f run.edr -o selected.xvg
```

For scripts, bypass the prompt with an explicit selection:

```bash
dps energy -f run.edr -o thermo.xvg \
  --terms temperature potentialEnergy volume \
  -b 1000 -e 5000 -dt 10
```

### Configurational rerun

`rerun` re-evaluates potential energy, box properties, and optionally the
configurational pressure tensor from saved XTC coordinates:

```bash
dps rerun -s run.tpr -f run.xtc -o rerun.edr \
  --terms potentialEnergy volume configPxx configPyy configPzz
```

XTC does not store velocities. Consequently, rerun EDR files cannot contain
kinetic energy, instantaneous temperature, the kinetic pressure tensor, or the
full instantaneous pressure tensor required for Green-Kubo viscosity. The
`configP*` names intentionally make that limitation explicit.

### Coexisting dense and dilute phases

Analyze an equilibrated, orthorhombic planar slab with two interfaces using:

```bash
dps coexistence \
  -s slab.tpr \
  -f slab.xtc \
  -o coexistence.xvg \
  -ref 0 \
  -sel 1 2 \
  --axis z \
  --bin-width 0.2 \
  --blocks 5
```

The reference group defines the dense slab. By default, each selected frame is
centered with the reference group's periodic first Fourier component, preventing
slab translation from broadening the average profile. DROPPS then fits the
reference density to a symmetric double-hyperbolic-tangent profile and reports
the dense and dilute densities, slab width, 10--90% interface width, fit
quality, and contiguous-block standard errors.

Three centering modes are available:

- `--center-mode frame` aligns every analyzed frame and is the default for
  coexistence concentrations.
- `--center-mode block` determines one center for each contiguous segment
  defined by `--blocks`; this retains within-block slab motion and provides a
  centering-sensitivity check for interface widths.
- `--center-mode none` applies no shift and requires a trajectory whose slab is
  already centered at the middle of the periodic box.

Compare `frame` and `block` for production interface-width estimates. Dense and
dilute plateau concentrations should be insensitive to the centering mode, while
the apparent width can increase when longer-wavelength slab motion is retained.

`coexistence.xvg` contains the centered mean profiles and the fitted reference
profile. A companion `coexistence.csv` contains dense- and dilute-plateau
densities for the reference and every `-sel` group, together with their block
errors and dense-to-dilute ratios. Use `-tp number` for selected-particle number
density in `nm^-3`; the default is mass density in `mg/mL`.

The method assumes one dense planar slab in an orthorhombic periodic cell. A
low fit R-squared value or low centering order can indicate insufficient
equilibration, a nearly homogeneous system, multiple domains, or an unsuitable
reference group. Use `--plateau-buffer` to change how far plateau averages stay
away from the fitted interfaces.

### Interphase exchange and residence times

Track molecule-resolved phase changes in the same planar-slab geometry using:

```bash
dps exchange \
  -s slab.tpr \
  -f slab.xtc \
  -n groups.ndx \
  -o exchange \
  -ref 0 \
  -sel 1 2 \
  --axis z \
  --center-mode frame \
  --interface-threshold 0.1 \
  --bin-width 0.2 \
  --blocks 5
```

The reference group is used only to locate the slab and fit its mean mass
density. Each `-sel` group is split into topology molecules (chains), and the
mass-weighted periodic center of mass of every selected molecule is tracked.
For partial-molecule selections, the center uses only the selected atoms.

The fitted double-tanh profile defines three spatial states. With the default
`--interface-threshold 0.1`, dense and dilute bulk states begin outside the
10--90% interfacial transition; positions between those cutoffs are classified
as `interface`. These fixed cutoffs are obtained from the trajectory-mean
profile rather than fitted separately on every frame. `--center-mode` has the
same `frame`, `block`, and `none` meanings as `coexistence`.

Two related but distinct records are produced:

- A residence is every continuous interval in one of the three spatial states.
  Initial intervals are marked left-censored and final intervals are marked
  right-censored.
- An exchange event is confirmed only when a molecule leaves one bulk phase and
  reaches the opposite bulk phase. An interface excursion that returns to the
  same bulk phase is not counted.

The output prefix creates `.profile.xvg`, `.states.csv`, `.events.csv`,
`.residence.csv`, `.survival.csv`, and `.summary.csv`. The survival file uses a
Kaplan--Meier estimate after excluding left-censored intervals while retaining
right-censored intervals. The summary reports occupancy, completed residence
statistics, the Kaplan--Meier median when observable, confirmed exchange counts,
and source-phase event rates per microsecond of exposure. Use `--no-states` to
omit the potentially large frame-by-frame state table.

Temporal resolution limits the kinetics: a molecule observed directly in the
opposite bulk state has an unresolved transition time, reported as zero with a
warning. Use a smaller trajectory output interval for interface-passage times
and short residences. The method assumes each selected molecule spans less than
half the box length along the interface normal so its periodic center can be
unwrapped unambiguously.

### Phase-resolved molecular dynamics

Calculate molecular MSD separately in dense and dilute regions with:

```bash
dps phase-msd \
  -s slab.tpr \
  -f slab.xtc \
  -n groups.ndx \
  -o phase_msd.xvg \
  -ref 0 \
  -sel 1 2 \
  --axis z \
  -t xy \
  --phases dense dilute \
  --conditioning both \
  --center-mode frame \
  --max-lag 100 \
  --blocks 5
```

Phase boundaries use the same trajectory-mean double-tanh fit as `coexistence`
and `exchange`. Centering is used only to decide whether each molecule is in the
dense, interface, or dilute region. Molecular dynamics are calculated from
periodically unwrapped laboratory-frame centers, so slab motion is not silently
removed by the spatial classification step.

For every group, phase, time origin, and lag, DROPPS calculates the
mass-weighted displacement of the eligible molecular cohort and subtracts it:

```text
delta r_intrinsic,i = delta r_i
                      - sum_j(m_j delta r_j) / sum_j(m_j)

MSD_raw = MSD_intrinsic + MSD_phase_translation
```

This removes coherent translation of the phase cohort in all dimensions chosen
by `-t`, including tangential drift that slab-normal recentering cannot remove.
The raw, intrinsic, and phase-translation terms are all retained in the CSV so
the correction can be audited.

Two membership definitions are available:

- `continuous` uses a molecule only if it remains in the requested phase for
  the complete interval.
- `intermittent` requires membership at both endpoints but allows departure and
  return between them.

The default `--conditioning both` reports both definitions, while the default
phases are `dense dilute`; add `interface` to `--phases` when enough interfacial
molecules are present. At least two eligible molecules are required at each time
origin by default. Increase `--min-particles` for a more stable phase-translation
estimate.

The main XVG contains intrinsic and raw MSD curves. A tidy companion CSV adds
the translation term, contiguous time-origin block errors, contributing-pair
counts, and the mean number of eligible molecules. A `.profile.xvg` companion
records the fitted density and phase cutoffs.

Use a fixed-volume trajectory when possible. With a varying box, pure affine
box rescaling is excluded from the unwrapped increments and a warning is issued
above 1% length variation. Sampling must also be frequent enough that a molecule
does not move more than half a box length between saved frames. Phase-specific
MSD becomes survivor-biased at long lag times, so always inspect the reported
pair and time-origin counts rather than fitting a diffusion coefficient blindly.

### Planar surface tension

Use an equilibrated, orthorhombic slab trajectory with two planar interfaces
and a fixed simulation box (`pcoulp = False`):

```bash
dps surftension \
  -s slab.tpr \
  -f slab.xtc \
  -o surface_tension.xvg \
  -b 500 \
  -e 3000 \
  -dt 1 \
  --axis z \
  --strain 1e-3 3e-4 1e-4 \
  --blocks 5
```

The command applies a volume-preserving affine strain to every selected frame,
re-evaluates the OpenMM potential energy, and calculates

```text
gamma = -L_normal / (n_interfaces * V) * dU/d(strain).
```

The XVG file contains the instantaneous estimate and running mean for every
requested strain. Its header reports the full-trajectory mean and the standard
error calculated from contiguous block means. Multiple strain values should
give consistent means before a result is treated as converged.

This configurational estimator assumes that the equilibrium-averaged kinetic
pressure anisotropy cancels. It does not support systems built with constrained
bonds because the constraint virial is not available from XTC coordinates.

### Runtime pressure tensor and zero-shear viscosity

Green–Kubo viscosity requires the microscopic pressure tensor, including
velocities and the configurational virial. An XTC trajectory does not contain
enough information, so enable runtime recording in the production MDP:

```ini
pcoulp          = False
bondtype        = bond
nst-stress      = 10
stress-strain   = 0.0003
stress-output   = auto
stress-platform = auto
stress-precision = double
```

`stress-output = auto` writes `<mdrun-output-prefix>.stress.xvg`. Recording is
attached after minimization and warming, so the file contains only production
samples. Each row contains time, volume, and `Pxx Pyy Pzz Pxy Pxz Pyz` in
`kJ mol^-1 nm^-3`.

The instantaneous tensor is evaluated as

```text
P_ab = (1/V) sum_i m_i v_ia v_ib - (1/V) dU/d(epsilon_ab).
```

DROPPS obtains the configurational derivative in an isolated OpenMM Context by
central finite differences: three diagonal deformations and three
volume-preserving simple shears. This needs 12 potential-energy evaluations per
reported sample, so `nst-stress` trades temporal resolution against runtime.
Check that the result is insensitive to `stress-strain` and the sampling
interval.

Calculate the zero-shear viscosity with:

```bash
dps viscosity-gk \
  -f run.stress.xvg \
  -o viscosity.xvg \
  --blocks 5
```

The output contains the shear relaxation modulus and its running integral. The
estimator uses all six independent tensor components:

```text
G(t) = V/(5RT)  * [Cxy(t) + Cxz(t) + Cyz(t)]
     + V/(30RT) * [Cxx-yy(t) + Cxx-zz(t) + Cyy-zz(t)]

eta_0 = integral_0^infinity G(t) dt.
```

By default, DROPPS averages the last 20% of the available running integral and
reports a standard error from contiguous trajectory blocks. Inspect the curve
and specify a physically stable plateau for the final estimate, for example:

```bash
dps viscosity-gk \
  -f run.stress.xvg \
  -o viscosity.xvg \
  --max-lag 500 \
  --plateau-start 200 \
  --plateau-end 400 \
  --blocks 5
```

All correlation-time arguments are in ps. Use a fixed, orthorhombic NVT box;
NPT data and constrained bonds are rejected. Individual-particle Langevin
damping can change stress dynamics, so viscosity should also be checked for
thermostat/friction sensitivity. Convergence should be tested against total
trajectory length, block size, maximum lag, plateau window, stress sampling
interval, and finite-difference strain.

## Notes

- The command name is `dps` (not legacy `cgps`).
- For exact and current argument details, always use `dps <command> -h`.
- Some analysis commands expect consistent `.tpr`/`.xtc`/`.ndx` group definitions.
- Refresh the auto-generated command catalog with: `python3 scripts/update_readme_commands.py`.

## Author

- Yiming Tang
- ymtang@fudan.edu.cn
