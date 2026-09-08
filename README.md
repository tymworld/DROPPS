# DROPPS

Distributed Rapid Operation Platform for Phase-separation Simulations.

[![Open Quickstart in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tymworld/DROPPS/blob/v1.0.0/examples/colab/00_DROPPS_1_0_Quickstart.ipynb)

## Overview

DROPPS is a command-line toolkit for coarse-grained biomolecular simulation workflows, including:

- System modeling and preprocessing
- Simulation packaging and running
- Trajectory conversion and analysis

CLI entrypoint:

```bash
dps
```

## Google Colab tutorials

The versioned Colab suite covers model construction, the compact-box NPT →
z-axis ×10 expansion → elongated-box NVT protocol, trajectory preparation, and
phase-separation analysis. Its default demonstration system contains 80 copies
of `FWFWFWFWFWFWFWFW`; both Lennard-Jones and Coulomb cutoffs default to
1.5 nm. The analysis notebook always writes a z-axis density profile as
`density.xvg` and `density_z.png`.

| Tutorial | Open version 1.0 in Colab |
|---|---|
| End-to-end quickstart | [Open](https://colab.research.google.com/github/tymworld/DROPPS/blob/v1.0.0/examples/colab/00_DROPPS_1_0_Quickstart.ipynb) |
| Model and system builder | [Open](https://colab.research.google.com/github/tymworld/DROPPS/blob/v1.0.0/examples/colab/01_DROPPS_1_0_Model_and_System_Builder.ipynb) |
| NPT → slab → NVT simulation | [Open](https://colab.research.google.com/github/tymworld/DROPPS/blob/v1.0.0/examples/colab/02_DROPPS_1_0_Simulation.ipynb) |
| Trajectory preparation | [Open](https://colab.research.google.com/github/tymworld/DROPPS/blob/v1.0.0/examples/colab/03_DROPPS_1_0_Trajectory_Preparation.ipynb) |
| Phase-separation analysis | [Open](https://colab.research.google.com/github/tymworld/DROPPS/blob/v1.0.0/examples/colab/04_DROPPS_1_0_Phase_Separation_Analysis.ipynb) |

The short mode validates the workflow. The manuscript-scale 300 ns NPT and
3 μs NVT runs should use checkpoints and a persistent GPU/HPC environment;
they are not expected to finish in one free Colab session.

## Scope of the accompanying manuscript

The submitted article describes the following command-line workflow:

- Molecular and system construction: `pdb2dps`, `genelastic`, `addangle`,
  `modifyres`, `genmesh`, and `editconf`
- Run preparation and execution: `grompp` and `mdrun`
- Trajectory preparation and validation: `trjconv`, `make_ndx`, `check`, and
  `gsd2xtc`
- Trajectory analysis: `density`, `contact`, `idist`, `odist`, `gyrate`,
  `angle`, `msd`, and `assembly`

Additional commands remain available for backward compatibility and ongoing
research. They are separated in
[`UNDOCUMENTED_COMMANDS.md`](UNDOCUMENTED_COMMANDS.md) for traceability. The
authors have confirmed that these commands remain in DROPPS 1.0; the article
does not need to describe every auxiliary CLI utility.

DROPPS uses the manuscript-defined `.ff`, `.itp`, `.top`, `.mdp`, and portable
`.tpr` formats for model and run inputs; `.pdb` and `.xtc` for coordinates and
trajectories; and `.xvg`, `.xpm`, and `.dat` for analysis output.

## Website

For tutorials, updates, and online resources, visit:

- https://dropps.online

## Installation

Requirements:

- Python `>=3.9`
- Python 3.10+ current/default stack: NumPy `>=2,<3`, OpenMM `>=8.5,<9`,
  and MDAnalysis `>=2.9` (tested with
  NumPy `2.5.1` and OpenMM `8.5.2`)
- Python 3.9 compatibility and legacy migration stack: NumPy `>=1.26.4,<2`,
  OpenMM `>=8.1,<8.2`, and MDAnalysis `>=2.7,<2.8`

Install the immutable 1.0 release wheel directly:

```bash
pip install 'dropps[latest] @ https://github.com/tymworld/DROPPS/releases/download/v1.0.0/dropps-1.0-py3-none-any.whl'
```

Install from source:

```bash
python scripts/install_dropps.py
```

The installer uses `nvidia-smi` to detect visible NVIDIA GPUs. It selects the
OpenMM CUDA 12 or CUDA 13 package supported by the installed driver, installs
DROPPS, and creates a real CUDA Context to verify the result. If a machine is
expected to have a GPU, make detection mandatory:

```bash
python scripts/install_dropps.py --require-cuda
```

To install a release wheel from a source checkout or an unpacked source archive,
pass it explicitly with `--package`. For a manual installation, choose the
matching extra:

```bash
pip install './dist/dropps-1.0-py3-none-any.whl[cuda12]'
# or, with a CUDA 13-compatible NVIDIA driver:
pip install './dist/dropps-1.0-py3-none-any.whl[cuda13]'
```

For a CPU/OpenCL development installation:

```bash
pip install -e '.[latest]'
```

OpenMM 8.1's XTC extension was built against the NumPy 1.x ABI and cannot be
imported with NumPy 2.x. To open an old native checkpoint in its original
OpenMM generation before migration, use the constrained legacy stack:

```bash
pip install -e '.[openmm81]'
```

The base package metadata spans both tested stacks; the extras above constrain
the resolver to a valid pair instead of allowing OpenMM 8.1 with NumPy 2.x.

## Reproducibility and input validation

Random construction is reproducible by default. `pdb2dps` and shuffled
`genmesh` runs use seed `1215`; record a different value with `--seed` when
independent replicas are needed. `grompp` validates force-field compatibility,
topology completeness, bead ordering, masses and charges, orthorhombic periodic
boxes, and the minimum-image cutoff requirement before writing a `.tpr` file.
The `.tpr` container also records source-file checksums and the OpenMM system in
portable XML form.

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

Run the publication contract and smoke tests from the source tree with the
environment in which DROPPS is installed:

```bash
python -m unittest discover -s tests -v
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
- `convert-tpr`: Convert a trusted legacy DROPPS TPR into the portable TPR v2 format.
  Required args: `-s/--run-input`, `-o/--output`
- `modifyres`: Apply residue modifications consistently to an ITP topology and PDB structure.
  Required args: `-ip/--input-topology`, `-if/--input-structure`, `-op/--output-topology`, `-of/--output-structure`, `-m/--modifications`
- `trjconv`: Select, transform, and convert a DROPPS trajectory.
  Required args: `-s/--run-input`, `-f/--input`, `-o/--output`
- `gsd2xtc`: Convert a GSD trajectory to XTC format.
  Required args: `-f/--input`, `-o/--output`

### Simulation

- `mdrun`: Run or resume a molecular-dynamics simulation from a DROPPS run-input file.
  Required args: `-s/--run-input`, `-o/--output-prefix/-deffnm`
- `rerun`: Re-evaluate configurational observables from an XTC trajectory.
  Required args: `-s/--run-input`, `-f/--input`, `-o/--output`

### Analysis

- `make_ndx`: Create or extend an atom-group index file interactively.
  Required args: `-s/--run-input`, `-o/--output`
- `check`: Inspect DROPPS run, trajectory, and index files before analysis.
  Required args: `-s/--run-input`
- `energy`: Interactively select observables from a DROPPS EDR and export XVG.
  Required args: `-f/--input`
- `density`: Calculate a one-dimensional mass- or charge-density profile.
  Required args: `-s/--run-input`, `-f/--input`, `-o/--output`
- `exchange`: Analyze molecular exchange and phase residence times across a planar slab interface.
  Required args: `-s/--run-input`, `-f/--input`, `-o/--output-prefix`
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
- `cstat`: Calculate residue-level statistics from a precomputed contact map.
  Required args: `-m/--map-input`, `-s/--run-input`, `-o/--output`
- `msd`: Calculate the mean-square displacement of a selected atom group.
  Required args: `-s/--run-input`, `-f/--input`, `-o/--output`
- `rmsd`: Calculate PBC-aware RMSD with configurable fitting and output granularity.
  Required args: `-s/--run-input`, `-f/--input`, `-o/--output`
- `assembly`: Analyze the formation, size, composition, and shape of molecular assemblies.
  Required args: `-s/--run-input`, `-f/--input`
- `pdb2bond`: Add PDB CONECT records from a DROPPS run file or system topology.
  Required args: one of `-s/--run-input` or `-p/--topology`, `-f/--input`, `-o/--output`

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

`density` evaluates every selected frame with its own orthorhombic box and
volume, then averages the per-frame profiles. `-dt` is an optional physical
sampling interval; omitting it analyzes every saved frame. The default
`--center-mode frame` centers every analyzed frame using the selected
reference-density profile before averaging.

### TPR portability and legacy conversion

New `grompp` runs write TPR v2, a versioned ZIP container. The OpenMM `System`
is stored as XML and the remaining data as JSON/NumPy arrays, so the file does
not depend on Python pickle class paths. TPR v2 is intended for moving an older
OpenMM 8.x run input to a newer OpenMM 8.x installation; a file produced by a
newer OpenMM release is not guaranteed to open in an older release.

DROPPS also reads trusted legacy pickle TPRs from the generation that uses
`Atomtype_AH_DH` and `Atomtype_WF_DH`. Convert one before moving environments:

```bash
dps convert-tpr -s legacy.tpr -o portable.tpr
```

Because pickle can execute code while loading, convert only files from a
trusted source. Still older TPRs that reference the original `Atomtype` class
are intentionally unsupported and must be regenerated from their PDB, TOP/ITP,
and MDP inputs.

### Convert and subset trajectories

`trjconv` is the single trajectory-conversion workflow for XTC and PDB. Frame
selection is time-based: use equal `-b` and `-e` values to write the saved frame
nearest that time.

```bash
dps trjconv -s run.tpr -f run.xtc -o frame.pdb \
  -b 100 -e 100 -sel "group 0"
```

Use `-b/-e/-dt` for a time range. Available PBC modes are `none`, `whole`,
`atom`, `res`, `mol`, and `nojump`; unlike the old implementation, `whole`
only reconstructs bonded molecules while `mol` also packs their centers into
the primary cell. Geometric, mass, or dense-slab centering and reference fits
can be composed in a fixed order:

```bash
dps trjconv -s slab.tpr -f slab.xtc -o centered.xtc \
  -sel "group 0" -pbc mol \
  --center dense --center-select "mol PROT" --center-axis z
```

PDB ranges are written as multi-model files; use `--separate` for one numbered
PDB per selected time and `--conect` to include selected topology bonds.

### Configure `mdrun`

Scientific settings such as the integration time step, temperature, pressure,
and output sampling intervals remain in the TPR generated by `grompp`.
Operational settings can be selected when `mdrun` starts. For example:

```bash
dps mdrun -s run.tpr -deffnm run \
  --platform CUDA -gpu_id 0 --precision mixed \
  -cpt 10 -maxh 23.5
```

Use explicit `--platform CUDA` for production jobs that must use NVIDIA GPUs.
Unlike `--platform auto`, it fails immediately if CUDA cannot be initialized
instead of falling back to another OpenMM platform.

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

Periodic restarts use wall-clock time. `-cpt` is measured in minutes and
defaults to 5; `-cpt 0` disables periodic saves. Each save produces a native
checkpoint (`.chk`), a portable OpenMM State (`.state.xml`), and a checksum/
compatibility manifest (`.restart.json`). A final set is still written after
successful completion, a `-maxh` stop, or Ctrl-C. `-maxh` stops after
approximately 99% of the requested hours so a batch job has time to write its
restart files. Add `-cpnum` to retain step-numbered copies alongside the latest
files.

### Resume an interrupted simulation

`mdrun` writes `<output-prefix>.chk`, `<output-prefix>.state.xml`, and
`<output-prefix>.restart.json` every five wall-clock minutes by default.
To continue the same run, keep the original run input and output prefix and use:

```bash
dps mdrun -s run.tpr -o run -cpi
```

With no path after `-cpi`, DROPPS loads `run.chk`. A checkpoint at another
location can be selected explicitly:

```bash
dps mdrun -s run.tpr -o run -cpi /path/to/run.chk
```

DROPPS first tries the native `.chk`. If OpenMM rejects it because the version,
platform, or hardware changed, DROPPS validates and loads the paired
`.state.xml` automatically. The State preserves positions, velocities, box,
time, step, and context parameters, but not the integrator's internal random-
number state. Therefore native checkpoint continuation is exact, while State
fallback starts a statistically continuous but not bitwise-identical trajectory
segment. A State can also be selected directly with `-cpi run.state.xml`.

On restart, `nsteps` remains the target total number of production steps, not
the number of extra steps to add. A production checkpoint skips minimization
and warming; a checkpoint written during warming resumes its recorded ramp
segment before production. DROPPS appends to the existing XTC, fixed
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
compute platform, and compatible hardware. Keep the complete three-file restart
set. When intentionally extending a run, use `-nsteps` or regenerate a compatible
TPR whose simulation definition changes only by increasing `nsteps`.

An old standalone `.chk` has no portable fallback. Load it once with its
original OpenMM/DROPPS environment and stop normally (for example with a very
short `-maxh`) to produce the new `.state.xml` and `.restart.json` pair before
upgrading.

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

### Configurable molecular/selection RMSD

Select one region across all molecule copies; DROPPS partitions it by topology
molecule and handles PBC before applying the requested fit:

```bash
dps rmsd \
  -s run.tpr \
  -f run.xtc \
  -o rmsd.xvg \
  --select "(mol PROT) & (resid 20-80)"
```

The complete bonded molecule is reconstructed through the periodic box before
the selected region is extracted. `--fit-mode none` preserves physical
translation and rotation, `molecule` (the default) applies a separate Kabsch
fit to every molecule, and `selection` applies one fit to all atoms in each
selection. PBC-equivalent molecule images are normalized in every mode, so box
crossings do not create false RMSD jumps.

`--output-mode molecule` (the default) reports every molecule and puts its mean
and population standard deviation in the XVG. `--output-mode selection` pools
all weighted atomic residuals into one RMSD per selection. Thus molecule fit
plus selection output is a weighted residual RMSD, not a simple average of
molecular RMSDs; selection fit plus molecule output partitions the one global
fit's residuals by molecule.

By default each molecule is compared with its own conformation in the first
analyzed frame; use `-ref` to choose another reference time. Equal atom weights
are the default, while `--mass-weighted` uses masses for fitting and RMSD.

The XVG contains the molecular mean and population standard deviation for each
selection. A matching CSV follows the selected output scope; add
`--summary-only` when that table would be too large. Multiple index groups can
be passed once with `-sel`, and `--select` can be repeated for several regions.
No per-molecule index groups are needed.

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
profile rather than fitted separately on every frame. `--center-mode` uses
the same `frame`, `block`, and `none` meanings as `density`.

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

### Runtime pressure tensor

Some transport analyses require the microscopic pressure tensor, including
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

For the leapfrog/Langevin-middle integrator, DROPPS synchronizes the stored
half-step velocities to the current-position time slice with the current force
before evaluating both kinetic energy and this kinetic pressure term.

DROPPS obtains the configurational derivative in an isolated OpenMM Context by
central finite differences: three diagonal deformations and three
volume-preserving simple shears. This needs 12 potential-energy evaluations per
reported sample, so `nst-stress` trades temporal resolution against runtime.
Check that the result is insensitive to `stress-strain` and the sampling
interval.

## Notes

- The command name is `dps` (not legacy `cgps`).
- For exact and current argument details, always use `dps <command> -h`.
- Some analysis commands expect consistent `.tpr`/`.xtc`/`.ndx` group definitions.

## License

DROPPS is open-source software distributed under the Apache License 2.0. See
[`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

## Author

- Yiming Tang
- ymtang@fudan.edu.cn
