# DROPPS 1.0.0

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

Download the wheel or source archive from the [1.0.0 release](https://github.com/tymworld/DROPPS/releases/tag/v1.0.0). For a current CPU installation:

```bash
python -m pip install --upgrade --force-reinstall "dropps[latest] @ https://github.com/tymworld/DROPPS/releases/download/v1.0.0/dropps-1.0.0-py3-none-any.whl"
```

For NVIDIA GPUs, replace `latest` with `cuda12` or `cuda13` according to the driver, or use the hardware-aware source installer below.

Requirements:

- Python `>=3.9`
- Python 3.10+ current/default stack: NumPy `>=2,<3`, OpenMM `>=8.5,<9`,
  and MDAnalysis `>=2.9` (tested with
  the runtime versions recorded in the [release validation](docs/release-checklist.md))
- Python 3.9 compatibility and legacy migration stack: NumPy `>=1.26.4,<2`,
  OpenMM `>=8.1,<8.2`, and MDAnalysis `>=2.7,<2.8`

The replacement uses the same release tag. If you installed the earlier `1.0`
wheel, reinstall explicitly: Python packaging treats `1.0` and `1.0.0` as the
same version.

Install from source:

```bash
git clone --branch v1.0.0 https://github.com/tymworld/DROPPS.git
cd DROPPS
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
pip install './release/1.0.0/dropps-1.0.0-py3-none-any.whl[cuda12]'
# or, with a CUDA 13-compatible NVIDIA driver:
pip install './release/1.0.0/dropps-1.0.0-py3-none-any.whl[cuda13]'
```

For a CPU/OpenCL development installation:

```bash
pip install -r requirements.txt
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

Documentation:

- [Complete reference: all 43 commands](docs/command-reference.md)
- [User guide / 中文用户指南](docs/user-guide.md)
- [Five Google Colab workflows](examples/colab/README.md)
- [Architecture](docs/architecture.md)
- [Developer guide](docs/developer-guide.md)
- [1.0.0 changelog](CHANGELOG.md)
- [Release checklist](docs/release-checklist.md)

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

Every command accepts `-db/--debug` before or after the command name:

```bash
dps --debug mdrun -s run.tpr -o run
dps mdrun -s run.tpr -o run -db
```

Runtime messages use consistent levels: blue `[INFO]`, green `[OK]`, yellow
`[WARNING]`, red `[ERROR]`, and dim cyan `[DEBUG]`. Colors are enabled only for
interactive terminals, so redirected output remains plain text. Set `NO_COLOR=1`
or `DROPPS_COLOR=never` to disable colors, and `DROPPS_COLOR=always` to retain
them when piping to a color-aware consumer.

Every public input and output option has a lowercase one- or two-letter short
form such as `-f`, `-o`, or `-op`, and every public option provides a lowercase
kebab-case long form such as `--run-input`. Every output short form starts with
`o`. Older lowercase nonstandard spellings remain accepted for 1.0.0
compatibility, but are omitted from help and new documentation; uppercase
option names are not part of the CLI.

Frequently used unambiguous options also receive mnemonic shortcuts when the
letters are free in that command, including `-c/--center`, `-se/--selection`
or `--select`, `-r/--reference`, `-dt/--delta-time`, `-tu/--time-unit`,
`-ca/--center-axis`, `-p/--pbc`, `-bw/--bin-width`, and `-ff/--forcefield`.
Short names are command-local; `dps COMMAND -h` is authoritative when two
meanings would compete for the same alias. For example:

```bash
dps trjconv -s run.tpr -f run.xtc -o centered.xtc \
  -se "group 0" -p mol -c mass -ca z
```

## Commands

<!-- BEGIN AUTO COMMANDS -->
Every public command is listed below with a minimal example and a link to its complete options, defaults, units, and workflow notes. See the [complete command reference](docs/command-reference.md) for task-based navigation.

### Modeling

- `help`: Show the command catalog or detailed help for one command.
  Required args: None
  [Full reference](docs/commands/help.md).

  ```bash
  dps help commands
  ```

- `pdb2dps`: Generate coarse-grained PDB conformations and an ITP topology from a protein sequence.
  Required args: `-s/--sequence`
  [Full reference](docs/commands/pdb2dps.md).

  ```bash
  dps pdb2dps -s MSEQNNTEMTFQIQRIYTKDISFEAPNAPHVFQKDW -o chain.pdb -op chain.itp
  ```

- `genelastic`: Add a distance-based elastic network to an ITP topology.
  Required args: `-f/--structure`, `-p/--topology`, `-o/--output`, `-r/--elastic-residues`
  [Full reference](docs/commands/genelastic.md).

  ```bash
  dps genelastic -f chain.pdb -p chain.itp -o elastic.itp -r elastic-groups.dat
  ```

- `genmesh`: Pack one or more molecule types onto a three-dimensional simulation-box mesh.
  Required args: `-f/--structure`, `-p/--topology`, `-n/--number`
  [Full reference](docs/commands/genmesh.md).

  ```bash
  dps genmesh -f chain.pdb -p chain.itp -n 100 -o system.pdb -op system.top
  ```

- `addangle`: Add angle restraints to an ITP topology.
  Required args: `-p/--input-topology`, `-op/--output-topology`, `-a/--angle-list`
  [Full reference](docs/commands/addangle.md).

  ```bash
  dps addangle -p chain.itp -op chain-angles.itp -a angles.dat
  ```

- `editconf`: Resize a PDB simulation box and optionally unwrap coordinates along selected axes.
  Required args: `-f/--structure`, `-o/--output`
  [Full reference](docs/commands/editconf.md).

  ```bash
  dps editconf -f system.pdb -o resized.pdb --multiply-x-axis 2 --multiply-y-axis 2 --multiply-z-axis 2
  ```

- `grompp`: Build a DROPPS run-input file from structure, topology, and simulation parameters.
  Required args: `-f/--structure`, `-p/--topology`, `-m/--parameter`, `-o/--output`
  [Full reference](docs/commands/grompp.md).

  ```bash
  dps grompp -f system.pdb -p system.top -m md.mdp -o run.tpr
  ```

- `convert-tpr`: Convert a trusted legacy DROPPS TPR into the portable TPR v2 format.
  Required args: `-s/--run-input`, `-o/--output`
  [Full reference](docs/commands/convert-tpr.md).

  ```bash
  dps convert-tpr -s legacy.tpr -o portable.tpr
  ```

- `modifyres`: Apply residue modifications consistently to an ITP topology and PDB structure.
  Required args: `-p/--input-topology`, `-f/--input-structure`, `-op/--output-topology`, `-o/--output-structure`, `-m/--modifications`
  [Full reference](docs/commands/modifyres.md).

  ```bash
  dps modifyres -p chain.itp -f chain.pdb -op modified.itp -o modified.pdb -m S129SMP
  ```

- `trjconv`: Select, transform, and convert a DROPPS trajectory.
  Required args: `-s/--run-input`, `-f/--input`, `-o/--output`
  [Full reference](docs/commands/trjconv.md).

  ```bash
  dps trjconv -s run.tpr -f run.xtc -o frame.pdb -b 100 -e 100 --select 'group 0'
  ```

- `gsd2xtc`: Convert a GSD trajectory to XTC format.
  Required args: `-f/--input`, `-o/--output`
  [Full reference](docs/commands/gsd2xtc.md).

  ```bash
  dps gsd2xtc -f run.gsd -o run.xtc
  ```

- `trjcat`: Concatenate DROPPS XTC trajectory segments in time order.
  Required args: `-f/--input`, `-o/--output`
  [Full reference](docs/commands/trjcat.md).

  ```bash
  dps trjcat -f run.xtc run.part0002.xtc -o complete.xtc
  ```

- `eneconv`: Concatenate and time-filter DROPPS CSV-format energy files.
  Required args: `-f/--input`, `-o/--output`
  [Full reference](docs/commands/eneconv.md).

  ```bash
  dps eneconv -f run.edr run.part0002.edr -o complete.edr
  ```

- `report-methods`: Report the system, simulation methods, and provenance stored in a DROPPS TPR.
  Required args: `-s/--run-input`
  [Full reference](docs/commands/report-methods.md).

  ```bash
  dps report-methods -s run.tpr -o methods.out -m methods.tex
  ```


### Simulation

- `mdrun`: Run or resume a molecular-dynamics simulation from a DROPPS run-input file.
  Required args: `-s/--run-input`, `-o/--output-prefix`
  [Full reference](docs/commands/mdrun.md).

  ```bash
  dps mdrun -s run.tpr -o run
  ```

- `rerun`: Re-evaluate configurational observables from an XTC trajectory.
  Required args: `-s/--run-input`, `-f/--input`, `-o/--output`
  [Full reference](docs/commands/rerun.md).

  ```bash
  dps rerun -s run.tpr -f run.xtc -o rerun.edr
  ```


### Analysis

- `make_ndx`: Create or extend an atom-group index file interactively.
  Required args: `-s/--run-input`, `-o/--output`
  [Full reference](docs/commands/make_ndx.md).

  ```bash
  dps make_ndx -s run.tpr -o index.ndx
  ```

- `check`: Inspect DROPPS run, trajectory, and index files before analysis.
  Required args: `-s/--run-input`
  [Full reference](docs/commands/check.md).

  ```bash
  dps check -s run.tpr -f run.xtc -n index.ndx
  ```

- `energy`: Interactively select observables from a DROPPS EDR and export XVG.
  Required args: `-f/--input`
  [Full reference](docs/commands/energy.md).

  ```bash
  dps energy -f run.edr --list
  ```

- `density`: Calculate a one-dimensional mass- or charge-density profile.
  Required args: `-s/--run-input`, `-f/--input`, `-o/--output`
  [Full reference](docs/commands/density.md).

  ```bash
  dps density -s run.tpr -f run.xtc -o density.xvg -x z
  ```

- `densmap`: Calculate planar two-dimensional mass, number, or charge density maps.
  Required args: `-s/--run-input`, `-f/--input`, `-o/--output`
  [Full reference](docs/commands/densmap.md).

  ```bash
  dps densmap -s run.tpr -f run.xtc -o densmap.dat --selection 0 --average-axis z --type number
  ```

- `coexistence`: Analyze dense/dilute coexistence and interface width in a planar slab trajectory.
  Required args: `-s/--run-input`, `-f/--input`, `-o/--output`
  [Full reference](docs/commands/coexistence.md).

  ```bash
  dps coexistence -s run.tpr -f run.xtc -o coexistence.xvg --axis z
  ```

- `exchange`: Analyze molecular exchange and phase residence times across a planar slab interface.
  Required args: `-s/--run-input`, `-f/--input`, `-o/--output-prefix`
  [Full reference](docs/commands/exchange.md).

  ```bash
  dps exchange -s run.tpr -f run.xtc -o exchange --axis z
  ```

- `phase-msd`: Calculate phase-conditioned molecular MSD with phase-translation removal in a planar slab.
  Required args: `-s/--run-input`, `-f/--input`, `-o/--output`
  [Full reference](docs/commands/phase-msd.md).

  ```bash
  dps phase-msd -s run.tpr -f run.xtc -o phase-msd.xvg --axis z
  ```

- `gyrate`: Calculate chain radii of gyration, including optional dense/dilute resolution.
  Required args: `-s/--run-input`, `-f/--input`
  [Full reference](docs/commands/gyrate.md).

  ```bash
  dps gyrate -s run.tpr -f run.xtc --output-average rg-average.xvg
  ```

- `polystat`: Calculate size, shape, and backbone statistics for many protein copies.
  Required args: `-s/--run-input`, `-f/--input`, `-o/--output`
  [Full reference](docs/commands/polystat.md).

  ```bash
  dps polystat -s run.tpr -f run.xtc -o polystat.xvg --select "mol PROT"
  ```

- `ss`: Infer H/E/C states from residue-level coarse-grained backbone geometry.
  Required args: `-s/--run-input`, `-f/--input`, `-o/--output`
  [Full reference](docs/commands/ss.md).

  ```bash
  dps ss -s run.tpr -f run.xtc -o ss-state.csv --select "mol PROT" --map-output ss-map.xpm --number-output ss-count.xvg --occupancy-output ss-occupancy.xvg
  ```

- `angle`: Calculate backbone-angle profiles and time series for equivalent protein chains.
  Required args: `-s/--run-input`, `-f/--input`
  [Full reference](docs/commands/angle.md).

  ```bash
  dps angle -s run.tpr -f run.xtc --output-residue angles-by-residue.xvg
  ```

- `distance`: Calculate fixed, intra-chain, inter-chain, minimum, and maximum distances.
  Required args: `-s/--run-input`, `-f/--input`, `-m/--mode`
  [Full reference](docs/commands/distance.md).

  ```bash
  dps distance -s run.tpr -f run.xtc --mode pair --scope intra-chain --reference 1 --selection 2 --output-statistics intra.csv
  ```

- `rdf`: Calculate radial distribution functions and cumulative coordination numbers.
  Required args: `-s/--run-input`, `-f/--input`, `-o/--output`
  [Full reference](docs/commands/rdf.md).

  ```bash
  dps rdf -s run.tpr -f run.xtc --reference 1 --selection 2 -o rdf.xvg
  ```

- `contact`: Calculate residue-level contact maps and contact-number time series.
  Required args: `-s/--run-input`, `-f/--input`, `-cs/--cutoff-scheme`
  [Full reference](docs/commands/contact.md).

  ```bash
  dps contact -s run.tpr -f run.xtc --reference-group 0 --selection-group 1 --cutoff-scheme global --output-inter-reference-selection contacts.dat
  ```

- `contact.ng`: Calculate residue-level contact maps with memory-efficient aggregation.
  Required args: `-s/--run-input`, `-f/--input`, `-cs/--cutoff-scheme`
  [Full reference](docs/commands/contact.ng.md).

  ```bash
  dps contact.ng -s run.tpr -f run.xtc --reference-group 0 --selection-group 1 --cutoff-scheme global --output-inter-reference-selection contacts.dat --search-mode stream
  ```

- `timecorr`: Calculate residue-contact correlation functions and lifetime matrices.
  Required args: `-s/--run-input`, `-f/--input`, `-cs/--cutoff-scheme`
  [Full reference](docs/commands/timecorr.md).

  ```bash
  dps timecorr -s run.tpr -f run.xtc --reference-group 0 --selection-group 1 --cutoff-scheme global --output-corr-inter-reference-selection contact-correlation.dat
  ```

- `cstat`: Calculate residue-level statistics from a precomputed contact map.
  Required args: `-m/--map-input`, `-s/--run-input`, `-o/--output`
  [Full reference](docs/commands/cstat.md).

  ```bash
  dps cstat -m contacts.dat -s run.tpr -o contact-statistics.xlsx
  ```

- `msd`: Calculate the mean-square displacement of a selected atom group.
  Required args: `-s/--run-input`, `-f/--input`, `-o/--output`
  [Full reference](docs/commands/msd.md).

  ```bash
  dps msd -s run.tpr -f run.xtc -o msd.xvg -t xyz
  ```

- `rmsd`: Calculate PBC-aware RMSD with configurable fitting and output granularity.
  Required args: `-s/--run-input`, `-f/--input`, `-o/--output`
  [Full reference](docs/commands/rmsd.md).

  ```bash
  dps rmsd -s run.tpr -f run.xtc -o rmsd.xvg --select "(mol PROT) & (resid 20-80)"
  ```

- `rmsf`: Calculate PBC-aware atomic or residue RMS fluctuations.
  Required args: `-s/--run-input`, `-f/--input`, `-o/--output`
  [Full reference](docs/commands/rmsf.md).

  ```bash
  dps rmsf -s run.tpr -f run.xtc -o rmsf.xvg --select "(mol PROT) & (resid 20-80)" --residue
  ```

- `sasa`: Calculate coarse-grained solvent-accessible surface area.
  Required args: `-s/--run-input`, `-f/--input`, `-o/--output`
  [Full reference](docs/commands/sasa.md).

  ```bash
  dps sasa -s run.tpr -f run.xtc -o sasa.xvg --surface 0 --output-selection 1 2
  ```

- `assembly`: Analyze the formation, size, composition, and shape of molecular assemblies.
  Required args: `-s/--run-input`, `-f/--input`
  [Full reference](docs/commands/assembly.md).

  ```bash
  dps assembly -s run.tpr -f run.xtc --cluster-number clusters.xvg --cluster-size largest.xvg
  ```

- `pbcontact`: Measure direct and network-mediated interactions with periodic molecular images.
  Required args: `-s/--run-input`, `-f/--input`, `-o/--output-prefix`
  [Full reference](docs/commands/pbcontact.md).

  ```bash
  dps pbcontact -s run.tpr -f run.xtc -o periodic-images
  ```

- `pdb2bond`: Add PDB CONECT records from a DROPPS run file or system topology.
  Required args: one of `-s/--run-input` or `-p/--topology`, `-f/--input`, `-o/--output`
  [Full reference](docs/commands/pdb2bond.md).

  ```bash
  dps pdb2bond -s run.tpr -f frame.pdb -o bonded.pdb
  ```

- `surftension`: Calculate planar-interface surface tension from a DROPPS TPR and XTC trajectory.
  Required args: `-s/--run-input`, `-f/--input`, `-o/--output`
  [Full reference](docs/commands/surftension.md).

  ```bash
  dps surftension -s run.tpr -f run.xtc -o surface-tension.xvg --axis z
  ```

- `viscosity-gk`: Calculate zero-shear viscosity from a DROPPS runtime pressure tensor.
  Required args: `-f/--input`, `-o/--output`
  [Full reference](docs/commands/viscosity-gk.md).

  ```bash
  dps viscosity-gk -f run.stress.xvg -o viscosity.xvg
  ```


<!-- END AUTO COMMANDS -->































## Typical Workflow

1. Build molecules and system.
2. Prepare simulation runtime files.
3. Run simulation.
4. Analyze trajectories.

Example:

```bash
# 1) Build single-chain model
dps pdb2dps -s "MSEQNNTEMTFQIQRIYTKDISFEAPNAPHVFQKDW" \
  -o chain.pdb -op chain.top

# 2) Build multi-molecule system
dps genmesh -f chain.pdb -p chain.itp -n 100 \
  -o system.pdb -op system.top

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
`--center-mode frame`, density threshold, and bin width are shared with
`coexistence`, so the reference mass-density curves agree when both commands
use the same selections and options.

For droplets, non-planar domains, lateral interface heterogeneity, or
component sorting, `densmap` projects the selected particles onto a two-
dimensional plane. It follows the planar conventions of `gmx densmap` while
adding mass and charge weights and multiple output selections:

```bash
# x-y number-density map, averaged over z
dps densmap -s run.tpr -f run.xtc -o number.dat \
  --selection 0 --average-axis z --type number

# Center a drifting droplet in x-y and write one mass map per component
dps densmap -s run.tpr -f run.xtc -o components.dat \
  --selection 1 2 --average-axis z --type mass --center-mode frame \
  --selection-fit 1 --xpm-output components.xpm
```

`--normalization volume` (alias `nm-3`) divides each frame by the physical 3D grid-cell
volume; `area` (`nm-2`) gives the projected surface density; `count` keeps the
mean weight in each cell. Box fluctuations are handled by normalizing each
frame before averaging in fractional coordinates. The `.dat` file is a numeric
matrix whose first row and column contain cell-center coordinates in nm. With
multiple `--selection` groups, stable numbered group names are appended to the output
stem. `--center-mode frame` uses the periodic mass center of `--selection-fit` in both
retained directions, so a compact droplet can cross a box boundary without
being split or smeared; fixed or multi-domain morphologies should normally use
the default `--center-mode none`.

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
  -b 100 -e 100 --select "group 0"
```

Use `-b`, `-e`, and `--delta-time` for a time range. Available PBC modes are `none`, `whole`,
`atom`, `res`, `mol`, and `nojump`; unlike the old implementation, `whole`
only reconstructs bonded molecules while `mol` also packs their centers into
the primary cell. Geometric, mass, or dense-slab centering and reference fits
can be composed in a fixed order:

```bash
dps trjconv -s slab.tpr -f slab.xtc -o centered.xtc \
  --select "group 0" --pbc mol \
  --center dense --center-select "mol PROT" --center-axis z
```

PDB ranges are written as multi-model files; use `--separate` for one numbered
PDB per selected time and `--conect` to include selected topology bonds.

### Concatenate run segments

`trjcat` concatenates XTC segments in chronological order. By default, an
identical boundary time is written once and the frame from the later segment
wins. The output is written atomically, and every selected frame retains its
stored step, time, box, coordinates, and XTC precision:

```bash
dps trjcat \
  -f run.xtc run.part0002.xtc run.part0003.xtc \
  -o complete.xtc
```

Use `-b`, `-e`, `--delta-time`, and `--time-unit` to select output frames,
`--no-sort` only when the input order is already chronological, and
`--keep-duplicates` when duplicate times must be kept.
All inputs must have the same atom count.

`eneconv` performs the corresponding operation for DROPPS CSV-EDR files:

```bash
dps eneconv \
  -f run.edr run.part0002.edr run.part0003.edr \
  -o complete.edr
```

The field order and units must match. Key scientific metadata such as target
temperature, pressure coupling, velocity availability, and pressure-estimator
semantics must also agree unless `--allow-incompatible` is supplied explicitly.
For a repeated step at the same time, the sample from the later input file is
used.

### Generate a simulation-methods report

`report-methods` reads the complete simulation definition and provenance stored
in a TPR. With no output option it prints a structured report to the terminal;
`-o` writes plain text and `-m` writes LaTeX:

```bash
dps report-methods -s run.tpr
dps report-methods -s run.tpr -o methods.out -m methods.tex
```

The report includes system composition, integration and ensemble settings,
nonbonded models and cutoffs, output intervals, software versions, run/system
identifiers, embedded source-file names, and a ready-to-edit Methods paragraph.

### Configure `mdrun`

Scientific settings such as the integration time step, temperature, pressure,
and output sampling intervals remain in the TPR generated by `grompp`.
Operational settings can be selected when `mdrun` starts. For example:

```bash
dps mdrun -s run.tpr --output-prefix run \
  --platform CUDA --device-index 0 --precision mixed \
  --checkpoint-minutes 10 --max-hours 23.5
```

Use explicit `--platform CUDA` for production jobs that must use NVIDIA GPUs.
Unlike `--platform auto`, it fails immediately if CUDA cannot be initialized
instead of falling back to another OpenMM platform.

The historical `-deffnm` remains an alias of `-o/--output-prefix`; the meaning
of `-o` has not changed. The platform can be `auto`, `CUDA`, `OpenCL`, `CPU`, or
`Reference`. Use `--threads` to select CPU-platform threads. GPU device and
precision can be set with `--device-index` and `--precision`.

Individual output paths can override the prefix defaults:

```bash
dps mdrun -s run.tpr -o run \
  -x trajectory.xtc -e thermo.edr -g progress.log \
  --checkpoint-output state.chk -c final.pdb --stress-output pressure.xvg
```

`--nsteps` temporarily overrides the production target stored in the TPR, while
`--seed` overrides its random seed for a new run. Neither option modifies the
TPR. The resolved seed controls the Langevin integrator, velocity generation,
and Monte Carlo barostat.

Periodic restarts use wall-clock time. `--checkpoint-minutes` is measured in
minutes and defaults to 5; `--checkpoint-minutes 0` disables periodic saves. Each save produces a native
checkpoint (`.chk`), a portable OpenMM State (`.state.xml`), and a checksum/
compatibility manifest (`.restart.json`). A final set is still written after
successful completion, a `-maxh` stop, or Ctrl-C. `-maxh` stops after
approximately 99% of the requested hours so a batch job has time to write its
restart files. Add `--number-checkpoints` to retain step-numbered copies alongside the latest
files.

### Resume an interrupted simulation

`mdrun` writes `<output-prefix>.chk`, `<output-prefix>.state.xml`, and
`<output-prefix>.restart.json` every five wall-clock minutes by default.
To continue the same run, keep the original run input and output prefix and use:

```bash
dps mdrun -s run.tpr -o run --checkpoint
```

With no path after `--checkpoint`, DROPPS loads `run.chk`. A checkpoint at another
location can be selected explicitly:

```bash
dps mdrun -s run.tpr -o run --checkpoint /path/to/run.chk
```

DROPPS first tries the native `.chk`. If OpenMM rejects it because the version,
platform, or hardware changed, DROPPS validates and loads the paired
`.state.xml` automatically. The State preserves positions, velocities, box,
time, step, and context parameters, but not the integrator's internal random-
number state. Therefore native checkpoint continuation is exact, while State
fallback starts a statistically continuous but not bitwise-identical trajectory
segment. A State can also be selected directly with `--checkpoint run.state.xml`.

On restart, `nsteps` remains the target total number of production steps, not
the number of extra steps to add. A production checkpoint skips minimization
and warming; a checkpoint written during warming resumes its recorded ramp
segment before production. DROPPS appends to the existing XTC, fixed
progress log, CSV-format EDR, and stress-tensor outputs. If any output is newer
than the checkpoint, it is first
rolled back to the checkpoint step so the resumed files do not contain
duplicate or out-of-order samples. A final checkpoint is also written after a
successful run or a Ctrl-C interruption.

To keep the original outputs unchanged, use `--no-append`. Continuation outputs
are then written as `run.part0002.xtc`, `run.part0002.edr`, and so on, while the
latest checkpoint remains `run.chk`:

```bash
dps mdrun -s run.tpr -o run --checkpoint --no-append
```

OpenMM binary checkpoints require the same system definition, OpenMM version,
compute platform, and compatible hardware. Keep the complete three-file restart
set. When intentionally extending a run, use `--nsteps` or regenerate a compatible
TPR whose simulation definition changes only by increasing `nsteps`.

An old standalone `.chk` has no portable fallback. Load it once with its
original OpenMM/DROPPS environment and stop normally (for example with a very
short `--max-hours`) to produce the new `.state.xml` and `.restart.json` pair before
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
centered from the largest periodic reference-density region above
`--dense-phase-threshold`, using the same definition as `density`. This prevents
slab translation from broadening either average profile. DROPPS then fits the
reference density to a symmetric double-hyperbolic-tangent profile and reports
the dense and dilute densities, slab width, 10--90% interface width, fit quality,
and contiguous-block standard errors.

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

### Phase-resolved molecular radius of gyration

Use `-op/--output-phase` to compare single-chain conformations in the dense and
dilute phases of a planar slab:

```bash
dps gyrate \
  -s slab.tpr \
  -f slab.xtc \
  -n groups.ndx \
  -op phase-rg.xvg \
  -ref 0 \
  -sel 1 2 \
  --axis z \
  --phases dense dilute \
  --center-mode frame \
  --density-bin-width 0.05 \
  --blocks 5
```

The reference group locates and fits the dense slab. Each `-sel` group is split
into molecules, every molecule is reconstructed across periodic boundaries,
and its mass-weighted Rg and center are calculated independently in every
frame. The center is then classified as dense, interface, or dilute using the
same fitted boundaries as `exchange` and `phase-msd`. Thus, a molecule that
crosses an interface changes cohorts instead of remaining in a static index
group. Add `interface` to `--phases` to report the interfacial cohort.

The requested `phase-rg.xvg` contains frame-local mean Rg series. Companion
files contain all molecule observations (`.observations.csv`), pooled and
contiguous-block summary statistics (`.summary.csv`), phase-conditioned PDFs
(`.histogram.xvg`), and the fitted reference density and phase cutoffs
(`.profile.xvg`). In addition, every selected group gets a
`.groupNN-<name>.values.csv` file containing all `dense_rg_nm` and
`dilute_rg_nm` values as two independently populated columns. Use
`--no-phase-observations` when the detailed provenance CSV would be too large.
The older `-ov`, `-oa`, and `-oh` outputs retain their original group-level
behavior and can be requested together with `-op`.

Phase-resolved mode assumes one orthorhombic planar slab and that the selected
part of each molecule spans less than half the box length along each periodic
direction, so minimum-image reconstruction is unambiguous. Empty group/phase
cohorts are written as `nan` in the frame-resolved XVG rather than silently
treated as zero.

### Unified many-chain distances

`distance` replaces the chain-specific `idist`/`odist` workflow. Select each
region once across all copies of a molecule; `--scope` determines which chain
pairs are admitted, while `--mode` determines how positions inside each pair
are compared.

For 100 `PROT` chains, this produces 100 intra-chain residue-center distances:

```bash
dps distance -s run.tpr -f run.xtc \
  -ref "(mol PROT) & (resid 10)" \
  -sel "(mol PROT) & (resid 50)" \
  --scope intra-chain --position cog --mode pair \
  -ostat intra.csv --statistics-scope both
```

Changing the scope to `inter-chain` produces the 9900 directed
`A(chain i)-B(chain j), i != j` channels. `--statistics-scope pooled` summarizes
all channels and frames without retaining or writing their individual time
series; `channel` reports one time statistic per chain pair, and `both` writes
both levels. `-oav` writes the per-frame cross-channel mean and population
standard deviation, while `-oall` and `-oxyz` are available when every detailed
time series is really needed.

Use `--position atom`, `cog`, or `com`. With atom positions, `--mode pair`
zips fixed positions and `min`/`max` reduces all atom pairs inside each admitted
entity pair. Add `--grouping residue` when a residue-name selection contains
multiple residue instances per chain. No per-chain NDX groups are required.

`idist` and `odist` remain temporary hidden compatibility aliases and print a
deprecation warning; both now use the unified numerical core.

### Radial distribution functions

`rdf` follows the central `gmx rdf` model: one reference selection is compared
with every repeated target selection, and `-cn` optionally writes cumulative
coordination numbers:

```bash
dps rdf -s run.tpr -f run.xtc \
  -ref "mol PROT" -sel "mol SOL" \
  -o protein-solvent.xvg -cn protein-solvent-cn.xvg
```

Selections can remain atom positions or be converted to COG/COM positions with
`--position` and `--grouping all|chain|residue|atom`; side-specific overrides
are available. `--norm` selects dimensionless `rdf`, radial `number-density`,
or unnormalized pair-count density. Use `--geometry xy` for an in-plane RDF,
`--exclude topology` for ITP 1-2/1-3 exclusions, or
`--exclude same-molecule` for an intermolecular distribution.

PBC is enabled by default for orthorhombic and triclinic cells. An omitted
`--rmax` is chosen from the smallest safe periodic radius over all analyzed
frames, so NPT box contraction cannot silently make the minimum image
ambiguous. `--surface chain|residue` instead bins each target's closest
selected atom in every reference entity and therefore requires `--norm none`.
See [the RDF design](docs/rdf-design.md) for the precise bin grid,
normalization, and exclusion semantics.

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

### PBC-aware multi-molecule RMSF

`rmsf` follows the core `gmx rmsf` definition but defaults to an independent
fit for every molecule copy, which isolates conformational flexibility from
copy diffusion and rearrangement:

```bash
dps rmsf \
  -s run.tpr \
  -f run.xtc \
  -o rmsf.xvg \
  --select "(mol PROT) & (resid 20-80)"
```

Complete bonded molecules are reconstructed across PBC before fitting. The
default mass-weighted `--fit-mode molecule` compares each copy with its own
coordinates in the first analyzed frame. Use `--fit-mode selection` for one
fit over the complete selection, `none` to retain translation and rotation,
or `-ref TIME_NS` to select another saved reference frame.

Atomic profiles are the default. `-res` mass-averages the atomic covariance
tensors within each residue before calculating RMSF, matching the GROMACS
residue convention. `--fit-weighting uniform` and `--group-weighting uniform`
make either operation equal-weighted.

The XVG stores the mean and population standard deviation across equivalent
molecule copies. Its companion CSV retains every molecule/site result together
with reference deviation, B factor, and the full Cartesian covariance tensor.
`-od` writes reference-deviation profiles; `-ox` and `-oq` write average and
reference coordinates with isotropic B factors to PDB. Add `--summary-only` to
suppress the detailed CSV. See [the RMSF design](docs/rmsf-design.md) for the
formulas, output schema, and current limitations.

### Coarse-grained solvent-accessible surface area

`sasa` follows the `gmx sasa` surface/output model. The surface selection
defines every bead that constructs and occludes the calculated surface. Each
additional output must be a subset of that surface and is extracted from the
same per-bead areas:

```bash
dps sasa \
  -s run.tpr \
  -f run.xtc \
  -n index.ndx \
  -o sasa.xvg \
  -surface "group 1 | group 2" \
  -output "group 1" "group 2"
```

The surface total is always the first XVG series. `-output` accepts multiple
quoted selections and can be repeated. In an interactive terminal, omitting
`-surface` prompts for one selection (an empty response means all atoms), then
prompts for zero or more output subsets. Non-interactive use defaults to all
topology atoms and no additional subsets, independently of NDX group 0.

By default, bead radii are `sigma/2`, the probe radius is `0.14 nm`, and every
expanded sphere uses 960 deterministic antipodal Fibonacci points. Use
`--radius-convention lj-minimum-half`, `--radius-scale`, `--fixed-radius`,
`--probe-radius`, and `--sphere-points` to make those geometric conventions
explicit. Ashbaugh--Hatch topologies use per-type sigma, while
Wang--Frenkel/MPiPi topologies use the self-pair `sigma_ii`.

Every frame is evaluated with its own periodic box. Coordinates many box
lengths outside the primary cell are reduced before neighbor search, and
surface-point occlusion uses minimum-image geometry. The calculation stops if
the largest expanded-sphere overlap cutoff reaches half the shortest lattice
vector, where a unique minimum image can no longer be assumed. Results are in
`nm^2`; the XVG header records the radius convention, probe, sampling count,
selection, and trajectory mean and population standard deviation.

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

For the leapfrog/Langevin-middle integrator, DROPPS synchronizes the stored
half-step velocities to the current-position time slice with the current force
before evaluating both kinetic energy and this kinetic pressure term.

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

## License and citation

DROPPS is distributed under the [Apache License 2.0](LICENSE), with attribution in [NOTICE](NOTICE). Software citation metadata is provided in [CITATION.cff](CITATION.cff).
