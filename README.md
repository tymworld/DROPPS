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
pip install dist/dropps-0.4.0-py3-none-any.whl
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

- `pdb2dps`: This program create pdb and top files for a given sequence.
  Required args: `-s/--sequence`
- `genelastic`: This program generate elastic network for an itp file.
  Required args: `-f/--structure`, `-p/--topology`, `-o/--output`, `-er/--elastic-residues`
- `genmesh`: This program generate mesh for a given set of molecules.
  Required args: `-f/--structure`, `-p/--topology`, `-n/--number`, `-oc/--output-conformation`, `-op/--output-topology`
- `addangle`: This program add angle terms for an itp file.
  Required args: `-ip/--input-topology`, `-op/--output-topology`, `-al/--angle-list`
- `editconf`: This program edit box configuration for a pdb file.
  Required args: `-f/--structure`, `-o/--output`
- `grompp`: This program initates and runs an molecular dynamics simulation.
  Required args: `-f/--structure`, `-p/--topology`, `-m/--parameter`, `-o/--output`
- `modifyres`: This program add residue modifications for an itp and a pdb file.
  Required args: `-ip/--input-topology`, `-if/--input-structure`, `-op/--output-topology`, `-of/--output-structure`, `-m/--modifications`
- `trjconv`: This program read and convert trajectory files in many ways.
  Required args: `-s/--run-input`, `-f/--input`, `-pbc/--pbc`
- `gsd2xtc`: This program read and convert gsd file to xtc file.
  Required args: `-f/--input`

### Simulation

- `mdrun`: This program initates and runs an molecular dynamics simulation.
  Required args: `-s/--run-input`, `-o/--output-prefix`

### Analysis

- `make_ndx`: This program generate an index file for data analysis.
  Required args: `-s/--run-input`, `-o/--output`
- `extract`: This program extract a single frame from a HPS trajectory.
  Required args: `-s/--run-input`, `-f/--input`, `-o/--output`
- `check`: This program check tpr, xtc, and ndx files for data analysis.
  Required args: `-s/--run-input`
- `density`: This program calculate the density profile of a simulation box along one axis.
  Required args: `-s/--run-input`, `-f/--input`, `-o/--output`
- `gyrate`: This program calculate the radius of gyration of all chains in a HPS trajectory.
  Required args: `-s/--run-input`, `-f/--input`
- `angle`: This program calculate the profile of angles along amino acid sequence of a protein.
  Required args: `-s/--run-input`, `-f/--input`
- `idist`: This program calculate the intra-chain distance profiles of a number of atom pairs.
  Required args: `-s/--run-input`, `-f/--input`
- `odist`: This program calculate the inter-chain distance profiles of atom pair.
  Required args: `-s/--run-input`, `-f/--input`
- `contact`: This program calculate contact map in a residue-level resolution.
  Required args: `-s/--run-input`, `-f/--input`, `-cs/--cutoff-scheme`
- `contact.ng`: This program calculates residue-level contact maps with memory-efficient aggregation.
  Required args: `-s/--run-input`, `-f/--input`, `-cs/--cutoff-scheme`
- `timecorr`: This program calculate residue-level time correlation functions and lifetime matrices for contact interactions.
  Required args: `-s/--run-input`, `-f/--input`, `-cs/--cutoff-scheme`
- `cstat`: This program perform residue-specific statistic on pre-calculated contact maps.
  Required args: `-m/--map-input`, `-s/--run-input`, `-o/--output`
- `msd`: This program calculate mean square deviation of molecules within a trajectory.
  Required args: `-s/--run-input`, `-f/--input`
- `assembly`: This program calculate the time evoluted formation of molecular assemblies.
  Required args: `-s/--run-input`, `-f/--input`
- `pdb2bond`: This program add conect record to a pdb file.
  Required args: `-f/--input`, `-o/--output`
- `surftension`: Calculate planar-interface surface tension from a DROPPS TPR and XTC trajectory.
  Required args: `-s/--run-input`, `-f/--input`, `-o/--output`

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

## Notes

- The command name is `dps` (not legacy `cgps`).
- For exact and current argument details, always use `dps <command> -h`.
- Some analysis commands expect consistent `.tpr`/`.xtc`/`.ndx` group definitions.
- Refresh the auto-generated command catalog with: `python3 scripts/update_readme_commands.py`.

## Author

- Yiming Tang
- ymtang@fudan.edu.cn
