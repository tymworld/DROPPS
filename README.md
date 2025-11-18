# DROPPS

**Distributed Rapid Operation Platform for Phase-separation Simulations**

---

## Distributions

### V0.3.0
#### Add: 
- dps gsd2xtc command which converts gsd to xtc file.
- dps cmap now allows global and residue specific contacts.
- dps trjconv command which converts trajectory and treat pbc
- Now trajectory tool allows gsd input

#### Fix:
- Fix contact cutoff unit issue in dps cmap

#### Deprecated: 
- dps contact command is deprecated because of unfixed bug.

### V0.2.1
#### Add:
- Construction tool for multi-bead-per-residue forcefield
- dps extract which extract a single frame

#### Update:
- MSD calculation program is updated

#### Fix:
- Fix misused sigma and lambda in HPS-T forcefield
- Fix angle constraints
- Parameter for PTM is fixed

---

## Overview

**DROPPS** is a high-throughput, distributed framework designed for performing coarse-grained simulations of protein and biomolecular phase separation. It provides an efficient, scalable, and user-friendly platform to explore the phase behavior of intrinsically disordered proteins, protein condensates, and other biomolecular systems.

---

## Usage

**One and single command:** ```src/dps```
**Show help:** ```dps help```
**Show all available commands:** ```dps help commands```
**Show help for a single command (such as mdrun):** ```dps help mdrun```
**Run a single command (such as mdrun):** ```dps mdrun -h```

---

## Authors
- **Leading developer:** Yiming Tang
- **E-mail:** ymtang@fudan.edu.cn
- **In case of questions or suggestions:** Please feel free to reach out but there is significant chance that I'll ignore you. Bite me.

---

## Features

- 🚀 **High-throughput simulation pipeline** for large-scale screening of biomolecular phase separation.
- ⚖️ **Coarse-grained modeling support**, enabling efficient simulation of large systems and long timescales.
- 🌐 **Distributed and parallel execution**, designed for HPC clusters, GPUs,  or local multi-core setups.
- 🧬 **Customizable molecular systems**, supporting different protein sequences and interaction models.
- 📊 **Automated analysis tools** for computing phase diagrams, density profiles, contact maps, and more.
- 💻 **Extensible framework**, easy to integrate with new force fields or analysis modules.

---

## Installation

```bash
pip install -r requirements.txt
pip install -e dist/dropps-0.1.0-py3-none-any.whl
```

---

## Commands
### Usage: dps command options
### Existing commands:
- addangle       : This program add angle terms for an itp file.
- editconf       : This program edit box configuration for a pdb file.
- getelastic     : This program generate elastic network for an itp file.
- genmesh        : This program generate mesh for a given set of molecules.
- grompp         : This program initates and runs an molecular dynamics simulation.
- mdrun          : This program initates and runs an molecular dynamics simulation.
- pdb2cgps       : This program create pdb and top files for a given sequence.
- help           : help

### pdb2cgps
#### This program create pdb and top files for a given sequence.

```
usage: pdb2cgps [-h] -s SEQUENCE [-f INPUT_PDB] [-ri RESIDUE_INDEX]
                [-ptm POST_TRANSLATIONAL_MODIFICATION] [-r RADIUS] [-n NUMBER] [-e DEGREE_EXTEND]
                [-ff {}] [-oc OUTPUT_CONFORMATION] [-op OUTPUT_TOPOLOGY] [-on OUTPUT_NAME] [-cNTD]
                [-cCTD]

This program create pdb and top files for a given sequence.

options:
  -h, --help            show this help message and exit
  -s SEQUENCE, --sequence SEQUENCE
                        Protein sequence
  -f INPUT_PDB, --input-pdb INPUT_PDB
                        PDB file containing structure information of a chain.
  -ri RESIDUE_INDEX, --residue-index RESIDUE_INDEX
                        Residue number of the first residue
  -ptm POST_TRANSLATIONAL_MODIFICATION, --post-translational-modification POST_TRANSLATIONAL_MODIFICATION
                        Post translational modification to add
  -r RADIUS, --radius RADIUS
                        Maximum radius of gyration of generated chain
  -n NUMBER, --number NUMBER
                        Number of conformations to generate
  -e DEGREE_EXTEND, --degree-extend DEGREE_EXTEND
                        Degree of extend for generated chain (0~1)
  -ff {}, --forcefield {}
                        Forcefield selection
  -oc OUTPUT_CONFORMATION, --output-conformation OUTPUT_CONFORMATION
                        File prefix to write output configuration
  -op OUTPUT_TOPOLOGY, --output-topology OUTPUT_TOPOLOGY
                        File prefix to write topology file
  -on OUTPUT_NAME, --output-name OUTPUT_NAME
                        Molecule name.
  -cNTD, --charged-NTD  Whether N terminal is patched by an additional positive charge.
  -cCTD, --charged-CTD  Whether C terminal is patched by an additional negative charge.
```

### editconf
#### This program edit box configuration for a pdb file.

```
usage: editconf [-h] -f STRUCTURE -o OUTPUT [-x X_AXIS] [-y Y_AXIS] [-z Z_AXIS] [-mx MULTIPLY_X_AXIS]
                [-my MULTIPLY_Y_AXIS] [-mz MULTIPLY_Z_AXIS] [-px] [-py] [-pz]

This program edit box configuration for a pdb file.

options:
  -h, --help            show this help message and exit
  -f STRUCTURE, --structure STRUCTURE
                        PDB file which is taken as input.
  -o OUTPUT, --output OUTPUT
                        PDB file which to write editted configuration.
  -x X_AXIS, --x-axis X_AXIS
                        Length of the simulation box to output in the x axis.
  -y Y_AXIS, --y-axis Y_AXIS
                        Length of the simulation box to output in the y axis.
  -z Z_AXIS, --z-axis Z_AXIS
                        Length of the simulation box to output in the z axis.
  -mx MULTIPLY_X_AXIS, --multiply-x-axis MULTIPLY_X_AXIS
                        Multiplier to act on the length of the simulation box in the x axis.
  -my MULTIPLY_Y_AXIS, --multiply-y-axis MULTIPLY_Y_AXIS
                        Multiplier to act on the length of the simulation box in the y axis.
  -mz MULTIPLY_Z_AXIS, --multiply-z-axis MULTIPLY_Z_AXIS
                        Multiplier to act on the length of the simulation box in the z axis.
  -px, --treat-pbc-x    Whether peridic images are treated in the x directions.
  -py, --treat-pbc-y    Whether peridic images are treated in the y directions.
  -pz, --treat-pbc-z    Whether peridic images are treated in the z directions.
```

### addangle
#### This program add angle terms for an itp file.

```
usage: addangle [-h] -ip INPUT_TOPOLOGY -op OUTPUT_TOPOLOGY -al ANGLE_LIST

This program add angle terms for an itp file.

options:
  -h, --help            show this help message and exit
  -ip INPUT_TOPOLOGY, --input-topology INPUT_TOPOLOGY
                        Input itp file.
  -op OUTPUT_TOPOLOGY, --output-topology OUTPUT_TOPOLOGY
                        Output itp file with secondary structure constrains added.
  -al ANGLE_LIST, --angle-list ANGLE_LIST
                        ASCII File containing angle information. Format: "ID-start-with-1 theta k"
```

### getelastic
#### This program generate elastic network for an itp file.

```
usage: genelastic [-h] -f STRUCTURE -p TOPOLOGY -o OUTPUT -er ELASTIC_RESIDUES
                  [-ef ELASTIC_FORCE_CONSTANT] [-el ELASTIC_LOWER] [-eu ELASTIC_UPPER]

This program generate elastic network for an itp file.

options:
  -h, --help            show this help message and exit
  -f STRUCTURE, --structure STRUCTURE
                        PDB file which is taken as input for structure.
  -p TOPOLOGY, --topology TOPOLOGY
                        ITP file which is taken as input for topology.
  -o OUTPUT, --output OUTPUT
                        ITP file which to write topology with elastic network added.
  -er ELASTIC_RESIDUES, --elastic-residues ELASTIC_RESIDUES
                        ASCII File each line of which contains group of bead on which elastic network
                        will be added. Start with 1.
  -ef ELASTIC_FORCE_CONSTANT, --elastic-force-constant ELASTIC_FORCE_CONSTANT
                        Elastic bond force constant Fc, default: 5000
  -el ELASTIC_LOWER, --elastic-lower ELASTIC_LOWER
                        Elastic bond lower cutoff: F = Fc if rij < lo, default: 0.5
  -eu ELASTIC_UPPER, --elastic-upper ELASTIC_UPPER
                        Elastic bond upper cutoff: F = 0 if rij > up, default: 0.9
```

### genmesh
#### This program generate mesh for a given set of molecules.

```
usage: genmesh [-h] -f STRUCTURE [STRUCTURE ...] -p TOPOLOGY [TOPOLOGY ...] -n NUMBER [NUMBER ...]
               [-g GAP] [-mesh MESH MESH MESH] [-bt {xy,cubic,anosotropy}] [-mx MINIMUM_X]
               [-my MINIMUM_Y] [-mz MINIMUM_Z] [-s] -oc OUTPUT_CONFORMATION -op OUTPUT_TOPOLOGY

This program generate mesh for a given set of molecules.

options:
  -h, --help            show this help message and exit
  -f STRUCTURE [STRUCTURE ...], --structure STRUCTURE [STRUCTURE ...]
                        PDB files which are taken as input. Allow multiple file.
  -p TOPOLOGY [TOPOLOGY ...], --topology TOPOLOGY [TOPOLOGY ...]
                        ITP files which are taken as input. Allow multiple file (one for each pdb
                        file).
  -n NUMBER [NUMBER ...], --number NUMBER [NUMBER ...]
                        Number of molecules to add. Allow multiple input (one number for each pdb
                        file).
  -g GAP, --gap GAP     Minimum distance between each two molecules.
  -mesh MESH MESH MESH, --mesh MESH MESH MESH
                        Three integers representing the size of the mesh in x/y/z direction. If not
                        specified, mesh will be guessed.
  -bt {xy,cubic,anosotropy}, --box-type {xy,cubic,anosotropy}
                        The type of the size of box.
  -mx MINIMUM_X, --minimum-x MINIMUM_X
                        Minimum length of the box in the x direction. Zero is as small as posible.
  -my MINIMUM_Y, --minimum-y MINIMUM_Y
                        Minimum length of the box in the y direction. Zero is as small as posible.
  -mz MINIMUM_Z, --minimum-z MINIMUM_Z
                        Minimum length of the box in the z direction. Zero is as small as posible.
  -s, --shuffle         Whether the molecules to input are shuffled before insertion.
  -oc OUTPUT_CONFORMATION, --output-conformation OUTPUT_CONFORMATION
                        File prefix to write output configuration
  -op OUTPUT_TOPOLOGY, --output-topology OUTPUT_TOPOLOGY
                        File prefix to write output configuration
```

### grompp
#### This program initates and runs an molecular dynamics simulation.

```
usage: mdrun [-h] -f STRUCTURE -p TOPOLOGY -m PARAMETER -o OUTPUT

This program initates and runs an molecular dynamics simulation.

options:
  -h, --help            show this help message and exit
  -f STRUCTURE, --structure STRUCTURE
                        GSD file containing the initial configuration of the system.
  -p TOPOLOGY, --topology TOPOLOGY
                        GSD file containing the initial configuration of the system.
  -m PARAMETER, --parameter PARAMETER
                        Parameter file (mdp) for this simulation.
  -o OUTPUT, --output OUTPUT
                        Path to output tpr files.
```

### mdrun
#### This program initates and runs an molecular dynamics simulation.

```
usage: mdrun [-h] -s RUN_INPUT [-cpi CHECKPOINT] -o OUTPUT_PREFIX

This program initates and runs an molecular dynamics simulation.

options:
  -h, --help            show this help message and exit
  -s RUN_INPUT, --run-input RUN_INPUT
                        TPR file containing all information for this simulation run.
  -cpi CHECKPOINT, --checkpoint CHECKPOINT
                        Checkpoint file to initialize the simulation.
  -o OUTPUT_PREFIX, --output-prefix OUTPUT_PREFIX
                        Prefix of all output files.
```
