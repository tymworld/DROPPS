# Complete DROPPS command reference

DROPPS 1.0.0 exposes **43 public commands**. The reference below includes every command and **679 option entries** (including help/debug on each applicable command).

Start with the [end-to-end tutorial](user-guide.md), [MDP settings](mdp-reference.md), or [Colab workflows](../examples/colab/README.md).

## Conventions

- Run `dps help COMMAND` or `dps COMMAND --help` for the same reference in the terminal.
- Prefer long options in scripts. An omitted selection may prompt interactively; supply groups explicitly for batch jobs.
- DROPPS group IDs, NDX atom indices, and selection `index`/`chainid` values start at **0**. PDB/ITP bead IDs and angle/elastic-file bead IDs start at **1**. `resid` follows the topology's residue numbers.
- Distances normally use nm, energies kJ/mol, and temperatures K. Analysis times normally use ns; commands with `--time-unit` document alternatives. PDB coordinate fields use Angstrom and XTC timestamps use ps internally.
- `.tpr` is DROPPS run input and `.edr` is DROPPS CSV energy data, not the binary GROMACS formats with the same extensions.
- Output options choose which results are written. Read the per-command output descriptions; several analysis commands require at least one output even when individual flags are optional.
- `--debug` shows runtime context and tracebacks. Normal output is plain text when redirected; use `NO_COLOR` to disable terminal colors.

## Catalog

- [help](commands/help.md): command catalog and detailed help.

### Build a simulation system

Create molecular models, pack a box, and compile the run input.

| Command | Purpose |
| --- | --- |
| [pdb2dps](commands/pdb2dps.md) | Generate coarse-grained PDB conformations and an ITP topology from a protein sequence. |
| [modifyres](commands/modifyres.md) | Apply residue modifications consistently to an ITP topology and PDB structure. |
| [genelastic](commands/genelastic.md) | Add a distance-based elastic network to an ITP topology. |
| [addangle](commands/addangle.md) | Add angle restraints to an ITP topology. |
| [genmesh](commands/genmesh.md) | Pack one or more molecule types onto a three-dimensional simulation-box mesh. |
| [editconf](commands/editconf.md) | Resize a PDB simulation box and optionally unwrap coordinates along selected axes. |
| [grompp](commands/grompp.md) | Build a DROPPS run-input file from structure, topology, and simulation parameters. |

### Run or recompute a simulation

Integrate a trajectory, resume a run, or recompute saved observables.

| Command | Purpose |
| --- | --- |
| [mdrun](commands/mdrun.md) | Run or resume a molecular-dynamics simulation from a DROPPS run-input file. |
| [rerun](commands/rerun.md) | Re-evaluate configurational observables from an XTC trajectory. |

### Inspect and convert files

Check inputs, create selections, process trajectories, and convert formats.

| Command | Purpose |
| --- | --- |
| [check](commands/check.md) | Inspect DROPPS run, trajectory, and index files before analysis. |
| [energy](commands/energy.md) | Interactively select observables from a DROPPS EDR and export XVG. |
| [make_ndx](commands/make_ndx.md) | Create or extend an atom-group index file interactively. |
| [trjconv](commands/trjconv.md) | Select, transform, and convert a DROPPS trajectory. |
| [gsd2xtc](commands/gsd2xtc.md) | Convert a GSD trajectory to XTC format. |
| [trjcat](commands/trjcat.md) | Concatenate DROPPS XTC trajectory segments in time order. |
| [eneconv](commands/eneconv.md) | Concatenate and time-filter DROPPS CSV-format energy files. |
| [convert-tpr](commands/convert-tpr.md) | Convert a trusted legacy DROPPS TPR into the portable TPR v2 format. |
| [report-methods](commands/report-methods.md) | Report the system, simulation methods, and provenance stored in a DROPPS TPR. |
| [pdb2bond](commands/pdb2bond.md) | Add PDB CONECT records from a DROPPS run file or system topology. |

### Analyze structure and contacts

Measure conformations, contacts, correlations, and assemblies.

| Command | Purpose |
| --- | --- |
| [gyrate](commands/gyrate.md) | Calculate chain radii of gyration, including optional dense/dilute resolution. |
| [polystat](commands/polystat.md) | Calculate size, shape, and backbone statistics for many protein copies. |
| [ss](commands/ss.md) | Infer H/E/C states from residue-level coarse-grained backbone geometry. |
| [angle](commands/angle.md) | Calculate backbone-angle profiles and time series for equivalent protein chains. |
| [distance](commands/distance.md) | Calculate fixed, intra-chain, inter-chain, minimum, and maximum distances. |
| [rdf](commands/rdf.md) | Calculate radial distribution functions and cumulative coordination numbers. |
| [contact](commands/contact.md) | Calculate residue-level contact maps and contact-number time series. |
| [contact.ng](commands/contact.ng.md) | Calculate residue-level contact maps with memory-efficient aggregation. |
| [timecorr](commands/timecorr.md) | Calculate residue-contact correlation functions and lifetime matrices. |
| [cstat](commands/cstat.md) | Calculate residue-level statistics from a precomputed contact map. |
| [rmsd](commands/rmsd.md) | Calculate PBC-aware RMSD with configurable fitting and output granularity. |
| [rmsf](commands/rmsf.md) | Calculate PBC-aware atomic or residue RMS fluctuations. |
| [sasa](commands/sasa.md) | Calculate coarse-grained solvent-accessible surface area. |
| [assembly](commands/assembly.md) | Analyze the formation, size, composition, and shape of molecular assemblies. |
| [pbcontact](commands/pbcontact.md) | Measure direct and network-mediated interactions with periodic molecular images. |

### Analyze phase behavior and transport

Measure slab phases, molecular motion, interfaces, and viscosity.

| Command | Purpose |
| --- | --- |
| [density](commands/density.md) | Calculate a one-dimensional mass- or charge-density profile. |
| [densmap](commands/densmap.md) | Calculate planar two-dimensional mass, number, or charge density maps. |
| [coexistence](commands/coexistence.md) | Analyze dense/dilute coexistence and interface width in a planar slab trajectory. |
| [exchange](commands/exchange.md) | Analyze molecular exchange and phase residence times across a planar slab interface. |
| [phase-msd](commands/phase-msd.md) | Calculate phase-conditioned molecular MSD with phase-translation removal in a planar slab. |
| [msd](commands/msd.md) | Calculate the mean-square displacement of a selected atom group. |
| [surftension](commands/surftension.md) | Calculate planar-interface surface tension from a DROPPS TPR and XTC trajectory. |
| [viscosity-gk](commands/viscosity-gk.md) | Calculate zero-shear viscosity from a DROPPS runtime pressure tensor. |

## Compatibility commands

`intra-distance` / `idist` and `inter-distance` / `odist` remain accepted with their legacy parameters. Use `dps ALIAS --help` for those interfaces; new scripts should use [distance](commands/distance.md) with `--scope intra-chain` or `--scope inter-chain`.

The former `extract` workflow is handled by [trjconv](commands/trjconv.md). The old `pdb2cgps` command is named [pdb2dps](commands/pdb2dps.md).

## Keeping this reference current

Run `python scripts/update_command_docs.py` after a CLI change and `python scripts/update_command_docs.py --check` in validation. The check covers the registry, practical notes, examples, and rendered help for every command.
