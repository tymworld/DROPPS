# DROPPS 1.0 Google Colab tutorials

These five notebooks use the public `dps` CLI from DROPPS 1.0.0. The [complete command reference](../../docs/command-reference.md) covers the full 43-command toolkit beyond the tutorial examples.

For a single-file handoff, use `DROPPS-1.0-Colab-Package.zip`; it contains all
five notebooks, this guide, the generator/validator, checksums, and the matching
DROPPS 1.0 wheel.

## Notebook suite

| Notebook | Purpose | Principal DROPPS commands |
|---|---|---|
| `00_DROPPS_1_0_Quickstart.ipynb` | Tiny end-to-end interface and file-flow validation | `pdb2dps`, `genmesh`, `grompp`, `mdrun`, `check`, `density` |
| `01_DROPPS_1_0_Model_and_System_Builder.ipynb` | Many copies of one protein in a compact cubic box; optional co-components, PTMs, and structural restraints | `pdb2dps`, `addangle`, `genelastic`, `modifyres`, `genmesh` |
| `02_DROPPS_1_0_Simulation.ipynb` | Required compact-box NPT → one-axis ×10 expansion → elongated-box NVT protocol, with TPR v2, CPU/GPU, checkpoint, and restart support | `grompp`, `mdrun`, `check`, `editconf` |
| `03_DROPPS_1_0_Trajectory_Preparation.ipynb` | NVT-only input validation, reusable NDX selections, PBC processing, centering, fitting, conversion, snapshots | `check`, `make_ndx`, `trjconv` |
| `04_DROPPS_1_0_Phase_Separation_Analysis.ipynb` | NVT-only phase behavior, interactions, conformation, dynamics, and spontaneous assembly analysis | `density`, `contact`, `cstat`, `gyrate`, `angle`, `idist`, `odist`, `msd`, `rmsd`, `assembly` |

The builder defaults to 80 copies of the demonstration sequence
`FWFWFWFWFWFWFWFW` on a 5×5×5 grid. The simulation notebook uses 1.5 nm as
the default cutoff for both Lennard-Jones and Coulomb interactions. It displays
the manuscript-scale defaults (300 ns NPT at 0.01 ps, then 3 μs NVT at 0.02 ps)
but initially selects a 100-step-per-stage validation mode. This verifies
interfaces and stage transitions; it does not reproduce scientific conclusions
or equilibrated phase behavior.

## Phase-coexistence protocol

The primary workflow is deliberately strict:

1. Pack many monomers of one protein into a compact cubic box.
2. Run the compact box under NPT to form/equilibrate the dense phase.
3. Use `dps editconf -mz 10` to elongate z exactly tenfold and center
   the NPT configuration in the new box.
4. Compile a new run input with pressure coupling disabled and run the elongated
   box under NVT.
5. Process and analyze only `slab_nvt.tpr` + `slab_nvt.xtc`. The trajectory and
   analysis notebooks reject a TPR whose `pcoulp` setting is true.

The elongated axis is fixed to z, and both the tenfold expansion and the
NPT-before-NVT order are enforced. Co-phase separation is available by enabling
additional components; it is not the default construction example. The analysis
notebook always computes the density profile along z and exports both
`density.xvg` and a directly viewable `density_z.png`; other enabled plotting
analyses also save PNG figures alongside their numeric outputs.

## Running in Google Colab

1. Open a versioned notebook from the badges in the GitHub repository, or upload
   one notebook to Google Colab for offline use.
2. For simulation notebooks, select a GPU runtime when CUDA is required.
3. The installation cell defaults to the versioned wheel attached to the public
   GitHub `v1.0.0` release. Uploading a local wheel remains available as an
   offline fallback.
4. Run cells from top to bottom. Every notebook writes a JSON provenance
   manifest and offers a ZIP download.
5. Pass the ZIP from one focused notebook to the next, or reuse the prior
   notebook's `/content/dropps_*` directory while the same Colab runtime remains
   active.

The model and simulation notebooks retain PDB, ITP, TOP, MDP, TPR, checkpoints,
portable state, and manifests together so a run can be transferred to an HPC
system without reconstructing its scientific inputs.

## Further analyses

The complete release also includes `densmap`, `coexistence`, `exchange`, `phase-msd`, `polystat`, `ss`, `rdf`, `rmsf`, `sasa`, `contact.ng`, `timecorr`, `pbcontact`, `surftension`, and `viscosity-gk`. See their [individual references](../../docs/command-reference.md) for inputs, assumptions, units, and examples.

## Maintenance and validation

`generate_notebooks.py` is the source of the generated notebooks. Edit the
generator, run it, and then run:

```bash
python3 validate_notebooks.py
```

The validator checks notebook structure, Python syntax, Colab form annotations,
forbidden legacy APIs/version strings, current CLI command names, and recorded
SHA-256 checksums. `SHA256SUMS` covers the five generated notebooks.
