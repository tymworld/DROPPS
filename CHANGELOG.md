# Changelog

## 1.0 — 2026-09-08

- Published a stable `dps` command-line interface for model construction,
  simulation, trajectory preparation, and phase-separation analysis.
- Introduced checksum-verified portable TPR v2 run inputs and explicit
  validation of force fields, topologies, coordinates, boxes, and cutoffs.
- Added reproducible random seeds, CPU/OpenCL/CUDA execution, checkpoints, and
  restart support.
- Added five versioned Google Colab tutorials spanning quick validation, system
  construction, compact-box NPT, z-axis tenfold expansion, elongated-box NVT,
  trajectory processing, and analysis with PNG output.
- Set the Colab demonstration defaults to 80 copies of
  `FWFWFWFWFWFWFWFW`, with 1.5 nm Lennard-Jones and Coulomb cutoffs.
- Added publication metadata, an Apache-2.0 license, automated tests, and
  source/package validation.

The 0.x repository history remains available through earlier commits. Version
1.0 removes obsolete experimental CLI commands and is not intended as a
drop-in replacement for private implementation imports used by old notebooks.
