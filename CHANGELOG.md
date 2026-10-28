# Changelog

All notable release changes are recorded here. The complete 0.5.0 development line is now published as 1.0.0.

## 1.0.0 — 2026-10-28

This owner-authorized replacement of the earlier `v1.0.0` release integrates the
full 0.5.0b4 feature set and preserves the earlier public runtime improvements.

### Added and completed

- All 43 public commands, with complete terminal help, task-based navigation,
  examples, input/output notes, units, defaults and 43 individual reference pages.
- Five refreshed Colab workflows and checksummed downloadable notebooks.
- Source distribution documentation and tests; CI for Python 3.9, 3.10 and 3.12.
- Portable TPR v2, restart support, migration, analysis and conversion commands
  described in the 0.5.0b4 entry below.

### Fixed

- MSD unwrapping before time-window/stride selection, FFT dependency declaration,
  and minimum-image handling across multiple box crossings.
- TPR initial-coordinate conversion to MDAnalysis Angstrom units.
- Genmesh coordinate/topology order for repeated molecule types, relative ITP
  references, force-field consistency and box-type handling.
- First-model C-alpha mapping, reproducible chain generation, bounded placement
  retries, fractional box multipliers and stricter malformed-input validation.
- Minimization-only execution and accurate methods reports for that mode.
- Fractional contact times and Python 3.9 RMSF option exclusivity.
- Clean source staging for release builds, wheel RECORD integrity checks and
  wheel/source runtime-payload comparison.

### Compatibility

- Canonical version strings and assets now use `1.0.0` (previously `1.0`). Because
  this replaces an existing release, reinstall the new wheel explicitly when
  upgrading from that release; PEP 440 treats `1.0` and `1.0.0` as equivalent.
- Apache-2.0 licensing and attribution are retained from the public 1.0 release.
- The dependency matrix and validation scope are recorded in
  [the release checklist](docs/release-checklist.md).

## 0.5.0b4 — 2026-08-14

### Added

- Portable, checksummed TPR v2 containers and the `convert-tpr` migration
  command for trusted legacy run inputs.
- Hardware-aware CUDA 12/13 installer with real OpenMM Context validation.
- Unified time-based trajectory conversion, concatenation, energy conversion,
  and reproducible simulation-method reporting.
- New density-map, distance, RDF, RMSD, RMSF, SASA, polymer-statistics,
  secondary-structure, and periodic-image interaction analyses.
- Shared numerical cores for geometry, time selection, density, fitting, and
  streaming statistics, with expanded regression coverage.
- Global `--debug` diagnostics with argv, environment, dependency versions,
  parsed parameters, elapsed time, and exception tracebacks.

### Changed

- Python support now starts at 3.9. Python 3.10+ uses the current OpenMM 8.5 /
  NumPy 2 stack by default; a constrained OpenMM 8.1 / NumPy 1 migration stack
  remains available.
- `trjconv` and analysis commands select frames from physical timestamps and
  reject incompatible nonuniform sampling where a constant interval is
  required.
- `density` and coexistence-family analyses share periodic centering, binning,
  phase definitions, and per-frame box geometry.
- Command registration is centralized and package initializers no longer
  eagerly import the entire scientific command graph.
- Runtime configuration, restart validation, append rollback, pressure-tensor
  reporting, and analysis output provenance are more explicit.
- Terminal output now uses consistent colored `DEBUG`, `INFO`, `OK`, `WARNING`,
  and `ERROR` levels, with stable plain text for redirected output. Every
  functional command reports its inputs, outputs, parameters, and result.
  Progress renderers may continue to use stderr without being mislabeled as
  warnings.
- Public CLI help now guarantees a conflict-free lowercase one- or two-letter
  short option for every input and output, alongside lowercase kebab-case long
  options. Every output short option begins with `o`; historical lowercase
  nonstandard aliases remain accepted but are hidden, while uppercase option
  names are removed.
- Frequently used options now have conflict-aware mnemonic shortcuts, including
  `trjconv -c/--center`, `-se/--select`, `-p/--pbc`, `-ca/--center-axis`,
  `-ff/--forcefield`, and `pdb2dps -pt/--post-translational-modification`.
- Topology inputs use `-p`, while topology outputs consistently use
  `-op/--output-topology`.

### Removed

- The legacy `extract` command; equivalent single-frame and range extraction is
  provided by `trjconv`.
- Generated Python bytecode and Synology/macOS metadata from the source tree.

### Compatibility notes

- OpenMM 8.1 must not be combined with NumPy 2 because its XTC extension uses
  the NumPy 1 ABI.
- Legacy pickle TPR files must be treated as trusted input and converted to TPR
  v2 before long-term use.
- `intra-distance` and `inter-distance` remain hidden compatibility aliases for
  the unified `distance` command.
- Existing multi-character single-dash CLI options remain compatibility aliases
  for 0.5.0b4; scripts should migrate to the long forms shown by `dps COMMAND -h`.
