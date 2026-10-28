# DROPPS 1.0.0 release validation

Date: 2026-10-28. Commands below are run from the source root.

## Runtime matrix

| Python | OpenMM | NumPy | MDAnalysis | Result |
| --- | --- | --- | --- | --- |
| 3.9.6 | 8.1.1 | 1.26.4 | 2.7.0 | 366 tests passed |
| 3.12.13 | 8.1.0 | 1.26.4 | 2.10.0 | 366 tests passed |
| 3.12.13 | 8.6.1 | 2.5.4 | 2.11.0 | 366 tests passed |

The suite exercises simulation preparation, a short Reference-platform run,
minimization, checkpoint/State continuation, TPR migration, trajectory analysis,
CLI contracts, numerical regressions and generated-document consistency.
Historical CUDA validation on 2026-10-10 used the 0.5.0b4 development snapshot,
OpenMM 8.5.2, CUDA 12.8 and an NVIDIA RTX 5080: a 100-step HPS simulation with
20 residues produced a readable XTC trajectory. The CUDA-aware installer also
provides a real Context probe for deployment environments.

## Reproduction

```bash
python -m unittest discover -s tests -p 'test_*.py'
python scripts/update_readme_commands.py --check
python scripts/update_command_docs.py --check
python examples/colab/validate_notebooks.py
ruff check --isolated --select E9,F63,F7,F82 --exclude backups,'._*' dropps scripts tests examples/colab
python scripts/prepare_release.py --expected-version 1.0.0
```

The GitHub Actions matrix additionally runs on Linux with Python 3.9, 3.10 and
3.12. The local runtime matrix above records the completed macOS checks.

## Documentation and packaging checks

- 43 public commands and 679 option entries; examples parse using the real CLI.
- Full README catalog and generated reference pages match the command registry.
- Five Colab notebooks pass syntax, form, command-presence and checksum checks.
- Wheel metadata, RECORD hashes/sizes and source-archive metadata are checked.
- Wheel and source archive contain identical runtime code.
- Source archive includes the command reference, tutorials, tests, scripts,
  Colab notebooks, license and citation information.
- Builds stage clean source files, excluding caches, NAS/macOS metadata and
  obsolete backup modules.

## Publication procedure

1. Preserve the previous release assets, metadata, working tree and Git history.
2. Integrate the earlier public main commit into the reviewed source history.
3. Build from the final source, test clean wheel and source installations, and
   record the source commit and asset SHA-256 values in the release manifest.
4. Replace `v1.0.0` and its release assets with the owner's authorization.
5. Download the published assets and verify their checksums and metadata.

Installations of the earlier version `1.0` must explicitly reinstall the new
wheel because PEP 440 compares `1.0` and `1.0.0` as the same version.
