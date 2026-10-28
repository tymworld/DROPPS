# DROPPS 1.0.0 integration record

Updated 2026-10-28. This release integrates the complete local 0.5.0b4 development line with the previously published 1.0 runtime fixes and Colab workflows. At the owner’s request it replaces the existing `v1.0.0` release.

## Scope

- 43 public commands, including the phase, transport, contact and structural-analysis tools developed on the 0.5.0 line.
- Full terminal help, README examples and 43 generated command pages, checked against the actual argument parsers.
- Portable TPR v2, legacy migration, restart validation, trajectory and energy conversion, and reproducible methods reports.
- Apache-2.0 license and author attribution retained from the existing public release.

## Corrections found during integration

- MSD now unwraps every saved frame before applying time-window or stride selection; starting partway through a trajectory no longer initializes unwrapping at the wrong frame. The FFT dependency is declared.
- Initial TPR positions are converted from OpenMM units to the Angstrom coordinates expected by MDAnalysis.
- Repeated molecule types in `genmesh` produce the same coordinate ordering as the merged TOP. ITP references are relative to the output TOP directory.
- All-atom mapping uses only the first model, validates the C-alpha count and centers coordinates. Sequence construction has a reproducible seed and bounded retries.
- Mesh box type, fractional box multipliers, topology consistency, cutoffs and selection bounds are validated.
- Minimization-only runs are supported and described accurately by `report-methods`.
- Fractional contact-analysis times are accepted. RMSF grouping options remain mutually exclusive on Python 3.9.
- Release builds use a clean staging tree, preventing NAS/macOS metadata from entering source archives. Wheel RECORD, wheel/sdist runtime equality and SHA-256 checks are enforced.

## Validation and traceability

See [release-checklist.md](release-checklist.md) for the tested dependency matrix and explicit limitations. Local audit backups preserve the initial working tree, Git metadata, previous release metadata and its four assets. The previous public commit remains in the integrated Git history.

The [historical beta audit](history/release-audit-20260814.md) describes the earlier 0.5.0 state; it is not the current release status.
