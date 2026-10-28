# dps rmsf

[All commands](../command-reference.md) · [User guide](../user-guide.md)

Calculate PBC-aware atomic or residue RMS fluctuations.

DROPPS 1.0.0. This page is generated from the executable CLI; defaults, choices, required inputs, and output options below match the installed command.

## Usage and options

```text
usage: dps rmsf [-h] [-db] -s FILE -f FILE [-n FILE] -o FILE [-od FILE]
                [-oa FILE] [-or FILE] [-se GROUP [GROUP ...] | -ex EXPR]
                [-b START_TIME] [-e END_TIME] [-dt DELTA_TIME]
                [-r REFERENCE_TIME] [--fit-mode {none,molecule,selection}]
                [--fit-weighting {mass,uniform}]
                [--grouping {atom,residue} | --residue]
                [--group-weighting {mass,uniform}] [--summary-only]

Calculate PBC-aware atomic or residue RMS fluctuations.

Input:
  -s FILE, --run-input FILE
                        Input DROPPS run file (.tpr) containing topology and
                        coordinates. (required)
  -f FILE, --input FILE
                        Input trajectory file (.xtc or .pdb). (required)
  -n FILE, --index FILE
                        Optional index file (.ndx) defining additional atom
                        groups. (default: None)
  -se GROUP [GROUP ...], --selection-groups GROUP [GROUP ...]
                        Index groups containing atoms across one or more
                        molecule copies; molecules are split automatically.
                        (default: None)
  -ex EXPR, --select EXPR, --selection-expression EXPR
                        DROPPS selection expression; repeat the option to
                        calculate more than one RMSF profile. (default: None)

Output:
  -o FILE, --output FILE
                        Output RMSF profile (.xvg); a matching per-molecule
                        .csv is also written unless --summary-only is used.
                        (required)
  -od FILE, --output-deviation FILE
                        Optional positional RMS deviation from the reference
                        (.xvg). (default: None)
  -oa FILE, --output-average FILE
                        Optional average aligned coordinates with B factors
                        (.pdb). (default: None)
  -or FILE, --output-reference FILE
                        Optional reference coordinates with B factors (.pdb).
                        (default: None)

Parameters:
  -h, --help            Show this help message and exit.
  -db, --debug          Show runtime context, parsed arguments, timing, and
                        tracebacks. (default: False)
  -b START_TIME, --start-time START_TIME
                        First trajectory time to analyze, in ns. (default:
                        None)
  -e END_TIME, --end-time END_TIME
                        Last trajectory time to analyze, in ns. (default:
                        None)
  -dt DELTA_TIME, --delta-time DELTA_TIME
                        Approximate interval between analyzed frames, in ns.
                        (default: None)
  -r REFERENCE_TIME, --reference-time REFERENCE_TIME
                        Reference time in ns; the nearest saved frame is used.
                        By default use the first analyzed frame.
  --fit-mode {none,molecule,selection}
                        Least-squares fit scope: none, every molecule
                        independently, or all atoms in each selection
                        together. (default: molecule)
  --fit-weighting {mass,uniform}
                        Weights used for fit centers and rotations. (default:
                        mass)
  --grouping {atom,residue}
                        Report fluctuations for individual atoms or residues;
                        defaults to atom.
  --residue             Calculate mass-weighted residue RMSF (gmx rmsf
                        compatibility). (default: None)
  --group-weighting {mass,uniform}
                        Weights used to average atomic covariance within each
                        residue. (default: mass)
  --summary-only        Skip the detailed CSV and write only requested summary
                        outputs. (default: False)

RMSF is sqrt(trace(<(x-<x>)(x-<x>)^T>)). Complete molecules are reconstructed across PBC before every molecule is fitted to its own reference coordinates by default.

Notes:
  Measure time-averaged positional fluctuations after the chosen PBC/fitting procedure, either per particle/residue or pooled across compatible molecules.

  RMSF uses nm; optional outputs include the mean structure and deviation from a reference. Translation/rotation left in the coordinates contributes to the fluctuation.

Examples:
  dps rmsf -s run.tpr -f run.xtc -o rmsf.xvg --select "(mol PROT) & (resid 20-80)" --residue
  dps rmsf -s run.tpr -f run.xtc -o rmsf.xvg --selection-groups 1 --output-deviation reference-deviation.xvg --output-average average.pdb
```
