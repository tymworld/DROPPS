# dps rmsd

[All commands](../command-reference.md) · [User guide](../user-guide.md)

Calculate PBC-aware RMSD with configurable fitting and output granularity.

DROPPS 1.0.0. This page is generated from the executable CLI; defaults, choices, required inputs, and output options below match the installed command.

## Usage and options

```text
usage: dps rmsd [-h] [-db] -s FILE -f FILE [-n FILE] -o FILE
                [-se GROUP [GROUP ...] | -ex EXPR] [-b START_TIME]
                [-e END_TIME] [-dt DELTA_TIME] [-r REFERENCE_TIME]
                [--fit-mode {none,molecule,selection}]
                [-om {molecule,selection}] [--mass-weighted] [--summary-only]

Calculate PBC-aware RMSD with configurable fitting and output granularity.

Input:
  -s FILE, --run-input FILE
                        Input DROPPS run file (.tpr) containing topology and
                        coordinates. (required)
  -f FILE, --input FILE
                        Input trajectory file (.xtc). (required)
  -n FILE, --index FILE
                        Optional index file (.ndx) defining additional atom
                        groups. (default: None)
  -se GROUP [GROUP ...], --selection-groups GROUP [GROUP ...]
                        Index groups containing selected atoms across any
                        number of molecules; molecules are split
                        automatically. (default: None)
  -ex EXPR, --select EXPR, --selection-expression EXPR
                        DROPPS selection expression applied across all
                        molecules; repeat the option to analyze more than one
                        region. (default: None)

Output:
  -o FILE, --output FILE
                        Output RMSD time series (.xvg); a matching detailed
                        .csv is also written unless --summary-only is used.
                        (required)
  -om {molecule,selection}, --output-mode {molecule,selection}
                        RMSD calculation/output granularity: one value per
                        molecule or one pooled value for all atoms in each
                        selection. (default: molecule)

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
                        Coordinate fitting before RMSD: none preserves
                        physical translation/rotation, molecule fits every
                        molecule independently, and selection applies one fit
                        to all atoms in each selection. (default: molecule)
  --mass-weighted       Use atom masses for centering, fitting, and RMSD.
                        (default: False)
  --summary-only        Skip the detailed CSV and write only the XVG time
                        series. (default: False)

Select a region once across all molecule copies. DROPPS splits it by topology molecule and unwraps each complete molecule. Fitting and RMSD reporting can then operate per molecule or per selection.

Notes:
  Compare selected coordinates with an explicit reference frame/time, optionally after translation/rotation fitting and molecular PBC reconstruction.

  Fit and measurement selections can differ. Per-molecule and whole-selection outputs answer different questions; all RMSD distances are in nm.

Examples:
  dps rmsd -s run.tpr -f run.xtc -o rmsd.xvg --select "(mol PROT) & (resid 20-80)"
  dps rmsd -s run.tpr -f run.xtc -o rmsd-system.xvg --selection-groups 1 --fit-mode selection --output-mode selection
```
