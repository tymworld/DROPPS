# dps angle

[All commands](../command-reference.md) · [User guide](../user-guide.md)

Calculate backbone-angle profiles and time series for equivalent protein chains.

DROPPS 1.0.0. This page is generated from the executable CLI; defaults, choices, required inputs, and output options below match the installed command.

## Usage and options

```text
usage: dps angle [-h] [-db] -s FILE -f FILE [-n FILE] [-se SELECTION]
                 [-b START_TIME] [-e END_TIME] [-dt DELTA_TIME] [-p]
                 [-ot FILE] [-or FILE] [-os FILE] [-ov FILE]

Calculate backbone-angle profiles and time series for equivalent protein chains.

Input:
  -s FILE, --run-input FILE
                        Input DROPPS run file (.tpr) containing the system and
                        simulation settings. (required)
  -f FILE, --input FILE
                        Input trajectory file (.xtc). (required)
  -n FILE, --index FILE
                        Optional index file (.ndx) defining additional atom
                        groups. (default: None)

Output:
  -ot FILE, --output-time FILE
                        Output chain-averaged angle time series (.xvg); the
                        extension is added if omitted. (default: None)
  -or FILE, --output-residue FILE
                        Output time-averaged angle profile by residue (.xvg);
                        the extension is added if omitted. (default: None)
  -os FILE, --output-residue-statistic FILE
                        Output residue-wise angle mean and standard deviation
                        (.xvg); the extension is added if omitted. (default:
                        None)
  -ov FILE, --output-verbose FILE
                        Output per-chain, per-residue angle time series
                        (.xvg); the extension is added if omitted. (default:
                        None)

Parameters:
  -h, --help            Show this help message and exit.
  -db, --debug          Show runtime context, parsed arguments, timing, and
                        tracebacks. (default: False)
  -se SELECTION, --selection SELECTION
                        Index group containing angle-center beads; chains are
                        split and terminal beads are ignored. If omitted,
                        prompt interactively. (default: None)
  -b START_TIME, --start-time START_TIME
                        First trajectory time to analyze, in ns. (default:
                        None)
  -e END_TIME, --end-time END_TIME
                        Last trajectory time to analyze, in ns. (default:
                        None)
  -dt DELTA_TIME, --delta-time DELTA_TIME
                        Approximate interval between analyzed frames, in ns.
                        (default: None)
  -p, --treat-pbc       Unwrap molecules across periodic boundaries before
                        analysis. (default: False)

Notes:
  Measure consecutive intrachain bead angles for a selected group. Chain-terminal beads have no centered triplet and are excluded.

  Choose at least one output for the time series, residue profile, standard deviation, or verbose per-chain values. Output angles are in degrees and trajectory time is in ns.

Examples:
  dps angle -s run.tpr -f run.xtc --output-residue angles-by-residue.xvg
```
