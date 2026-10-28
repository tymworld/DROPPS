# dps energy

[All commands](../command-reference.md) · [User guide](../user-guide.md)

Interactively select observables from a DROPPS EDR and export XVG.

DROPPS 1.0.0. This page is generated from the executable CLI; defaults, choices, required inputs, and output options below match the installed command.

## Usage and options

```text
usage: dps energy [-h] [-db] -f FILE [-o FILE] [--list]
                  [--terms TERMS [TERMS ...]] [-b START_TIME] [-e END_TIME]
                  [-dt DELTA_TIME] [--x-axis {time,step}]

Interactively select observables from a DROPPS EDR and export XVG.

Input:
  -f FILE, --input FILE
                        Input DROPPS CSV-format energy file (.edr). (required)

Output:
  -o FILE, --output FILE
                        Output selected energy time series (.xvg). (default:
                        None)

Parameters:
  -h, --help            Show this help message and exit.
  -db, --debug          Show runtime context, parsed arguments, timing, and
                        tracebacks. (default: False)
  --list                List available terms and exit without writing XVG.
                        (default: False)
  --terms TERMS [TERMS ...]
                        Terms to export by name, unique prefix, or 1-based
                        number. If omitted, select terms interactively.
                        (default: None)
  -b START_TIME, --start-time START_TIME
                        First sample time to export, in ps. (default: None)
  -e END_TIME, --end-time END_TIME
                        Last sample time to export, in ps. (default: None)
  -dt DELTA_TIME, --delta-time DELTA_TIME
                        Approximate output interval, in ps. (default: None)
  --x-axis {time,step}  XVG x-axis quantity. (default: time)

Notes:
  Read the versioned CSV table written by DROPPS mdrun/rerun. --list displays available terms; --terms selects the columns to export as XVG.

  The output header records units. Potential/kinetic/total energies use kJ/mol, temperature K, box lengths nm, volume nm^3, and pressure bar where available.

Examples:
  dps energy -f run.edr --list
  dps energy -f run.edr -o thermo.xvg --terms temperature volume
```
