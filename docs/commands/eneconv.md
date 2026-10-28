# dps eneconv

[All commands](../command-reference.md) · [User guide](../user-guide.md)

Concatenate and time-filter DROPPS CSV-format energy files.

DROPPS 1.0.0. This page is generated from the executable CLI; defaults, choices, required inputs, and output options below match the installed command.

## Usage and options

```text
usage: dps eneconv [-h] [-db] -f FILE [FILE ...] -o FILE [-b START_TIME]
                   [-e END_TIME] [-dt DELTA_TIME] [-tu {fs,ps,ns,us,ms,s}]
                   [--allow-incompatible]

Concatenate and time-filter DROPPS CSV-format energy files.

Input:
  -f FILE [FILE ...], --input FILE [FILE ...]
                        Input DROPPS CSV-format energy files (.edr).
                        (required)

Output:
  -o FILE, --output FILE
                        Output concatenated DROPPS energy file (.edr).
                        (required)

Parameters:
  -h, --help            Show this help message and exit.
  -db, --debug          Show runtime context, parsed arguments, timing, and
                        tracebacks. (default: False)
  -b START_TIME, --start-time START_TIME
                        First energy-sample time to write. (default: None)
  -e END_TIME, --end-time END_TIME
                        Last energy-sample time to write. (default: None)
  -dt DELTA_TIME, --delta-time DELTA_TIME
                        Approximate physical interval between output samples.
                        (default: None)
  -tu {fs,ps,ns,us,ms,s}, --time-unit {fs,ps,ns,us,ms,s}
                        Unit used by -b, -e, and -dt. (default: ps)
  --allow-incompatible  Allow differing scientific metadata while still
                        requiring identical fields and units. (default: False)

Notes:
  Merge DROPPS CSV EDR segments with matching columns and units. This command does not read binary GROMACS EDR.

  Overlapping times prefer the later segment unless duplicate retention is enabled. Check chronology and step numbers before combining unrelated runs.

Examples:
  dps eneconv -f run.edr run.part0002.edr -o complete.edr
```
