# dps report-methods

[All commands](../command-reference.md) · [User guide](../user-guide.md)

Report the system, simulation methods, and provenance stored in a DROPPS TPR.

DROPPS 1.0.0. This page is generated from the executable CLI; defaults, choices, required inputs, and output options below match the installed command.

## Usage and options

```text
usage: dps report-methods [-h] [-db] -s FILE [-o FILE] [-ol FILE]

Report the system, simulation methods, and provenance stored in a DROPPS TPR.

Input:
  -s FILE, --run-input FILE
                        Input DROPPS run file (.tpr) containing the recorded
                        methods. (required)

Output:
  -o FILE, --output FILE
                        Output unformatted methods report (.out); when neither
                        output option is used, print to the terminal.
                        (default: None)
  -ol FILE, --latex-output FILE
                        Optional LaTeX-formatted methods report (.tex).
                        (default: None)

Parameters:
  -h, --help            Show this help message and exit.
  -db, --debug          Show runtime context, parsed arguments, timing, and
                        tracebacks. (default: False)

Notes:
  Generate text and optional LaTeX methods from TPR settings and provenance. The report does not read restart metadata or record the actual runtime platform.

  The report describes stored inputs; supply run overrides and subsequent analysis choices when preparing a complete experimental record.

Examples:
  dps report-methods -s run.tpr -o methods.out -ol methods.tex
```
