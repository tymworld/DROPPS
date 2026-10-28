# dps convert-tpr

[All commands](../command-reference.md) · [User guide](../user-guide.md)

Convert a trusted legacy DROPPS TPR into the portable TPR v2 format.

DROPPS 1.0.0. This page is generated from the executable CLI; defaults, choices, required inputs, and output options below match the installed command.

## Usage and options

```text
usage: dps convert-tpr [-h] [-db] -s FILE -o FILE

Convert a trusted legacy DROPPS TPR into the portable TPR v2 format.

Input:
  -s FILE, --run-input FILE
                        Trusted legacy or portable DROPPS TPR to read.
                        (required)

Output:
  -o FILE, --output FILE
                        Portable TPR v2 output file. (required)

Parameters:
  -h, --help            Show this help message and exit.
  -db, --debug          Show runtime context, parsed arguments, timing, and
                        tracebacks. (default: False)

Notes:
  Convert a trusted legacy pickle TPR into ZIP/XML/JSON/NumPy TPR v2. Legacy pickle loading can execute code, so use files from a trusted source.

  For an old OpenMM checkpoint, first export a portable State using its original environment. Converting the TPR alone does not convert the native checkpoint.

Examples:
  dps convert-tpr -s legacy.tpr -o portable.tpr
```
