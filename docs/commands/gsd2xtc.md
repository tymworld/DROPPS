# dps gsd2xtc

[All commands](../command-reference.md) · [User guide](../user-guide.md)

Convert a GSD trajectory to XTC format.

DROPPS 1.0.0. This page is generated from the executable CLI; defaults, choices, required inputs, and output options below match the installed command.

## Usage and options

```text
usage: dps gsd2xtc [-h] [-db] -f FILE -o FILE [-ts TIME_STEP] [--box X Y Z]

Convert a GSD trajectory to XTC format.

Input:
  -f FILE, --input FILE
                        Input trajectory file (.gsd). (required)

Output:
  -o FILE, --output FILE
                        Output trajectory file (.xtc); the extension is added
                        if omitted. (required)

Parameters:
  -h, --help            Show this help message and exit.
  -db, --debug          Show runtime context, parsed arguments, timing, and
                        tracebacks. (default: False)
  -ts TIME_STEP, --time-step TIME_STEP
                        Time between consecutive GSD frames, in ns. (default:
                        0.1)
  --box X Y Z           Box lengths in nm; required when the GSD file has no
                        periodic box. (default: None)

Notes:
  Convert a GSD trajectory whose coordinates are expressed in nm to XTC. The GSD box is used unless --box overrides its lengths in nm.

  --time-step is the spacing between saved frames in ns; output starts at time zero. Confirm the units used by the GSD producer before conversion.

Examples:
  dps gsd2xtc -f run.gsd -o run.xtc
```
