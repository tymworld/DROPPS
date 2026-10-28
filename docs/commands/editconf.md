# dps editconf

[All commands](../command-reference.md) · [User guide](../user-guide.md)

Resize a PDB simulation box and optionally unwrap coordinates along selected axes.

DROPPS 1.0.0. This page is generated from the executable CLI; defaults, choices, required inputs, and output options below match the installed command.

## Usage and options

```text
usage: dps editconf [-h] [-db] -f FILE -o FILE
                    [-x X_AXIS | -mx MULTIPLY_X_AXIS]
                    [-y Y_AXIS | -my MULTIPLY_Y_AXIS]
                    [-z Z_AXIS | -mz MULTIPLY_Z_AXIS] [-px] [-py] [-pz]

Resize a PDB simulation box and optionally unwrap coordinates along selected axes.

Input:
  -f FILE, --structure FILE
                        Input structure file (.pdb). (required)

Output:
  -o FILE, --output FILE
                        Output structure file (.pdb); the extension is added
                        if omitted. (required)

Parameters:
  -h, --help            Show this help message and exit.
  -db, --debug          Show runtime context, parsed arguments, timing, and
                        tracebacks. (default: False)
  -x X_AXIS, --x-axis X_AXIS
                        New box length along x, in nm. (default: None)
  -mx MULTIPLY_X_AXIS, --multiply-x-axis MULTIPLY_X_AXIS
                        Multiplier applied to the current x box length.
                        (default: None)
  -y Y_AXIS, --y-axis Y_AXIS
                        New box length along y, in nm. (default: None)
  -my MULTIPLY_Y_AXIS, --multiply-y-axis MULTIPLY_Y_AXIS
                        Multiplier applied to the current y box length.
                        (default: None)
  -z Z_AXIS, --z-axis Z_AXIS
                        New box length along z, in nm. (default: None)
  -mz MULTIPLY_Z_AXIS, --multiply-z-axis MULTIPLY_Z_AXIS
                        Multiplier applied to the current z box length.
                        (default: None)
  -px, --treat-pbc-x    Unwrap coordinates across periodic boundaries along x.
                        (default: False)
  -py, --treat-pbc-y    Unwrap coordinates across periodic boundaries along y.
                        (default: False)
  -pz, --treat-pbc-z    Unwrap coordinates across periodic boundaries along z.
                        (default: False)

Notes:
  Read a periodic PDB and write an enlarged box with translated coordinates. Use either an absolute length in nm or a multiplier for each axis.

  Expanded axes are unwrapped automatically. Unchanged axes are unwrapped only when requested. A changed box requires a new 'grompp' run before simulation.

Examples:
  dps editconf -f system.pdb -o resized.pdb --multiply-x-axis 2 --multiply-y-axis 2 --multiply-z-axis 2
```
