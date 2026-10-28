# dps make_ndx

[All commands](../command-reference.md) · [User guide](../user-guide.md)

Create or extend an atom-group index file interactively.

DROPPS 1.0.0. This page is generated from the executable CLI; defaults, choices, required inputs, and output options below match the installed command.

## Usage and options

```text
usage: dps make_ndx [-h] [-db] -s FILE [-n FILE] -o FILE

Create or extend an atom-group index file interactively.

Input:
  -s FILE, --run-input FILE
                        Input DROPPS run file (.tpr) containing the system
                        topology. (required)
  -n FILE, --index FILE
                        Optional existing index file (.ndx) to extend.
                        (default: None)

Output:
  -o FILE, --output FILE
                        Output index file (.ndx). (required)

Parameters:
  -h, --help            Show this help message and exit.
  -db, --debug          Show runtime context, parsed arguments, timing, and
                        tracebacks. (default: False)

Notes:
  Build an index interactively from TPR. Expressions support mol, name, abbr, res, resid, chainid, index, group, parentheses, & and |.

  DROPPS atom and group indices are zero-based. Enter an expression to add a group, 'name NUMBER LABEL' to rename, 'splitch NUMBER' to split by molecule, and 'q' to save.

Examples:
  dps make_ndx -s run.tpr -o index.ndx
```
