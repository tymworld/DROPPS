# dps trjcat

[All commands](../command-reference.md) · [User guide](../user-guide.md)

Concatenate DROPPS XTC trajectory segments in time order.

DROPPS 1.0.0. This page is generated from the executable CLI; defaults, choices, required inputs, and output options below match the installed command.

## Usage and options

```text
usage: dps trjcat [-h] [-db] -f FILE [FILE ...] -o FILE [-b START_TIME]
                  [-e END_TIME] [-dt DELTA_TIME] [-tu {fs,ps,ns,us,ms,s}]
                  [--sort | --no-sort] [--keep-duplicates]

Concatenate DROPPS XTC trajectory segments in time order.

Input:
  -f FILE [FILE ...], --input FILE [FILE ...]
                        Input DROPPS XTC trajectory segments. (required)

Output:
  -o FILE, --output FILE
                        Output concatenated XTC trajectory. (required)

Parameters:
  -h, --help            Show this help message and exit.
  -db, --debug          Show runtime context, parsed arguments, timing, and
                        tracebacks. (default: False)
  -b START_TIME, --start-time START_TIME
                        First trajectory time to write. (default: None)
  -e END_TIME, --end-time END_TIME
                        Last trajectory time to write. (default: None)
  -dt DELTA_TIME, --delta-time DELTA_TIME
                        Approximate physical interval between output frames.
                        (default: None)
  -tu {fs,ps,ns,us,ms,s}, --time-unit {fs,ps,ns,us,ms,s}
                        Unit used by -b, -e, and -dt. (default: ps)
  --sort                Sort input segments by their first frame time before
                        merging. (default: True)
  --no-sort             Preserve command-line input order; frame times must
                        remain monotonic. The default is to sort.
  --keep-duplicates     Keep frames with duplicate times instead of using the
                        later segment. (default: False)

Notes:
  Concatenate XTC segments with identical atom counts. Inputs are ordered by their first timestamp by default; overlapping timestamps prefer the later segment.

  Use --keep-duplicates only when repeated timestamps are intended. Time selection uses --time-unit, and the destination is replaced atomically after successful writing.

Examples:
  dps trjcat -f run.xtc run.part0002.xtc -o complete.xtc
```
