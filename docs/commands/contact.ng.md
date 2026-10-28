# dps contact.ng

[All commands](../command-reference.md) · [User guide](../user-guide.md)

Calculate residue-level contact maps with memory-efficient aggregation.

DROPPS 1.0.0. This page is generated from the executable CLI; defaults, choices, required inputs, and output options below match the installed command.

## Usage and options

```text
usage: dps contact.ng [-h] [-db] -s FILE -f FILE [-n FILE]
                      [-r REFERENCE_GROUP] [-sg SELECTION_GROUP] -cs
                      {global,residue} [-c CUTOFF] [-cm CUTOFF_MULTIPLIER]
                      [-rd REMOVE_DIAGONAL] [-b START_TIME] [-e END_TIME]
                      [-dt DELTA_TIME] [-p] [-oa FILE] [-ob FILE] [-oc FILE]
                      [-oi FILE] [-oj FILE] [-oe FILE] [-of FILE] [-og FILE]
                      [-oh FILE] [-ol FILE] [--intra-average]
                      [-oy {dat,xpm,xlsx}] [--search-mode {fast,stream}]
                      [--search-batch-size N] [--pair-chunk-size N]

Calculate residue-level contact maps with memory-efficient aggregation.

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
  -oa FILE, --output-inter-reference-selection FILE
                        Output inter-chain reference-selection contact map;
                        the --output-type extension is added if omitted.
                        (default: None)
  -ob FILE, --output-inter-reference-reference FILE
                        Output inter-chain reference-reference contact map;
                        the --output-type extension is added if omitted.
                        (default: None)
  -oc FILE, --output-inter-selection-selection FILE
                        Output inter-chain selection-selection contact map;
                        the --output-type extension is added if omitted.
                        (default: None)
  -oi FILE, --output-intra-reference FILE
                        Output intra-chain reference contact map; the
                        --output-type extension is added if omitted. (default:
                        None)
  -oj FILE, --output-intra-selection FILE
                        Output intra-chain selection contact map; the
                        --output-type extension is added if omitted. (default:
                        None)
  -oe FILE, --output-time-inter-reference-selection FILE
                        Output inter-chain reference-selection contact-number
                        time series (.xvg); the extension is added if omitted.
                        (default: None)
  -of FILE, --output-time-inter-reference-reference FILE
                        Output inter-chain reference-reference contact-number
                        time series (.xvg); the extension is added if omitted.
                        (default: None)
  -og FILE, --output-time-inter-selection-selection FILE
                        Output inter-chain selection-selection contact-number
                        time series (.xvg); the extension is added if omitted.
                        (default: None)
  -oh FILE, --output-time-intra-reference FILE
                        Output intra-chain reference contact-number time
                        series (.xvg); the extension is added if omitted.
                        (default: None)
  -ol FILE, --output-time-intra-selection FILE
                        Output intra-chain selection contact-number time
                        series (.xvg); the extension is added if omitted.
                        (default: None)
  -oy {dat,xpm,xlsx}, --output-type {dat,xpm,xlsx}
                        File format for contact-map outputs. (default: dat)

Parameters:
  -h, --help            Show this help message and exit.
  -db, --debug          Show runtime context, parsed arguments, timing, and
                        tracebacks. (default: False)
  -r REFERENCE_GROUP, --reference-group REFERENCE_GROUP
                        Reference index group forming the x axis of each
                        contact map. (default: None)
  -sg SELECTION_GROUP, --selection-group SELECTION_GROUP
                        Optional selection index group forming the y axis of
                        cross-group maps. (default: None)
  -cs {global,residue}, --cutoff-scheme {global,residue}
                        Contact-cutoff scheme: one global distance or residue-
                        specific distances. (required)
  -c CUTOFF, --cutoff CUTOFF
                        Global contact cutoff, in nm. (default: 0.7)
  -cm CUTOFF_MULTIPLIER, --cutoff-multiplier CUTOFF_MULTIPLIER
                        Multiplier applied to residue sigma values in the
                        residue cutoff scheme. (default: 1.2)
  -rd REMOVE_DIAGONAL, --remove-diagonal REMOVE_DIAGONAL
                        Number of main and neighboring diagonals removed from
                        intra-chain maps. (default: 2)
  -b START_TIME, --start-time START_TIME
                        First trajectory time to analyze, in ns. (default:
                        None)
  -e END_TIME, --end-time END_TIME
                        Last trajectory time to analyze, in ns. (default:
                        None)
  -dt DELTA_TIME, --delta-time DELTA_TIME
                        Approximate interval between analyzed frames, in ns.
                        (default: None)
  -p, --treat-pbc       Apply periodic-boundary distances during contact
                        calculation. (default: False)
  --intra-average       Average intra-chain contact maps instead of summing
                        them. (default: False)
  --search-mode {fast,stream}
                        Neighbor-search strategy: fast stores all pairs from
                        one frame; stream bounds pair memory by searching in
                        query batches. (default: fast)
  --search-batch-size N
                        Number of query atoms per neighborhood-search batch in
                        stream mode. Smaller values reduce peak memory at some
                        performance cost. (default: 4096)
  --pair-chunk-size N   Maximum number of neighbor pairs aggregated at once.
                        (default: 250000)

At least one contact-map or contact-number output option is required.

Notes:
  Compute the contact workflow with neighbor search and streaming aggregation for larger systems. Cutoff definitions and output choices are shared with contact.

  --search-mode chooses the computation strategy; the map remains a residue-level aggregate. Use time-series outputs when temporal changes are the quantity of interest.

Examples:
  dps contact.ng -s run.tpr -f run.xtc --reference-group 0 --selection-group 1 --cutoff-scheme global --output-inter-reference-selection contacts.dat --search-mode stream
```
