# dps pbcontact

[All commands](../command-reference.md) · [User guide](../user-guide.md)

Measure direct and network-mediated interactions with periodic molecular images.

DROPPS 1.0.0. This page is generated from the executable CLI; defaults, choices, required inputs, and output options below match the installed command.

## Usage and options

```text
usage: dps pbcontact [-h] [-db] -s FILE -f FILE [-n FILE] -o PREFIX [-om FILE]
                     [-sg SELECTION_GROUP] [-c CUTOFF] [-mc MINIMUM_CONTACTS]
                     [-b START_TIME] [-e END_TIME] [-dt DELTA_TIME]

Measure direct and network-mediated interactions with periodic molecular images.

Input:
  -s FILE, --run-input FILE
                        Input DROPPS run file (.tpr) containing topology and
                        coordinates. (required)
  -f FILE, --input FILE
                        Input trajectory file (.xtc). (required)
  -n FILE, --index FILE
                        Optional index file (.ndx) defining additional atom
                        groups. (default: None)

Output:
  -o PREFIX, --output-prefix PREFIX
                        Output prefix for .frames.csv and .summary.csv files.
                        (required)
  -om FILE, --molecule-output FILE
                        Optional detailed CSV with the image-interaction state
                        of every selected molecule in every analyzed frame.
                        (default: None)

Parameters:
  -h, --help            Show this help message and exit.
  -db, --debug          Show runtime context, parsed arguments, timing, and
                        tracebacks. (default: False)
  -sg SELECTION_GROUP, --selection-group SELECTION_GROUP
                        Index group whose atoms define contacts; molecules
                        represented by the group are analyzed. If omitted,
                        prompt interactively. (default: None)
  -c CUTOFF, --cutoff CUTOFF
                        Atom-contact cutoff in nm. (default: 0.7)
  -mc MINIMUM_CONTACTS, --minimum-contacts MINIMUM_CONTACTS
                        Minimum atom-pair contacts required for one molecule-
                        to-image edge. (default: 1)
  -b START_TIME, --start-time START_TIME
                        First trajectory time to analyze, in ns. (default:
                        None)
  -e END_TIME, --end-time END_TIME
                        Last trajectory time to analyze, in ns. (default:
                        None)
  -dt DELTA_TIME, --delta-time DELTA_TIME
                        Approximate interval between analyzed frames, in ns.
                        (default: None)

A non-zero-winding cycle in the molecule contact graph is counted as an image interaction, including paths through any number of intermediate molecules.

Notes:
  Detect contacts between molecules and periodic images to assess finite-box self-interaction. The output prefix produces detailed and summarized interaction data.

  Interpret the results together with chain extension and box size. This diagnostic complements ordinary minimum-image contact maps.

Examples:
  dps pbcontact -s run.tpr -f run.xtc -o periodic-images
```
