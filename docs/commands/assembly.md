# dps assembly

[All commands](../command-reference.md) · [User guide](../user-guide.md)

Analyze the formation, size, composition, and shape of molecular assemblies.

DROPPS 1.0.0. This page is generated from the executable CLI; defaults, choices, required inputs, and output options below match the installed command.

## Usage and options

```text
usage: dps assembly [-h] [-db] -s FILE -f FILE [-n FILE] [-r REFERENCE_GROUP]
                    [-sg SELECTION_GROUP [SELECTION_GROUP ...]] [-p]
                    [-c CUTOFF] [-t THRESHOLD] [-b START_TIME] [-e END_TIME]
                    [-dt DELTA_TIME] [-on FILE] [-oz FILE] [-oq FILE]
                    [-om FILE]
                    [--molecule-fraction-cutoff MOLECULE_FRACTION_CUTOFF]
                    [-og FILE] [-oa FILE] [-oe FILE]

Analyze the formation, size, composition, and shape of molecular assemblies.

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
  -on FILE, --cluster-number FILE
                        Output cluster-count time series (.xvg); the extension
                        is added if omitted. (default: None)
  -oz FILE, --cluster-size FILE
                        Output largest-cluster size time series (.xvg); the
                        extension is added if omitted. (default: None)
  -oq FILE, --cluster-size-distribution FILE
                        Output per-frame cluster-size distributions (.xvg);
                        the extension is added if omitted. (default: None)
  -om FILE, --molecule-fraction FILE
                        Output selected-group fractions in large clusters
                        (.xvg); the extension is added if omitted. (default:
                        None)
  -og FILE, --radius-gyration-largest FILE
                        Output largest-cluster radius-of-gyration time series
                        (.xvg); the extension is added if omitted. (default:
                        None)
  -oa FILE, --asphericity FILE
                        Output asphericity of large clusters (.xvg); the
                        extension is added if omitted. (default: None)
  -oe FILE, --ellipticity FILE
                        Output ellipticity of large clusters (.xvg); the
                        extension is added if omitted. (default: None)

Parameters:
  -h, --help            Show this help message and exit.
  -db, --debug          Show runtime context, parsed arguments, timing, and
                        tracebacks. (default: False)
  -r REFERENCE_GROUP, --reference-group REFERENCE_GROUP
                        Reference index group used to determine clusters; if
                        omitted, prompt interactively. (default: None)
  -sg SELECTION_GROUP [SELECTION_GROUP ...], --selection-group SELECTION_GROUP [SELECTION_GROUP ...]
                        Index groups whose molecular fractions in clusters are
                        reported; if omitted, prompt interactively. (default:
                        None)
  -p, --treat-pbc       Apply periodic-boundary distances during cluster
                        detection. (default: False)
  -c CUTOFF, --cutoff CUTOFF
                        Bead-contact cutoff used for cluster detection, in nm.
                        (default: 0.7)
  -t THRESHOLD, --threshold THRESHOLD
                        Minimum number of bead contacts required to connect
                        two chains. (default: 5)
  -b START_TIME, --start-time START_TIME
                        First trajectory time to analyze, in ns. (default:
                        None)
  -e END_TIME, --end-time END_TIME
                        Last trajectory time to analyze, in ns. (default:
                        None)
  -dt DELTA_TIME, --delta-time DELTA_TIME
                        Approximate interval between analyzed frames, in ns.
                        (default: None)
  --molecule-fraction-cutoff MOLECULE_FRACTION_CUTOFF
                        Minimum cluster size used for molecular-fraction and
                        shape analyses. (default: 10)

Notes:
  Build a graph of molecular groups connected by enough bead contacts, then report cluster count, size, composition, and optional shape measures.

  Cutoffs are in nm. Enable PBC for periodic contacts; shape reconstruction uses contact-connected images. Specify output options explicitly and inspect percolating clusters before interpreting finite-cluster shape.

Examples:
  dps assembly -s run.tpr -f run.xtc --cluster-number clusters.xvg --cluster-size largest.xvg
```
