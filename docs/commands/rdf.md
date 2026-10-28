# dps rdf

[All commands](../command-reference.md) · [User guide](../user-guide.md)

Calculate radial distribution functions and cumulative coordination numbers.

DROPPS 1.0.0. This page is generated from the executable CLI; defaults, choices, required inputs, and output options below match the installed command.

## Usage and options

```text
usage: dps rdf [-h] [-db] -s FILE -f FILE [-n FILE] -o FILE [-oc FILE]
               [-r SELECTION] [-se SELECTION] [--position {atom,cog,com}]
               [--ref-position {atom,cog,com}] [--sel-position {atom,cog,com}]
               [--grouping {all,chain,molecule,residue,atom}]
               [--ref-grouping {all,chain,molecule,residue,atom}]
               [--sel-grouping {all,chain,molecule,residue,atom}]
               [--geometry {3d,xy}] [-p | --pbc | --no-pbc] [-b START_TIME]
               [-e END_TIME] [-dt DELTA_TIME] [-bw BIN_WIDTH] [--cut CUT]
               [--rmax RMAX] [--exclude {none,topology,same-molecule}]
               [-sf {none,chain,molecule,residue}]
               [--norm {rdf,number-density,number_density,none}]

Calculate radial distribution functions and cumulative coordination numbers.

Input:
  -s FILE, --run-input FILE
                        Input DROPPS run file (.tpr) containing topology and
                        coordinates. (required)
  -f FILE, --input FILE
                        Input trajectory file (.xtc or .pdb). (required)
  -n FILE, --index FILE
                        Optional index file (.ndx) defining additional atom
                        groups. (default: None)
  -r SELECTION, --reference SELECTION
                        Reference DROPPS selection expression or index-group
                        number. (default: None)
  -se SELECTION, --selection SELECTION
                        Target DROPPS selection; repeat to write multiple RDF
                        curves. (default: None)

Output:
  -o FILE, --output FILE
                        Output radial-distribution curves (.xvg). (required)
  -oc FILE, --output-cumulative FILE
                        Optional cumulative coordination-number curves (.xvg).
                        (default: None)

Parameters:
  -h, --help            Show this help message and exit.
  -db, --debug          Show runtime context, parsed arguments, timing, and
                        tracebacks. (default: False)
  --position {atom,cog,com}
                        Common position construction for reference and
                        targets. (default: atom)
  --ref-position {atom,cog,com}
                        Reference position construction overriding --position.
                        (default: None)
  --sel-position {atom,cog,com}
                        Target position construction overriding --position.
                        (default: None)
  --grouping {all,chain,molecule,residue,atom}
                        Common grouping applied before COG/COM position
                        construction. (default: all)
  --ref-grouping {all,chain,molecule,residue,atom}
                        Reference grouping overriding --grouping. (default:
                        None)
  --sel-grouping {all,chain,molecule,residue,atom}
                        Target grouping overriding --grouping. (default: None)
  --geometry {3d,xy}    Use spherical 3D distances or distances projected into
                        x-y. (default: 3d)
  -p, --pbc, --no-pbc   Use periodic minimum-image distances. (default: True)
  -b START_TIME, --start-time START_TIME
                        First trajectory time to analyze, in ns. (default:
                        None)
  -e END_TIME, --end-time END_TIME
                        Last trajectory time to analyze, in ns. (default:
                        None)
  -dt DELTA_TIME, --delta-time DELTA_TIME
                        Approximate interval between analyzed frames, in ns.
                        (default: None)
  -bw BIN_WIDTH, --bin-width BIN_WIDTH
                        Radial bin width in nm. (default: 0.002)
  --cut CUT             Strict lower distance cutoff in nm. (default: 0.0)
  --rmax RMAX           Requested maximum RDF radius in nm; otherwise choose a
                        safe value. (default: None)
  --exclude {none,topology,same-molecule}
                        Candidate-pair exclusion policy; normalization remains
                        unchanged. (default: none)
  -sf {none,chain,molecule,residue}, --surface {none,chain,molecule,residue}
                        Use closest selected reference atom per chain or
                        residue. (default: none)
  --norm {rdf,number-density,number_density,none}
                        Normalization. Default: rdf; surface mode defaults to
                        none.

One reference selection is compared with every repeated target selection. Atom selections can be retained as positions or grouped into topology-aware COG/COM positions.

Notes:
  Measure radial distributions between atoms or grouped centers with 3D spherical or xy planar geometry, optional exclusions, and cumulative coordination numbers.

  RDF normalization is dimensionless; number-density and raw-count modes have different meanings. Grouping, surface selection, and exclusion rules affect the denominator and must match the scientific question.

Examples:
  dps rdf -s run.tpr -f run.xtc --reference 1 --selection 2 -o rdf.xvg
  dps rdf -s run.tpr -f run.xtc --reference 1 --selection 1 -o rdf-com.xvg --position com --grouping chain --exclude same-molecule
```
