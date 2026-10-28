# dps distance

[All commands](../command-reference.md) · [User guide](../user-guide.md)

Calculate fixed, intra-chain, inter-chain, minimum, and maximum distances.

DROPPS 1.0.0. This page is generated from the executable CLI; defaults, choices, required inputs, and output options below match the installed command.

## Usage and options

```text
usage: dps distance [-h] [-db] -s FILE -f FILE [-n FILE] [-r SELECTION]
                    [-se SELECTION] [--label LABEL] -m {pair,min,max}
                    [--scope {corresponding,intra-chain,inter-chain,all}]
                    [--unique-entity-pairs] [--position {atom,cog,com}]
                    [--ref-position {atom,cog,com}]
                    [--sel-position {atom,cog,com}]
                    [--grouping {all,chain,molecule,residue,atom}]
                    [--ref-grouping {all,chain,molecule,residue,atom}]
                    [--sel-grouping {all,chain,molecule,residue,atom}]
                    [--exclude {none,same-atom,same-molecule}]
                    [-p | --pbc | --no-pbc] [-b START_TIME] [-e END_TIME]
                    [-dt DELTA_TIME] [-o FILE] [-oa FILE] [-oc FILE]
                    [-os FILE] [-oh FILE]
                    [--statistics-scope {pooled,channel,both}]
                    [--hist-scope {job,channel}]
                    [--hist-normalization {probability,count,density}]
                    [-bw BIN_WIDTH] [--hist-min HIST_MIN]
                    [--hist-max HIST_MAX] [--allow-large-output]

Calculate fixed, intra-chain, inter-chain, minimum, and maximum distances.

Input:
  -s FILE, --run-input FILE
                        Input DROPPS run file (.tpr) containing topology and
                        coordinates. (required)
  -f FILE, --input FILE
                        Input trajectory file (.xtc or .pdb). (required)
  -n FILE, --index FILE
                        Optional index file (.ndx) defining additional atom
                        groups. (default: None)

Output:
  -o FILE, --output-all FILE
                        Output every scalar distance channel as a time series
                        (.xvg). (default: None)
  -oa FILE, --output-average FILE
                        Output per-frame channel mean and population std per
                        job (.xvg). (default: None)
  -oc FILE, --output-components FILE
                        Output dx, dy, and dz for every distance channel
                        (.xvg). (default: None)
  -os FILE, --output-statistics FILE
                        Output pooled and/or per-channel time statistics
                        (.csv). (default: None)
  -oh FILE, --output-histogram FILE
                        Output pooled or per-channel distance histograms
                        (.xvg). (default: None)

Parameters:
  -h, --help            Show this help message and exit.
  -db, --debug          Show runtime context, parsed arguments, timing, and
                        tracebacks. (default: False)
  -r SELECTION, --reference SELECTION
                        Reference DROPPS selection expression or index-group
                        number; repeat to define multiple jobs. (default:
                        None)
  -se SELECTION, --selection SELECTION
                        Selection-side DROPPS expression or index-group
                        number; repeat to define multiple jobs. (default:
                        None)
  --label LABEL         Optional output label repeated once per resulting
                        distance job. (default: None)
  -m {pair,min,max}, --mode {pair,min,max}
                        Distance within each entity pair: zip corresponding
                        positions, or choose the minimum/maximum Cartesian
                        position-pair norm. (required)
  --scope {corresponding,intra-chain,inter-chain,all}
                        Entity relationship used before the distance mode is
                        applied. (default: corresponding)
  --unique-entity-pairs
                        Keep only one direction of a symmetric all/inter-chain
                        entity product; endpoints must be identical. (default:
                        False)
  --position {atom,cog,com}
                        Common position construction for both endpoint sides.
                        (default: atom)
  --ref-position {atom,cog,com}
                        Reference-side position construction overriding
                        --position. (default: None)
  --sel-position {atom,cog,com}
                        Selection-side position construction overriding
                        --position. (default: None)
  --grouping {all,chain,molecule,residue,atom}
                        Common endpoint grouping. Default: chain for
                        intra/inter-chain scope, otherwise all. 'molecule' is
                        an alias for chain.
  --ref-grouping {all,chain,molecule,residue,atom}
                        Reference-side grouping overriding --grouping.
                        (default: None)
  --sel-grouping {all,chain,molecule,residue,atom}
                        Selection-side grouping overriding --grouping.
                        (default: None)
  --exclude {none,same-atom,same-molecule}
                        Candidate position pairs to remove before
                        output/reduction. (default: none)
  -p, --pbc, --no-pbc   Use minimum-image periodic distances and PBC-aware
                        centers. (default: True)
  -b START_TIME, --start-time START_TIME
                        First trajectory time to analyze, in ns. (default:
                        None)
  -e END_TIME, --end-time END_TIME
                        Last trajectory time to analyze, in ns. (default:
                        None)
  -dt DELTA_TIME, --delta-time DELTA_TIME
                        Approximate interval between analyzed frames, in ns.
                        (default: None)
  --statistics-scope {pooled,channel,both}
                        Granularity written by --output-statistics. (default:
                        pooled)
  --hist-scope {job,channel}
                        Pool histogram samples per job or retain every
                        channel. (default: job)
  --hist-normalization {probability,count,density}
                        Histogram y-axis normalization. (default: probability)
  -bw BIN_WIDTH, --bin-width BIN_WIDTH
                        Histogram bin width in nm. (default: 0.01)
  --hist-min HIST_MIN   Optional histogram lower edge in nm. (default: None)
  --hist-max HIST_MAX   Optional histogram upper edge in nm. (default: None)
  --allow-large-output  Allow detailed outputs containing more than 100000
                        channels. (default: False)

Select each region once across all chain copies. --scope controls which chains are related; --mode controls how positions inside each admitted entity pair are compared.

Notes:
  Select each region once across all molecule copies, then choose intra-chain or inter-chain scope and pair or center-based measurement.

  Distances use nm. Select output statistics, time series, or pair details as needed; PBC and pair matching are explicit options. Hidden intra-distance/inter-distance and idist/odist aliases retain the older interfaces.

Examples:
  dps distance -s run.tpr -f run.xtc --mode pair --scope intra-chain --reference 1 --selection 2 --output-statistics intra.csv
  dps distance -s run.tpr -f run.xtc --mode pair --scope inter-chain --reference 1 --selection 2 --output-statistics inter.csv
```
