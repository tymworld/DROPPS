# dps polystat

[All commands](../command-reference.md) · [User guide](../user-guide.md)

Calculate size, shape, and backbone statistics for many protein copies.

DROPPS 1.0.0. This page is generated from the executable CLI; defaults, choices, required inputs, and output options below match the installed command.

## Usage and options

```text
usage: dps polystat [-h] [-db] -s FILE -f FILE [-n FILE] -o FILE [-oi FILE]
                    [-op FILE] [-ot FILE] [-om FILE] [-se GROUP [GROUP ...] |
                    -ex EXPR] [--metrics {ree,rg,shape} [{ree,rg,shape} ...]]
                    [--weight {mass,uniform}] [-ml MAX_LAG]
                    [--lag-step LAG_STEP] [--allow-heterogeneous-cohorts]
                    [--allow-large-output] [--pbc | --no-pbc] [-b START_TIME]
                    [-e END_TIME] [-dt DELTA_TIME]

Calculate size, shape, and backbone statistics for many protein copies.

Input:
  -s FILE, --run-input FILE
                        Input DROPPS run file (.tpr) containing topology and
                        coordinates. (required)
  -f FILE, --input FILE
                        Input trajectory file (.xtc or .pdb). (required)
  -n FILE, --index FILE
                        Optional index file (.ndx) defining additional atom
                        groups. (default: None)
  -se GROUP [GROUP ...], --selection-groups GROUP [GROUP ...]
                        Index groups spanning any number of molecule copies;
                        each group is one analysis job. (default: None)
  -ex EXPR, --select EXPR, --selection-expression EXPR
                        DROPPS selection expression; repeat to analyze more
                        than one molecule type or region. (default: None)

Output:
  -o FILE, --output FILE
                        Compact per-frame cohort statistics (.xvg). (required)
  -oi FILE, --internal-distance-output FILE
                        Pooled mean-square internal-distance curve (.csv).
                        (default: None)
  -op FILE, --persistence-output FILE
                        Per-frame 1/e persistence length in steps and nm
                        (.xvg). (default: None)
  -ot FILE, --tangent-output FILE
                        Pooled backbone tangent-correlation curve (.csv).
                        (default: None)
  -om FILE, --molecule-output FILE
                        Optional long-form per-molecule, per-frame values
                        (.csv). (default: None)

Parameters:
  -h, --help            Show this help message and exit.
  -db, --debug          Show runtime context, parsed arguments, timing, and
                        tracebacks. (default: False)
  --metrics {ree,rg,shape} [{ree,rg,shape} ...]
                        Primary time-series metrics: end-to-end distance,
                        radius of gyration, and rotation-invariant principal
                        shape. (default: ('ree', 'rg', 'shape'))
  --weight {mass,uniform}
                        Weights used only for gyration centers and tensors.
                        (default: mass)
  -ml MAX_LAG, --max-lag MAX_LAG
                        Largest backbone lag; by default use every available
                        lag.
  --lag-step LAG_STEP   Backbone lag spacing; use 2 for the legacy all-trans
                        convention. (default: 1)
  --allow-heterogeneous-cohorts
                        Allow one molecule name to have different topology or
                        local selection signatures and label them as separate
                        cohorts. (default: False)
  --allow-large-output  Allow an estimated per-molecule output larger than
                        safety limits. (default: False)
  --pbc, --no-pbc       Reconstruct complete molecules from topology bonds
                        before analysis. (default: True)
  -b START_TIME, --start-time START_TIME
                        First trajectory time to analyze, in ns. (default:
                        None)
  -e END_TIME, --end-time END_TIME
                        Last trajectory time to analyze, in ns. (default:
                        None)
  -dt DELTA_TIME, --delta-time DELTA_TIME
                        Approximate interval between analyzed frames, in ns.
                        (default: None)

Select a molecule type or region once across all copies. DROPPS splits it into topology molecules and keeps different molecule types in separate cohorts.

Notes:
  Split selected molecules into compatible cohorts and measure per-chain Rg, end-to-end length, gyration-tensor shape, and optional internal-distance/persistence statistics.

  The compact XVG reports cohort summaries; optional CSV/XVG files retain per-molecule observations, tangent correlations, or internal distances. Lengths use nm; inspect censored or invalid persistence estimates.

Examples:
  dps polystat -s run.tpr -f run.xtc -o polystat.xvg --select "mol PROT"
  dps polystat -s run.tpr -f run.xtc -o polystat.xvg --selection-groups 1 -oi internal-distance.csv -op persistence.xvg
```
