# dps sasa

[All commands](../command-reference.md) · [User guide](../user-guide.md)

Calculate coarse-grained solvent-accessible surface area.

DROPPS 1.0.0. This page is generated from the executable CLI; defaults, choices, required inputs, and output options below match the installed command.

## Usage and options

```text
usage: dps sasa [-h] [-db] -s FILE -f FILE [-n FILE] -o FILE [-sf SELECTION]
                [--output-selection SELECTION [SELECTION ...]] [-b START_TIME]
                [-e END_TIME] [-dt DELTA_TIME] [--probe-radius PROBE_RADIUS]
                [--sphere-points SPHERE_POINTS]
                [--radius-convention {sigma-half,lj-minimum-half}]
                [--radius-scale RADIUS_SCALE] [--fixed-radius FIXED_RADIUS]

Calculate coarse-grained solvent-accessible surface area.

Input:
  -s FILE, --run-input FILE
                        Input DROPPS run file (.tpr) containing topology and
                        bead parameters. (required)
  -f FILE, --input FILE
                        Input trajectory file (.xtc or .pdb). (required)
  -n FILE, --index FILE
                        Optional index file (.ndx) defining additional atom
                        groups. (default: None)

Output:
  -o FILE, --output FILE
                        Output surface-area time series (.xvg). (required)

Parameters:
  -h, --help            Show this help message and exit.
  -db, --debug          Show runtime context, parsed arguments, timing, and
                        tracebacks. (default: False)
  -sf SELECTION, --surface SELECTION
                        One DROPPS selection expression or index-group number
                        defining the calculated surface. Default: choose
                        interactively in a terminal, with an empty response
                        meaning all atoms; otherwise all atoms.
  --output-selection SELECTION [SELECTION ...]
                        Additional surface subsets reported as separate
                        series; accepts one or more quoted selections and may
                        be repeated. Default: choose zero or more groups
                        interactively in a terminal; otherwise none.
  -b START_TIME, --start-time START_TIME
                        First trajectory time to analyze, in ns. (default:
                        None)
  -e END_TIME, --end-time END_TIME
                        Last trajectory time to analyze, in ns. (default:
                        None)
  -dt DELTA_TIME, --delta-time DELTA_TIME
                        Approximate interval between analyzed frames, in ns.
                        (default: None)
  --probe-radius PROBE_RADIUS
                        Solvent-probe radius in nm. (default: 0.14)
  --sphere-points SPHERE_POINTS
                        Number of deterministic sampling points on every
                        expanded bead sphere. (default: 960)
  --radius-convention {sigma-half,lj-minimum-half}
                        Mapping from the force-field sigma to a geometric bead
                        radius: sigma-half uses sigma/2 and lj-minimum-half
                        uses 2^(1/6)*sigma/2. (default: sigma-half)
  --radius-scale RADIUS_SCALE
                        Positive multiplier applied to all resolved bead
                        radii. (default: 1.0)
  --fixed-radius FIXED_RADIUS
                        Use this radius in nm for every bead instead of force-
                        field sigma values; --radius-scale is still applied.
                        (default: None)

The surface selection defines all beads that can occlude one another. Each output selection must be a subset of that surface; its area is extracted without recalculating an isolated surface.

Notes:
  Estimate coarse-grained solvent-accessible surface area by sampling expanded bead spheres. Choose the surface group, output groups, probe radius, and sphere resolution.

  Areas use nm^2. Occlusion uses the full selected surface even when reporting a smaller output group; force-field radii and probe size determine the physical interpretation.

Examples:
  dps sasa -s run.tpr -f run.xtc -o sasa.xvg --surface 0 --output-selection 1 2
```
