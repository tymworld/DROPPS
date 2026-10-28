# dps trjconv

[All commands](../command-reference.md) · [User guide](../user-guide.md)

Select, transform, and convert a DROPPS trajectory.

DROPPS 1.0.0. This page is generated from the executable CLI; defaults, choices, required inputs, and output options below match the installed command.

## Usage and options

```text
usage: dps trjconv [-h] [-db] -s FILE -f FILE [-n FILE] [-se EXPR] -o FILE
                   [-b START_TIME] [-e END_TIME] [-dt DELTA_TIME]
                   [-tu {fs,ps,ns,us,ms,s}]
                   [-p {none,whole,atom,res,mol,nojump}]
                   [-c {none,geometry,mass,dense}] [-cs EXPR]
                   [-ca {x,y,z,xyz}]
                   [--dense-phase-threshold DENSE_PHASE_THRESHOLD]
                   [--density-bin-width DENSITY_BIN_WIDTH]
                   [--fit {none,translation,transxy,rot+trans,rotxy+transxy,progressive}]
                   [-fs EXPR] [--fit-reference {tpr,first}]
                   [--fit-weighting {mass,uniform}] [--translate DX DY DZ]
                   [--shift DX DY DZ] [--precision PRECISION] [--separate]
                   [--zero-pad ZERO_PAD] [--conect]

Select, transform, and convert a DROPPS trajectory.

Input:
  -s FILE, --run-input FILE
                        Input DROPPS run file (.tpr) containing topology and
                        coordinates. (required)
  -f FILE, --input FILE
                        Input trajectory or structure file (.xtc or .pdb).
                        (required)
  -n FILE, --index FILE
                        Optional index file (.ndx) defining additional atom
                        groups. (default: None)
  -se EXPR, --select EXPR
                        DROPPS selection expression for atoms to write; when
                        omitted, prompt interactively. (default: None)
  -cs EXPR, --center-select EXPR
                        Selection used for centering; defaults to the output
                        selection.
  -fs EXPR, --fit-select EXPR
                        Selection used for fitting; defaults to the output
                        selection.

Output:
  -o FILE, --output FILE
                        Output trajectory or structure file (.xtc or .pdb).
                        (required)

Parameters:
  -h, --help            Show this help message and exit.
  -db, --debug          Show runtime context, parsed arguments, timing, and
                        tracebacks. (default: False)
  -b START_TIME, --start-time START_TIME
                        First trajectory time to write. (default: None)
  -e END_TIME, --end-time END_TIME
                        Last trajectory time to write; set equal to -b to
                        select the nearest saved frame. (default: None)
  -dt DELTA_TIME, --delta-time DELTA_TIME
                        Time interval between output frames, using nearest
                        saved frames. (default: None)
  -tu {fs,ps,ns,us,ms,s}, --time-unit {fs,ps,ns,us,ms,s}
                        Unit used by -b, -e, and -dt. (default: ns)
  -p {none,whole,atom,res,mol,nojump}, --pbc {none,whole,atom,res,mol,nojump}
                        Periodic-boundary treatment: whole reconstructs bonded
                        molecules; atom packs atoms; res/mol pack residue or
                        whole-molecule centers; nojump removes temporal box
                        jumps. (default: none)
  -c {none,geometry,mass,dense}, --center {none,geometry,mass,dense}
                        Center the selected group geometrically, by mass, or
                        as a dense slab. (default: none)
  -ca {x,y,z,xyz}, --center-axis {x,y,z,xyz}
                        Axes to center; defaults to xyz for geometry/mass and
                        z for dense centering.
  --dense-phase-threshold DENSE_PHASE_THRESHOLD
                        Dense bins must exceed this fraction of the maximum
                        mass density. (default: 0.5)
  --density-bin-width DENSITY_BIN_WIDTH
                        Dense-phase histogram bin width in nm. (default: 0.05)
  --fit {none,translation,transxy,rot+trans,rotxy+transxy,progressive}
                        Fit coordinates to the TPR or first processed frame.
                        (default: none)
  --fit-reference {tpr,first}
                        Reference coordinates for ordinary or progressive
                        fitting. (default: tpr)
  --fit-weighting {mass,uniform}
                        Weights used to determine fit centers and rotations.
                        (default: mass)
  --translate DX DY DZ  Final constant Cartesian translation vector in nm.
                        (default: None)
  --shift DX DY DZ      Additional translation in nm multiplied by the input
                        frame index. (default: None)
  --precision PRECISION
                        Number of decimal places used for XTC output
                        precision. (default: 3)
  --separate            Write each selected frame to a separate numbered PDB
                        file. (default: False)
  --zero-pad ZERO_PAD   Number of digits in filenames created by --separate.
                        (default: 6)
  --conect              Add reindexed topology bonds as CONECT records to PDB
                        output. (default: False)

Frame selection is time-based. Set -b and -e to the same value to write the saved frame nearest that time. Transform order is: make whole/no-jump, center, pack into the box, fit, translate.

Notes:
  Process XTC using the TPR topology and write XTC or PDB. --select chooses output atoms; centering and fitting may use separate selections.

  Time bounds use --time-unit (ns by default). Equal start/end selects the nearest saved frame. PBC modes include whole molecules, wrapping, and temporal no-jump tracking.

  No-jump and progressive fitting visit intermediate frames before output subsampling. A subset trajectory requires a matching subset topology for later analysis.

Examples:
  dps trjconv -s run.tpr -f run.xtc -o frame.pdb -b 100 -e 100 --select 'group 0'
  dps trjconv -s run.tpr -f run.xtc -o centered.xtc -se 'group 0' -p mol -c mass -ca z
```
