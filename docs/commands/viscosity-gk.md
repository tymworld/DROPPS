# dps viscosity-gk

[All commands](../command-reference.md) · [User guide](../user-guide.md)

Calculate zero-shear viscosity from a DROPPS runtime pressure tensor.

DROPPS 1.0.0. This page is generated from the executable CLI; defaults, choices, required inputs, and output options below match the installed command.

## Usage and options

```text
usage: dps viscosity-gk [-h] [-db] -f FILE -o FILE [-tm TEMPERATURE]
                        [-b START_TIME] [-e END_TIME] [-ml MAX_LAG]
                        [--plateau-start PLATEAU_START]
                        [--plateau-end PLATEAU_END] [-bl BLOCKS]

Calculate zero-shear viscosity from a DROPPS runtime pressure tensor.

Input:
  -f FILE, --input FILE
                        Input runtime pressure-tensor file (.stress.xvg).
                        (required)

Output:
  -o FILE, --output FILE
                        Output Green-Kubo viscosity time series (.xvg); the
                        extension is added if omitted. (required)

Parameters:
  -h, --help            Show this help message and exit.
  -db, --debug          Show runtime context, parsed arguments, timing, and
                        tracebacks. (default: False)
  -tm TEMPERATURE, --temperature TEMPERATURE
                        Temperature, in K; if omitted, use the value stored in
                        the stress file. (default: None)
  -b START_TIME, --start-time START_TIME
                        First stress-sample time to analyze, in ps. (default:
                        None)
  -e END_TIME, --end-time END_TIME
                        Last stress-sample time to analyze, in ps. (default:
                        None)
  -ml MAX_LAG, --max-lag MAX_LAG
                        Maximum correlation lag, in ps. (default: None)
  --plateau-start PLATEAU_START
                        Start of the viscosity averaging plateau, in ps.
                        (default: None)
  --plateau-end PLATEAU_END
                        End of the viscosity averaging plateau, in ps.
                        (default: None)
  -bl BLOCKS, --blocks BLOCKS
                        Number of contiguous blocks used for the standard
                        error. (default: 5)

Notes:
  Integrate stress autocorrelations from the full runtime pressure tensor to obtain Green-Kubo shear viscosity. Use uniformly sampled equilibrated stress data.

  The running integral and plateau estimate are in mPa s. Select a plateau window and blocks; output from configurational rerun lacks the full tensor required here.

Examples:
  dps viscosity-gk -f run.stress.xvg -o viscosity.xvg
```
