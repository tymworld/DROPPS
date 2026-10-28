# dps surftension

[All commands](../command-reference.md) · [User guide](../user-guide.md)

Calculate planar-interface surface tension from a DROPPS TPR and XTC trajectory.

DROPPS 1.0.0. This page is generated from the executable CLI; defaults, choices, required inputs, and output options below match the installed command.

## Usage and options

```text
usage: dps surftension [-h] [-db] -s FILE -f FILE -o FILE [-b START_TIME]
                       [-e END_TIME] [-dt DELTA_TIME] [-a {x,y,z}]
                       [--strain STRAIN [STRAIN ...]]
                       [--interfaces INTERFACES] [-bl BLOCKS] [-p PLATFORM]
                       [--precision {single,mixed,double}] [--device DEVICE]
                       [-t THREADS]

Calculate planar-interface surface tension from a DROPPS TPR and XTC trajectory.

Input:
  -s FILE, --run-input FILE
                        Input DROPPS run file (.tpr) containing the OpenMM
                        system and topology. (required)
  -f FILE, --input FILE
                        Input equilibrated planar-slab trajectory file (.xtc).
                        (required)

Output:
  -o FILE, --output FILE
                        Output surface-tension time series (.xvg); the
                        extension is added if omitted. (required)

Parameters:
  -h, --help            Show this help message and exit.
  -db, --debug          Show runtime context, parsed arguments, timing, and
                        tracebacks. (default: False)
  -b START_TIME, --start-time START_TIME
                        First trajectory time to analyze, in ns. (default:
                        None)
  -e END_TIME, --end-time END_TIME
                        Last trajectory time to analyze, in ns. (default:
                        None)
  -dt DELTA_TIME, --delta-time DELTA_TIME
                        Approximate interval between analyzed frames, in ns.
                        (default: None)
  -a {x,y,z}, --axis {x,y,z}
                        Interface-normal axis. (default: z)
  --strain STRAIN [STRAIN ...]
                        One or more positive finite-difference strains.
                        Supplying several values provides a convergence check.
                        (default: [0.0003])
  --interfaces INTERFACES
                        Number of planar interfaces in the periodic cell.
                        (default: 2)
  -bl BLOCKS, --blocks BLOCKS
                        Number of contiguous blocks used for the standard
                        error. (default: 5)
  -p PLATFORM, --platform PLATFORM
                        OpenMM platform name, or auto for automatic selection.
                        (default: auto)
  --precision {single,mixed,double}
                        GPU/OpenCL energy precision. (default: double)
  --device DEVICE       OpenMM GPU/OpenCL device index. (default: None)
  -t THREADS, --threads THREADS
                        Number of OpenMM CPU platform threads. (default: None)

Notes:
  Estimate planar interfacial tension by finite-strain configurational energy evaluation along the interface normal. Choose the number of interfaces and strain values consistently with the slab geometry.

  Output is in mN/m with block uncertainty and strain-convergence information. Inspect convergence and the pressure/geometry assumptions before comparing state points.

Examples:
  dps surftension -s run.tpr -f run.xtc -o surface-tension.xvg --axis z
```
