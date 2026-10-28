# dps rerun

[All commands](../command-reference.md) · [User guide](../user-guide.md)

Re-evaluate configurational observables from an XTC trajectory.

DROPPS 1.0.0. This page is generated from the executable CLI; defaults, choices, required inputs, and output options below match the installed command.

## Usage and options

```text
usage: dps rerun [-h] [-db] -s FILE -f FILE -o FILE
                 [--terms TERMS [TERMS ...]] [-b START_TIME] [-e END_TIME]
                 [-dt DELTA_TIME] [--strain STRAIN] [-p PLATFORM]
                 [--precision {single,mixed,double}] [--device DEVICE]
                 [-t THREADS]

Re-evaluate configurational observables from an XTC trajectory.

Input:
  -s FILE, --run-input FILE
                        Input DROPPS run file (.tpr). (required)
  -f FILE, --input FILE
                        Input coordinate trajectory (.xtc). (required)

Output:
  -o FILE, --output FILE
                        Output rerun energy file (.edr). (required)

Parameters:
  -h, --help            Show this help message and exit.
  -db, --debug          Show runtime context, parsed arguments, timing, and
                        tracebacks. (default: False)
  --terms TERMS [TERMS ...]
                        Configurational terms to calculate. Defaults to
                        potential energy, box lengths, volume, and density.
  -b START_TIME, --start-time START_TIME
                        First trajectory time, in ps. (default: None)
  -e END_TIME, --end-time END_TIME
                        Last trajectory time, in ps. (default: None)
  -dt DELTA_TIME, --delta-time DELTA_TIME
                        Approximate frame interval, in ps. (default: None)
  --strain STRAIN       Finite-difference strain for configurational pressure
                        terms. (default: 0.0003)
  -p PLATFORM, --platform PLATFORM
                        OpenMM platform name, or auto. (default: auto)
  --precision {single,mixed,double}
                        OpenMM energy-evaluation precision. (default: double)
  --device DEVICE       CUDA/OpenCL device index, or auto. (default: auto)
  -t THREADS, --threads THREADS
                        CPU threads; 0 uses the OpenMM default. (default: 0)

Notes:
  Re-evaluate saved XTC coordinates using the system stored in TPR; this does not integrate new dynamics. Output is a DROPPS EDR table.

  Configurational pressure terms omit kinetic and constraint virials. Use runtime stress output, rather than these terms, for Green-Kubo viscosity.

Examples:
  dps rerun -s run.tpr -f run.xtc -o rerun.edr
```
