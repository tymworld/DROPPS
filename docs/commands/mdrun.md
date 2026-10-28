# dps mdrun

[All commands](../command-reference.md) · [User guide](../user-guide.md)

Run or resume a molecular-dynamics simulation from a DROPPS run-input file.

DROPPS 1.0.0. This page is generated from the executable CLI; defaults, choices, required inputs, and output options below match the installed command.

## Usage and options

```text
usage: dps mdrun [-h] [-db] -s FILE [-cp [FILE]] -o PREFIX [-ox FILE]
                 [-oe FILE] [-ol FILE] [-ok FILE] [-oc FILE] [-os FILE]
                 [-p {auto,CUDA,OpenCL,CPU,Reference}]
                 [--device-index DEVICE_INDEX]
                 [--precision {single,mixed,double}] [-t THREADS] [-n NSTEPS]
                 [--seed SEED] [--checkpoint-minutes CHECKPOINT_MINUTES]
                 [--number-checkpoints] [-mh MAX_HOURS] [-a | --no-append]

Run or resume a molecular-dynamics simulation from a DROPPS run-input file.

Input:
  -s FILE, --run-input FILE
                        Input DROPPS run file (.tpr) containing the system and
                        simulation settings. (required)
  -cp [FILE], --checkpoint [FILE]
                        Resume from a native checkpoint (.chk) or portable
                        State (.state.xml). If FILE is omitted, use the
                        resolved checkpoint path and fall back to its paired
                        State XML. (default: None)

Output:
  -o PREFIX, --output-prefix PREFIX
                        Default output prefix for .log, .edr, production .xtc,
                        optional .warming.xtc, .chk, final .pdb, and optional
                        .stress.xvg files. (required)
  -ox FILE, --trajectory-output FILE
                        Override the compressed trajectory output path.
                        (default: None)
  -oe FILE, --energy-output FILE
                        Override the DROPPS EDR output path. (default: None)
  -ol FILE, --log-output FILE
                        Override the fixed-schema progress-log output path.
                        (default: None)
  -ok FILE, --checkpoint-output FILE
                        Override the latest checkpoint output path. (default:
                        None)
  -oc FILE, --final-structure FILE
                        Override the final PDB structure output path.
                        (default: None)
  -os FILE, --stress-output FILE
                        Override the standalone pressure-tensor output path.
                        (default: None)

Parameters:
  -h, --help            Show this help message and exit.
  -db, --debug          Show runtime context, parsed arguments, timing, and
                        tracebacks. (default: False)
  -p {auto,CUDA,OpenCL,CPU,Reference}, --platform {auto,CUDA,OpenCL,CPU,Reference}
                        OpenMM compute platform. (default: auto)
  --device-index DEVICE_INDEX
                        OpenMM CUDA/OpenCL device index or comma-separated
                        indices. (default: None)
  --precision {single,mixed,double}
                        CUDA/OpenCL arithmetic precision; GPU default is
                        mixed.
  -t THREADS, --threads THREADS
                        CPU platform thread count; 0 uses the OpenMM default.
                        (default: 0)
  -n NSTEPS, --nsteps NSTEPS
                        Override the TPR production target in steps for this
                        run. (default: None)
  --seed SEED           Override the TPR random seed for a new run. (default:
                        None)
  --checkpoint-minutes CHECKPOINT_MINUTES
                        Wall-clock minutes between periodic checkpoints; 0
                        disables periodic saves, but a final or interrupt
                        checkpoint is still saved. (default: 5.0)
  --number-checkpoints  Keep step-numbered checkpoint copies in addition to
                        the latest file. (default: False)
  -mh MAX_HOURS, --max-hours MAX_HOURS
                        Stop production safely after 0.99 times this many
                        hours; -1 disables the limit. (default: -1.0)
  -a, --append          Append restart outputs after rolling them back to the
                        checkpoint. Default: append mode.
  --no-append           Write restart outputs to the next .partNNNN files.
                        Default: append mode.

Notes:
  A new Langevin run minimizes/warms as configured, then integrates to the total production nsteps. --nsteps sets the total target, including steps already completed on resume.

  --checkpoint resumes from the output prefix or an explicit restart path. Append mode rolls outputs back to the checkpoint; --no-append writes a new part. Native checkpoints have priority over portable State fallback.

  The output prefix produces log, DROPPS CSV EDR, XTC, checkpoint, State XML, restart JSON, and final PDB. Optional stress output contains the pressure tensor for viscosity analysis.

  Set --platform CUDA when GPU execution is required. integrator=steep runs only OpenMM energy minimization and writes the final structure/restart without a dynamics trajectory.

Examples:
  dps mdrun -s run.tpr -o run
  dps mdrun -s run.tpr -o run --checkpoint
```
