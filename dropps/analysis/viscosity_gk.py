"""Green-Kubo zero-shear viscosity from a runtime pressure-tensor file."""

from __future__ import annotations

from dropps.share.argument_parser import ArgumentParser
from pathlib import Path
import re

import numpy as np

from dropps.analysis.viscosity_gk_core import (
    plateau_average,
    shear_relaxation_modulus,
    viscosity_curve,
)
from dropps.fileio.filename_control import validate_extension
from dropps.share.command_class import single_command


prog = "viscosity-gk"
desc = "Calculate zero-shear viscosity from a DROPPS runtime pressure tensor."


def read_stress_tensor(path):
    """Read a StressTensorReporter XVG file and its scalar metadata."""

    rows = []
    metadata = {}
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("@"):
                continue
            if stripped.startswith("#"):
                for key, value in re.findall(r"([A-Za-z_]+)=([^\s]+)", stripped):
                    metadata[key] = value
                continue
            fields = stripped.split()
            if len(fields) != 9:
                raise ValueError(
                    f"{path}:{line_number}: expected 9 numeric columns, "
                    f"found {len(fields)}."
                )
            rows.append([float(value) for value in fields])

    if len(rows) < 4:
        raise ValueError("The stress file must contain at least four samples.")
    data = np.asarray(rows, dtype=float)
    if not np.all(np.isfinite(data)):
        raise ValueError("The stress file contains non-finite values.")
    if np.any(np.diff(data[:, 1]) <= 0.0):
        raise ValueError("Stress sample times must be strictly increasing.")
    return data, metadata


def _select_time_window(data, start_ps, end_ps):
    start = data[0, 1] if start_ps is None else float(start_ps)
    end = data[-1, 1] if end_ps is None else float(end_ps)
    if end < start:
        raise ValueError("End time must not precede start time.")
    selected = data[(data[:, 1] >= start) & (data[:, 1] <= end)]
    if len(selected) < 8:
        raise ValueError(
            "The selected time window must contain at least eight samples."
        )
    return selected


def _sample_interval(times_ps):
    intervals = np.diff(times_ps)
    interval = float(np.median(intervals))
    if not np.allclose(intervals, interval, rtol=1.0e-5, atol=1.0e-9):
        raise ValueError("Green-Kubo analysis requires uniformly sampled stress data.")
    return interval


def analyze_green_kubo(
    data,
    temperature_k,
    blocks=5,
    max_lag_ps=None,
    plateau_start_ps=None,
    plateau_end_ps=None,
):
    """Calculate the full viscosity curve and block standard error."""

    if temperature_k <= 0.0:
        raise ValueError("Temperature must be positive.")
    if blocks <= 0 or blocks > len(data):
        raise ValueError("Number of blocks is incompatible with the selected data.")

    times = data[:, 1]
    volumes = data[:, 2]
    pressure = data[:, 3:9]
    interval = _sample_interval(times)
    relative_volume_span = float(np.ptp(volumes) / np.mean(volumes))
    if relative_volume_span > 1.0e-3:
        raise ValueError(
            "Cell volume varies by more than 0.1%. Green-Kubo viscosity must be "
            "calculated from an NVT production run (pcoulp = False)."
        )

    block_indices = np.array_split(np.arange(len(data)), blocks)
    shortest_block = min(len(indices) for indices in block_indices)
    if shortest_block < 8:
        raise ValueError("Each statistical block must contain at least eight samples.")

    if max_lag_ps is None:
        max_lag_steps = max(2, (shortest_block - 1) // 2)
    else:
        if max_lag_ps <= 0.0:
            raise ValueError("Maximum lag must be positive.")
        max_lag_steps = int(np.floor(max_lag_ps / interval + 1.0e-10))
    if not 2 <= max_lag_steps < shortest_block:
        maximum = (shortest_block - 1) * interval
        raise ValueError(
            f"Maximum lag must span at least two samples and be below {maximum:g} ps "
            "so it can be evaluated in every block."
        )

    lag_times = np.arange(max_lag_steps + 1, dtype=float) * interval
    lag_end = float(lag_times[-1])
    if plateau_start_ps is None:
        plateau_start_index = max(1, int(np.floor(0.8 * max_lag_steps)))
        plateau_start = float(lag_times[plateau_start_index])
    else:
        plateau_start = float(plateau_start_ps)
    plateau_end = lag_end if plateau_end_ps is None else float(plateau_end_ps)
    if plateau_start < 0.0 or plateau_end <= plateau_start or plateau_end > lag_end:
        raise ValueError(
            f"Plateau window must satisfy 0 <= start < end <= {lag_end:g} ps."
        )

    mean_volume = float(np.mean(volumes))
    modulus = shear_relaxation_modulus(
        pressure,
        mean_volume,
        temperature_k,
        max_lag_steps=max_lag_steps,
    )
    viscosity = viscosity_curve(modulus, interval)
    eta = plateau_average(lag_times, viscosity, plateau_start, plateau_end)

    block_etas = []
    for indices in block_indices:
        block_modulus = shear_relaxation_modulus(
            pressure[indices],
            float(np.mean(volumes[indices])),
            temperature_k,
            max_lag_steps=max_lag_steps,
        )
        block_viscosity = viscosity_curve(block_modulus, interval)
        block_etas.append(
            plateau_average(lag_times, block_viscosity, plateau_start, plateau_end)
        )
    block_etas = np.asarray(block_etas, dtype=float)
    sem = (
        float(np.std(block_etas, ddof=1) / np.sqrt(blocks))
        if blocks > 1
        else float("nan")
    )
    return {
        "lag_times_ps": lag_times,
        "modulus_kj_mol_nm3": modulus,
        "viscosity_pa_s": viscosity,
        "eta_pa_s": eta,
        "sem_pa_s": sem,
        "block_etas_pa_s": block_etas,
        "sample_interval_ps": interval,
        "volume_nm3": mean_volume,
        "relative_volume_span": relative_volume_span,
        "plateau_start_ps": plateau_start,
        "plateau_end_ps": plateau_end,
    }


def _write_xvg(path, result, metadata):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# DROPPS Green-Kubo zero-shear viscosity\n")
        handle.write(
            "# G(t)=V/(5RT)*sum(C_offdiag)+V/(30RT)*sum(C_normal-difference)\n"
        )
        handle.write(
            f"# temperature_K={metadata['temperature_k']:.10g} "
            f"volume_nm3={result['volume_nm3']:.10g} "
            f"sample_interval_ps={result['sample_interval_ps']:.10g}\n"
        )
        handle.write(
            f"# plateau_start_ps={result['plateau_start_ps']:.10g} "
            f"plateau_end_ps={result['plateau_end_ps']:.10g} "
            f"eta0_Pa_s={result['eta_pa_s']:.10g} "
            f"block_sem_Pa_s={result['sem_pa_s']:.10g} "
            f"blocks={metadata['blocks']}\n"
        )
        handle.write("# columns: lag_ps G_kJ_mol^-1_nm^-3 eta_Pa_s eta_mPa_s\n")
        handle.write('@ title "Green-Kubo zero-shear viscosity"\n')
        handle.write('@ xaxis label "Lag time (ps)"\n')
        handle.write('@ yaxis label "Running viscosity (Pa s)"\n')
        handle.write("@TYPE xy\n")
        for lag, modulus, eta in zip(
            result["lag_times_ps"],
            result["modulus_kj_mol_nm3"],
            result["viscosity_pa_s"],
        ):
            handle.write(f"{lag:.12g} {modulus:.12g} {eta:.12g} {eta * 1000:.12g}\n")


def getargs_viscosity_gk(argv):
    parser = ArgumentParser(prog=prog, description=desc)
    parser.add_argument(
        "-f", "--input", required=True, help="Input runtime pressure-tensor file (.stress.xvg)."
    )
    parser.add_argument("-o", "--output", required=True,
                        help="Output Green-Kubo viscosity time series (.xvg); the extension is added if omitted.")
    parser.add_argument(
        "-tm",
        "--temperature",
        type=float,
        help="Temperature, in K; if omitted, use the value stored in the stress file.",
    )
    parser.add_argument("-b", "--start-time", type=float, help="First stress-sample time to analyze, in ps.")
    parser.add_argument("-e", "--end-time", type=float, help="Last stress-sample time to analyze, in ps.")
    parser.add_argument("--max-lag", type=float, help="Maximum correlation lag, in ps.")
    parser.add_argument(
        "--plateau-start", type=float, help="Start of the viscosity averaging plateau, in ps."
    )
    parser.add_argument(
        "--plateau-end", type=float, help="End of the viscosity averaging plateau, in ps."
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=5,
        help="Number of contiguous blocks used for the standard error.",
    )
    return parser.parse_args(argv)


def viscosity_gk(args):
    data, metadata = read_stress_tensor(args.input)
    if metadata.get("pressure_coupling", "False").lower() == "true":
        raise ValueError(
            "This stress file was recorded with pressure coupling enabled. "
            "Run an NVT production simulation (pcoulp = False) for Green-Kubo viscosity."
        )
    data = _select_time_window(data, args.start_time, args.end_time)
    if args.temperature is not None:
        temperature = args.temperature
    else:
        stored_temperature = metadata.get("temperature_K")
        if stored_temperature in (None, "unknown"):
            raise ValueError(
                "No temperature is stored in the stress file; supply --temperature."
            )
        temperature = float(stored_temperature)

    result = analyze_green_kubo(
        data,
        temperature,
        blocks=args.blocks,
        max_lag_ps=args.max_lag,
        plateau_start_ps=args.plateau_start,
        plateau_end_ps=args.plateau_end,
    )
    output_path = validate_extension(args.output, "xvg")
    _write_xvg(
        output_path,
        result,
        {"temperature_k": temperature, "blocks": args.blocks},
    )

    eta_mpa_s = result["eta_pa_s"] * 1000.0
    sem_mpa_s = result["sem_pa_s"] * 1000.0
    print(f"## Green-Kubo curve written to {Path(output_path).resolve()}.")
    if np.isfinite(sem_mpa_s):
        print(
            f"## eta_0 = {eta_mpa_s:.8g} +/- {sem_mpa_s:.4g} mPa s "
            f"(block SEM, {args.blocks} blocks)."
        )
    else:
        print(f"## eta_0 = {eta_mpa_s:.8g} mPa s (one block; SEM unavailable).")
    print(
        f"## Plateau average: {result['plateau_start_ps']:.6g} to "
        f"{result['plateau_end_ps']:.6g} ps. Inspect the running integral and "
        "set --plateau-start/--plateau-end explicitly for production results."
    )


viscosity_gk_commands = single_command(
    "viscosity-gk", getargs_viscosity_gk, viscosity_gk, desc
)
