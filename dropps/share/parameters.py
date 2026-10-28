from copy import deepcopy
import math

parameters_dict_template = {
    # Integration
    "integrator": "Langevin",
    "friction": 1.0,
    "dt": 0.01,
    "nsteps": 100000,
    "minimize": True,
    "max_step": 100000,
    "seed": 1215,
    "comm_mode": "Linear",
    "nstcomm": 100,
    "bondtype": "bond",
    # Energy minimization
    # "energytol"  : 10.0,
    "forcetol": 100.0,
    # Output control
    "nst_xout": 1000,
    # Deprecated compatibility key; mdrun -cpt now controls wall-clock
    # checkpointing and always saves a final checkpoint.
    "nst_cp": 0,
    "nst_energy": 10000,
    "energy_grps": "potentialEnergy,kineticEnergy,totalEnergy,temperature,boxX,boxY,boxZ,volume,density",
    "nst_stress": 0,
    "stress_strain": 0.0003,
    "stress_output": "auto",
    "stress_platform": "auto",
    "stress_precision": "double",
    "stress_device": "auto",
    "stress_threads": 0,
    # "nstvout"    : 10000,
    "nst_screenlog": 1000,
    "nst_filelog": 10000,
    # Deprecated compatibility keys. Progress log fields are now fixed;
    # filelog_grps is treated as an energy_grps alias for old MDP files.
    "screenlog_grps": "step,elapsedTime,remainingTime,speed,progress",
    "filelog_grps": "step,potentialEnergy,kineticEnergy",
    # Neighbor searching
    "nstlist": 10,
    "cutoff_scheme_lj": "static",
    # "cutoff_scheme_coul": "static",
    "cutoff_lj_multi": 3.0,
    # "cutoff_coul_multi": 3.0,
    "cutoff_lj": 1.5,
    "cutoff_coul": 1.5,
    "shift_lj": True,
    "shift_coul": True,
    "buffer": 0.5,
    # Force calculations
    "coulombtype": "yukawa",
    "vdwtype": "pLJ",
    "salt_conc": 0.1,
    # Temperature coupling
    "tcoulp": "Bussi",
    "production_temperature": 300.0,
    "gen_vel": True,
    "initial_temperature": 150.0,
    "warming_speed": 1.0,
    "keep_warming_trajectory": False,
    # Pressure coupling
    "pcoulp": True,
    "ref_P": 1.0,
    "tau_P": 5.0,
}


def getparameter(mdp_file):
    parameters = deepcopy(parameters_dict_template)

    try:
        with open(mdp_file, "r", encoding="utf-8") as stream:
            raw_parameters = {}
            for line_number, raw_line in enumerate(stream, start=1):
                line = raw_line.split("#", 1)[0].strip()
                if not line:
                    continue
                if "=" not in line:
                    raise ValueError(
                        f"Malformed MDP line {line_number}: expected 'name = value'."
                    )
                raw_name, raw_value = line.split("=", 1)
                name = raw_name.strip().replace("-", "_")
                value = raw_value.strip()
                if not name or not value:
                    raise ValueError(
                        f"Malformed MDP line {line_number}: empty name or value."
                    )
                if name in raw_parameters:
                    raise ValueError(
                        f"Duplicate MDP parameter {name.replace('_', '-')} "
                        f"on line {line_number}."
                    )
                raw_parameters[name] = value
    except OSError as exc:
        raise ValueError(f"Cannot read parameter file {mdp_file}: {exc}") from exc

    for param in raw_parameters:
        if param not in parameters:
            raise ValueError(
                f"Unknown key {param.replace('_', '-')} in parameter file."
            )
        if isinstance(parameters[param], bool):
            if raw_parameters[param] == "True":
                parameters[param] = True
            elif raw_parameters[param] == "False":
                parameters[param] = False
            else:
                raise ValueError(
                    f"Cannot process {param.replace('_', '-')}="
                    f"{raw_parameters[param]!r}; booleans must be True or False."
                )
        else:
            try:
                parameters[param] = type(parameters[param])(raw_parameters[param])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Cannot process {param.replace('_', '-')}="
                    f"{raw_parameters[param]!r}."
                ) from exc

    if "nst_filelog" in raw_parameters and "nst_energy" not in raw_parameters:
        parameters["nst_energy"] = parameters["nst_filelog"]
        print(
            "## WARNING: nst-energy is absent; using deprecated nst-filelog "
            "as the energy-output interval."
        )
    if "filelog_grps" in raw_parameters and "energy_grps" not in raw_parameters:
        parameters["energy_grps"] = parameters["filelog_grps"]
        print("## WARNING: filelog-grps is deprecated; treating it as energy-grps.")
    if "screenlog_grps" in raw_parameters:
        print(
            "## WARNING: screenlog-grps is deprecated and ignored because "
            "progress log fields are fixed."
        )
    if "nst_cp" in raw_parameters:
        print(
            "## WARNING: nst-cp is deprecated and ignored; use mdrun -cpt "
            "to configure checkpoint minutes."
        )

    for deprecated_parameter, replacement in (
        ("tcoulp", "the Langevin integrator's friction parameter"),
        ("nstlist", "OpenMM's internal neighbor-list scheduler"),
        ("buffer", "OpenMM's internal neighbor-list buffering"),
    ):
        if deprecated_parameter in raw_parameters:
            print(
                f"## WARNING: {deprecated_parameter.replace('_', '-')} is "
                f"deprecated and ignored; DROPPS uses {replacement}."
            )

    validate_parameters(parameters)
    return parameters


def _require_choice(parameters, name, choices):
    value = parameters[name]
    if value not in choices:
        allowed = ", ".join(repr(choice) for choice in choices)
        raise ValueError(
            f"Invalid {name.replace('_', '-')}={value!r}; allowed values are {allowed}."
        )


def _require_finite(parameters, name, *, minimum=None, strict=False):
    value = parameters[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name.replace('_', '-')} must be numeric.")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name.replace('_', '-')} must be finite.")
    if minimum is not None:
        invalid = value <= minimum if strict else value < minimum
        if invalid:
            comparison = ">" if strict else ">="
            raise ValueError(
                f"{name.replace('_', '-')} must be {comparison} {minimum}."
            )


def _require_integer(parameters, name, *, minimum=0, strict=False):
    value = parameters[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name.replace('_', '-')} must be an integer.")
    if not math.isfinite(float(value)) or int(value) != value:
        raise ValueError(f"{name.replace('_', '-')} must be an integer.")
    invalid = value <= minimum if strict else value < minimum
    if invalid:
        comparison = ">" if strict else ">="
        raise ValueError(f"{name.replace('_', '-')} must be {comparison} {minimum}.")


def validate_parameters(parameters):
    """Validate the complete simulation configuration before system creation."""

    missing = sorted(set(parameters_dict_template).difference(parameters))
    if missing:
        raise ValueError(
            "Simulation parameters are missing required keys: "
            + ", ".join(name.replace("_", "-") for name in missing)
            + "."
        )

    _require_choice(parameters, "integrator", ("Langevin", "steep"))
    _require_choice(parameters, "comm_mode", ("Linear", "none"))
    _require_choice(parameters, "bondtype", ("bond", "constraint"))
    _require_choice(parameters, "cutoff_scheme_lj", ("static", "dynamic"))
    _require_choice(parameters, "coulombtype", ("yukawa", "no"))
    _require_choice(parameters, "vdwtype", ("pLJ", "MPiPi"))
    _require_choice(
        parameters,
        "stress_platform",
        ("auto", "CUDA", "OpenCL", "CPU", "Reference"),
    )
    _require_choice(parameters, "stress_precision", ("single", "mixed", "double"))

    _require_finite(parameters, "friction", minimum=0.0)
    _require_finite(parameters, "dt", minimum=0.0, strict=True)
    _require_finite(parameters, "forcetol", minimum=0.0, strict=True)
    _require_finite(parameters, "cutoff_lj", minimum=0.0, strict=True)
    _require_finite(parameters, "cutoff_lj_multi", minimum=0.0, strict=True)
    if parameters["coulombtype"] == "yukawa":
        _require_finite(parameters, "cutoff_coul", minimum=0.0, strict=True)
        _require_finite(parameters, "salt_conc", minimum=0.0, strict=True)
    _require_finite(parameters, "production_temperature", minimum=0.0, strict=True)
    _require_finite(parameters, "initial_temperature", minimum=0.0, strict=True)
    _require_finite(parameters, "ref_P")
    _require_finite(parameters, "stress_strain", minimum=0.0, strict=True)
    if parameters["stress_strain"] >= 0.02:
        raise ValueError("stress-strain must be smaller than 0.02.")

    for name in (
        "nsteps",
        "max_step",
        "seed",
        "nst_xout",
        "nst_cp",
        "nst_energy",
        "nst_stress",
        "nst_screenlog",
        "nst_filelog",
        "stress_threads",
    ):
        _require_integer(parameters, name)
    _require_integer(
        parameters,
        "nstcomm",
        minimum=0,
        strict=parameters["comm_mode"] == "Linear",
    )
    _require_integer(
        parameters,
        "tau_P",
        minimum=0,
        strict=parameters["pcoulp"] is True,
    )

    if parameters["initial_temperature"] != parameters["production_temperature"]:
        _require_finite(parameters, "warming_speed", minimum=0.0, strict=True)

    if parameters["nst_energy"] > 0:
        from dropps.share.energy_reporter import normalize_field_names

        normalize_field_names(parameters["energy_grps"])

    return parameters


if __name__ == "__main__":
    print(getparameter("../templates/md.mdp"))
