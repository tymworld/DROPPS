"""Canonical DROPPS command registry."""

from dropps.analysis import (
    angle_commands,
    assembly_commands,
    check_commands,
    contact_commands,
    contact_statistic_commands,
    density_commands,
    energy_commands,
    exchange_commands,
    gyrate_commands,
    inter_distance_commands,
    intra_distance_commands,
    make_ndx_commands,
    msd_commands,
    pdb2bond_commands,
    rmsd_commands,
)
from dropps.commands import (
    addangle_commands,
    convert_tpr_commands,
    editconf_commands,
    genelastic_commands,
    genmesh_commands,
    grompp_commands,
    gsd2xtc_commands,
    help_commands,
    mdrun_commands,
    modifyres_commands,
    pdb2dps_commands,
    rerun_commands,
    trjconv_commands,
)
from dropps.share.command_class import all_commands_class

commands_modelling = [
    help_commands,
    pdb2dps_commands,
    genelastic_commands,
    genmesh_commands,
    addangle_commands,
    editconf_commands,
    grompp_commands,
    convert_tpr_commands,
    modifyres_commands,
    trjconv_commands,
    gsd2xtc_commands,
]

commands_simulation = [
    mdrun_commands,
    rerun_commands,
]

commands_analysis = [
    make_ndx_commands,
    check_commands,
    energy_commands,
    density_commands,
    exchange_commands,
    gyrate_commands,
    angle_commands,
    intra_distance_commands,
    inter_distance_commands,
    contact_commands,
    contact_statistic_commands,
    msd_commands,
    rmsd_commands,
    assembly_commands,
    pdb2bond_commands,
]

all_commands = all_commands_class(
    commands_modelling + commands_simulation + commands_analysis
)
