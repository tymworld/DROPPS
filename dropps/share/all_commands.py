"""The single authoritative registry for all public DROPPS commands."""

from dropps.analysis.angle import angle_commands
from dropps.analysis.assembly import assembly_commands
from dropps.analysis.check import check_commands
from dropps.analysis.coexistence import coexistence_commands
from dropps.analysis.contact import contact_commands
from dropps.analysis.contact_ng import contact_ng_commands
from dropps.analysis.contact_statistic import contact_statistic_commands
from dropps.analysis.densmap import densmap_commands
from dropps.analysis.density import density_commands
from dropps.analysis.distance import distance_commands
from dropps.analysis.energy import energy_commands
from dropps.analysis.exchange import exchange_commands
from dropps.analysis.gyrate import gyrate_commands
from dropps.analysis.image_interaction import image_interaction_commands
from dropps.analysis.inter_distance import inter_distance_commands
from dropps.analysis.intra_distance import intra_distance_commands
from dropps.analysis.make_ndx import make_ndx_commands
from dropps.analysis.msd import msd_commands
from dropps.analysis.pdb2bond import pdb2bond_commands
from dropps.analysis.phase_msd import phase_msd_commands
from dropps.analysis.polystat import polystat_commands
from dropps.analysis.rdf import rdf_commands
from dropps.analysis.rmsd import rmsd_commands
from dropps.analysis.rmsf import rmsf_commands
from dropps.analysis.sasa import sasa_commands
from dropps.analysis.ss import ss_commands
from dropps.analysis.surface_tension import surface_tension_commands
from dropps.analysis.timecorr import timecorr_commands
from dropps.analysis.viscosity_gk import viscosity_gk_commands
from dropps.commands.addangle import addangle_commands
from dropps.commands.convert_tpr import convert_tpr_commands
from dropps.commands.editconf import editconf_commands
from dropps.commands.eneconv import eneconv_commands
from dropps.commands.genelastic import genelastic_commands
from dropps.commands.genmesh import genmesh_commands
from dropps.commands.grompp import grompp_commands
from dropps.commands.gsd2xtc import gsd2xtc_commands
from dropps.commands.help import help_commands
from dropps.commands.mdrun import mdrun_commands
from dropps.commands.modifyres import modifyres_commands
from dropps.commands.pdb2cgps import pdb2dps_commands
from dropps.commands.report_methods import report_methods_commands
from dropps.commands.rerun import rerun_commands
from dropps.commands.trjcat import trjcat_commands
from dropps.commands.trjconv import trjconv_commands
from dropps.share.command_class import CommandRegistry, CommandSpec


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
    trjcat_commands,
    eneconv_commands,
    report_methods_commands,
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
    densmap_commands,
    coexistence_commands,
    exchange_commands,
    phase_msd_commands,
    gyrate_commands,
    polystat_commands,
    ss_commands,
    angle_commands,
    distance_commands,
    rdf_commands,
    contact_commands,
    contact_ng_commands,
    timecorr_commands,
    contact_statistic_commands,
    msd_commands,
    rmsd_commands,
    rmsf_commands,
    sasa_commands,
    assembly_commands,
    image_interaction_commands,
    pdb2bond_commands,
    surface_tension_commands,
    viscosity_gk_commands,
]

all_commands = CommandRegistry(
    commands_modelling + commands_simulation + commands_analysis,
    hidden_commands=(
        intra_distance_commands,
        inter_distance_commands,
        CommandSpec("intra-distance", intra_distance_commands.getargs,
                    intra_distance_commands.main_func, intra_distance_commands.desc),
        CommandSpec("inter-distance", inter_distance_commands.getargs,
                    inter_distance_commands.main_func, inter_distance_commands.desc),
    ),
)
