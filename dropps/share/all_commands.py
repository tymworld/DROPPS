import dropps.commands
import dropps.analysis

from dropps.share.command_class import all_commands_class

commands_modelling = [dropps.commands.help.help_commands,
                      dropps.commands.pdb2cgps.pdb2cgps_commands,
                      dropps.commands.genelastic.genelastic_commands,
                      dropps.commands.genmesh.genmesh_commands,
                      dropps.commands.addangle.addangle_commands,
                      dropps.commands.editconf.editconf_commands,
                      dropps.commands.grompp.grompp_commands,
                      dropps.commands.modifyres.modifyres_commands,
                      dropps.commands.trjconv.trjconv_commands]

commands_simulation = [dropps.commands.mdrun.mdrun_commands]

commands_analysis = [dropps.analysis.make_ndx.make_ndx_commands,
                     dropps.analysis.extract.extract_commands,
                     dropps.analysis.check.check_commands,
                     dropps.analysis.density.density_commands,
                     dropps.analysis.gyrate.gyrate_commands,
                     dropps.analysis.contact_map.contactmap_commands,
                     dropps.analysis.angle.angle_commands,
                     dropps.analysis.intra_distance.intra_distance_commands,
                     dropps.analysis.inter_distance.inter_distance_commands,
                     dropps.analysis.contact_number.contact_number_commands,
                     dropps.analysis.msd.msd_commands]

all_commands = all_commands_class(commands_modelling + commands_simulation + commands_analysis)
