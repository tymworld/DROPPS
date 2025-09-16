def getargs_help(argv):

    from dropps.share.all_commands import all_commands
    if len(argv) == 0:
        args = ["commands"]
    else:
        args = []
        for command_name in argv:
            if command_name in all_commands.command_names:
                args.append(command_name)
            elif command_name == "commands":
                args.append(command_name)
            else:
                print(f"## WARNING: Unknown command {command_name}.")
        
    args = list(set(args))
    return args 

def help(args):

    from dropps.share.all_commands import all_commands, commands_simulation, commands_modelling, commands_analysis

    if "commands" in args:
    
        print(f"## Usage: dps command options")
        print(f"## Existing commands:")
        print(f"## Commands for system modelling:")
        for command in commands_modelling:
            print(f"## {command.name:15s}: {command.desc}")
        print(f"## Commands for simulation:")
        for command in commands_simulation:
            print(f"## {command.name:15s}: {command.desc}")
        print(f"## Commands for simulation analysis:")
        for command in commands_analysis:
            print(f"## {command.name:15s}: {command.desc}")
    
    else:
        for command_name in args:
            print("#" * (len(all_commands.desc(command_name)) + 11))
            print(f"# COMMAND: {command_name}")
            print(f"# DESC:    {all_commands.desc(command_name)}")
            print("#" * (len(all_commands.desc(command_name)) + 11))
            help_command = all_commands.getargs(command_name)
            help_command(["-h"])

from dropps.share.command_class import single_command
help_commands = single_command("help", getargs_help, help, "help")
