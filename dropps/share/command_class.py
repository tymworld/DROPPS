class single_command:
    def __init__(self, name, getargs, main_func, desc=""):
        self.name = name
        self.getargs = getargs
        self.main_func = main_func
        self.desc = desc

class all_commands_class():
    def __init__(self, command_list):
        self.commmand_dict = {
            command.name : command
            for command in command_list
        }
        self.command_names = [command.name for command in command_list]
    
    def names(self):
        return self.command_names
    
    def exist(self, name):
        return name in self.command_names
    
    def getargs(self, name):
        if not self.exist(name):
            print(f"ERROR: Command {name} not exist.")
            quit()
        else:
            return self.commmand_dict[name].getargs

    def command(self, name):
        if not self.exist(name):
            print(f"ERROR: Command {name} not exist.")
            quit()
        else:
            return self.commmand_dict[name].main_func
    
    def desc(self, name):
        if not self.exist(name):
            print(f"ERROR: Command {name} not exist.")
            quit()
        else:
            return self.commmand_dict[name].desc