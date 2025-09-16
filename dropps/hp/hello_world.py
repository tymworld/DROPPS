package_name = "DROPPS"
package_abbr = "DPS"
package_version = "0.1.0"

from sys import executable
from os import path, getcwd

def print_hello(entry, cmd):

    package_info = f":-) {package_name} - {package_abbr}, {package_version} (:-"
    author_info = f":-> Developer: Yiming Tang @ Fudan (:-"
    contact_into = f":-> Contact: ymtang@fudan.edu.cn (:-"

    title = package_info.center(60," ") + "\n"
    title += author_info.center(60, " ") + "\n"
    title += contact_into.center(60, " ") + "\n\n"

    title += f"Executable:   {executable}\n"
    title += f"Entry dir:    {entry}\n"
    title += f"Working dir:  {getcwd()}\n"
    title += f"Command line: \n  {cmd}\n\n"
    title += f"SYNOPSIS\n\ndps command options\n"
    title += f"  use \"dps help command\"   to see a list of available commands.\n"
    title += f"  use \"dps help \'command\'\" to see help on the specific command.\n\n"
    title += f"DROPPS reminds you: \"Once there is a smart boy who loves publishing his papers in Nature.\"\n"

    print(title)
