"""Model-building, conversion, and simulation command modules.

Command registration is intentionally centralized in
``dropps.share.all_commands``.  Keeping this package initializer lightweight
allows individual command modules and their tests to be imported without
loading every scientific runtime dependency.
"""
