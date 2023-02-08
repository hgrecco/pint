import pint

from . import util


def time_create_registry(args):
    if len(args) == 2:
        pint.UnitRegistry(args[0], cache_folder=args[1])
    else:
        pint.UnitRegistry(*args)


time_create_registry.params = [[(None,), tuple(), (util.get_tiny_def(),), ("", None)]]
