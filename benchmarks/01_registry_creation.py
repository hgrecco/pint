import pint

from . import util


def time_create_registry(args):
    pint.UnitRegistry(*args)


time_create_registry.params = [[(None,), tuple(), (util.get_tiny_def(),)]]
