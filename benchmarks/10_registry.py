import pint

from . import util

units = ("meter", "kilometer", "second", "minute", "angstrom")

other_units = ("meter", "angstrom", "kilometer/second", "angstrom/minute")

all_values = ("int", "float", "complex")

ureg = None
data = {}


def setup(*args):

    global ureg, data

    data["int"] = 1
    data["float"] = 1.0
    data["complex"] = complex(1, 2)

    ureg = pint.UnitRegistry(util.get_tiny_def())


def my_setup(*args):
    global data
    setup(*args)
    for unit in units + other_units:
        data["uc_%s" % unit] = pint.registry.to_units_container(unit, ureg)


def time_build_cache():
    ureg._build_cache()


def time_getattr(key):
    getattr(ureg, key)


time_getattr.params = units


def time_getitem(key):
    ureg[key]


time_getitem.params = units


def time_parse_unit_name(key):
    ureg.parse_unit_name(key)


time_parse_unit_name.params = units


def time_parse_units(key):
    ureg.parse_units(key)


time_parse_units.params = units


def time_parse_expression(key):
    ureg.parse_expression("1.0 " + key)


time_parse_expression.params = units


def time_base_units(unit):
    ureg.get_base_units(unit)


time_base_units.params = other_units


def time_to_units_container_registry(unit):
    pint.registry.to_units_container(unit, ureg)


time_to_units_container_registry.params = other_units


def time_to_units_container_detached(unit):
    pint.registry.to_units_container(unit, ureg)


time_to_units_container_detached.params = other_units


def time_convert_from_uc(key):
    src, dst = key
    ureg._convert(1.0, data[src], data[dst])


time_convert_from_uc.setup = my_setup
time_convert_from_uc.params = [
    (("uc_meter", "uc_kilometer"), ("uc_kilometer/second", "uc_angstrom/minute"))
]


def time_parse_math_expression():
    ureg.parse_expression("3 + 5 * 2 + value", value=10)
