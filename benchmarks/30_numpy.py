import itertools as it
import operator

import numpy as np

import pint

from . import util

lengths = ("short", "mid")
all_values = tuple(
    "%s_%s" % (a, b) for a, b in it.product(lengths, ("list", "tuple", "array"))
)
all_arrays = ("short_array", "mid_array")
units = ("meter", "kilometer")
all_arrays_q = tuple("%s_%s" % (a, b) for a, b in it.product(all_arrays, units))

ureg = None
data = {}
op1 = (operator.neg,)  # operator.truth,
op2_cmp = (operator.eq, operator.lt)
op2_math = (operator.add, operator.sub, operator.mul, operator.truediv)
numpy_op2_cmp = (np.equal, np.less)
numpy_op2_math = (np.add, np.subtract, np.multiply, np.true_divide)


def float_range(n):
    return (float(x) for x in range(1, n + 1))


def setup(*args):

    global ureg, data
    short = list(float_range(3))
    mid = list(float_range(1_000))

    data["short_list"] = short
    data["short_tuple"] = tuple(short)
    data["short_array"] = np.asarray(short)
    data["mid_list"] = mid
    data["mid_tuple"] = tuple(mid)
    data["mid_array"] = np.asarray(mid)

    ureg = pint.UnitRegistry(util.get_tiny_def())

    for key in all_arrays:
        data[key + "_meter"] = data[key] * ureg.meter
        data[key + "_kilometer"] = data[key] * ureg.kilometer


def time_finding_meter_getattr():
    ureg.meter


def time_finding_meter_getitem():
    ureg["meter"]


def time_base_units(unit):
    ureg.get_base_units(unit)


time_base_units.params = ["meter", "angstrom", "meter/second", "angstrom/minute"]


def time_build_by_mul(key):
    data[key] * ureg.meter


time_build_by_mul.params = all_arrays


def time_op1(key, op):
    op(data[key])


time_op1.params = [all_arrays_q, op1 + (np.sqrt, np.square)]


def time_op2(keys, op):
    key1, key2 = keys
    op(data[key1], data[key2])


time_op2.params = [
    (
        ("short_array_meter", "short_array_meter"),
        ("short_array_meter", "short_array_kilometer"),
        ("short_array_kilometer", "short_array_meter"),
        ("short_array_kilometer", "short_array_kilometer"),
        ("mid_array_meter", "mid_array_meter"),
        ("mid_array_meter", "mid_array_kilometer"),
        ("mid_array_kilometer", "mid_array_meter"),
        ("mid_array_kilometer", "mid_array_kilometer"),
    ),
    op2_math + op2_cmp + numpy_op2_math + numpy_op2_cmp,
]
