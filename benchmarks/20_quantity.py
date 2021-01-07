import itertools as it
import operator

import pint

from . import util

units = ("meter", "kilometer", "second", "minute", "angstrom")
all_values = ("int", "float", "complex")
all_values_q = tuple(
    "%s_%s" % (a, b) for a, b in it.product(all_values, ("meter", "kilometer"))
)

op1 = (operator.neg, operator.truth)
op2_cmp = (operator.eq,)  # operator.lt)
op2_math = (operator.add, operator.sub, operator.mul, operator.truediv)

ureg = None
data = {}


def setup(*args):

    global ureg, data

    data["int"] = 1
    data["float"] = 1.0
    data["complex"] = complex(1, 2)

    ureg = pint.UnitRegistry(util.get_tiny_def())

    for key in all_values:
        data[key + "_meter"] = data[key] * ureg.meter
        data[key + "_kilometer"] = data[key] * ureg.kilometer


def time_build_by_mul(key):
    data[key] * ureg.meter


time_build_by_mul.params = all_values


def time_op1(key, op):
    op(data[key])


time_op1.params = [all_values_q, op1]


def time_op2(keys, op):
    key1, key2 = keys
    op(data[key1], data[key2])


time_op2.params = [tuple(it.product(all_values_q, all_values_q)), op2_math + op2_cmp]
