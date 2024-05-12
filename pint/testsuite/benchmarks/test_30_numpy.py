from __future__ import annotations

import itertools as it
import operator
from collections.abc import Generator
from typing import Any

import pytest

import pint
from pint.compat import np

from ..helpers import requires_numpy

SMALL_VEC_LEN = 3
MID_VEC_LEN = 1_000
LARGE_VEC_LEN = 1_000_000

LENGTHS = ("short", "mid")
ALL_VALUES = tuple(
    f"{a}_{b}" for a, b in it.product(LENGTHS, ("list", "tuple", "array"))
)
ALL_ARRAYS = ("short_array", "mid_array")
UNITS = ("meter", "kilometer")
ALL_ARRAYS_Q = tuple(f"{a}_{b}" for a, b in it.product(ALL_ARRAYS, UNITS))

OP1 = (operator.neg,)  # operator.truth,
OP2_CMP = (operator.eq, operator.lt)
OP2_MATH = (operator.add, operator.sub, operator.mul, operator.truediv)

if np is None:
    NUMPY_OP1_MATH = NUMPY_OP2_CMP = NUMPY_OP2_MATH = ()
else:
    NUMPY_OP1_MATH = (np.sqrt, np.square)
    NUMPY_OP2_CMP = (np.equal, np.less)
    NUMPY_OP2_MATH = (np.add, np.subtract, np.multiply, np.true_divide)


def float_range(n: int) -> Generator[float, None, None]:
    return (float(x) for x in range(1, n + 1))


@pytest.fixture
def setup(registry_tiny) -> tuple[pint.UnitRegistry, dict[str, Any]]:
    data = {}
    short = list(float_range(3))
    mid = list(float_range(1_000))

    data["short_list"] = short
    data["short_tuple"] = tuple(short)
    data["short_array"] = np.asarray(short)
    data["mid_list"] = mid
    data["mid_tuple"] = tuple(mid)
    data["mid_array"] = np.asarray(mid)

    ureg = registry_tiny

    for key in ALL_ARRAYS:
        data[key + "_meter"] = data[key] * ureg.meter
        data[key + "_kilometer"] = data[key] * ureg.kilometer

    return ureg, data


@requires_numpy
def test_finding_meter_getattr(benchmark, setup):
    ureg, _ = setup
    benchmark(getattr, ureg, "meter")


@requires_numpy
def test_finding_meter_getitem(benchmark, setup):
    ureg, _ = setup
    benchmark(operator.getitem, ureg, "meter")


@requires_numpy
@pytest.mark.parametrize(
    "unit", ["meter", "angstrom", "meter/second", "angstrom/minute"]
)
def test_base_units(benchmark, setup, unit):
    ureg, _ = setup
    benchmark(ureg.get_base_units, unit)


@requires_numpy
@pytest.mark.parametrize("key", ALL_ARRAYS)
def test_build_by_mul(benchmark, setup, key):
    ureg, data = setup
    benchmark(operator.mul, data[key], ureg.meter)


@requires_numpy
@pytest.mark.parametrize("key", ALL_ARRAYS_Q)
@pytest.mark.parametrize("op", OP1 + NUMPY_OP1_MATH)
def test_op1(benchmark, setup, key, op):
    _, data = setup
    benchmark(op, data[key])


@requires_numpy
@pytest.mark.parametrize(
    "keys",
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
)
@pytest.mark.parametrize("op", OP2_MATH + OP2_CMP + NUMPY_OP2_MATH + NUMPY_OP2_CMP)
def test_op2(benchmark, setup, keys, op):
    _, data = setup
    key1, key2 = keys
    benchmark(op, data[key1], data[key2])
