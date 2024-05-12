from __future__ import annotations

import itertools as it
import operator
from typing import Any

import pytest

import pint

UNITS = ("meter", "kilometer", "second", "minute", "angstrom")
ALL_VALUES = ("int", "float", "complex")
ALL_VALUES_Q = tuple(
    f"{a}_{b}" for a, b in it.product(ALL_VALUES, ("meter", "kilometer"))
)

OP1 = (operator.neg, operator.truth)
OP2_CMP = (operator.eq,)  # operator.lt)
OP2_MATH = (operator.add, operator.sub, operator.mul, operator.truediv)


@pytest.fixture
def setup(registry_tiny) -> tuple[pint.UnitRegistry, dict[str, Any]]:
    data = {}
    data["int"] = 1
    data["float"] = 1.0
    data["complex"] = complex(1, 2)

    ureg = registry_tiny

    for key in ALL_VALUES:
        data[key + "_meter"] = data[key] * ureg.meter
        data[key + "_kilometer"] = data[key] * ureg.kilometer

    return ureg, data


@pytest.mark.parametrize("key", ALL_VALUES)
def test_build_by_mul(benchmark, setup, key):
    ureg, data = setup
    benchmark(operator.mul, data[key], ureg.meter)


@pytest.mark.parametrize("key", ALL_VALUES_Q)
@pytest.mark.parametrize("op", OP1)
def test_op1(benchmark, setup, key, op):
    _, data = setup
    benchmark(op, data[key])


@pytest.mark.parametrize("keys", tuple(it.product(ALL_VALUES_Q, ALL_VALUES_Q)))
@pytest.mark.parametrize("op", OP2_MATH + OP2_CMP)
def test_op2(benchmark, setup, keys, op):
    _, data = setup
    key1, key2 = keys
    benchmark(op, data[key1], data[key2])


@pytest.mark.parametrize("key", ALL_VALUES_Q)
def test_wrapper(benchmark, setup, key):
    ureg, data = setup
    value, unit = key.split("_")

    @ureg.wraps(None, (unit,))
    def f(a):
        pass

    benchmark(f, data[key])


@pytest.mark.parametrize("key", ALL_VALUES_Q)
def test_wrapper_nonstrict(benchmark, setup, key):
    ureg, data = setup
    value, unit = key.split("_")

    @ureg.wraps(None, (unit,), strict=False)
    def f(a):
        pass

    benchmark(f, data[value])


@pytest.mark.parametrize("key", ALL_VALUES_Q)
def test_wrapper_ret(benchmark, setup, key):
    ureg, data = setup
    value, unit = key.split("_")

    @ureg.wraps(unit, (unit,))
    def f(a):
        return a

    benchmark(f, data[key])
