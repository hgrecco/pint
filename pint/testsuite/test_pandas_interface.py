# -*- coding: utf-8 -*-

# - pandas test resources https://github.com/pandas-dev/pandas/blob/master/pandas/tests/extension/base/__init__.py

import pytest
from pandas.compat import PY3
from pandas.tests.extension import base

import numpy as np
import pint
import pint.pandas_interface as ppi

ureg = pint.UnitRegistry()

@pytest.fixture
def dtype():
    return ppi.PintType()


@pytest.fixture
def data():
    return ppi.PintArray(np.arange(100) * ureg.meter)


@pytest.fixture
def data_missing():
    return ppi.PintArray([np.nan, 1] * ureg.meter)


@pytest.fixture(params=['data', 'data_missing'])
def all_data(request, data, data_missing):
    """Parametrized fixture giving 'data' and 'data_missing'"""
    if request.param == 'data':
        return data
    elif request.param == 'data_missing':
        return data_missing


@pytest.fixture
def data_repeated(data):
    """Return different versions of data for count times"""
    def gen(count):
        for _ in range(count):
            yield data  # no idea what I'm meant to put here, try just copying from https://github.com/pandas-dev/pandas/blob/master/pandas/tests/extension/integer/test_integer.py
    yield gen


@pytest.fixture
def data_for_sorting():
    return ppi.PintArray([0.3, 10, -50])
    # should probably get fancy and do something like
    # [1 * ureg.meter, 3*ureg.meter, 10 * ureg.centimeter]


@pytest.fixture
def data_missing_for_sorting():
    return ppi.PintArray([4, np.nan, -5])
    # should probably get fancy and do something like
    # [1 * ureg.meter, 3*ureg.meter, 10 * ureg.centimeter]


@pytest.fixture
def na_cmp():
    """Binary operator for comparing NA values.

    Should return a function of two arguments that returns
    True if both arguments are (scalar) NA for your type.

    By defult, uses ``operator.or``
    """
    return lambda x, y: bool(np.isnan(x)) & bool(np.isnan(y))


@pytest.fixture
def na_value():
    return ppi.PintType.na_value


@pytest.fixture
def data_for_grouping():
    a = 1
    b = 2 ** 32 + 1
    c = 2 ** 32 + 10
    return ppi.PintArray([
        b, b, np.nan, np.nan, a, a, b, c
    ])

# === missing from docs about what has to be included in tests ===
# copied from pandas/pandas/conftest.py
_all_arithmetic_operators = ['__add__', '__radd__',
                             '__sub__', '__rsub__',
                             '__mul__', '__rmul__',
                             '__floordiv__', '__rfloordiv__',
                             '__truediv__', '__rtruediv__',
                             '__pow__', '__rpow__',
                             '__mod__', '__rmod__']
if not PY3:
    _all_arithmetic_operators.extend(['__div__', '__rdiv__'])

@pytest.fixture(params=_all_arithmetic_operators)
def all_arithmetic_operators(request):
    """
    Fixture for dunder names for common arithmetic operations
    """
    return request.param

@pytest.fixture(params=['__eq__', '__ne__', '__le__',
                        '__lt__', '__ge__', '__gt__'])
def all_compare_operators(request):
    """
    Fixture for dunder names for common compare operations

    * >=
    * >
    * ==
    * !=
    * <
    * <=
    """
    return request.param
# =================================================================


class TestCasting(base.BaseCastingTests):
    pass


class TestConstructors(base.BaseConstructorsTests):
    pass


class TestDtype(base.BaseDtypeTests):
    pass


class TestGetitem(base.BaseGetitemTests):
    pass


class TestGroupby(base.BaseGroupbyTests):
    pass


class TestInterface(base.BaseInterfaceTests):
    pass


class TestMethods(base.BaseMethodsTests):
    pass


class TestArithmeticOps(base.BaseArithmeticOpsTests):
    pass


class TestComparisonOps(base.BaseComparisonOpsTests):
    pass


class TestOpsUtil(base.BaseOpsUtil):
    pass


# class TestMissing(base.BaseMissingTests):
#     pass


class TestReshaping(base.BaseReshapingTests):
    pass


# class TestSetitem(base.BaseSetitemTests):
#     pass

