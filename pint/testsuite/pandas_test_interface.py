# -*- coding: utf-8 -*-

# - pandas test resources https://github.com/pandas-dev/pandas/blob/master/pandas/tests/extension/base/__init__.py

import sys

import pint
import pytest
import pandas as pd
from pandas.compat import PY3
from pandas.tests.extension import base
from pandas.core import ops

import numpy as np
import pint.pandas_interface as ppi
import operator
from .test_quantity import QuantityTestCase
from ..errors import DimensionalityError
from ..pandas_interface import PintArray


ureg = pint.UnitRegistry()


@pytest.fixture
def dtype():
    return ppi.PintType()


@pytest.fixture
def data():
    return ppi.PintArray(np.arange(100) * ureg.kilogram)


@pytest.fixture
def data_missing():
    return ppi.PintArray([np.nan, 1] * ureg.meter)


@pytest.fixture(params=['data', 'data_missing'])
def all_data(request, data, data_missing):
    if request.param == 'data':
        return data
    elif request.param == 'data_missing':
        return data_missing


@pytest.fixture
def data_repeated(data):
    """Return different versions of data for count times"""
    # no idea what I'm meant to put here, try just copying from https://github.com/pandas-dev/pandas/blob/master/pandas/tests/extension/integer/test_integer.py
    def gen(count):
        for _ in range(count):
            yield data
    yield gen


@pytest.fixture
def data_for_sorting():
    return ppi.PintArray([0.3, 10, -50])
    # should probably get more sophisticated and do something like
    # [1 * ureg.meter, 3 * ureg.meter, 10 * ureg.centimeter]


@pytest.fixture
def data_missing_for_sorting():
    return ppi.PintArray([4, np.nan, -5])
    # should probably get more sophisticated and do something like
    # [4 * ureg.meter, np.nan, 10 * ureg.centimeter]


@pytest.fixture
def na_cmp():
    """Binary operator for comparing NA values.
    """
    return lambda x, y: bool(np.isnan(x)) & bool(np.isnan(y))


@pytest.fixture
def na_value():
    return ppi.PintType.na_value


@pytest.fixture
def data_for_grouping():
    # should probably get more sophisticated here and use units on all these
    # quantities
    a = 1
    b = 2 ** 32 + 1
    c = 2 ** 32 + 10
    return ppi.PintArray([
        b, b, np.nan, np.nan, a, a, b, c
    ])

# === missing from pandas extension docs about what has to be included in tests ===
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
    def check_opname(self, s, op_name, other, exc=None):
        op = self.get_op_from_name(op_name)

        self._check_op(s, op, other, exc)

    def _check_op(self, s, op, other, exc=None):
        if exc is None:
            result = op(s, other)
            expected = s.combine(other, op)
            self.assert_series_equal(result, expected)
        else:
            with pytest.raises(exc):
                op(s, other)

    def _check_divmod_op(self, s, op, other, exc=None):
        # divmod has multiple return values, so check separately
        if exc is None:
            result_div, result_mod = op(s, other)
            if op is divmod:
                expected_div, expected_mod = s // other, s % other
            else:
                expected_div, expected_mod = other // s, other % s
            self.assert_series_equal(result_div, expected_div)
            self.assert_series_equal(result_mod, expected_mod)
        else:
            with pytest.raises(exc):
                divmod(s, other)

    def _get_exception(self, data, op_name):
        if op_name in ["__pow__", "__rpow__"]:
            return op_name, DimensionalityError
        else:
            return op_name, None

    def test_arith_series_with_scalar(self, data, all_arithmetic_operators):
        # series & scalar
        op_name, exc = self._get_exception(data, all_arithmetic_operators)
        s = pd.Series(data)
        self.check_opname(s, op_name, s.iloc[0], exc=exc)

    @pytest.mark.xfail(run=True, reason="_reduce needs implementation")
    def test_arith_frame_with_scalar(self, data, all_arithmetic_operators):
        # frame & scalar
        op_name, exc = self._get_exception(data, all_arithmetic_operators)
        df = pd.DataFrame({'A': data})
        self.check_opname(df, op_name, data[0], exc=exc)

    @pytest.mark.xfail(run=True, reason="s.combine does not accept arrays")
    def test_arith_series_with_array(self, data, all_arithmetic_operators):
        # ndarray & other series
        op_name, exc = self._get_exception(data, all_arithmetic_operators)
        s = pd.Series(data)
        self.check_opname(s, op_name, data, exc=exc)

    # parameterise this to try divisor not equal to 1
    def test_divmod(self, data):
        s = pd.Series(data)
        self._check_divmod_op(s, divmod, 1*ureg.kg)
        self._check_divmod_op(1*ureg.kg, ops.rdivmod, s)

    def test_error(self, data, all_arithmetic_operators):
        # invalid ops

        op = all_arithmetic_operators
        s = pd.Series(data)
        ops = getattr(s, op)
        opa = getattr(data, op)

        # invalid scalars
        # TODO: work out how to make this more specific/test for the two
        #       different possible errors here
        with pytest.raises(Exception):
            ops('foo')

        # TODO: work out how to make this more specific/test for the two
        #       different possible errors here
        with pytest.raises(Exception):
            ops(pd.Timestamp('20180101'))

        # invalid array-likes
        # TODO: work out how to make this more specific/test for the two
        #       different possible errors here
        with pytest.raises(Exception):
            ops(pd.Series('foo', index=s.index))

        # 2d
        with pytest.raises(KeyError):
            opa(pd.DataFrame({'A': s}))

        with pytest.raises(ValueError):
            opa(np.arange(len(s)).reshape(-1, len(s)))


class TestComparisonOps(base.BaseComparisonOpsTests):
    def _compare_other(self, s, data, op_name, other):
        op = self.get_op_from_name(op_name)

        result = op(s,other)
        expected = op(s.values.data, other)
        assert (result==expected).all()

    def test_compare_scalar(self, data, all_compare_operators):
        op_name = all_compare_operators
        s = pd.Series(data)
        other = data[0]
        self._compare_other(s, data, op_name, other)

    def test_compare_array(self, data, all_compare_operators):
        # nb this compares an quantity containing array
        # eg Q_([1,2],"m")
        op_name = all_compare_operators
        s = pd.Series(data)
        other = data.data
        self._compare_other(s, data, op_name, other)


class TestOpsUtil(base.BaseOpsUtil):
    pass

class TestMissing(base.BaseMissingTests):
    pass

class TestReshaping(base.BaseReshapingTests):
    pass

class TestSetitem(base.BaseSetitemTests):
    pass


class TestUserInterface(object):
    def test_get_underlying_data(self, data):
        ser = pd.Series(data)
        # this first test creates an array of bool (which is desired, eg for indexing)
        assert all(ser.values == data)
        assert ser.values[23] == data[23]

    def test_arithmetic(self, data):
        ser = pd.Series(data)
        ser2 = ser + ser
        assert all(ser2.values == 2*data)

    def test_initialisation(self, data):
        # fails with plain array
        # works with PintArray
        strt = np.arange(100) * ureg.newton

        # it is sad this doesn't work
        with pytest.raises(ValueError):
            ser_fail = pd.Series(strt, dtype=ppi.PintType())
            assert all(ser_fail.values == strt)

        # This needs to be a list of scalar quantities to work :<
        ser = pd.Series([q for q in strt], dtype=ppi.PintType())
        assert all(ser.values == strt)


arithmetic_ops = [
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.floordiv,
    operator.pow,
]

comparative_ops = [
    operator.eq,
    operator.le,
    operator.lt,
    operator.ge,
    operator.gt,
]

class TestPintArrayQuantity(QuantityTestCase):
    FORCE_NDARRAY = True

    def test_pintarray_creation(self):
        x = self.Q_([1, 2, 3],"m")
        ys = [
            PintArray(x),
            PintArray._from_sequence([item for item in x])
        ]
        for y in ys:
            self.assertQuantityAlmostEqual(x,y.data)

    def test_pintarray_operations(self):
        # Perform operations with Quantities and PintArrays
        # The resulting Quantity and PintArray.Data should be the same
        # a op b == c
        def test_op(a_pint, a_pint_array, b_, coerce=True):
            try:
                result_pint = op(a_pint, b_)
                if coerce:
                    # a PintArray is returned from arithmetics, so need the data
                    c_pint_array = op(a_pint_array, b_).data
                else:
                    # a boolean array is returned from comparatives
                    c_pint_array = op(a_pint_array, b_)

                self.assertQuantityAlmostEqual(result_pint, c_pint_array)

            except Exception as caught_exception:
                self.assertRaises(type(caught_exception), op, a_pint_array, b_)


        a_pints = [
            self.Q_([3, 4], "m"),
            self.Q_([3, 4], ""),
        ]

        a_pint_arrays = [PintArray(q) for q in a_pints]

        bs = [
            2,
            self.Q_(3, "m"),
            [1., 3.],
            [3.3, 4.4],
            self.Q_([6, 6], "m"),
            self.Q_([7., np.nan]),
            # PintArray(self.Q_([6,6],"m")),
            # PintArray(self.Q_([7.,np.nan])),
        ]

        for a_pint, a_pint_array in zip(a_pints, a_pint_arrays):
            for b in bs:
                for op in arithmetic_ops:
                    test_op(a_pint, a_pint_array, b)
                for op in comparative_ops:
                    test_op(a_pint, a_pint_array, b, coerce=False)

    def test_mismatched_dimensions(self):
        x_and_ys=[
            (PintArray(self.Q_([5], "m")), [1, 1]),
            (PintArray(self.Q_([5, 5, 5], "m")), [1, 1]),
            (PintArray(self.Q_([5, 5], "m")), [1]),
        ]
        for x, y in x_and_ys:
            for op in comparative_ops + arithmetic_ops:
                self.assertRaises(ValueError, op, x, y)
