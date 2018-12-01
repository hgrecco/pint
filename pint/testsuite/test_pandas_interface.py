# -*- coding: utf-8 -*-

# - pandas test resources https://github.com/pandas-dev/pandas/blob/master/pandas/tests/extension/base/__init__.py

from pint.compat import HAS_PROPER_PANDAS, HAS_PYTEST

# I can't see how else to do this without this massive if clause...
if not (HAS_PYTEST and HAS_PROPER_PANDAS):
    from pint.testsuite import BaseTestCase
    class TestPandasException(BaseTestCase):
        def test_pandas_exception(self):
            expected_error_msg = (
                "The installed version of Pandas is not compatible with Pint, please "
                "check the docs."
            )
            with self.assertRaises(ImportError) as cm:
                import pint.pandas_interface

            self.assertEqual(str(cm.exception), expected_error_msg)

    if not (HAS_PYTEST and HAS_PROPER_PANDAS):
        msg_end = "the correct version of Pandas and pytest installed"
    elif not HAS_PROPER_PANDAS:
        msg_end = "the latest version of Pandas installed"
    elif not HAS_PYTEST:
        msg_end = "pytest installed"

    print("Skipping all Pandas tests except exception raising as we don't have {}".format(msg_end))


else:
    import sys
    from os.path import join, dirname


    import numpy as np
    import pytest
    import operator
    import warnings

    import pint
    import pint.pandas_interface as ppi
    from pint.testsuite import helpers

    import pandas as pd
    from pandas.compat import PY3
    from pandas.tests.extension import base
    from pandas.core import ops


    from pint.testsuite.test_quantity import QuantityTestCase
    from pint.errors import DimensionalityError
    from pint.pandas_interface import PintArray


    ureg = pint.UnitRegistry()


    @pytest.fixture
    def dtype():
        return ppi.PintType()


    def get_tdata():
        return ppi.PintArray(np.arange(start=1., stop=101.) * ureg.kilogram)


    @pytest.fixture
    def data():
        return get_tdata()


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
        return lambda x, y: bool(np.isnan(x.magnitude)) & bool(np.isnan(y))


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

        @pytest.mark.filterwarnings("ignore::pint.UnitStrippedWarning")
        # See test_setitem_mask_broadcast note
        @pytest.mark.parametrize('dropna', [True, False])
        def test_value_counts(self, all_data, dropna):
            all_data = all_data[:10]
            if dropna:
                other = all_data[~all_data.isna()]
            else:
                other = all_data

            result = pd.Series(all_data).value_counts(dropna=dropna).sort_index()
            expected = pd.Series(other).value_counts(
                dropna=dropna).sort_index()

            self.assert_series_equal(result, expected)

        @pytest.mark.filterwarnings("ignore::pint.UnitStrippedWarning")
        # See test_setitem_mask_broadcast note
        @pytest.mark.parametrize('box', [pd.Series, lambda x: x])
        @pytest.mark.parametrize('method', [lambda x: x.unique(), pd.unique])
        def test_unique(self, data, box, method):
            duplicated = box(data._from_sequence([data[0], data[0]]))

            result = method(duplicated)

            assert len(result) == 1
            assert isinstance(result, type(data))
            assert result[0] == duplicated[0]


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
        @pytest.mark.parametrize('setter', ['loc', None])
        @pytest.mark.filterwarnings("ignore::pint.UnitStrippedWarning")
        # Pandas performs a hasattr(__array__), which triggers the warning
        # Debugging it does not pass through a PintArray, so
        # I think this needs changing in pint quantity
        # eg s[[True]*len(s)]=Q_(1,"m")
        def test_setitem_mask_broadcast(self, data, setter):
            ser = pd.Series(data)
            mask = np.zeros(len(data), dtype=bool)
            mask[:2] = True

            if setter:   # loc
                target = getattr(ser, setter)
            else:  # __setitem__
                target = ser

            operator.setitem(target, mask, data[10])
            assert ser[0] == data[10]
            assert ser[1] == data[10]


    # would be ideal to just test all of this by running the example notebook
    # but this isn't a discussion we've had yet

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

        def test_df_operations(self):
            # simply a copy of what's in the notebook
            Q_ = ureg.Quantity
            df = pd.DataFrame({
                "torque": PintArray(Q_([1, 2, 2, 3], "lbf ft")),
                "angular_velocity": PintArray(Q_([1000, 2000, 2000, 3000], "rpm"))
            })

            df['power'] = df['torque'] * df['angular_velocity']

            df.power.values.data
            df.torque.values.data
            df.angular_velocity.values.data

            df.power.values.data.to("kW")

            test_csv = join(
                dirname(__file__),
                "test-data", "pandas_test.csv"
            )

            df = pd.read_csv(test_csv, header=[0,1])
            df_ = df.pint.quantify(ureg, level=-1)

            df_['mech power'] = df_.speed*df_.torque
            df_['fluid power'] = df_['fuel flow rate'] * df_['rail pressure']

            df_.pint.dequantify()

            df_['fluid power'] = df_['fluid power'].pint.to("kW")
            df_['mech power'] = df_['mech power'].pint.to("kW")
            df_.pint.dequantify()

            df_.pint.to_base_units().pint.dequantify()


    class TestDataFrameAccessor(object):
        def test_index_maintained(self):
            test_csv = join(
                dirname(__file__),
                "test-data", "pandas_test.csv"
            )

            df = pd.read_csv(test_csv, header=[0, 1])
            df.columns = pd.MultiIndex.from_arrays(
                [
                    ['Holden', 'Holden', 'Holden', 'Ford', 'Ford', 'Ford'],
                    ['speed', 'mech power', 'torque', 'rail pressure', 'fuel flow rate' ,'fluid power'],
                    ['rpm', 'kW', 'N m', 'bar', 'l/min', 'kW'],
                ],
                names = ['Car type', 'metric', 'unit']
            )
            df.index = pd.MultiIndex.from_arrays(
                [
                    [1, 12, 32, 48],
                    ['Tim', 'Tim', 'Jane', 'Steve'],
                ],
                names = ['Measurement number', 'Measurer']

            )


            expected = df.copy()

            # we expect the result to come back with pint names, not input
            # names
            def get_pint_value(in_str):
                return str(ureg.Quantity(1, in_str).units)

            units_level = [
                i for i, name in enumerate(df.columns.names) if name == 'unit'
            ][0]

            expected.columns = df.columns.set_levels(
                df.columns.levels[units_level].map(get_pint_value),
                level='unit'
            )


            result = df.pint.quantify(ureg, level=-1).pint.dequantify()

            pd.testing.assert_frame_equal(result, expected)


    class TestSeriesAccessors(object):
        @pytest.mark.parametrize('attr', [
            'debug_used',
            'default_format',
            'dimensionality',
            'dimensionless',
            'force_ndarray',
            'shape',
            'u',
            'unitless',
            'units',
        ])
        def test_series_scalar_property_accessors(self, data, attr):
            s = pd.Series(data)
            assert getattr(s.pint, attr) == getattr(data._data,attr)

        @pytest.mark.parametrize('attr', [
            'm',
            'magnitude',
            #'imag', # failing, not sure why
            #'real', # failing, not sure why
        ])
        def test_series_property_accessors(self, data, attr):
            s = pd.Series(data)
            assert all(getattr(s.pint, attr) == pd.Series(getattr(data._data,attr)))

        @pytest.mark.parametrize('attr_args', [
            ('check', ({"[length]": 1})),
            ('compatible_units', ()),
            # ('format_babel', ()), Needs babel installed?
            # ('plus_minus', ()), Needs uncertanties
            ('to_tuple', ()),
            ('tolist', ())
        ])
        def test_series_scalar_method_accessors(self, data, attr_args):
            attr = attr_args[0]
            args = attr_args[1]
            s = pd.Series(data)
            assert getattr(s.pint, attr)(*args) == getattr(data._data, attr)(*args)

        @pytest.mark.parametrize('attr_args', [
            ('ito', ("g",)),
            ('ito_base_units', ()),
            ('ito_reduced_units', ()),
            ('ito_root_units', ()),
            ('put', (1, get_tdata()[0]))
        ])
        def test_series_inplace_method_accessors(self, data, attr_args):
            attr = attr_args[0]
            args = attr_args[1]
            from copy import deepcopy
            s = pd.Series(deepcopy(data))
            getattr(s.pint, attr)(*args)
            getattr(data._data, attr)(*args)
            assert all(s.values == data)

        @pytest.mark.parametrize('attr_args', [
            ('clip', (get_tdata()[10], get_tdata()[20])),
            ('from_tuple', (get_tdata().data.to_tuple(),)),
            ('m_as', ("g",)),
            ('searchsorted', (get_tdata()[10],)),
            ('to', ("g")),
            ('to_base_units', ()),
            ('to_compact', ()),
            ('to_reduced_units', ()),
            ('to_root_units', ()),
            # ('to_timedelta', ()),
        ])
        def test_series_method_accessors(self, data, attr_args):
            attr=attr_args[0]
            args=attr_args[1]
            s = pd.Series(data)
            assert all(getattr(s.pint, attr)(*args) == getattr(data._data,attr)(*args))


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


        @pytest.mark.filterwarnings("ignore::pint.UnitStrippedWarning")
        @pytest.mark.filterwarnings("ignore::RuntimeWarning")
        def test_pintarray_operations(self):
            # Perform operations with Quantities and PintArrays
            # The resulting Quantity and PintArray.Data should be the same
            # a op b == c
            # warnings ignored here as it these tests are to ensure
            # pint array behaviour is the same as quantity
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
