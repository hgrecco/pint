from unittest.mock import patch

import pytest

import pint.facets.numpy.numpy_func
from pint import DimensionalityError, OffsetUnitCalculusError
from pint.compat import np
from pint.facets.numpy.numpy_func import (
    _is_quantity,
    _is_sequence_with_quantity_elements,
    convert_to_consistent_units,
    get_op_output_unit,
    implements,
    numpy_wrap,
    unwrap_and_wrap_consistent_units,
)
from pint.testsuite import helpers
from pint.testsuite.test_numpy import TestNumpyMethods


class TestNumPyFuncUtils(TestNumpyMethods):
    @patch("pint.facets.numpy.numpy_func.HANDLED_FUNCTIONS", {})
    @patch("pint.facets.numpy.numpy_func.HANDLED_UFUNCS", {})
    def test_implements(self):
        # Test for functions
        @implements("test", "function")
        def test_function():
            pass

        assert pint.facets.numpy.numpy_func.HANDLED_FUNCTIONS["test"] == test_function

        # Test for ufuncs
        @implements("test", "ufunc")
        def test_ufunc():
            pass

        assert pint.facets.numpy.numpy_func.HANDLED_UFUNCS["test"] == test_ufunc

        # Test for invalid func type
        with pytest.raises(ValueError):

            @implements("test", "invalid")
            def test_invalid():
                pass

    def test_is_quantity(self):
        assert _is_quantity(self.Q_(0))
        assert _is_quantity(np.arange(4) * self.ureg.m)
        assert not _is_quantity(1.0)
        assert not _is_quantity(np.array([1, 1, 2, 3, 5, 8]))
        assert not _is_quantity("not-a-quantity")
        # TODO (#905 follow-up): test other duck arrays that wrap or are wrapped by Pint

    def test_is_sequence_with_quantity_elements(self):
        assert _is_sequence_with_quantity_elements(
            (self.Q_(0, "m"), self.Q_(32.0, "degF"))
        )
        assert _is_sequence_with_quantity_elements(np.arange(4) * self.ureg.m)
        assert _is_sequence_with_quantity_elements((self.Q_(0), 0))
        assert _is_sequence_with_quantity_elements((0, self.Q_(0)))
        assert not _is_sequence_with_quantity_elements([1, 3, 5])
        assert not _is_sequence_with_quantity_elements(9 * self.ureg.m)
        assert not _is_sequence_with_quantity_elements(np.arange(4))
        assert not _is_sequence_with_quantity_elements("0123")
        assert not _is_sequence_with_quantity_elements([])
        assert not _is_sequence_with_quantity_elements(np.array([]))

    def test_convert_to_consistent_units_with_pre_calc_units(self):
        args, kwargs = convert_to_consistent_units(
            self.Q_(50, "cm"),
            np.arange(4).reshape(2, 2) * self.ureg.m,
            [0.042] * self.ureg.km,
            (self.Q_(0, "m"), self.Q_(1, "dam")),
            a=6378 * self.ureg.km,
            pre_calc_units=self.ureg.meter,
        )
        assert args[0] == 0.5
        self.assertNDArrayEqual(args[1], np.array([[0, 1], [2, 3]]))
        self.assertNDArrayEqual(args[2], np.array([42]))
        assert args[3][0] == 0
        assert args[3][1] == 10
        assert kwargs["a"] == 6.378e6

    def test_convert_to_consistent_units_with_dimensionless(self):
        args, kwargs = convert_to_consistent_units(
            np.arange(2), pre_calc_units=self.ureg.g / self.ureg.kg
        )
        self.assertNDArrayEqual(args[0], np.array([0, 1000]))
        assert kwargs == {}

    def test_convert_to_consistent_units_with_dimensionality_error(self):
        with pytest.raises(DimensionalityError):
            convert_to_consistent_units(
                self.Q_(32.0, "degF"),
                pre_calc_units=self.ureg.meter,
            )
        with pytest.raises(DimensionalityError):
            convert_to_consistent_units(
                np.arange(4),
                pre_calc_units=self.ureg.meter,
            )

    def test_convert_to_consistent_units_without_pre_calc_units(self):
        args, kwargs = convert_to_consistent_units(
            (self.Q_(0), self.Q_(10, "degC")),
            [1, 2, 3, 5, 7] * self.ureg.m,
            pre_calc_units=None,
        )
        assert args[0][0] == 0
        assert args[0][1] == 10
        self.assertNDArrayEqual(args[1], np.array([1, 2, 3, 5, 7]))
        assert kwargs == {}

    def test_unwrap_and_wrap_constistent_units(self):
        (a,), output_wrap_a = unwrap_and_wrap_consistent_units([2, 4, 8] * self.ureg.m)
        (b, c), output_wrap_c = unwrap_and_wrap_consistent_units(
            np.arange(4), self.Q_(1, "g/kg")
        )

        self.assertNDArrayEqual(a, np.array([2, 4, 8]))
        self.assertNDArrayEqual(b, np.array([0, 1000, 2000, 3000]))
        assert c == 1

        helpers.assert_quantity_equal(output_wrap_a(0), 0 * self.ureg.m)
        helpers.assert_quantity_equal(output_wrap_c(0), self.Q_(0, "g/kg"))

    def test_op_output_unit_sum(self):
        assert get_op_output_unit("sum", self.ureg.m) == self.ureg.m
        with pytest.raises(OffsetUnitCalculusError):
            get_op_output_unit("sum", self.ureg.degC)

    def test_op_output_unit_mul(self):
        assert (
            get_op_output_unit(
                "mul", self.ureg.s, (self.Q_(1, "m"), self.Q_(1, "m**2"))
            )
            == self.ureg.m**3
        )

    def test_op_output_unit_delta(self):
        assert get_op_output_unit("delta", self.ureg.m) == self.ureg.m
        assert get_op_output_unit("delta", self.ureg.degC) == self.ureg.delta_degC

    def test_op_output_unit_delta_div(self):
        assert (
            get_op_output_unit(
                "delta,div", self.ureg.m, (self.Q_(1, "m"), self.Q_(1, "s"))
            )
            == self.ureg.m / self.ureg.s
        )
        assert (
            get_op_output_unit(
                "delta,div", self.ureg.degC, (self.Q_(1, "degC"), self.Q_(1, "m"))
            )
            == self.ureg.delta_degC / self.ureg.m
        )

    def test_op_output_unit_div(self):
        assert (
            get_op_output_unit(
                "div", self.ureg.m, (self.Q_(1, "m"), self.Q_(1, "s"), self.Q_(1, "K"))
            )
            == self.ureg.m / self.ureg.s / self.ureg.K
        )
        assert (
            get_op_output_unit("div", self.ureg.s, (1, self.Q_(1, "s")))
            == self.ureg.s**-1
        )

    def test_op_output_unit_variance(self):
        assert get_op_output_unit("variance", self.ureg.m) == self.ureg.m**2
        with pytest.raises(OffsetUnitCalculusError):
            get_op_output_unit("variance", self.ureg.degC)

    def test_op_output_unit_square(self):
        assert get_op_output_unit("square", self.ureg.m) == self.ureg.m**2

    def test_op_output_unit_sqrt(self):
        assert get_op_output_unit("sqrt", self.ureg.m) == self.ureg.m**0.5

    def test_op_output_unit_reciprocal(self):
        assert get_op_output_unit("reciprocal", self.ureg.m) == self.ureg.m**-1

    def test_op_output_unit_size(self):
        assert get_op_output_unit("size", self.ureg.m, size=3) == self.ureg.m**3
        with pytest.raises(ValueError):
            get_op_output_unit("size", self.ureg.m)

    def test_numpy_wrap(self):
        with pytest.raises(ValueError):
            numpy_wrap("invalid", np.ones, [], {}, [])
        # TODO (#905 follow-up): test that NotImplemented is returned when upcast types
        # present
