import collections
import copy
import math
import operator as op

import pytest

from pint.util import (
    ParserHelper,
    UnitsContainer,
    find_connected_nodes,
    find_shortest_path,
    iterable,
    matrix_to_string,
    sized,
    string_preprocessor,
    to_units_container,
    tokenizer,
    transpose,
)


class TestUnitsContainer:
    def _test_inplace(self, operator, value1, value2, expected_result):
        value1 = copy.copy(value1)
        value2 = copy.copy(value2)
        id1 = id(value1)
        id2 = id(value2)
        value1 = operator(value1, value2)
        value2_cpy = copy.copy(value2)
        assert value1 == expected_result
        # Inplace operation creates copies
        assert id1 != id(value1)
        assert value2 == value2_cpy
        assert id2 == id(value2)

    def _test_not_inplace(self, operator, value1, value2, expected_result):
        id1 = id(value1)
        id2 = id(value2)

        value1_cpy = copy.copy(value1)
        value2_cpy = copy.copy(value2)

        result = operator(value1, value2)

        assert expected_result == result
        assert value1 == value1_cpy
        assert value2 == value2_cpy
        assert id(result) != id1
        assert id(result) != id2

    def test_unitcontainer_creation(self):
        x = UnitsContainer(meter=1, second=2)
        y = UnitsContainer({"meter": 1, "second": 2})
        assert isinstance(x["meter"], int)
        assert x == y
        assert x is not y
        z = copy.copy(x)
        assert x == z
        assert x is not z
        z = UnitsContainer(x)
        assert x == z
        assert x is not z

    def test_unitcontainer_repr(self):
        x = UnitsContainer()
        assert str(x) == "dimensionless"
        assert repr(x) == "<UnitsContainer({})>"
        x = UnitsContainer(meter=1, second=2)
        assert str(x) == "meter * second ** 2"
        assert repr(x) == "<UnitsContainer({'meter': 1, 'second': 2})>"
        x = UnitsContainer(meter=1, second=2.5)
        assert str(x) == "meter * second ** 2.5"
        assert repr(x) == "<UnitsContainer({'meter': 1, 'second': 2.5})>"

    def test_unitcontainer_bool(self):
        assert UnitsContainer(meter=1, second=2)
        assert not UnitsContainer()

    def test_unitcontainer_comp(self):
        x = UnitsContainer(meter=1, second=2)
        y = UnitsContainer(meter=1.0, second=2)
        z = UnitsContainer(meter=1, second=3)
        assert x == y
        assert not (x != y)
        assert not (x == z)
        assert x != z

    def test_unitcontainer_arithmetic(self):
        x = UnitsContainer(meter=1)
        y = UnitsContainer(second=1)
        z = UnitsContainer(meter=1, second=-2)

        self._test_not_inplace(op.mul, x, y, UnitsContainer(meter=1, second=1))
        self._test_not_inplace(op.truediv, x, y, UnitsContainer(meter=1, second=-1))
        self._test_not_inplace(op.pow, z, 2, UnitsContainer(meter=2, second=-4))
        self._test_not_inplace(op.pow, z, -2, UnitsContainer(meter=-2, second=4))

        self._test_inplace(op.imul, x, y, UnitsContainer(meter=1, second=1))
        self._test_inplace(op.itruediv, x, y, UnitsContainer(meter=1, second=-1))
        self._test_inplace(op.ipow, z, 2, UnitsContainer(meter=2, second=-4))
        self._test_inplace(op.ipow, z, -2, UnitsContainer(meter=-2, second=4))

    def test_string_comparison(self):
        x = UnitsContainer(meter=1)
        y = UnitsContainer(second=1)
        z = UnitsContainer(meter=1, second=-2)
        assert x == "meter"
        assert "meter" == x
        assert x != "meter ** 2"
        assert x != "meter * meter"
        assert x != "second"
        assert y == "second"
        assert z == "meter/second/second"

    def test_invalid(self):
        with pytest.raises(TypeError):
            UnitsContainer({1: 2})
        with pytest.raises(TypeError):
            UnitsContainer({"1": "2"})
        d = UnitsContainer()
        with pytest.raises(TypeError):
            d.__mul__(list())
        with pytest.raises(TypeError):
            d.__pow__(list())
        with pytest.raises(TypeError):
            d.__truediv__(list())
        with pytest.raises(TypeError):
            d.__rtruediv__(list())


class TestToUnitsContainer:
    def test_str_conversion(self):
        assert to_units_container("m") == UnitsContainer(m=1)

    def test_uc_conversion(self):
        a = UnitsContainer(m=1)
        assert to_units_container(a) is a

    def test_quantity_conversion(self):
        from pint.registry import UnitRegistry

        ureg = UnitRegistry()
        assert to_units_container(
            ureg.Quantity(1, UnitsContainer(m=1))
        ) == UnitsContainer(m=1)

    def test_unit_conversion(self):
        from pint import Unit

        assert to_units_container(Unit(UnitsContainer(m=1))) == UnitsContainer(m=1)

    def test_dict_conversion(self):
        assert to_units_container(dict(m=1)) == UnitsContainer(m=1)


class TestParseHelper:
    def test_basic(self):
        # Parse Helper ar mutables, so we build one everytime
        x = lambda: ParserHelper(1, meter=2)
        xp = lambda: ParserHelper(1, meter=2)
        y = lambda: ParserHelper(2, meter=2)

        assert x() == xp()
        assert x() != y()
        assert ParserHelper.from_string("") == ParserHelper()
        assert repr(x()) == "<ParserHelper(1, {'meter': 2})>"

        assert ParserHelper(2) == 2

        assert x() == dict(meter=2)
        assert x() == "meter ** 2"
        assert y() != dict(meter=2)
        assert y() != "meter ** 2"

        assert xp() != object()

    def test_calculate(self):
        # Parse Helper ar mutables, so we build one everytime
        x = lambda: ParserHelper(1.0, meter=2)
        y = lambda: ParserHelper(2.0, meter=-2)
        z = lambda: ParserHelper(2.0, meter=2)

        assert x() * 4.0 == ParserHelper(4.0, meter=2)
        assert x() * y() == ParserHelper(2.0)
        assert x() * "second" == ParserHelper(1.0, meter=2, second=1)

        assert x() / 4.0 == ParserHelper(0.25, meter=2)
        assert x() / "second" == ParserHelper(1.0, meter=2, second=-1)
        assert x() / z() == ParserHelper(0.5)

        assert 4.0 / z() == ParserHelper(2.0, meter=-2)
        assert "seconds" / z() == ParserHelper(0.5, seconds=1, meter=-2)
        assert dict(seconds=1) / z() == ParserHelper(0.5, seconds=1, meter=-2)

    def _test_eval_token(self, expected, expression, use_decimal=False):
        token = next(tokenizer(expression))
        actual = ParserHelper.eval_token(token, use_decimal=use_decimal)
        assert expected == actual
        assert type(expected) == type(actual)

    def test_eval_token(self):
        self._test_eval_token(1000.0, "1e3")
        self._test_eval_token(1000.0, "1E3")
        self._test_eval_token(1000, "1000")

    def test_nan(self, subtests):
        for s in ("nan", "NAN", "NaN", "123 NaN nan NAN 456"):
            with subtests.test(s):
                p = ParserHelper.from_string(s + " kg")
                assert math.isnan(p.scale)
                assert dict(p) == {"kg": 1}


class TestStringProcessor:
    def _test(self, bef, aft):
        for pattern in ("{}", "+{}+"):
            b = pattern.format(bef)
            a = pattern.format(aft)
            assert string_preprocessor(b) == a

    def test_square_cube(self):
        self._test("bcd^3", "bcd**3")
        self._test("bcd^ 3", "bcd** 3")
        self._test("bcd ^3", "bcd **3")
        self._test("bcd squared", "bcd**2")
        self._test("bcd squared", "bcd**2")
        self._test("bcd cubed", "bcd**3")
        self._test("sq bcd", "bcd**2")
        self._test("square bcd", "bcd**2")
        self._test("cubic bcd", "bcd**3")
        self._test("bcd efg", "bcd*efg")

    def test_per(self):
        self._test("miles per hour", "miles/hour")

    def test_numbers(self):
        self._test("1,234,567", "1234567")
        self._test("1e-24", "1e-24")
        self._test("1e+24", "1e+24")
        self._test("1e24", "1e24")
        self._test("1E-24", "1E-24")
        self._test("1E+24", "1E+24")
        self._test("1E24", "1E24")

    def test_space_multiplication(self):
        self._test("bcd efg", "bcd*efg")
        self._test("bcd  efg", "bcd*efg")
        self._test("1 hour", "1*hour")
        self._test("1. hour", "1.*hour")
        self._test("1.1 hour", "1.1*hour")
        self._test("1E24 hour", "1E24*hour")
        self._test("1E-24 hour", "1E-24*hour")
        self._test("1E+24 hour", "1E+24*hour")
        self._test("1.2E24 hour", "1.2E24*hour")
        self._test("1.2E-24 hour", "1.2E-24*hour")
        self._test("1.2E+24 hour", "1.2E+24*hour")

    def test_joined_multiplication(self):
        self._test("1hour", "1*hour")
        self._test("1.hour", "1.*hour")
        self._test("1.1hour", "1.1*hour")
        self._test("1h", "1*h")
        self._test("1.h", "1.*h")
        self._test("1.1h", "1.1*h")

    def test_names(self):
        self._test("g_0", "g_0")
        self._test("g0", "g0")
        self._test("g", "g")
        self._test("water_60F", "water_60F")


class TestGraph:
    def test_start_not_in_graph(self):
        g = collections.defaultdict(set)
        g[1] = {2}
        g[2] = {3}
        assert find_connected_nodes(g, 9) is None

    def test_shortest_path(self):
        g = collections.defaultdict(set)
        g[1] = {2}
        g[2] = {3}
        p = find_shortest_path(g, 1, 2)
        assert p == [1, 2]
        p = find_shortest_path(g, 1, 3)
        assert p == [1, 2, 3]
        p = find_shortest_path(g, 3, 1)
        assert p is None

        g = collections.defaultdict(set)
        g[1] = {2}
        g[2] = {3, 1}
        g[3] = {2}
        p = find_shortest_path(g, 1, 2)
        assert p == [1, 2]
        p = find_shortest_path(g, 1, 3)
        assert p == [1, 2, 3]
        p = find_shortest_path(g, 3, 1)
        assert p == [3, 2, 1]
        p = find_shortest_path(g, 2, 1)
        assert p == [2, 1]


class TestMatrix:
    def test_matrix_to_string(self):
        assert (
            matrix_to_string([[1, 2], [3, 4]], row_headers=None, col_headers=None)
            == "1\t2\n"
            "3\t4"
        )

        assert (
            matrix_to_string(
                [[1, 2], [3, 4]],
                row_headers=None,
                col_headers=None,
                fmtfun=lambda x: f"{x:.2f}",
            )
            == "1.00\t2.00\n"
            "3.00\t4.00"
        )

        assert (
            matrix_to_string([[1, 2], [3, 4]], row_headers=["c", "d"], col_headers=None)
            == "c\t1\t2\n"
            "d\t3\t4"
        )

        assert (
            matrix_to_string([[1, 2], [3, 4]], row_headers=None, col_headers=["a", "b"])
            == "a\tb\n"
            "1\t2\n"
            "3\t4"
        )

        assert (
            matrix_to_string(
                [[1, 2], [3, 4]], row_headers=["c", "d"], col_headers=["a", "b"]
            )
            == "\ta\tb\n"
            "c\t1\t2\n"
            "d\t3\t4"
        )

    def test_transpose(self):
        assert transpose([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]


class TestOtherUtils:
    def test_iterable(self):
        # Test with list, string, generator, and scalar
        assert iterable([0, 1, 2, 3])
        assert iterable("test")
        assert iterable((i for i in range(5)))
        assert not iterable(0)

    def test_sized(self):
        # Test with list, string, generator, and scalar
        assert sized([0, 1, 2, 3])
        assert sized("test")
        assert not sized((i for i in range(5)))
        assert not sized(0)
