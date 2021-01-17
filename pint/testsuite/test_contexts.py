import itertools
import logging
import math
import re
from collections import defaultdict

import pytest

from pint import (
    DefinitionSyntaxError,
    DimensionalityError,
    UndefinedUnitError,
    UnitRegistry,
)
from pint.context import Context
from pint.testsuite import helpers
from pint.util import UnitsContainer


def add_ctxs(ureg):
    a, b = UnitsContainer({"[length]": 1}), UnitsContainer({"[time]": -1})
    d = Context("lc")
    d.add_transformation(a, b, lambda ureg, x: ureg.speed_of_light / x)
    d.add_transformation(b, a, lambda ureg, x: ureg.speed_of_light / x)

    ureg.add_context(d)

    a, b = UnitsContainer({"[length]": 1}), UnitsContainer({"[current]": 1})
    d = Context("ab")
    d.add_transformation(a, b, lambda ureg, x: ureg.ampere * ureg.meter / x)
    d.add_transformation(b, a, lambda ureg, x: ureg.ampere * ureg.meter / x)

    ureg.add_context(d)


def add_arg_ctxs(ureg):
    a, b = UnitsContainer({"[length]": 1}), UnitsContainer({"[time]": -1})
    d = Context("lc")
    d.add_transformation(a, b, lambda ureg, x, n: ureg.speed_of_light / x / n)
    d.add_transformation(b, a, lambda ureg, x, n: ureg.speed_of_light / x / n)

    ureg.add_context(d)

    a, b = UnitsContainer({"[length]": 1}), UnitsContainer({"[current]": 1})
    d = Context("ab")
    d.add_transformation(a, b, lambda ureg, x: ureg.ampere * ureg.meter / x)
    d.add_transformation(b, a, lambda ureg, x: ureg.ampere * ureg.meter / x)

    ureg.add_context(d)


def add_argdef_ctxs(ureg):
    a, b = UnitsContainer({"[length]": 1}), UnitsContainer({"[time]": -1})
    d = Context("lc", defaults=dict(n=1))
    assert d.defaults == dict(n=1)

    d.add_transformation(a, b, lambda ureg, x, n: ureg.speed_of_light / x / n)
    d.add_transformation(b, a, lambda ureg, x, n: ureg.speed_of_light / x / n)

    ureg.add_context(d)

    a, b = UnitsContainer({"[length]": 1}), UnitsContainer({"[current]": 1})
    d = Context("ab")
    d.add_transformation(a, b, lambda ureg, x: ureg.ampere * ureg.meter / x)
    d.add_transformation(b, a, lambda ureg, x: ureg.ampere * ureg.meter / x)

    ureg.add_context(d)


def add_sharedargdef_ctxs(ureg):
    a, b = UnitsContainer({"[length]": 1}), UnitsContainer({"[time]": -1})
    d = Context("lc", defaults=dict(n=1))
    assert d.defaults == dict(n=1)

    d.add_transformation(a, b, lambda ureg, x, n: ureg.speed_of_light / x / n)
    d.add_transformation(b, a, lambda ureg, x, n: ureg.speed_of_light / x / n)

    ureg.add_context(d)

    a, b = UnitsContainer({"[length]": 1}), UnitsContainer({"[current]": 1})
    d = Context("ab", defaults=dict(n=0))
    d.add_transformation(a, b, lambda ureg, x, n: ureg.ampere * ureg.meter * n / x)
    d.add_transformation(b, a, lambda ureg, x, n: ureg.ampere * ureg.meter * n / x)

    ureg.add_context(d)


class TestContexts:
    def test_known_context(self, func_registry):
        ureg = UnitRegistry()
        add_ctxs(ureg)
        with ureg.context("lc"):
            assert ureg._active_ctx
            assert ureg._active_ctx.graph

        assert not ureg._active_ctx
        assert not ureg._active_ctx.graph

        with ureg.context("lc", n=1):
            assert ureg._active_ctx
            assert ureg._active_ctx.graph

        assert not ureg._active_ctx
        assert not ureg._active_ctx.graph

    def test_known_context_enable(self, func_registry):
        ureg = UnitRegistry()
        add_ctxs(ureg)
        ureg.enable_contexts("lc")
        assert ureg._active_ctx
        assert ureg._active_ctx.graph
        ureg.disable_contexts(1)

        assert not ureg._active_ctx
        assert not ureg._active_ctx.graph

        ureg.enable_contexts("lc", n=1)
        assert ureg._active_ctx
        assert ureg._active_ctx.graph
        ureg.disable_contexts(1)

        assert not ureg._active_ctx
        assert not ureg._active_ctx.graph

    def test_graph(self, func_registry):
        ureg = UnitRegistry()
        add_ctxs(ureg)
        l = UnitsContainer({"[length]": 1.0})  # noqa: E741
        t = UnitsContainer({"[time]": -1.0})
        c = UnitsContainer({"[current]": 1.0})

        g_sp = defaultdict(set)
        g_sp.update({l: {t}, t: {l}})

        g_ab = defaultdict(set)
        g_ab.update({l: {c}, c: {l}})

        g = defaultdict(set)
        g.update({l: {t, c}, t: {l}, c: {l}})

        with ureg.context("lc"):
            assert ureg._active_ctx.graph == g_sp

        with ureg.context("lc", n=1):
            assert ureg._active_ctx.graph == g_sp

        with ureg.context("ab"):
            assert ureg._active_ctx.graph == g_ab

        with ureg.context("lc"):
            with ureg.context("ab"):
                assert ureg._active_ctx.graph == g

        with ureg.context("ab"):
            with ureg.context("lc"):
                assert ureg._active_ctx.graph == g

        with ureg.context("lc", "ab"):
            assert ureg._active_ctx.graph == g

        with ureg.context("ab", "lc"):
            assert ureg._active_ctx.graph == g

    def test_graph_enable(self, func_registry):
        ureg = UnitRegistry()
        add_ctxs(ureg)
        l = UnitsContainer({"[length]": 1.0})  # noqa: E741
        t = UnitsContainer({"[time]": -1.0})
        c = UnitsContainer({"[current]": 1.0})

        g_sp = defaultdict(set)
        g_sp.update({l: {t}, t: {l}})

        g_ab = defaultdict(set)
        g_ab.update({l: {c}, c: {l}})

        g = defaultdict(set)
        g.update({l: {t, c}, t: {l}, c: {l}})

        ureg.enable_contexts("lc")
        assert ureg._active_ctx.graph == g_sp
        ureg.disable_contexts(1)

        ureg.enable_contexts("lc", n=1)
        assert ureg._active_ctx.graph == g_sp
        ureg.disable_contexts(1)

        ureg.enable_contexts("ab")
        assert ureg._active_ctx.graph == g_ab
        ureg.disable_contexts(1)

        ureg.enable_contexts("lc")
        ureg.enable_contexts("ab")
        assert ureg._active_ctx.graph == g
        ureg.disable_contexts(2)

        ureg.enable_contexts("ab")
        ureg.enable_contexts("lc")
        assert ureg._active_ctx.graph == g
        ureg.disable_contexts(2)

        ureg.enable_contexts("lc", "ab")
        assert ureg._active_ctx.graph == g
        ureg.disable_contexts(2)

        ureg.enable_contexts("ab", "lc")
        assert ureg._active_ctx.graph == g
        ureg.disable_contexts(2)

    def test_known_nested_context(self, func_registry):
        ureg = UnitRegistry()
        add_ctxs(ureg)

        with ureg.context("lc"):
            x = dict(ureg._active_ctx)
            y = dict(ureg._active_ctx.graph)
            assert ureg._active_ctx
            assert ureg._active_ctx.graph

            with ureg.context("ab"):
                assert ureg._active_ctx
                assert ureg._active_ctx.graph
                assert x != ureg._active_ctx
                assert y != ureg._active_ctx.graph

            assert x == ureg._active_ctx
            assert y == ureg._active_ctx.graph

        assert not ureg._active_ctx
        assert not ureg._active_ctx.graph

    def test_unknown_context(self, func_registry):
        ureg = func_registry
        add_ctxs(ureg)
        with pytest.raises(KeyError):
            with ureg.context("la"):
                pass
        assert not ureg._active_ctx
        assert not ureg._active_ctx.graph

    def test_unknown_nested_context(self, func_registry):
        ureg = UnitRegistry()
        add_ctxs(ureg)

        with ureg.context("lc"):
            x = dict(ureg._active_ctx)
            y = dict(ureg._active_ctx.graph)
            with pytest.raises(KeyError):
                with ureg.context("la"):
                    pass

            assert x == ureg._active_ctx
            assert y == ureg._active_ctx.graph

        assert not ureg._active_ctx
        assert not ureg._active_ctx.graph

    def test_one_context(self, func_registry):
        ureg = UnitRegistry()

        add_ctxs(ureg)

        q = 500 * ureg.meter
        s = (ureg.speed_of_light / q).to("Hz")

        meter_units = ureg.get_compatible_units(ureg.meter)
        hertz_units = ureg.get_compatible_units(ureg.hertz)

        with pytest.raises(DimensionalityError):
            q.to("Hz")
        with ureg.context("lc"):
            assert q.to("Hz") == s
            assert ureg.get_compatible_units(q) == meter_units | hertz_units
        with pytest.raises(DimensionalityError):
            q.to("Hz")
        assert ureg.get_compatible_units(q) == meter_units

    def test_multiple_context(self, func_registry):
        ureg = UnitRegistry()

        add_ctxs(ureg)

        q = 500 * ureg.meter
        s = (ureg.speed_of_light / q).to("Hz")

        meter_units = ureg.get_compatible_units(ureg.meter)
        hertz_units = ureg.get_compatible_units(ureg.hertz)
        ampere_units = ureg.get_compatible_units(ureg.ampere)

        with pytest.raises(DimensionalityError):
            q.to("Hz")
        with ureg.context("lc", "ab"):
            assert q.to("Hz") == s
            assert (
                ureg.get_compatible_units(q) == meter_units | hertz_units | ampere_units
            )
        with pytest.raises(DimensionalityError):
            q.to("Hz")
        assert ureg.get_compatible_units(q) == meter_units

    def test_nested_context(self, func_registry):
        ureg = UnitRegistry()

        add_ctxs(ureg)

        q = 500 * ureg.meter
        s = (ureg.speed_of_light / q).to("Hz")

        with pytest.raises(DimensionalityError):
            q.to("Hz")
        with ureg.context("lc"):
            assert q.to("Hz") == s
            with ureg.context("ab"):
                assert q.to("Hz") == s
            assert q.to("Hz") == s

        with ureg.context("ab"):
            with pytest.raises(DimensionalityError):
                q.to("Hz")
            with ureg.context("lc"):
                assert q.to("Hz") == s
            with pytest.raises(DimensionalityError):
                q.to("Hz")

    def test_context_with_arg(self, func_registry):

        ureg = UnitRegistry()

        add_arg_ctxs(ureg)

        q = 500 * ureg.meter
        s = (ureg.speed_of_light / q).to("Hz")

        with pytest.raises(DimensionalityError):
            q.to("Hz")
        with ureg.context("lc", n=1):
            assert q.to("Hz") == s
            with ureg.context("ab"):
                assert q.to("Hz") == s
            assert q.to("Hz") == s

        with ureg.context("ab"):
            with pytest.raises(DimensionalityError):
                q.to("Hz")
            with ureg.context("lc", n=1):
                assert q.to("Hz") == s
            with pytest.raises(DimensionalityError):
                q.to("Hz")

        with ureg.context("lc"):
            with pytest.raises(TypeError):
                q.to("Hz")

    def test_enable_context_with_arg(self, func_registry):

        ureg = UnitRegistry()

        add_arg_ctxs(ureg)

        q = 500 * ureg.meter
        s = (ureg.speed_of_light / q).to("Hz")

        with pytest.raises(DimensionalityError):
            q.to("Hz")
        ureg.enable_contexts("lc", n=1)
        assert q.to("Hz") == s
        ureg.enable_contexts("ab")
        assert q.to("Hz") == s
        assert q.to("Hz") == s
        ureg.disable_contexts(1)
        ureg.disable_contexts(1)

        ureg.enable_contexts("ab")
        with pytest.raises(DimensionalityError):
            q.to("Hz")
        ureg.enable_contexts("lc", n=1)
        assert q.to("Hz") == s
        ureg.disable_contexts(1)
        with pytest.raises(DimensionalityError):
            q.to("Hz")
        ureg.disable_contexts(1)

        ureg.enable_contexts("lc")
        with pytest.raises(TypeError):
            q.to("Hz")
        ureg.disable_contexts(1)

    def test_context_with_arg_def(self, func_registry):

        ureg = UnitRegistry()

        add_argdef_ctxs(ureg)

        q = 500 * ureg.meter
        s = (ureg.speed_of_light / q).to("Hz")

        with pytest.raises(DimensionalityError):
            q.to("Hz")
        with ureg.context("lc"):
            assert q.to("Hz") == s
            with ureg.context("ab"):
                assert q.to("Hz") == s
            assert q.to("Hz") == s

        with ureg.context("ab"):
            with pytest.raises(DimensionalityError):
                q.to("Hz")
            with ureg.context("lc"):
                assert q.to("Hz") == s
            with pytest.raises(DimensionalityError):
                q.to("Hz")

        with pytest.raises(DimensionalityError):
            q.to("Hz")
        with ureg.context("lc", n=2):
            assert q.to("Hz") == s / 2
            with ureg.context("ab"):
                assert q.to("Hz") == s / 2
            assert q.to("Hz") == s / 2

        with ureg.context("ab"):
            with pytest.raises(DimensionalityError):
                q.to("Hz")
            with ureg.context("lc", n=2):
                assert q.to("Hz") == s / 2
            with pytest.raises(DimensionalityError):
                q.to("Hz")

    def test_context_with_sharedarg_def(self, func_registry):

        ureg = UnitRegistry()

        add_sharedargdef_ctxs(ureg)

        q = 500 * ureg.meter
        s = (ureg.speed_of_light / q).to("Hz")
        u = (1 / 500) * ureg.ampere

        with ureg.context("lc"):
            assert q.to("Hz") == s
            with ureg.context("ab"):
                assert q.to("ampere") == u

        with ureg.context("ab"):
            assert q.to("ampere") == 0 * u
            with ureg.context("lc"):
                with pytest.raises(ZeroDivisionError):
                    ureg.Quantity.to(q, "Hz")

        with ureg.context("lc", n=2):
            assert q.to("Hz") == s / 2
            with ureg.context("ab"):
                assert q.to("ampere") == 2 * u

        with ureg.context("ab", n=3):
            assert q.to("ampere") == 3 * u
            with ureg.context("lc"):
                assert q.to("Hz") == s / 3

        with ureg.context("lc", n=2):
            assert q.to("Hz") == s / 2
            with ureg.context("ab", n=4):
                assert q.to("ampere") == 4 * u

        with ureg.context("ab", n=3):
            assert q.to("ampere") == 3 * u
            with ureg.context("lc", n=6):
                assert q.to("Hz") == s / 6

    def test_anonymous_context(self, func_registry):
        ureg = UnitRegistry()
        c = Context()
        c.add_transformation("[length]", "[time]", lambda ureg, x: x / ureg("5 cm/s"))
        with pytest.raises(ValueError):
            ureg.add_context(c)

        x = ureg("10 cm")
        expect = ureg("2 s")
        helpers.assert_quantity_equal(x.to("s", c), expect)

        with ureg.context(c):
            helpers.assert_quantity_equal(x.to("s"), expect)

        ureg.enable_contexts(c)
        helpers.assert_quantity_equal(x.to("s"), expect)
        ureg.disable_contexts(1)
        with pytest.raises(DimensionalityError):
            x.to("s")

        # Multiple anonymous contexts
        c2 = Context()
        c2.add_transformation("[length]", "[time]", lambda ureg, x: x / ureg("10 cm/s"))
        c2.add_transformation("[mass]", "[time]", lambda ureg, x: x / ureg("10 kg/s"))
        with ureg.context(c2, c):
            helpers.assert_quantity_equal(x.to("s"), expect)
            # Transformations only in c2 are still working even if c takes priority
            helpers.assert_quantity_equal(ureg("100 kg").to("s"), ureg("10 s"))
        with ureg.context(c, c2):
            helpers.assert_quantity_equal(x.to("s"), ureg("1 s"))

    def _test_ctx(self, ctx):
        ureg = UnitRegistry()
        q = 500 * ureg.meter
        s = (ureg.speed_of_light / q).to("Hz")

        nctx = len(ureg._contexts)

        assert ctx.name not in ureg._contexts
        ureg.add_context(ctx)

        assert ctx.name in ureg._contexts
        assert len(ureg._contexts) == nctx + 1 + len(ctx.aliases)

        with ureg.context(ctx.name):
            assert q.to("Hz") == s
            assert s.to("meter") == q

        ureg.remove_context(ctx.name)
        assert ctx.name not in ureg._contexts
        assert len(ureg._contexts) == nctx

    @pytest.mark.parametrize(
        "badrow",
        (
            "[length] = 1 / [time]: c / value",
            "1 / [time] = [length]: c / value",
            "[length] <- [time] = c / value",
            "[length] - [time] = c / value",
        ),
    )
    def test_parse_invalid(self, badrow):
        with pytest.raises(DefinitionSyntaxError):
            Context.from_lines(["@context c", badrow])

    def test_parse_simple(self):

        a = Context.__keytransform__(
            UnitsContainer({"[time]": -1}), UnitsContainer({"[length]": 1})
        )
        b = Context.__keytransform__(
            UnitsContainer({"[length]": 1}), UnitsContainer({"[time]": -1})
        )

        s = [
            "@context longcontextname",
            "[length] -> 1 / [time]: c / value",
            "1 / [time] -> [length]: c / value",
        ]

        c = Context.from_lines(s)
        assert c.name == "longcontextname"
        assert c.aliases == ()
        assert c.defaults == {}
        assert c.funcs.keys() == {a, b}
        self._test_ctx(c)

        s = ["@context longcontextname = lc", "[length] <-> 1 / [time]: c / value"]

        c = Context.from_lines(s)
        assert c.name == "longcontextname"
        assert c.aliases == ("lc",)
        assert c.defaults == {}
        assert c.funcs.keys() == {a, b}
        self._test_ctx(c)

        s = [
            "@context longcontextname = lc = lcn",
            "[length] <-> 1 / [time]: c / value",
        ]

        c = Context.from_lines(s)
        assert c.name == "longcontextname"
        assert c.aliases == ("lc", "lcn")
        assert c.defaults == {}
        assert c.funcs.keys() == {a, b}
        self._test_ctx(c)

    def test_parse_auto_inverse(self):

        a = Context.__keytransform__(
            UnitsContainer({"[time]": -1.0}), UnitsContainer({"[length]": 1.0})
        )
        b = Context.__keytransform__(
            UnitsContainer({"[length]": 1.0}), UnitsContainer({"[time]": -1.0})
        )

        s = ["@context longcontextname", "[length] <-> 1 / [time]: c / value"]

        c = Context.from_lines(s)
        assert c.defaults == {}
        assert c.funcs.keys() == {a, b}
        self._test_ctx(c)

    def test_parse_define(self):
        a = Context.__keytransform__(
            UnitsContainer({"[time]": -1}), UnitsContainer({"[length]": 1.0})
        )
        b = Context.__keytransform__(
            UnitsContainer({"[length]": 1}), UnitsContainer({"[time]": -1.0})
        )

        s = ["@context longcontextname", "[length] <-> 1 / [time]: c / value"]
        c = Context.from_lines(s)
        assert c.defaults == {}
        assert c.funcs.keys() == {a, b}
        self._test_ctx(c)

    def test_parse_parameterized(self):
        a = Context.__keytransform__(
            UnitsContainer({"[time]": -1.0}), UnitsContainer({"[length]": 1.0})
        )
        b = Context.__keytransform__(
            UnitsContainer({"[length]": 1.0}), UnitsContainer({"[time]": -1.0})
        )

        s = ["@context(n=1) longcontextname", "[length] <-> 1 / [time]: n * c / value"]

        c = Context.from_lines(s)
        assert c.defaults == {"n": 1}
        assert c.funcs.keys() == {a, b}
        self._test_ctx(c)

        s = [
            "@context(n=1, bla=2) longcontextname",
            "[length] <-> 1 / [time]: n * c / value / bla",
        ]

        c = Context.from_lines(s)
        assert c.defaults == {"n": 1, "bla": 2}
        assert c.funcs.keys() == {a, b}

        # If the variable is not present in the definition, then raise an error
        s = ["@context(n=1) longcontextname", "[length] <-> 1 / [time]: c / value"]
        with pytest.raises(DefinitionSyntaxError):
            Context.from_lines(s)

    def test_warnings(self, caplog):

        ureg = UnitRegistry()

        with caplog.at_level(logging.DEBUG, "pint"):
            add_ctxs(ureg)

            d = Context("ab")
            ureg.add_context(d)

            assert len(caplog.records) == 1
            assert "ab" in str(caplog.records[-1].args)

            d = Context("ab1", aliases=("ab",))
            ureg.add_context(d)

            assert len(caplog.records) == 2
            assert "ab" in str(caplog.records[-1].args)


class TestDefinedContexts:
    @classmethod
    def setup_class(cls):
        cls.ureg = UnitRegistry()

    @classmethod
    def teardown_class(cls):
        cls.ureg = None

    def test_defined(self):
        ureg = self.ureg
        with ureg.context("sp"):
            pass

        a = Context.__keytransform__(
            UnitsContainer({"[time]": -1.0}), UnitsContainer({"[length]": 1.0})
        )
        b = Context.__keytransform__(
            UnitsContainer({"[length]": 1.0}), UnitsContainer({"[time]": -1.0})
        )
        assert a in ureg._contexts["sp"].funcs
        assert b in ureg._contexts["sp"].funcs
        with ureg.context("sp"):
            assert a in ureg._active_ctx
            assert b in ureg._active_ctx

    def test_spectroscopy(self):
        ureg = self.ureg
        eq = (532.0 * ureg.nm, 563.5 * ureg.terahertz, 2.33053 * ureg.eV)
        with ureg.context("sp"):
            from pint.util import find_shortest_path

            for a, b in itertools.product(eq, eq):
                for x in range(2):
                    if x == 1:
                        a = a.to_base_units()
                        b = b.to_base_units()
                    da, db = Context.__keytransform__(
                        a.dimensionality, b.dimensionality
                    )
                    p = find_shortest_path(ureg._active_ctx.graph, da, db)
                    assert p
                    msg = "{} <-> {}".format(a, b)
                    # assertAlmostEqualRelError converts second to first
                    helpers.assert_quantity_almost_equal(b, a, rtol=0.01, msg=msg)

        for a, b in itertools.product(eq, eq):
            helpers.assert_quantity_almost_equal(a.to(b.units, "sp"), b, rtol=0.01)

    def test_textile(self):
        ureg = self.ureg
        qty_direct = 1.331 * ureg.tex
        with pytest.raises(DimensionalityError):
            qty_indirect = qty_direct.to("Nm")

        with ureg.context("textile"):
            from pint.util import find_shortest_path

            qty_indirect = qty_direct.to("Nm")
            a = qty_direct.to_base_units()
            b = qty_indirect.to_base_units()
            da, db = Context.__keytransform__(a.dimensionality, b.dimensionality)
            p = find_shortest_path(ureg._active_ctx.graph, da, db)
            assert p
            msg = "{} <-> {}".format(a, b)
            helpers.assert_quantity_almost_equal(b, a, rtol=0.01, msg=msg)

            # Check RKM <-> cN/tex conversion
            helpers.assert_quantity_almost_equal(
                1 * ureg.RKM, 0.980665 * ureg.cN / ureg.tex
            )
            helpers.assert_quantity_almost_equal(
                (1 / 0.980665) * ureg.RKM, 1 * ureg.cN / ureg.tex
            )
            assert (
                round(abs((1 * ureg.RKM).to(ureg.cN / ureg.tex).m - 0.980665), 7) == 0
            )
            assert (
                round(abs((1 * ureg.cN / ureg.tex).to(ureg.RKM).m - 1 / 0.980665), 7)
                == 0
            )

    def test_decorator(self):
        ureg = self.ureg

        a = 532.0 * ureg.nm
        with ureg.context("sp"):
            b = a.to("terahertz")

        def f(wl):
            return wl.to("terahertz")

        with pytest.raises(DimensionalityError):
            f(a)

        @ureg.with_context("sp")
        def g(wl):
            return wl.to("terahertz")

        assert b == g(a)

    def test_decorator_composition(self):
        ureg = self.ureg

        a = 532.0 * ureg.nm
        with ureg.context("sp"):
            b = a.to("terahertz")

        @ureg.with_context("sp")
        @ureg.check("[length]")
        def f(wl):
            return wl.to("terahertz")

        @ureg.with_context("sp")
        @ureg.check("[length]")
        def g(wl):
            return wl.to("terahertz")

        assert b == f(a)
        assert b == g(a)


def test_redefine(subtests):
    ureg = UnitRegistry(
        """
        foo = [d] = f = foo_alias
        bar = 2 foo = b = bar_alias
        baz = 3 bar = _ = baz_alias
        asd = 4 baz

        @context c
            # Note how we're redefining a symbol, not the base name, as a
            # function of another name
            b = 5 f
        """.splitlines()
    )
    # Units that are somehow directly or indirectly defined as a function of the
    # overridden unit are also affected
    foo = ureg.Quantity(1, "foo")
    bar = ureg.Quantity(1, "bar")
    asd = ureg.Quantity(1, "asd")

    # Test without context before and after, to verify that the cache and units have
    # not been polluted
    for enable_ctx in (False, True, False):
        with subtests.test(enable_ctx):
            if enable_ctx:
                ureg.enable_contexts("c")
                k = 5
            else:
                k = 2

            assert foo.to("b").magnitude == 1 / k
            assert foo.to("bar").magnitude == 1 / k
            assert foo.to("bar_alias").magnitude == 1 / k
            assert foo.to("baz").magnitude == 1 / k / 3
            assert bar.to("foo").magnitude == k
            assert bar.to("baz").magnitude == 1 / 3
            assert asd.to("foo").magnitude == 4 * 3 * k
            assert asd.to("bar").magnitude == 4 * 3
            assert asd.to("baz").magnitude == 4

        ureg.disable_contexts()


def test_define_nan():
    ureg = UnitRegistry(
        """
        USD = [currency]
        EUR = nan USD
        GBP = nan USD

        @context c
            EUR = 1.11 USD
            # Note that we're changing which unit GBP is defined against
            GBP = 1.18 EUR
        @end
        """.splitlines()
    )

    q = ureg.Quantity("10 GBP")
    assert q.magnitude == 10
    assert q.units.dimensionality == {"[currency]": 1}
    assert q.to("GBP").magnitude == 10
    assert math.isnan(q.to("USD").magnitude)
    assert math.isclose(q.to("USD", "c").magnitude, 10 * 1.18 * 1.11)


def test_non_multiplicative(subtests):
    ureg = UnitRegistry(
        """
        kelvin = [temperature]
        fahrenheit = 5 / 9 * kelvin; offset: 255
        bogodegrees = 9 * kelvin

        @context nonmult_to_nonmult
            fahrenheit = 7 * kelvin; offset: 123
        @end
        @context nonmult_to_mult
            fahrenheit = 123 * kelvin
        @end
        @context mult_to_nonmult
            bogodegrees = 5 * kelvin; offset: 123
        @end
        """.splitlines()
    )
    k = ureg.Quantity(100, "kelvin")

    with subtests.test("baseline"):
        helpers.assert_quantity_almost_equal(
            k.to("fahrenheit").magnitude, (100 - 255) * 9 / 5
        )
        helpers.assert_quantity_almost_equal(k.to("bogodegrees").magnitude, 100 / 9)

    with subtests.test("nonmult_to_nonmult"):
        with ureg.context("nonmult_to_nonmult"):
            helpers.assert_quantity_almost_equal(
                k.to("fahrenheit").magnitude, (100 - 123) / 7
            )

    with subtests.test("nonmult_to_mult"):
        with ureg.context("nonmult_to_mult"):
            helpers.assert_quantity_almost_equal(
                k.to("fahrenheit").magnitude, 100 / 123
            )

    with subtests.test("mult_to_nonmult"):
        with ureg.context("mult_to_nonmult"):
            helpers.assert_quantity_almost_equal(
                k.to("bogodegrees").magnitude, (100 - 123) / 5
            )


def test_stack_contexts():
    ureg = UnitRegistry(
        """
        a = [dim1]
        b = 1/2 a
        c = 1/3 a
        d = [dim2]

        @context c1
            b = 1/4 a
            c = 1/6 a
            [dim1]->[dim2]: value * 2 d/a
        @end
        @context c2
            b = 1/5 a
            [dim1]->[dim2]: value * 3 d/a
        @end
        """.splitlines()
    )
    q = ureg.Quantity(1, "a")
    assert q.to("b").magnitude == 2
    assert q.to("c").magnitude == 3
    assert q.to("b", "c1").magnitude == 4
    assert q.to("c", "c1").magnitude == 6
    assert q.to("d", "c1").magnitude == 2
    assert q.to("b", "c2").magnitude == 5
    assert q.to("c", "c2").magnitude == 3
    assert q.to("d", "c2").magnitude == 3
    assert q.to("b", "c1", "c2").magnitude == 5  # c2 takes precedence
    assert q.to("c", "c1", "c2").magnitude == 6  # c2 doesn't change it, so use c1
    assert q.to("d", "c1", "c2").magnitude == 3  # c2 takes precedence


def test_err_change_base_unit():
    ureg = UnitRegistry(
        """
        foo = [d1]
        bar = [d2]

        @context c
            bar = foo
        @end
        """.splitlines()
    )

    expected = "Can't redefine a base unit to a derived one"
    with pytest.raises(ValueError, match=expected):
        ureg.enable_contexts("c")


def test_err_to_base_unit():
    expected = "Can't define base units within a context"
    with pytest.raises(DefinitionSyntaxError, match=expected):
        Context.from_lines(["@context c", "x = [d]"])


def test_err_change_dimensionality():
    ureg = UnitRegistry(
        """
        foo = [d1]
        bar = [d2]
        baz = foo

        @context c
            baz = bar
        @end
        """.splitlines()
    )

    expected = re.escape(
        "Can't change dimensionality of baz from [d1] to [d2] in a context"
    )
    with pytest.raises(ValueError, match=expected):
        ureg.enable_contexts("c")


def test_err_cyclic_dependency():
    ureg = UnitRegistry(
        """
        foo = [d]
        bar = foo
        baz = bar

        @context c
            bar = baz
        @end
        """.splitlines()
    )
    # TODO align this exception and the one you get when you implement a cyclic
    #      dependency within the base registry. Ideally this exception should be
    #      raised by enable_contexts.
    ureg.enable_contexts("c")
    q = ureg.Quantity("bar")
    with pytest.raises(RecursionError):
        q.to("foo")


def test_err_dimension_redefinition():
    expected = re.escape("Expected <unit> = <converter>; got [d1] = [d2] * [d3]")
    with pytest.raises(DefinitionSyntaxError, match=expected):
        Context.from_lines(["@context c", "[d1] = [d2] * [d3]"])


def test_err_prefix_redefinition():
    expected = re.escape("Expected <unit> = <converter>; got [d1] = [d2] * [d3]")
    with pytest.raises(DefinitionSyntaxError, match=expected):
        Context.from_lines(["@context c", "[d1] = [d2] * [d3]"])


def test_err_redefine_alias(subtests):
    expected = "Can't change a unit's symbol or aliases within a context"
    for s in ("foo = bar = f", "foo = bar = _ = baz"):
        with subtests.test(s):
            with pytest.raises(DefinitionSyntaxError, match=expected):
                Context.from_lines(["@context c", s])


def test_err_redefine_with_prefix():
    ureg = UnitRegistry(
        """
        kilo- = 1000
        gram = [mass]
        pound = 454 gram

        @context c
            kilopound = 500000 gram
        @end
        """.splitlines()
    )

    expected = "Can't redefine a unit with a prefix: kilopound"
    with pytest.raises(ValueError, match=expected):
        ureg.enable_contexts("c")


def test_err_new_unit():
    ureg = UnitRegistry(
        """
        foo = [d]
        @context c
            bar = foo
        @end
        """.splitlines()
    )
    expected = "'bar' is not defined in the unit registry"
    with pytest.raises(UndefinedUnitError, match=expected):
        ureg.enable_contexts("c")
