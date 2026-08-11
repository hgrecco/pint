from __future__ import annotations

import pytest

from pint import formatting as fmt
from pint.delegates.formatter._format_helpers import formatter, join_u
from pint.formatting import formatter as pf_formatter


class TestFormatter:
    def test_join(self):
        for empty in ((), []):
            assert join_u("s", empty) == ""
        assert join_u("*", "1 2 3".split()) == "1*2*3"
        assert join_u("{0}*{1}", "1 2 3".split()) == "1*2*3"

    def test_formatter(self):
        assert formatter({}.items(), ()) == ""
        assert formatter(dict(meter=1).items(), ()) == "meter"
        assert formatter((), dict(meter=-1).items()) == "1 / meter"
        assert formatter((), dict(meter=-1).items(), as_ratio=False) == "meter ** -1"

        assert (
            formatter((), dict(meter=-1, second=-1).items(), as_ratio=False)
            == "meter ** -1 * second ** -1"
        )
        assert (
            formatter(
                (),
                dict(meter=-1, second=-1).items(),
            )
            == "1 / meter / second"
        )
        assert (
            formatter((), dict(meter=-1, second=-1).items(), single_denominator=True)
            == "1 / (meter * second)"
        )
        assert (
            formatter((), dict(meter=-1, second=-2).items())
            == "1 / meter / second ** 2"
        )
        assert (
            formatter((), dict(meter=-1, second=-2).items(), single_denominator=True)
            == "1 / (meter * second ** 2)"
        )

    def test_formatter_fraction_exponent(self):
        # A Fraction exponent (used when a registry is created with
        # non_int_type=Fraction) does not accept the 'n' format spec, so the
        # default exponent formatting used to raise ValueError instead of
        # rendering. See GH #2386.
        from fractions import Fraction

        exp = Fraction(23, 10)
        assert formatter(dict(meter=exp).items(), ()) == "meter ** 23/10"
        assert (
            formatter((), dict(meter=-exp).items(), as_ratio=False)
            == "meter ** -23/10"
        )

    def test_unit_fraction_exponent_formatting(self):
        # End-to-end: formatting a unit whose exponent is a Fraction must not
        # raise for any of the built-in format specs. GH #2386.
        import fractions

        import pint

        ureg = pint.UnitRegistry(non_int_type=fractions.Fraction)
        u = ureg.Unit("m**2.3")
        assert str(u) == "meter ** 23/10"
        assert format(u, "~") == "m ** 23/10"
        # Pretty specs render the exponent as superscripts; assert they produce
        # (non-empty) output rather than raising.
        assert format(u, "P")
        assert format(u, "~P")

    def testparse_spec(self):
        assert fmt._parse_spec("") == ""
        assert fmt._parse_spec("") == ""
        with pytest.raises(ValueError):
            fmt._parse_spec("W")
        with pytest.raises(ValueError):
            fmt._parse_spec("PL")

    def test_format_unit(self):
        assert fmt.format_unit("", "C") == "dimensionless"
        with pytest.raises(ValueError):
            fmt.format_unit("m", "W")

    def test_pf_formatter(self):
        assert pf_formatter({}.items()) == ""
        assert pf_formatter(dict(meter=1).items()) == "meter"
        assert pf_formatter(dict(meter=-1).items()) == "1 / meter"
        assert pf_formatter(dict(meter=-1).items(), as_ratio=False) == "meter ** -1"

        assert (
            pf_formatter(dict(meter=-1, second=-1).items(), as_ratio=False)
            == "meter ** -1 * second ** -1"
        )
        assert (
            pf_formatter(
                dict(meter=-1, second=-1).items(),
            )
            == "1 / meter / second"
        )
        assert (
            pf_formatter(dict(meter=-1, second=-1).items(), single_denominator=True)
            == "1 / (meter * second)"
        )
        assert (
            pf_formatter(dict(meter=-1, second=-2).items()) == "1 / meter / second ** 2"
        )
        assert (
            pf_formatter(dict(second=-2, meter=-1).items(), sort=False)
            == "1 / second ** 2 / meter"
        )
        assert (
            pf_formatter(dict(meter=-1, second=-2).items(), single_denominator=True)
            == "1 / (meter * second ** 2)"
        )
