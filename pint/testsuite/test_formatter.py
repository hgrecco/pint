import pytest

from pint import formatting as fmt
import pint.delegates.formatter._format_helpers


class TestFormatter:
    def test_join(self):
        for empty in ((), []):
            assert fmt._join("s", empty) == ""
        assert fmt._join("*", "1 2 3".split()) == "1*2*3"
        assert fmt._join("{0}*{1}", "1 2 3".split()) == "1*2*3"

    def test_formatter(self):
        assert pint.delegates.formatter._format_helpers.formatter({}.items()) == ""
        assert (
            pint.delegates.formatter._format_helpers.formatter(dict(meter=1).items())
            == "meter"
        )
        assert (
            pint.delegates.formatter._format_helpers.formatter(dict(meter=-1).items())
            == "1 / meter"
        )
        assert (
            pint.delegates.formatter._format_helpers.formatter(
                dict(meter=-1).items(), as_ratio=False
            )
            == "meter ** -1"
        )

        assert (
            pint.delegates.formatter._format_helpers.formatter(
                dict(meter=-1, second=-1).items(), as_ratio=False
            )
            == "meter ** -1 * second ** -1"
        )
        assert (
            pint.delegates.formatter._format_helpers.formatter(
                dict(meter=-1, second=-1).items()
            )
            == "1 / meter / second"
        )
        assert (
            pint.delegates.formatter._format_helpers.formatter(
                dict(meter=-1, second=-1).items(), single_denominator=True
            )
            == "1 / (meter * second)"
        )
        assert (
            pint.delegates.formatter._format_helpers.formatter(
                dict(meter=-1, second=-2).items()
            )
            == "1 / meter / second ** 2"
        )
        assert (
            pint.delegates.formatter._format_helpers.formatter(
                dict(meter=-1, second=-2).items(), single_denominator=True
            )
            == "1 / (meter * second ** 2)"
        )

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
