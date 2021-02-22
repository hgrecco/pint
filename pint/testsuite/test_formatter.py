import pytest

from pint import formatting as fmt


class TestFormatter:
    def test_join(self):
        for empty in (tuple(), []):
            assert fmt._join("s", empty) == ""
        assert fmt._join("*", "1 2 3".split()) == "1*2*3"
        assert fmt._join("{0}*{1}", "1 2 3".split()) == "1*2*3"

    def test_formatter(self):
        assert fmt.formatter(dict().items()) == ""
        assert fmt.formatter(dict(meter=1).items()) == "meter"
        assert fmt.formatter(dict(meter=-1).items()) == "1 / meter"
        assert fmt.formatter(dict(meter=-1).items(), as_ratio=False) == "meter ** -1"

        assert (
            fmt.formatter(dict(meter=-1, second=-1).items(), as_ratio=False)
            == "meter ** -1 * second ** -1"
        )
        assert fmt.formatter(dict(meter=-1, second=-1).items()) == "1 / meter / second"
        assert (
            fmt.formatter(dict(meter=-1, second=-1).items(), single_denominator=True)
            == "1 / (meter * second)"
        )
        assert (
            fmt.formatter(dict(meter=-1, second=-2).items())
            == "1 / meter / second ** 2"
        )
        assert (
            fmt.formatter(dict(meter=-1, second=-2).items(), single_denominator=True)
            == "1 / (meter * second ** 2)"
        )

    def test_parse_spec(self):
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
