from __future__ import annotations

import pytest

from pint import formatting as fmt
from pint.delegates.formatter._format_helpers import formatter, join_u


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
