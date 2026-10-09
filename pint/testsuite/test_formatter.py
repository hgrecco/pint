from __future__ import annotations

import locale

import pytest

from pint import formatting as fmt
from pint.delegates.formatter import _format_helpers
from pint.delegates.formatter._format_helpers import formatter, join_u
from pint.formatting import formatter as pf_formatter


@pytest.fixture(params=["C", "fr_FR"])
def numeric_locale(monkeypatch, request):
    state = {"current": request.param}

    def setlocale(category, value=None):
        assert category == locale.LC_NUMERIC
        if value is None:
            return state["current"]
        if value == (None, None):
            value = "C"
        elif isinstance(value, tuple):
            raise locale.Error("tuple locale is not round-trippable")
        if value == "unavailable":
            raise locale.Error("unsupported locale")
        state["current"] = value
        return value

    monkeypatch.setattr(_format_helpers, "setlocale", setlocale)
    return state


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError, KeyboardInterrupt])
def test_override_locale_restores_after_error(numeric_locale, error_type):
    original = numeric_locale["current"]
    error = error_type("body error")
    with pytest.raises(error_type) as caught:
        with _format_helpers.override_locale(".2f", "outer"):
            assert numeric_locale["current"] == "outer"
            raise error
    assert caught.value is error
    assert numeric_locale["current"] == original


def test_override_locale_restores_after_success(numeric_locale):
    original = numeric_locale["current"]
    with _format_helpers.override_locale(".2f", "outer") as format_number:
        assert numeric_locale["current"] == "outer"
        assert format_number(1.25) == "1.25"
    assert numeric_locale["current"] == original


def test_override_locale_nested_error(numeric_locale):
    original = numeric_locale["current"]
    with _format_helpers.override_locale("", "outer"):
        with pytest.raises(ValueError, match="inner error"):
            with _format_helpers.override_locale("", "inner"):
                assert numeric_locale["current"] == "inner"
                raise ValueError("inner error")
        assert numeric_locale["current"] == "outer"
    assert numeric_locale["current"] == original


@pytest.mark.parametrize("raise_error", [False, True])
def test_override_locale_none_does_not_change_locale(monkeypatch, raise_error):
    def unexpected_call(*args):
        pytest.fail("locale=None must not read or change LC_NUMERIC")

    monkeypatch.setattr(_format_helpers, "setlocale", unexpected_call)
    if raise_error:
        with pytest.raises(ValueError, match="body error"):
            with _format_helpers.override_locale(".2f", None):
                raise ValueError("body error")
    else:
        with _format_helpers.override_locale(".2f", None) as format_number:
            assert format_number(1.25) == "1.25"


def test_override_locale_setup_error(numeric_locale):
    original = numeric_locale["current"]
    with pytest.raises(locale.Error, match="unsupported locale"):
        with _format_helpers.override_locale("", "unavailable"):
            pytest.fail("a failed locale setup must not enter the body")
    assert numeric_locale["current"] == original


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
