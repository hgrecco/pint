"""Tests for the shared registry preprocessing pipeline."""

import pytest

from pint import UnitRegistry
from pint.util import _symbol_preprocessor, string_preprocessor


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("m²", "m**(2)"),
        ("m^2", "m**2"),
        ("m × s", "m * s"),
        ("%", " percent "),
        ("‰", " permille "),
    ],
)
def test_builtin_preprocessors(sess_registry, text, expected):
    assert sess_registry._apply_preprocessors(text) == expected
    assert sess_registry.parse_units(text) == sess_registry.parse_units(expected)
    assert sess_registry.parse_expression(text) == sess_registry.parse_expression(
        expected
    )
    assert sess_registry.get_dimensionality(text) == sess_registry.get_dimensionality(
        expected
    )


def test_custom_preprocessor_order():
    seen = []

    def first(text):
        seen.append(text)
        return text.replace("area", "m²")

    def second(text):
        seen.append(text)
        return text.replace("m²", "s^2")

    custom = [first, second]
    registry = UnitRegistry(preprocessors=custom)
    seen.clear()
    assert registry._apply_preprocessors("area × % ‰") == "s**2 * percent permille "
    assert seen == ["area *  percent   permille ", "m² *  percent   permille "]
    assert registry.preprocessors == [
        _symbol_preprocessor,
        first,
        second,
        string_preprocessor,
    ]
    assert custom == [first, second]


def test_preprocessors_list_can_be_reused():
    custom = [lambda text: text.replace("area", "m²")]
    first = UnitRegistry(preprocessors=custom)
    second = UnitRegistry(preprocessors=custom)
    assert first.preprocessors is not second.preprocessors
    assert len(custom) == 1
    assert first.parse_units("area") == first.meter**2
    assert second.parse_expression("3 area") == second.Quantity(3, "m**2")
