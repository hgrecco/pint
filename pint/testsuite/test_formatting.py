from __future__ import annotations

import pytest

import pint.formatting as fmt


@pytest.mark.filterwarnings("ignore::DeprecationWarning:pint*")
@pytest.mark.parametrize(
    ["format", "default", "flag", "expected"],
    (
        pytest.param(".02fD", ".3fP", True, (".02f", "D"), id="both-both-separate"),
        pytest.param(".02fD", ".3fP", False, (".02f", "D"), id="both-both-combine"),
        pytest.param(".02fD", ".3fP", None, (".02f", "D"), id="both-both-default"),
        pytest.param("D", ".3fP", True, (".3f", "D"), id="unit-both-separate"),
        pytest.param("D", ".3fP", False, ("", "D"), id="unit-both-combine"),
        pytest.param("D", ".3fP", None, ("", "D"), id="unit-both-default"),
        pytest.param(".02f", ".3fP", True, (".02f", "P"), id="magnitude-both-separate"),
        pytest.param(".02f", ".3fP", False, (".02f", ""), id="magnitude-both-combine"),
        pytest.param(".02f", ".3fP", None, (".02f", ""), id="magnitude-both-default"),
        pytest.param("D", "P", True, ("", "D"), id="unit-unit-separate"),
        pytest.param("D", "P", False, ("", "D"), id="unit-unit-combine"),
        pytest.param("D", "P", None, ("", "D"), id="unit-unit-default"),
        pytest.param(
            ".02f", ".3f", True, (".02f", ""), id="magnitude-magnitude-separate"
        ),
        pytest.param(
            ".02f", ".3f", False, (".02f", ""), id="magnitude-magnitude-combine"
        ),
        pytest.param(
            ".02f", ".3f", None, (".02f", ""), id="magnitude-magnitude-default"
        ),
        pytest.param("D", ".3f", True, (".3f", "D"), id="unit-magnitude-separate"),
        pytest.param("D", ".3f", False, ("", "D"), id="unit-magnitude-combine"),
        pytest.param("D", ".3f", None, ("", "D"), id="unit-magnitude-default"),
        pytest.param(".02f", "P", True, (".02f", "P"), id="magnitude-unit-separate"),
        pytest.param(".02f", "P", False, (".02f", ""), id="magnitude-unit-combine"),
        pytest.param(".02f", "P", None, (".02f", ""), id="magnitude-unit-default"),
        pytest.param("", ".3fP", True, (".3f", "P"), id="none-both-separate"),
        pytest.param("", ".3fP", False, (".3f", "P"), id="none-both-combine"),
        pytest.param("", ".3fP", None, (".3f", "P"), id="none-both-default"),
        pytest.param("", "P", True, ("", "P"), id="none-unit-separate"),
        pytest.param("", "P", False, ("", "P"), id="none-unit-combine"),
        pytest.param("", "P", None, ("", "P"), id="none-unit-default"),
        pytest.param("", ".3f", True, (".3f", ""), id="none-magnitude-separate"),
        pytest.param("", ".3f", False, (".3f", ""), id="none-magnitude-combine"),
        pytest.param("", ".3f", None, (".3f", ""), id="none-magnitude-default"),
        pytest.param("", "", True, ("", ""), id="none-none-separate"),
        pytest.param("", "", False, ("", ""), id="none-none-combine"),
        pytest.param("", "", None, ("", ""), id="none-none-default"),
    ),
)
def test_split_format(format, default, flag, expected):
    result = fmt.split_format(format, default, flag)

    assert result == expected


def test_register_unit_format(func_registry):
    @fmt.register_unit_format("custom")
    def format_custom(unit, registry, **options):
        # Ensure the registry is correct..
        registry.Unit(unit)
        return "<formatted unit>"

    quantity = 1.0 * func_registry.meter
    assert f"{quantity:custom}" == "1.0 <formatted unit>"

    with pytest.raises(ValueError, match="format 'custom' already exists"):

        @fmt.register_unit_format("custom")
        def format_custom_redefined(unit, registry, **options):
            return "<overwritten>"
