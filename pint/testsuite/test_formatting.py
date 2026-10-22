from __future__ import annotations

from fractions import Fraction

import pytest

import pint.formatting as fmt
from pint.compat import np
from pint.testsuite import helpers


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


def test_register_unit_format(registry_tiny):
    @fmt.register_unit_format("custom")
    def format_custom(unit, registry, **options):
        # Ensure the registry is correct..
        registry.Unit(unit)
        return "<formatted unit>"

    quantity = 1.0 * registry_tiny.meter
    assert f"{quantity:g#~custom}" == "1 <formatted unit>"
    assert f"{quantity:custom}" == "1.0 <formatted unit>"

    with pytest.raises(ValueError, match="format 'custom' already exists"):

        @fmt.register_unit_format("custom")
        def format_custom_redefined(unit, registry, **options):
            return "<overwritten>"


def test_format_unit_caret_negative_power(func_registry):
    """format_unit with ^ flag should use negative powers (fixes #2313)."""
    u = func_registry.Unit("second") ** -1
    # With ^ flag: s⁻¹ (negative power form)
    assert format(u, "~^P") == "s⁻¹"
    # Without ^ flag: 1/s (ratio form, unchanged)
    assert format(u, "~P") == "1/s"


def test_format_unit_fraction_exponent(func_registry):
    """format_unit with / flag renders fractional exponents as a pretty
    printed fraction (superscript numerator, superscript slash, superscript
    denominator) instead of a decimal (fixes #60).
    """
    u = func_registry.Unit("second") ** 1.5
    assert format(u, "~P") == "s¹⋅⁵"
    assert format(u, "~/P") == "s³ᐟ²"

    # Whole-number exponents are unaffected (still a plain superscript).
    u2 = func_registry.Unit("meter") ** 2
    assert format(u2, "~/P") == "m²"

    # An exact Fraction exponent renders the same as the equivalent float.
    u3 = func_registry.Unit("second") ** Fraction(3, 2)
    assert format(u3, "~/P") == "s³ᐟ²"


def test_format_unit_fraction_exponent_absorbs_noise(func_registry):
    """The / flag snaps near-integer floating-point noise back to a clean
    exponent, the same noise that caused DimensionalityError in #1487/#681.
    """
    # Noise near an integer, e.g. from combining fractional powers (#681).
    u = func_registry.Unit("second") ** -0.9999999999999998
    assert format(u, "~P") == "1/s¹"
    assert format(u, "~/P") == "1/s"

    # Noise near zero (#1487): the residual exponent snaps to 0.
    u2 = func_registry.Unit("meter") ** 5.55e-17
    assert format(u2, "~P") == "m⁵⋅⁵⁵e⁻¹⁷"
    assert format(u2, "~/P") == "m⁰"


@pytest.mark.parametrize(
    ("unit", "latex", "siunitx"),
    [
        ("degC", r"\mathrm{{}^{\circ}C}", r"\si[]{\degreeCelsius}"),
        ("degF", r"\mathrm{{}^{\circ}F}", r"\si[]{\degree F}"),
        ("degR", r"\mathrm{{}^{\circ}R}", r"\si[]{\degree R}"),
        ("delta_degC", r"\mathrm{\Delta {}^{\circ}C}", r"\si[]{\degreeCelsius}"),
        ("delta_degF", r"\mathrm{\Delta {}^{\circ}F}", r"\si[]{\degree F}"),
    ],
)
def test_format_temperature_latex(func_registry, unit, latex, siunitx):
    """Temperature units format as valid LaTeX and siunitx (#400)."""
    u = func_registry.Unit(unit)
    assert format(u, "~L") == latex
    assert format(u, "Lx") == siunitx
    assert format(u, "~Lx") == siunitx


def test_format_temperature_siunitx_compound(func_registry):
    u = func_registry.Unit("joule / delta_degC ** 2")
    assert format(u, "Lx") == r"\si[]{\joule\per\degreeCelsius\squared}"


@helpers.requires_numpy
@pytest.mark.parametrize(
    "spec, shape, expected",
    [
        ("~T", (2,), 'mat(0, 1) "m"'),
        ("~T", (2, 2), 'mat(0, 1 ;\n2, 3) "m"'),
        (
            "~T",
            (2, 1, 1),
            'arr[0,:,:] = mat(0)\narr[1,:,:] = mat(1) "m"',
        ),
        (
            "~T",
            (1, 2, 1, 1),
            'arr[0,0,:,:] = mat(0)\narr[0,1,:,:] = mat(1) "m"',
        ),
        ("~Tz", (2,), "mat(#quan[0], #quan[1]) #quan[m]"),
        ("~Tz", (2, 2), "mat(#quan[0], #quan[1] ;\n#quan[2], #quan[3]) #quan[m]"),
    ],
)
def test_format_typst_array(func_registry, spec, shape, expected):
    q = func_registry.Quantity(np.arange(np.prod(shape)).reshape(shape), "m")
    assert format(q, spec) == expected


@helpers.requires_numpy
def test_format_typst_array_mspec(func_registry):
    q = func_registry.Quantity(np.array([1.0, 2e20]), "m")
    assert format(q, ".1f~T") == 'mat(1.0, 200000000000000000000.0) "m"'
    assert format(q, "~T") == 'mat(1.0, 2 times 10^(20)) "m"'
    assert format(q, "~Tz") == "mat(#quan[1.0], #quan[2e20]) #quan[m]"
