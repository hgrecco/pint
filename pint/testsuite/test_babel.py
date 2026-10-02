from __future__ import annotations

import os

import pytest

from pint import UnitRegistry
from pint.testsuite import helpers


@helpers.requires_not_babel()
def test_no_babel(func_registry):
    ureg = func_registry
    distance = 24.0 * ureg.meter
    with pytest.raises(Exception):
        ureg.formatter.format_unit_babel(distance, locale="fr_FR", length="long")


@helpers.requires_babel(["fr_FR", "ro_RO"])
def test_format(func_registry):
    ureg = func_registry
    dirname = os.path.dirname(__file__)
    ureg.load_definitions(os.path.join(dirname, "../xtranslated.txt"))

    distance = 24.1 * ureg.meter
    assert distance.format_babel(locale="fr_FR", length="long") == "24,1 mètres"
    time = 8.1 * ureg.second
    assert time.format_babel(locale="fr_FR", length="long") == "8,1 secondes"
    assert time.format_babel(locale="ro_RO", length="short") == "8,1 s"
    acceleration = distance / time**2
    assert (
        acceleration.format_babel(spec=".3nP", locale="fr_FR", length="long")
        == "0,367 mètre par seconde²"
    )
    mks = ureg.get_system("mks")
    assert mks.format_babel(locale="fr_FR") == "métrique"


@helpers.requires_babel(["fr_FR", "ro_RO"])
def test_registry_locale():
    ureg = UnitRegistry(fmt_locale="fr_FR")
    dirname = os.path.dirname(__file__)
    ureg.load_definitions(os.path.join(dirname, "../xtranslated.txt"))

    distance = 24.1 * ureg.meter
    assert distance.format_babel(length="long") == "24,1 mètres"
    time = 8.1 * ureg.second
    assert time.format_babel(length="long") == "8,1 secondes"
    assert time.format_babel(locale="ro_RO", length="short") == "8,1 s"
    acceleration = distance / time**2
    assert (
        acceleration.format_babel(spec=".3nC", length="long")
        == "0,367 mètre/seconde**2"
    )
    assert (
        acceleration.format_babel(spec=".3nP", length="long")
        == "0,367 mètre par seconde²"
    )
    mks = ureg.get_system("mks")
    assert mks.format_babel(locale="fr_FR") == "métrique"


@helpers.requires_babel(["fr_FR"])
def test_unit_format_babel():
    ureg = UnitRegistry(fmt_locale="fr_FR")
    volume = ureg.Unit("ml")
    assert volume.format_babel() == "millilitre"

    ureg.formatter.default_format = "~"
    assert volume.format_babel() == "ml"

    dimensionless_unit = ureg.Unit("")
    assert dimensionless_unit.format_babel() == ""

    ureg.set_fmt_locale(None)
    with pytest.raises(ValueError):
        volume.format_babel()


@helpers.requires_babel()
def test_no_registry_locale(func_registry):
    ureg = func_registry
    distance = 24.0 * ureg.meter
    with pytest.raises(Exception):
        distance.format_babel()


@helpers.requires_babel(["fr_FR"])
def test_str(func_registry):
    ureg = func_registry
    d = 24.1 * ureg.meter

    s = "24.1 meter"
    assert str(d) == s
    assert "%s" % d == s
    assert f"{d}" == s

    ureg.set_fmt_locale("fr_FR")
    s = "24,1 mètres"
    assert str(d) == s
    assert "%s" % d == s
    assert f"{d}" == s

    ureg.set_fmt_locale(None)
    s = "24.1 meter"
    assert str(d) == s
    assert "%s" % d == s
    assert f"{d}" == s


@helpers.requires_babel(["fr_FR"])
def test_unit_with_empty_numerator(func_registry):
    # A unit whose exponents are all negative has an empty numerator, and the
    # pluralization step used to reach for its last element, raising IndexError.
    ureg = func_registry
    ureg.formatter.locale = "fr_FR"

    assert f"{3 / ureg.second}" == "3 1 par seconde"
    assert f"{3 / ureg.meter**2}" == "3 1 par mètre ** 2"
    assert ureg.formatter.format_unit(ureg.Unit("1 / second"), locale="fr_FR") == (
        "1 par seconde"
    )
