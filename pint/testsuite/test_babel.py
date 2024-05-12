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
        distance.format_babel(locale="fr_FR", length="long")


@helpers.requires_babel()
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


@helpers.requires_babel()
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


@helpers.requires_babel()
def test_unit_format_babel():
    ureg = UnitRegistry(fmt_locale="fr_FR")
    volume = ureg.Unit("ml")
    assert volume.format_babel() == "millilitre"

    ureg.default_format = "~"
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


@helpers.requires_babel()
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
