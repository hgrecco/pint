from __future__ import annotations

import pathlib
from decimal import Decimal

import pytest

import pint
from pint.errors import DefinitionSyntaxError

PINT_DIR = pathlib.Path(pint.__file__).parent
VERSION = 'version = "1.0"\n'


def summary(ureg):
    """Everything a definitions file contributes to a registry."""
    return dict(
        units={
            d.name: (d.defined_symbol, d.aliases, d.converter, d.reference)
            for d in ureg._units.values()
        },
        prefixes={d.name: (d.value, d.aliases) for d in ureg._prefixes.values()},
        dimensions={
            d.name: getattr(d, "reference", None) for d in ureg._dimensions.values()
        },
        groups={name: sorted(g.members) for name, g in ureg._groups.items()},
        systems={name: sorted(s.base_units) for name, s in ureg._systems.items()},
        contexts={
            name: (sorted(c.defaults.items()), sorted(map(str, c.funcs)))
            for name, c in ureg._contexts.items()
        },
        defaults=ureg._defaults,
        default_system=ureg.default_system,
    )


def test_default_toml_matches_txt():
    txt = pint.UnitRegistry(PINT_DIR / "default_en.txt")
    toml = pint.UnitRegistry(PINT_DIR / "default_en.toml")
    assert summary(toml) == summary(txt)
    with toml.context("sp"):
        assert toml.Quantity(500, "nm").to("THz").magnitude == pytest.approx(
            599.58, 1e-4
        )


def test_numbers_keep_their_digits():
    # TOML floats are read as text, so they keep every digit and get the
    # registry's number type, as in the text format
    txt = pint.UnitRegistry(PINT_DIR / "default_en.txt", non_int_type=Decimal)
    toml = pint.UnitRegistry(PINT_DIR / "default_en.toml", non_int_type=Decimal)
    assert summary(toml) == summary(txt)
    neper = toml._units["neper"].converter
    assert neper.logbase == Decimal(
        "2.71828182845904523536028747135266249775724709369995"
    )
    assert toml._units["degree_Celsius"].converter.offset == Decimal("273.15")
    assert toml._prefixes["kilo"].value == Decimal("1e3")


def test_modifier_keys(tmp_path):
    f = tmp_path / "defs.toml"
    f.write_text(
        VERSION + "[unit]\n"
        'kelvin = { value = "[temperature]", offset = 0, symbol = "K" }\n'
        'celsius = { value = "kelvin", offset = 273.15 }\n'
        'fahrenheit = { value = "5 / 9 * kelvin", offset = "233.15 + 200 / 9" }\n'
        'bel = { value = "1", logbase = 10, logfactor = 1, symbol = "B" }\n',
        encoding="utf-8",
    )
    ureg = pint.UnitRegistry(f)
    assert ureg.Quantity(0, "celsius").to("K").magnitude == pytest.approx(273.15)
    assert ureg.Quantity(32, "fahrenheit").to("celsius").magnitude == pytest.approx(0)
    assert ureg.Quantity(2, "B").to("").magnitude == pytest.approx(100)


def test_sections_are_optional(tmp_path):
    f = tmp_path / "defs.toml"
    f.write_text(VERSION + '[unit]\nmeter = { value = "[length]" }\n', encoding="utf-8")
    ureg = pint.UnitRegistry(f)
    assert ureg.Quantity(1, "meter").dimensionality == {"[length]": 1}


@pytest.mark.parametrize(
    "content",
    [
        VERSION + '[units]\nmeter = { value = "[length]" }\n',
        VERSION + "[unit]\nmeter = {}\n",
        VERSION + '[prefix]\nkilo = { value = "meter" }\n',
        # the version is required, and must have the supported major version
        '[unit]\nmeter = { value = "[length]" }\n',
        'version = "2.0"\n[unit]\nmeter = { value = "[length]" }\n',
    ],
)
def test_syntax_errors_name_the_file(tmp_path, content):
    f = tmp_path / "defs.toml"
    f.write_text(content, encoding="utf-8")
    with pytest.raises(DefinitionSyntaxError, match="defs.toml"):
        pint.UnitRegistry(f)
