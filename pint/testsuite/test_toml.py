from __future__ import annotations

import pathlib
from decimal import Decimal

import pytest

import pint
from pint.delegates import ParserConfig
from pint.delegates.toml_defparser import TomlParser
from pint.delegates.toml_writer import TomlWriter
from pint.facets.context.definitions import BidirectionalRelation, ContextDefinition
from pint.facets.plain import definitions
from pint.util import UnitsContainer

PINT_DIR = pathlib.Path(pint.__file__).parent
VERSION = 'version = "1.0"\n'
BASIC = (
    definitions.DefaultsDefinition,
    definitions.PrefixDefinition,
    definitions.DerivedDimensionDefinition,
    definitions.UnitDefinition,
)


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


@pytest.mark.parametrize("non_int_type", [float, Decimal])
def test_write_reads_back(tmp_path, non_int_type):
    parser = TomlParser(ParserConfig(non_int_type))
    defs = parser.parse_file(PINT_DIR / "default_en.toml")
    f = tmp_path / "defs.toml"
    TomlWriter().write_file(f, defs)
    assert parser.parse_file(f) == defs

    txt = pint.UnitRegistry(PINT_DIR / "default_en.txt", non_int_type=non_int_type)
    written = pint.UnitRegistry(f, non_int_type=non_int_type)
    assert summary(written) == summary(txt)


def test_write_rejects_other_definitions(tmp_path):
    # base dimensions come from base units, so they have no section
    dimension = definitions.DimensionDefinition("[length]")
    with pytest.raises(TypeError, match="DimensionDefinition"):
        TomlWriter().write_file(tmp_path / "defs.toml", [dimension])


def test_write_keeps_relation_text(tmp_path):
    defs = TomlParser(ParserConfig(float)).parse_file(PINT_DIR / "default_en.toml")
    text = TomlWriter().dumps(defs)
    assert '"[length] <-> [frequency]: speed_of_light / n / value"' in text
    assert "defaults = {n = 1}" in text

    # relations made in code are rebuilt
    length, time = UnitsContainer({"[length]": 1}), UnitsContainer({"[time]": 1})
    relation = BidirectionalRelation(length, time, "value / c")
    context = ContextDefinition("c", (), {}, (relation,), ())
    assert '"[length] <-> [time]: value / c"' in TomlWriter().dumps([context])


def test_write_keeps_value_text(tmp_path):
    f = tmp_path / "defs.toml"
    f.write_text(
        VERSION + "[prefix]\n"
        'kilo = { value = 1e3, symbol = "k" }\n'
        'kibi = { value = "2**10", symbol = "Ki" }\n'
        "[dimension]\n"
        'area = { value = "[length] ** 2" }\n'
        "[unit]\n"
        'kelvin = { value = "[temperature]", symbol = "K" }\n'
        'fahrenheit = { value = "5 / 9 * kelvin", offset = 255.37 }\n',
        encoding="utf-8",
    )
    text = TomlWriter().dumps(TomlParser(ParserConfig(float)).parse_file(f))
    assert 'kilo = {value = 1e3, symbol = "k"}' in text
    assert 'kibi = {value = "2**10", symbol = "Ki"}' in text
    assert 'area = {value = "[length] ** 2"}' in text
    assert 'fahrenheit = {value = "5 / 9 * kelvin", offset = 255.37}' in text


def test_write_txt_definitions(tmp_path):
    ureg = pint.UnitRegistry(None)
    parsed = ureg._def_parser.parse_file(PINT_DIR / "default_en.txt")
    basic = [d for d in parsed.iter_statements() if isinstance(d, BASIC)]
    f = tmp_path / "defs.toml"
    TomlWriter().write_file(f, basic)
    text = f.read_text(encoding="utf-8")
    assert 'kibi = {value = "2**10", symbol = "Ki"}' in text
    back = TomlParser(ParserConfig(float)).parse_file(f)
    # the writer gathers derived dimensions in one section, changing the order
    assert {d.name: d.value_text for d in back if hasattr(d, "value_text")} == {
        d.name: d.value_text for d in basic if hasattr(d, "value_text")
    }


@pytest.mark.parametrize("source", ["default_en.toml", "default_en.txt"])
def test_write_keeps_modifier_text(source):
    ureg = pint.UnitRegistry(None)
    if source.endswith(".toml"):
        defs = TomlParser(ParserConfig(float)).parse_file(PINT_DIR / source)
    else:
        defs = ureg._def_parser.parse_file(PINT_DIR / source).iter_statements()
    text = TomlWriter().dumps(d for d in defs if isinstance(d, BASIC))
    assert 'offset = "233.15 + 200 / 9"' in text
    assert "offset = 273.15," in text
    assert "logbase = 2.71828182845904523536028747135266249775724709369995" in text


@pytest.mark.parametrize("non_int_type", [float, Decimal])
def test_write_keeps_context_defaults_text(tmp_path, non_int_type):
    f = tmp_path / "defs.toml"
    f.write_text(
        VERSION + "[unit]\n"
        'meter = { value = "[length]" }\n'
        'second = { value = "[time]" }\n'
        "[[context]]\n"
        'name = "demo"\n'
        'defaults = { n = "2**10", k = 1.00000000000000000001 }\n'
        'relations = ["[length] <-> [time]: value * n / k"]\n',
        encoding="utf-8",
    )
    defs = TomlParser(ParserConfig(non_int_type)).parse_file(f)
    assert 'defaults = {n = "2**10", k = 1.00000000000000000001}' in (
        TomlWriter().dumps(defs)
    )


def test_txt_context_defaults_text():
    context = ContextDefinition.from_lines(
        ["@context(n = 1, k = 2**10) demo", "[length] <-> [time]: value * n / k"],
        float,
    )
    assert context.defaults == {"n": 1, "k": 1024}
    assert context.defaults_text == {"n": "1", "k": "2**10"}
