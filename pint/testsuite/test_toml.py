from __future__ import annotations

import pathlib
import sys
import tomllib

import pytest

import pint
from pint.errors import DefinitionSyntaxError

PINT_DIR = pathlib.Path(pint.__file__).parent

TINY_TXT = """
@defaults
    group = international
    system = mks
@end

kilo- = 1e3 = k
[speed] = [length] / [time]
meter = [length] = m = metre
second = [time] = s
kelvin = [temperature]; offset: 0 = K
degree_Celsius = kelvin; offset: 273.15 = °C = celsius
minute = 60 * second = min

@group international
    inch = meter / 39.37 = in
    foot = 12 * inch = ft
@end

@system mks using international
    meter
    second
@end

@context(n=1) spectroscopy = sp
    [length] <-> [time]: n * meter * second / value
@end
"""


def summary(ureg):
    """Everything a definitions file contributes to a registry."""
    return dict(
        units={
            d.name: (d.value_text, d.defined_symbol, d.aliases)
            for d in ureg._units.values()
            if d.value_text
        },
        prefixes={d.name: d.value_text for d in ureg._prefixes.values()},
        dimensions={d.name: d.value_text for d in ureg._dimensions.values()},
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


def test_default_toml_is_up_to_date(tmp_path):
    pytest.importorskip("tomlkit")
    out = tmp_path / "default_en.toml"
    pint.UnitRegistry(PINT_DIR / "default_en.txt").write_definitions(out)
    # compare parsed data, as the shipped file is reformatted by taplo
    written = tomllib.loads(out.read_text(encoding="utf-8"))
    shipped = tomllib.loads((PINT_DIR / "default_en.toml").read_text(encoding="utf-8"))
    assert written == shipped, (
        "Regenerate with UnitRegistry().write_definitions('pint/default_en.toml')"
    )


def test_round_trip(tmp_path):
    pytest.importorskip("tomlkit")
    txt_file = tmp_path / "defs.txt"
    txt_file.write_text(TINY_TXT, encoding="utf-8")
    txt = pint.UnitRegistry(txt_file)

    toml_file = tmp_path / "defs.toml"
    txt.write_definitions(toml_file)
    toml = pint.UnitRegistry(toml_file)

    assert summary(toml) == summary(txt)
    assert toml.Quantity(1, "foot").to("meter").magnitude == pytest.approx(12 / 39.37)
    assert toml.Quantity(0, "celsius").to("K").magnitude == pytest.approx(273.15)
    with toml.context("sp", n=2):
        assert toml.Quantity(4, "s").to("m").magnitude == pytest.approx(0.5)


def test_sections_are_optional(tmp_path):
    f = tmp_path / "defs.toml"
    f.write_text('[unit]\nmeter = { value = "[length]" }\n', encoding="utf-8")
    ureg = pint.UnitRegistry(f)
    assert ureg.Quantity(1, "meter").dimensionality == {"[length]": 1}


@pytest.mark.parametrize(
    "content",
    [
        '[units]\nmeter = { value = "[length]" }\n',
        '[prefix]\nkilo = { value = "meter" }\n',
        '[unit]\ncelsius = { value = "kelvin; offset" }\n',
        '[defaults]\nsystm = "mks"\n',
        '[[context]]\nname = "c"\nrelations = ["[length] <> [time]: value"]\n',
    ],
)
def test_syntax_errors(tmp_path, content):
    f = tmp_path / "defs.toml"
    f.write_text(content, encoding="utf-8")
    with pytest.raises(DefinitionSyntaxError):
        pint.UnitRegistry(f)


def test_write_without_tomlkit(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "tomlkit", None)
    with pytest.raises(ImportError, match="tomlkit"):
        pint.UnitRegistry(PINT_DIR / "default_en.toml").write_definitions(
            tmp_path / "out.toml"
        )
