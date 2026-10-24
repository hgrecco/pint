from __future__ import annotations

import functools
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

@context fast
    minute = 30 * second
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


@pytest.mark.filterwarnings("error")
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
        '[unit]\nmeter = { value = "[length]", aliases = "metre" }\n',
        '[unit]\nmeter = { defined_symbol = "m" }\n',
        "[unit]\nmeter = { value = 1 }\n",
        '[[unit]]\nmeter = "[length]"\n',
        '[unit]\nmeter = "[length]"\n',
        '[[context]]\nname = "c"\ndefaults = { n = "x" }\n',
        '[[system]]\nrules = ["meter"]\n',
        '[[system]]\nname = "s"\nrules = "meter"\n',
        '[[context]]\nname = "c"\nrelations = "[length] -> [time]: value"\n',
        '[[group]]\nname = "g"\ndefinitions = ["meter"]\n',
        '[[context]]\nname = "c"\ndefaults = ["n"]\n',
        '[[context]]\nname = "c"\nredefinitions = ["meter"]\n',
        '[unit]\nmeter = { value = "[length]", defined_symbol = 1 }\n',
        "[defaults]\ngroup = 1\n",
        "[unit\n",
        '[alias]\nmeter = { aliases = "metro" }\n',
        '[alias]\nmeter = ["metro"]\n',
    ],
)
def test_syntax_errors(tmp_path, content):
    f = tmp_path / "defs.toml"
    f.write_text(content, encoding="utf-8")
    with pytest.raises(DefinitionSyntaxError):
        pint.UnitRegistry(f)


def test_error_names_file_and_definition(tmp_path):
    f = tmp_path / "defs.toml"
    f.write_text('[unit]\nmeter = { defined_symbol = "m" }\n', encoding="utf-8")
    with pytest.raises(DefinitionSyntaxError, match=r"\[unit\] meter: missing 'value'"):
        pint.UnitRegistry(f)
    with pytest.raises(DefinitionSyntaxError, match="defs.toml"):
        pint.UnitRegistry(f)


def test_registry_error_notes_file(tmp_path):
    f = tmp_path / "defs.toml"
    f.write_text(
        '[unit]\nmeter = { value = "[length]" }\n'
        '[[system]]\nname = "s"\nrules = ["parsec"]\n',
        encoding="utf-8",
    )
    with pytest.raises(pint.UndefinedUnitError) as excinfo:
        pint.UnitRegistry(f)
    (note,) = excinfo.value.__notes__
    assert note == f"While adding SystemDefinition 's' from {f}"


def test_line_generator_is_not_a_path():
    ureg = pint.UnitRegistry(None)
    # same TypeError as before TOML support: only lists of lines are accepted
    with pytest.raises(TypeError):
        ureg.load_definitions(line for line in ["meter = [length]"])


def write_and_load(ureg, tmp_path):
    pytest.importorskip("tomlkit")
    f = tmp_path / "out.toml"
    ureg.write_definitions(f)
    return pint.UnitRegistry(f), tomllib.loads(f.read_text(encoding="utf-8"))


@pytest.fixture
def tiny(tmp_path):
    f = tmp_path / "defs.txt"
    f.write_text(TINY_TXT, encoding="utf-8")
    return pint.UnitRegistry(f)


def test_removed_and_replaced_contexts(tiny, tmp_path):
    from pint.facets.context import ContextDefinition

    tiny.remove_context("sp")
    tiny.add_context(ContextDefinition("fast", (), {}, (), ()))
    reloaded, data = write_and_load(tiny, tmp_path)
    assert data["context"] == [{"name": "fast"}]
    assert sorted(reloaded._contexts) == ["fast"]


def test_definitions_made_in_code(tiny, tmp_path):
    from pint.facets.context import ContextDefinition, ForwardRelation
    from pint.facets.nonmultiplicative.definitions import OffsetConverter
    from pint.facets.plain import ScaleConverter, UnitDefinition
    from pint.util import UnitsContainer

    tiny.define(
        UnitDefinition("double", "dbl", (), ScaleConverter(2), UnitsContainer(meter=1))
    )
    tiny.define(
        UnitDefinition(
            "warm", None, (), OffsetConverter(1, 10), UnitsContainer(kelvin=1)
        )
    )
    relation = ForwardRelation(
        UnitsContainer({"[length]": 1}), UnitsContainer({"[time]": 1}), "value"
    )
    tiny.add_context(ContextDefinition("code", (), {}, (relation,), ()))
    tiny.Quantity(1, "km")  # caches the prefixed unit kilometer

    reloaded, data = write_and_load(tiny, tmp_path)
    assert data["unit"]["double"] == {"value": "2 * meter", "defined_symbol": "dbl"}
    assert "kilometer" not in data["unit"]
    assert "delta_degree_Celsius" not in data["unit"]
    assert reloaded.Quantity(1, "dbl").to("m").magnitude == 2
    assert reloaded.Quantity(0, "warm").to("K").magnitude == 10
    assert reloaded._context_definitions["code"].relations[0].equation == "value"


def test_active_context_not_written(tiny, tmp_path):
    with tiny.context("fast"):
        reloaded, data = write_and_load(tiny, tmp_path)
    assert data["unit"]["minute"]["value"] == "60 * second"
    assert reloaded.Quantity(1, "minute").to("s").magnitude == 60


def test_more_definitions_made_in_code(tiny, tmp_path):
    from pint.facets.context import BidirectionalRelation, ContextDefinition
    from pint.facets.nonmultiplicative.definitions import LogarithmicConverter
    from pint.facets.plain import PrefixDefinition, ScaleConverter, UnitDefinition
    from pint.facets.plain.definitions import DerivedDimensionDefinition
    from pint.util import UnitsContainer

    tiny.define(PrefixDefinition("mega", 1e6, "M"))
    tiny.define(DerivedDimensionDefinition("[area]", UnitsContainer({"[length]": 2})))
    tiny.define(
        UnitDefinition(
            "mybel", None, (), LogarithmicConverter(1, 10, 1), UnitsContainer()
        )
    )
    # named like a prefixed unit, but defined in code
    tiny.define(
        UnitDefinition(
            "kilofoot", None, (), ScaleConverter(1000), UnitsContainer(foot=1)
        )
    )
    tiny.define(
        UnitDefinition(
            "delta_minute", None, (), ScaleConverter(2), UnitsContainer(minute=1)
        )
    )
    tiny.define("@alias meter = metro")
    relation = BidirectionalRelation(
        UnitsContainer({"[length]": 1}),
        UnitsContainer({"[time]": 1}),
        "meter * second / value",
    )
    tiny.add_context(ContextDefinition("both", (), {}, (relation,), ()))

    reloaded, data = write_and_load(tiny, tmp_path)
    assert data["prefix"]["mega"] == {"value": "1000000.0", "defined_symbol": "M"}
    assert data["dimension"]["area"] == {"value": "[length] ** 2"}
    assert data["unit"]["mybel"]["value"] == "1; logbase: 10; logfactor: 1"
    assert data["unit"]["meter"]["aliases"] == ["metre", "metro"]
    assert reloaded.Quantity(2, "mybel").to("").magnitude == pytest.approx(100)
    assert reloaded.Quantity(1, "kilofoot").to("foot").magnitude == 1000
    assert reloaded.Quantity(1, "delta_minute").to("minute").magnitude == 2
    assert reloaded.Quantity(1, "metro").to("meter").magnitude == 1
    with reloaded.context("both"):
        assert reloaded.Quantity(4, "s").to("m").magnitude == 0.25


def test_warns_about_what_cannot_be_written(tiny, tmp_path):
    tiny.add_context(pint.Context("adhoc"))
    tiny.get_group("international").add_units("minute")
    tiny.get_group("extra").add_units("second")
    with pytest.warns(UserWarning) as record:
        write_and_load(tiny, tmp_path)
    message = str(record[0].message)
    assert "context adhoc" in message
    assert "group extra" in message
    assert record[0].filename == __file__


def test_alias_updates_definition(tiny):
    tiny.define("@alias meter = metro")
    meter = tiny._units["meter"]
    assert meter.aliases == ("metre", "metro")
    assert tiny._units["m"] is tiny._units["metro"] is meter


def test_aliases_of_prefixed_and_group_units(tiny, tmp_path):
    tiny.define("@alias kilometer = kilometro")
    tiny.define("@alias inch = pulgada")
    reloaded, data = write_and_load(tiny, tmp_path)
    assert data["alias"] == {"kilometer": {"aliases": ["kilometro"]}}
    assert "kilometer" not in data["unit"]
    (international,) = data["group"]
    assert international["definitions"]["inch"]["aliases"] == ["pulgada"]
    assert reloaded.Quantity(1, "kilometro").to("m").magnitude == 1000
    assert reloaded.Quantity(1, "km").to("m").magnitude == 1000
    assert reloaded.Quantity(1, "pulgada").to("inch").magnitude == 1


@pytest.mark.filterwarnings("error")
def test_empty_on_demand_system_not_reported(tiny, tmp_path):
    tiny.get_system("nothing")
    tiny.get_group("nothing_either")
    write_and_load(tiny, tmp_path)


def test_direct_write_warns_at_caller(tiny, tmp_path):
    pytest.importorskip("tomlkit")
    from pint.delegates.toml_parser import write_definitions

    tiny.add_context(pint.Context("adhoc"))
    with pytest.warns(UserWarning, match="context adhoc") as record:
        write_definitions(tmp_path / "out.toml", tiny)
    assert record[0].filename == __file__


def test_alias_on_delta_unit_reloads_cleanly(tiny, tmp_path):
    tiny.define("@alias delta_degree_Celsius = dC")
    pytest.importorskip("tomlkit")
    f = tmp_path / "out.toml"
    tiny.write_definitions(f)
    reloaded = pint.UnitRegistry(f, on_redefinition="raise")
    assert "delta_degree_Celsius" not in tomllib.loads(f.read_text("utf-8"))["unit"]
    assert reloaded.Quantity(1, "dC").to("K").magnitude == 1


def test_failed_alias_leaves_definition_unchanged(tmp_path):
    ureg = pint.UnitRegistry(
        ["canonical = [] = can = alias1", "other = [] = oth = alias2"],
        on_redefinition="raise",
    )
    with pytest.raises(pint.RedefinitionError):
        ureg.define("@alias canonical = alias2")
    assert ureg._units["canonical"].aliases == ("alias1",)
    assert ureg._units["alias2"].name == "other"


def test_alias_applies_to_context_redefinition(tiny):
    tiny.define("@alias minute = minuto")
    with tiny.context("fast"):
        assert tiny.Quantity(1, "minuto").to("s").magnitude == 30
    assert tiny.Quantity(1, "minuto").to("s").magnitude == 60


def test_write_without_tomlkit(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "tomlkit", None)
    with pytest.raises(ImportError, match="tomlkit"):
        pint.UnitRegistry(PINT_DIR / "default_en.toml").write_definitions(
            tmp_path / "out.toml"
        )


def test_disk_cache(tmp_path, monkeypatch):
    from pint.delegates.toml_parser import TomlParser

    f = tmp_path / "defs.toml"
    f.write_text('[unit]\nx = { value = "1234" }\n', encoding="utf-8")
    cache = tmp_path / "cache"

    calls = []
    parse_path = TomlParser._parse_path

    @functools.wraps(parse_path)
    def counting(self, path):
        calls.append(path)
        return parse_path(self, path)

    monkeypatch.setattr(TomlParser, "_parse_path", counting)

    first = pint.UnitRegistry(f, cache_folder=cache)
    # one file for the definitions, one for the RegistryCache
    assert len(tuple(cache.glob("*.pickle"))) == 2
    second = pint.UnitRegistry(f, cache_folder=cache)
    assert len(calls) == 1
    assert second._cache.root_units == first._cache.root_units
    assert second.x == 1234

    f.write_text('[unit]\nx = { value = "1235" }\n', encoding="utf-8")
    assert pint.UnitRegistry(f, cache_folder=cache).x == 1235
    assert len(calls) == 2
