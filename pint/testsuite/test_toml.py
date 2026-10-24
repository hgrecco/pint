from __future__ import annotations

import pathlib

import pytest

import pint
from pint.errors import DefinitionSyntaxError

PINT_DIR = pathlib.Path(pint.__file__).parent


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


def test_sections_are_optional(tmp_path):
    f = tmp_path / "defs.toml"
    f.write_text('[unit]\nmeter = { value = "[length]" }\n', encoding="utf-8")
    ureg = pint.UnitRegistry(f)
    assert ureg.Quantity(1, "meter").dimensionality == {"[length]": 1}


@pytest.mark.parametrize(
    "content",
    [
        '[units]\nmeter = { value = "[length]" }\n',
        "[unit]\nmeter = {}\n",
        '[prefix]\nkilo = { value = "meter" }\n',
    ],
)
def test_syntax_errors_name_the_file(tmp_path, content):
    f = tmp_path / "defs.toml"
    f.write_text(content, encoding="utf-8")
    with pytest.raises(DefinitionSyntaxError, match="defs.toml"):
        pint.UnitRegistry(f)
