"""
pint.delegates.toml_parser.toml_writer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Write the definitions of a registry to a TOML file. Requires tomlkit.

Only definitions that keep their source text (``value_text``) are written,
so units derived by the registry (prefixed units, delta units) are skipped.

:copyright: 2025 by Pint Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...facets.plain import GenericPlainRegistry


def _unique(definitions) -> dict[str, Any]:
    # registry dicts map aliases and symbols to the same definition
    return {d.name: d for d in definitions if d.value_text}


def _definition(definition) -> dict[str, Any]:
    d: dict[str, Any] = {"value": definition.value_text}
    if getattr(definition, "defined_symbol", None):
        d["defined_symbol"] = definition.defined_symbol
    if getattr(definition, "aliases", ()):
        d["aliases"] = list(definition.aliases)
    return d


def _number(value):
    return value if isinstance(value, (int, float)) else str(value)


def to_dict(ureg: GenericPlainRegistry) -> dict[str, Any]:
    """Collect the definitions of a registry as plain TOML-ready data."""
    groups = getattr(ureg, "_group_definitions", [])
    systems = getattr(ureg, "_system_definitions", [])
    contexts = getattr(ureg, "_context_definitions", [])
    group_units = {u.name for g in groups for u in g.definitions}

    data: dict[str, Any] = {}
    if ureg._defaults:
        data["defaults"] = dict(ureg._defaults)
    data["prefix"] = {
        name: _definition(d) for name, d in _unique(ureg._prefixes.values()).items()
    }
    data["dimension"] = {
        name[1:-1]: {"value": d.value_text}
        for name, d in _unique(ureg._dimensions.values()).items()
        if not d.is_base
    }
    data["unit"] = {
        name: _definition(d)
        for name, d in _unique(ureg._units.values()).items()
        if name not in group_units
    }
    data["group"] = [
        {
            "name": g.name,
            "using_group_names": list(g.using_group_names),
            "definitions": {u.name: _definition(u) for u in g.definitions},
        }
        for g in groups
    ]
    data["system"] = [
        {
            "name": s.name,
            "using_group_names": list(s.using_group_names),
            "rules": [
                f"{r.new_unit_name}: {r.old_unit_name}"
                if r.old_unit_name
                else r.new_unit_name
                for r in s.rules
            ],
        }
        for s in systems
    ]
    data["context"] = [
        {
            "name": c.name,
            "aliases": list(c.aliases),
            "defaults": {k: _number(v) for k, v in c.defaults.items()},
            "relations": [r.value_text for r in c.relations],
            "redefinitions": {u.name: _definition(u) for u in c.redefinitions},
        }
        for c in contexts
    ]
    return data


def write_definitions(filename: str | pathlib.Path, ureg: GenericPlainRegistry):
    try:
        import tomlkit
    except ImportError:
        raise ImportError("tomlkit is required to write TOML definitions") from None

    def inline(d):
        t = tomlkit.inline_table()
        t.update(d)
        return t

    def table_of_inline(d):
        t = tomlkit.table()
        for k, v in d.items():
            t.add(k, inline(v))
        return t

    def multiline(values):
        a = tomlkit.array()
        a.extend(values)
        return a.multiline(len(values) > 1)

    data = to_dict(ureg)
    doc = tomlkit.document()
    if "defaults" in data:
        doc.add("defaults", data["defaults"])
    for key in ("prefix", "dimension", "unit"):
        doc.add(key, table_of_inline(data[key]))

    for key in ("group", "system", "context"):
        aot = tomlkit.aot()
        for item in data[key]:
            t = tomlkit.table()
            for k, v in item.items():
                if not v:
                    continue
                if k in ("definitions", "redefinitions"):
                    t.add(k, table_of_inline(v))
                elif k == "defaults":
                    t.add(k, inline(v))
                elif isinstance(v, list):
                    t.add(k, multiline(v))
                else:
                    t.add(k, v)
            aot.append(t)
        doc.add(key, aot)

    pathlib.Path(filename).write_text(tomlkit.dumps(doc), encoding="utf-8")
