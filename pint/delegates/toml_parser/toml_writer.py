"""
pint.delegates.toml_parser.toml_writer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Write the definitions of a registry to a TOML file. Requires tomlkit.

Values are written from their source text (``value_text``), or rebuilt for
definitions made in code. Units derived by the registry (prefixed units,
delta units) and the overlays of active contexts are not written.

:copyright: 2025 by Pint Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import pathlib
import warnings
from collections import ChainMap
from typing import TYPE_CHECKING, Any

from ...facets.context import BidirectionalRelation
from ...facets.nonmultiplicative.definitions import (
    LogarithmicConverter,
    OffsetConverter,
)

if TYPE_CHECKING:
    from ...facets.plain import GenericPlainRegistry


def _unit_value(d) -> str:
    """Rebuild the value of a unit defined in code, without source text."""
    if d.is_base:
        text = str(d.reference) if d.reference else "[]"
    else:
        scale = d.converter.scale
        parts = ([] if scale == 1 else [str(scale)]) + (
            [str(d.reference)] if d.reference else []
        )
        text = " * ".join(parts) or "1"
    c = d.converter
    if isinstance(c, LogarithmicConverter):
        text += f"; logbase: {c.logbase}; logfactor: {c.logfactor}"
    elif isinstance(c, OffsetConverter):
        text += f"; offset: {c.offset}"
    return text


def _relation_value(r) -> str:
    if r.value_text:
        return r.value_text
    separator = "<->" if isinstance(r, BidirectionalRelation) else "->"
    return f"{r.src} {separator} {r.dst}: {r.equation}"


def _definition(definition, value: str | None = None) -> dict[str, Any]:
    d: dict[str, Any] = {"value": definition.value_text or value}
    if getattr(definition, "defined_symbol", None):
        d["defined_symbol"] = definition.defined_symbol
    if getattr(definition, "aliases", ()):
        d["aliases"] = list(definition.aliases)
    return d


def _unit(d, extra_aliases=()) -> dict[str, Any]:
    out = _definition(d, d.value_text or _unit_value(d))
    if extra_aliases:
        out["aliases"] = out.get("aliases", []) + list(extra_aliases)
    return out


def _generated_by_registry(ureg, d) -> bool:
    """Prefixed units cached by get_name, and delta units of offset units."""
    if d.value_text:
        return False
    if d.name.startswith("delta_"):
        base = ureg._units.get(d.name[6:])
        if base is not None and not base.is_multiplicative:
            return d.reference == base.reference
    # get_name reuses the prefix's converter object
    return any(
        d.converter is p.converter and d.name.startswith(p.name)
        for p in ureg._prefixes.values()
        if p.name
    )


def _warn_unwritten(ureg, groups, systems, contexts) -> None:
    """Warn about groups, systems and contexts that were not added from
    definitions (or were changed afterwards), as they cannot be written."""
    skipped = []
    defined_groups = {g.name: g for g in groups}
    for name, group in getattr(ureg, "_groups", {}).items():
        gd = defined_groups.get(name)
        # these also collect the units defined outside any group
        if name in ("root", ureg._defaults.get("group")):
            continue
        if gd is None:
            # empty groups are created on demand, e.g. by `using`
            if group._unit_names or group._used_groups:
                skipped.append(f"group {name}")
        elif group.non_inherited_unit_names != set(gd.unit_names) or (
            group._used_groups != set(gd.using_group_names)
        ):
            skipped.append(f"changes to group {name}")
    defined_systems = {s.name for s in systems}
    skipped += [
        f"system {name}"
        for name in getattr(ureg, "_systems", {})
        if name not in defined_systems
    ]
    defined_contexts = {c.name for c in contexts}
    skipped += [
        f"context {name}"
        for name in dict.fromkeys(
            c.name for c in getattr(ureg, "_contexts", {}).values()
        )
        if name not in defined_contexts
    ]
    if skipped:
        warnings.warn(
            "Only definitions can be written to TOML; skipping " + ", ".join(skipped),
            stacklevel=5,
        )


def _number(value):
    return value if isinstance(value, (int, float)) else str(value)


def to_dict(ureg: GenericPlainRegistry) -> dict[str, Any]:
    """Collect the definitions of a registry as plain TOML-ready data."""
    groups = getattr(ureg, "_group_definitions", [])
    systems = getattr(ureg, "_system_definitions", [])
    contexts = list(getattr(ureg, "_context_definitions", {}).values())
    group_units = {u.name for g in groups for u in g.definitions}
    _warn_unwritten(ureg, groups, systems, contexts)

    # skip the overlays of active contexts; aliases map to the same definition
    units = ureg._units.maps[-1] if isinstance(ureg._units, ChainMap) else ureg._units
    # aliases added with @alias are only keys in the registry
    extra_aliases: dict[str, list[str]] = {}
    for key, d in units.items():
        if key not in (d.name, d.symbol) and key not in d.aliases:
            extra_aliases.setdefault(d.name, []).append(key)
    units = {d.name: d for d in units.values()}
    prefixes = {d.name: d for d in ureg._prefixes.values() if d.name}
    dimensions = {d.name: d for d in ureg._dimensions.values() if not d.is_base}

    data: dict[str, Any] = {}
    if ureg._defaults:
        data["defaults"] = dict(ureg._defaults)
    data["prefix"] = {
        name: _definition(d, str(d.value)) for name, d in prefixes.items()
    }
    data["dimension"] = {
        name[1:-1]: {"value": d.value_text or str(d.reference)}
        for name, d in dimensions.items()
    }
    data["unit"] = {
        name: _unit(d, extra_aliases.get(name, ()))
        for name, d in units.items()
        if name not in group_units and not _generated_by_registry(ureg, d)
    }
    data["group"] = [
        {
            "name": g.name,
            "using_group_names": list(g.using_group_names),
            "definitions": {
                u.name: _unit(u, extra_aliases.get(u.name, ())) for u in g.definitions
            },
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
            "relations": [_relation_value(r) for r in c.relations],
            "redefinitions": {u.name: _unit(u) for u in c.redefinitions},
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
