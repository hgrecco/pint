from __future__ import annotations

from dataclasses import fields

import tomli_w

from ...facets.plain import GenericPlainRegistry

keep_fields = [
    "value",
    "defined_symbol",
    "aliases",
]


def add_key_if_not_empty(dat, key, value):
    if value == () or value is None:
        return dat
    else:
        dat[key] = value
    return dat


def parse_simple_definition(definition, definition_type):
    attrs = [field.name for field in fields(definition) if field.name in keep_fields]
    dat = {}
    for attr in attrs:
        dat = add_key_if_not_empty(dat, attr, getattr(definition, attr))
    if definition_type in ["units", "dimensions", "prefixes"] and hasattr(
        definition, "raw"
    ):
        dat["value"] = definition.raw.split("=")[1].strip()
    return dat


def prefixes_units_dimensions(ureg):
    data = {
        "prefix": {},
        "unit": {},
        "dimension": {},
    }
    for definition_type, ureg_attr in zip(
        ["unit", "prefix", "dimension"],
        ["_units", "_prefixes", "_dimensions"],
    ):
        definitions = getattr(ureg, ureg_attr).values()
        for definition in definitions:
            for name, definition in getattr(ureg, ureg_attr).items():
                if hasattr(definition, "raw"):
                    data[definition_type][definition.name] = parse_simple_definition(
                        definition, definition_type
                    )
    return data


def groups(ureg):
    group_data = {}
    for group in ureg._group_definitions:
        dat = {}
        for attr in ["using_group_names"]:
            dat = add_key_if_not_empty(dat, attr, getattr(group, attr))
        dat["definitions"] = {}
        for definition in group.definitions:
            dat["definitions"][definition.name] = parse_simple_definition(
                definition, "_units"
            )
        group_data[group.name] = dat
    return group_data


def systems(ureg):
    system_data = {}
    for group in ureg._system_definitions:
        dat = {}
        for attr in ["using_group_names"]:
            dat = add_key_if_not_empty(dat, attr, getattr(group, attr))
        dat["rules"] = []
        for rule in group.rules:
            dat["rules"].append(rule.raw)
        system_data[group.name] = dat
    return system_data


def contexts(ureg):
    context_data = {}
    for group in ureg._context_definitions:
        dat = {}
        for attr in ["aliases", "defaults", "redefinitions"]:
            dat = add_key_if_not_empty(dat, attr, getattr(group, attr))
        dat["relations"] = []
        for rule in group.relations:
            dat["relations"].append(rule.raw)
        context_data[group.name] = dat
    return context_data


def write_definitions(filename: str, ureg: GenericPlainRegistry):
    data = prefixes_units_dimensions(ureg)
    data["group"] = groups(ureg)
    data["system"] = systems(ureg)
    data["context"] = contexts(ureg)
    with open("test.toml", "wb") as f:
        tomli_w.dump(data, f)
