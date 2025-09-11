from __future__ import annotations

from dataclasses import fields

from ...compat import tomli_w

from ...facets.plain import GenericPlainRegistry
from ...facets.plain.definitions import DimensionDefinition
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
    if definition_type in ["unit", "dimension", "prefix"] and hasattr(
        definition, "value_text"
    ):
        dat["value"] = definition.value_text
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
                if (isinstance(definition, DimensionDefinition) and definition.is_base) or definition.value_text == "":
                    # skip base dimensions
                    continue
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
            try: print(definition)
            except: pass
            dat["definitions"][definition.name] = parse_simple_definition(
                definition, "unit"
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
    with open(filename, "wb") as f:
        tomli_w.dump(data, f)


import tomlkit
from tomlkit import comment
from tomlkit import document
from tomlkit import nl
from tomlkit import table

def write_definitions(filename: str, ureg: GenericPlainRegistry):
    data = prefixes_units_dimensions(ureg)
    data["group"] = groups(ureg)
    data["system"] = systems(ureg)
    data["context"] = contexts(ureg)
    
    doc = document()
    for definition in ["prefix", "dimension", "unit",]:
        definition_table = tomlkit.table()
        for key, value in data[definition].items():
            definition_item = tomlkit.inline_table()
            definition_item.update(value)
            definition_table.add(key, definition_item)

        doc.add(definition, definition_table)        
        doc.add(nl())

    for definition in ["group", "system", "context"]:
        definition_table = tomlkit.table()
        for key, value in data[definition].items():
            definition_item = tomlkit.table()
            for k, v in value.items():
                if k == "definitions":
                    definition_item_item = tomlkit.table()
                    for kk, vv in v.items():
                        definition_item_item_item = tomlkit.inline_table()
                        definition_item_item_item.update(vv)
                        definition_item_item.add(kk, definition_item_item_item)
                    definition_item.add(k, definition_item_item)
                    continue
                elif k == "relations":
                    definition_item_item = tomlkit.array()
                    for vv in v:
                        definition_item_item.add_line(vv)
                    definition_item.add(k, definition_item_item)
                    continue
                else:
                    definition_item_item = tomlkit.inline_table()
                if isinstance(v, (list,tuple)):
                    definition_item.add(k, v)
                else:
                    definition_item_item.update(v)
                    definition_item.add(k, definition_item_item)
                    
                # 
                # definition_item_item.update(v)
                # definition_item.add(k, definition_item_item)
            definition_table.add(key, definition_item)
        doc.add(definition, definition_table)
        doc.add(nl())


    text = tomlkit.dumps(doc)
    with open(filename, "wb") as fp:
        fp.write(text.encode("utf-8"))
        