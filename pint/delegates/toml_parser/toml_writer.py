from __future__ import annotations

import tomlkit
from tomlkit import comment
from tomlkit import document
from tomlkit import nl
from tomlkit import table


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
    if value == () or value is None or value == {}:
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
                name_ = definition.name
                if definition_type == "dimension":
                    name_ = definition.name[1:-1]  # remove the brackets
                data[definition_type][name_] = parse_simple_definition(
                    definition, definition_type
                )
    return data


def groups(ureg):
    group_data = []
    for group in ureg._group_definitions:
        dat = {'name': group.name}
        for attr in ["using_group_names"]:
            dat = add_key_if_not_empty(dat, attr, getattr(group, attr))
        dat["definitions"] = {}
        for definition in group.definitions:
            dat["definitions"][definition.name] = parse_simple_definition(
                definition, "unit"
            )
        group_data.append(dat)
    return group_data


def systems(ureg):
    system_data = []
    for group in ureg._system_definitions:
        dat = {'name': group.name}
        for attr in ["using_group_names"]:
            dat = add_key_if_not_empty(dat, attr, getattr(group, attr))
        dat["rules"] = []
        for rule in group.rules:
            dat["rules"].append(rule.raw)
        system_data.append(dat)
    return system_data


def contexts(ureg):
    context_data = []
    for group in ureg._context_definitions:
        dat = {'name': group.name}
        for attr in ["aliases", "defaults", "redefinitions"]:
            dat = add_key_if_not_empty(dat, attr, getattr(group, attr))
        dat["relations"] = []
        for rule in group.relations:
            dat["relations"].append(rule.raw)
        context_data.append(dat)
    return context_data

def to_inline_table(d):
    definition_item = tomlkit.table()
    for k, v in d.items():
        definition_item_item = tomlkit.inline_table()
        definition_item_item.update(v)
        definition_item.add(k, definition_item_item)
    return definition_item

def write_definitions(filename: str, ureg: GenericPlainRegistry):
    data = prefixes_units_dimensions(ureg)
    data["group"] = groups(ureg)
    data["system"] = systems(ureg)
    data["context"] = contexts(ureg)
    
    doc = document()
    for definition in ["prefix", "dimension", "unit",]:
        definition_table = to_inline_table(data[definition])

        doc.add(definition, definition_table)        
        doc.add(nl())

    for definition in ["group", "system", "context"]:
        multiline_definition_table = tomlkit.aot()
        for item in data[definition]:
            if "definitions" in item:
                item['definitions'] = to_inline_table(item['definitions'])
            if "relations" in item:
                definition_item_item = tomlkit.array().multiline(True)
                for vv in item['relations']:
                    definition_item_item.append(vv)
                item['relations'] = definition_item_item
            if "defaults" in item:
                definition_item_item = tomlkit.inline_table()
                definition_item_item.update(item['defaults'])
                item['defaults'] = definition_item_item
            multiline_definition_table.append(item)
        doc.add(definition, multiline_definition_table)
        
    text = tomlkit.dumps(doc)
    text=text.replace('\n[group.', '[group.')
    with open(filename, "wb") as fp:
        fp.write(text.encode("utf-8"))
        