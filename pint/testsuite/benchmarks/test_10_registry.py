import pytest

import pathlib
from typing import Any

import pint

from operator import getitem

UNITS = ("meter", "kilometer", "second", "minute", "angstrom")

OTHER_UNITS = ("meter", "angstrom", "kilometer/second", "angstrom/minute")

ALL_VALUES = ("int", "float", "complex")


@pytest.fixture
def setup(registry_tiny) -> tuple[pint.UnitRegistry, dict[str, Any]]:
    data = {}
    data["int"] = 1
    data["float"] = 1.0
    data["complex"] = complex(1, 2)

    return registry_tiny, data


@pytest.fixture
def my_setup(setup) -> tuple[pint.UnitRegistry, dict[str, Any]]:
    ureg, data = setup
    for unit in UNITS + OTHER_UNITS:
        data["uc_%s" % unit] = pint.util.to_units_container(unit, ureg)
    return ureg, data


def test_build_cache(setup, benchmark):
    ureg, _ = setup
    benchmark(ureg._build_cache)


@pytest.mark.parametrize("key", UNITS)
def test_getattr(benchmark, setup, key):
    ureg, _ = setup
    benchmark(getattr, ureg, key)


@pytest.mark.parametrize("key", UNITS)
def test_getitem(benchmark, setup, key):
    ureg, _ = setup
    benchmark(getitem, ureg, key)


@pytest.mark.parametrize("key", UNITS)
def test_parse_unit_name(benchmark, setup, key):
    ureg, _ = setup
    benchmark(ureg.parse_unit_name, key)


@pytest.mark.parametrize("key", UNITS)
def test_parse_units(benchmark, setup, key):
    ureg, _ = setup
    benchmark(ureg.parse_units, key)


@pytest.mark.parametrize("key", UNITS)
def test_parse_expression(benchmark, setup, key):
    ureg, _ = setup
    benchmark(ureg.parse_expression, "1.0 " + key)


@pytest.mark.parametrize("unit", OTHER_UNITS)
def test_base_units(benchmark, setup, unit):
    ureg, _ = setup
    benchmark(ureg.get_base_units, unit)


@pytest.mark.parametrize("unit", OTHER_UNITS)
def test_to_units_container_registry(benchmark, setup, unit):
    ureg, _ = setup
    benchmark(pint.util.to_units_container, unit, ureg)


@pytest.mark.parametrize("unit", OTHER_UNITS)
def test_to_units_container_detached(benchmark, setup, unit):
    ureg, _ = setup
    benchmark(pint.util.to_units_container, unit, ureg)


@pytest.mark.parametrize(
    "key", (("uc_meter", "uc_kilometer"), ("uc_kilometer/second", "uc_angstrom/minute"))
)
def test_convert_from_uc(benchmark, my_setup, key):
    src, dst = key
    ureg, data = my_setup
    benchmark(ureg._convert, 1.0, data[src], data[dst])


def test_parse_math_expression(benchmark, my_setup):
    ureg, _ = my_setup
    benchmark(ureg.parse_expression, "3 + 5 * 2 + value", value=10)


# This code is duplicated with other benchmarks but simplify comparison


@pytest.fixture
def cache_folder(tmppath_factory: pathlib.Path):
    folder = tmppath_factory / "cache"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


@pytest.mark.parametrize("use_cache_folder", (None, True))
def test_load_definitions_stage_1(benchmark, cache_folder, use_cache_folder):
    """empty registry creation"""

    if use_cache_folder is True:
        use_cache_folder = cache_folder
    else:
        use_cache_folder = None
    benchmark(pint.UnitRegistry, None, cache_folder=use_cache_folder)


@pytest.mark.parametrize("use_cache_folder", (None, True))
def test_load_definitions_stage_2(benchmark, cache_folder, use_cache_folder):
    """empty registry creation + parsing default files + definition object loading"""

    if use_cache_folder is True:
        use_cache_folder = cache_folder
    else:
        use_cache_folder = None

    from pint import errors

    defpath = pathlib.Path(errors.__file__).parent / "default_en.txt"
    empty_registry = pint.UnitRegistry(None, cache_folder=use_cache_folder)
    benchmark(empty_registry.load_definitions, defpath, True)


@pytest.mark.parametrize("use_cache_folder", (None, True))
def test_load_definitions_stage_3(benchmark, cache_folder, use_cache_folder):
    """empty registry creation + parsing default files + definition object loading + cache building"""

    if use_cache_folder is True:
        use_cache_folder = cache_folder
    else:
        use_cache_folder = None

    from pint import errors

    defpath = pathlib.Path(errors.__file__).parent / "default_en.txt"
    empty_registry = pint.UnitRegistry(None, cache_folder=use_cache_folder)
    loaded_files = empty_registry.load_definitions(defpath, True)
    benchmark(empty_registry._build_cache, loaded_files)
