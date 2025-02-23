from __future__ import annotations

import pint


def test_create_empty_registry(benchmark):
    benchmark(pint.UnitRegistry, None)


def test_create_tiny_registry(benchmark, tiny_definition_file):
    benchmark(pint.UnitRegistry, tiny_definition_file)


def test_create_default_registry(benchmark):
    benchmark(
        pint.UnitRegistry,
        cache_folder=None,
    )


def test_create_default_registry_use_cache(benchmark, tmppath_factory):
    folder = tmppath_factory / "cache01"
    pint.UnitRegistry(cache_folder=tmppath_factory / "cache01")
    benchmark(pint.UnitRegistry, cache_folder=folder)
