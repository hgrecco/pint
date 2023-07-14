import pint


def test_create_empty_registry(benchmark):
    benchmark(
        pint.UnitRegistry,
    )


def test_create_tiny_registry(benchmark, tiny_definition_file):
    benchmark(pint.UnitRegistry, tiny_definition_file)


def test_create_default_registry(benchmark):
    benchmark(
        pint.UnitRegistry,
    )


def test_create_default_registry_no_cache(benchmark):
    benchmark(pint.UnitRegistry, cache_folder=None)
