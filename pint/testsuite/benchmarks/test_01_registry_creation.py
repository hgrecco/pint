import pytest

import pint


@pytest.mark.parametrize("args", [[(None,), tuple(), ("tiny",), ("", None)]])
def test_create_registry(benchmark, tiny_definition_file, args):
    if args[0] == "tiny":
        args = (tiny_definition_file, args[1])

    @benchmark
    def _():
        if len(args) == 2:
            pint.UnitRegistry(args[0], cache_folder=args[1])
        else:
            pint.UnitRegistry(*args)
