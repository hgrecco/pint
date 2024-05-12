from __future__ import annotations

import decimal
import pickle
import time

import flexparser as fp
import pytest

import pint
from pint.facets.plain import UnitDefinition

FS_SLEEP = 0.010


from .helpers import internal


@pytest.fixture
def float_cache_filename(tmp_path):
    ureg = pint.UnitRegistry(cache_folder=tmp_path / "cache_with_float")
    assert internal(ureg)._diskcache
    assert internal(ureg)._diskcache.cache_folder

    return tuple(internal(ureg)._diskcache.cache_folder.glob("*.pickle"))


def test_must_be_three_files(float_cache_filename):
    # There should be 3 files here:
    # - cached defaults_en.txt
    # - cached constants_en.txt
    # - cached RegistryCache
    assert len(float_cache_filename) == 3, float_cache_filename


def test_no_cache():
    ureg = pint.UnitRegistry(cache_folder=None)
    assert internal(ureg)._diskcache is None
    assert ureg.cache_folder is None


def test_decimal(tmp_path, float_cache_filename):
    ureg = pint.UnitRegistry(
        cache_folder=tmp_path / "cache_with_decimal", non_int_type=decimal.Decimal
    )
    assert internal(ureg)._diskcache
    assert internal(ureg)._diskcache.cache_folder == tmp_path / "cache_with_decimal"
    assert ureg.cache_folder == tmp_path / "cache_with_decimal"

    files = tuple(internal(ureg)._diskcache.cache_folder.glob("*.pickle"))
    assert len(files) == 3

    # check that the filenames with decimal are different to the ones with float
    float_filenames = tuple(p.name for p in float_cache_filename)
    for p in files:
        assert p.name not in float_filenames

    for p in files:
        with p.open(mode="rb") as fi:
            obj = pickle.load(fi)
            if not isinstance(obj, fp.ParsedSource):
                continue
            for definition in obj.parsed_source.filter_by(UnitDefinition):
                if definition.name == "pi":
                    assert isinstance(definition.converter.scale, decimal.Decimal)
                    return
    assert False


def test_auto(float_cache_filename):
    float_filenames = tuple(p.name for p in float_cache_filename)

    ureg = pint.UnitRegistry(cache_folder=":auto:")
    assert internal(ureg)._diskcache
    assert internal(ureg)._diskcache.cache_folder
    auto_files = tuple(
        p.name for p in internal(ureg)._diskcache.cache_folder.glob("*.pickle")
    )
    for file in float_filenames:
        assert file in auto_files


def test_change_file(tmp_path):
    # Generate a definition file
    dfile = tmp_path / "definitions.txt"
    dfile.write_text("x = 1234")

    # Load the definition file
    # (this will create two cache files, one for the file another for RegistryCache)
    ureg = pint.UnitRegistry(dfile, cache_folder=tmp_path)
    assert ureg.x == 1234
    files = tuple(internal(ureg)._diskcache.cache_folder.glob("*.pickle"))
    assert len(files) == 2

    # Modify the definition file
    # Add some sleep to make sure that the time stamp difference is signficant.
    time.sleep(FS_SLEEP)
    dfile.write_text("x = 1235")

    # Verify that the definiton file was loaded (the cache was invalidated).
    ureg = pint.UnitRegistry(dfile, cache_folder=tmp_path)
    assert ureg.x == 1235
    files = tuple(internal(ureg)._diskcache.cache_folder.glob("*.pickle"))
    assert len(files) == 4
