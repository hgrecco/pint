# pytest fixtures
from __future__ import annotations

import pathlib

import pytest

import pint

_TINY = """
yocto- = 1e-24 = y-
zepto- = 1e-21 = z-
atto- =  1e-18 = a-
femto- = 1e-15 = f-
pico- =  1e-12 = p-
nano- =  1e-9  = n-
micro- = 1e-6  = µ- = μ- = u-
milli- = 1e-3  = m-
centi- = 1e-2  = c-
deci- =  1e-1  = d-
deca- =  1e+1  = da- = deka-
hecto- = 1e2   = h-
kilo- =  1e3   = k-
mega- =  1e6   = M-
giga- =  1e9   = G-
tera- =  1e12  = T-
peta- =  1e15  = P-
exa- =   1e18  = E-
zetta- = 1e21  = Z-
yotta- = 1e24  = Y-

meter = [length] = m = metre
second = [time] = s = sec

angstrom = 1e-10 * meter = Å = ångström = Å
minute = 60 * second = min
"""


@pytest.fixture(scope="session")
def tmppath_factory(tmpdir_factory) -> pathlib.Path:
    tmp = tmpdir_factory.mktemp("pint")
    return pathlib.Path(tmp)


@pytest.fixture(scope="session")
def tiny_definition_file(tmppath_factory: pathlib.Path) -> pathlib.Path:
    folder = tmppath_factory / "definitions"
    folder.mkdir(exist_ok=True, parents=True)
    path = folder / "tiny.txt"
    if not path.exists():
        path.write_text(_TINY, encoding="utf-8")
    return path


@pytest.fixture
def registry_empty():
    return pint.UnitRegistry(None)


@pytest.fixture
def registry_tiny(tiny_definition_file: pathlib.Path):
    return pint.UnitRegistry(tiny_definition_file)


@pytest.fixture
def func_registry():
    return pint.UnitRegistry()


@pytest.fixture(scope="class")
def class_registry():
    """Only use for those test that do not modify the registry."""
    return pint.UnitRegistry()


@pytest.fixture(scope="module")
def module_registry():
    """Only use for those test that do not modify the registry."""
    return pint.UnitRegistry()


@pytest.fixture(scope="session")
def sess_registry():
    """Only use for those test that do not modify the registry."""
    return pint.UnitRegistry()


@pytest.fixture(scope="class")
def class_tiny_app_registry():
    ureg_bak = pint.get_application_registry()
    ureg = pint.UnitRegistry(None)
    ureg.define("foo = []")
    ureg.define("bar = foo / 2")
    pint.set_application_registry(ureg)
    assert pint.get_application_registry() is ureg
    yield ureg
    pint.set_application_registry(ureg_bak)
