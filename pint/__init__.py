# -*- coding: utf-8 -*-
"""
    pint
    ~~~~

    Pint is Python module/package to define, operate and manipulate
    **physical quantities**: the product of a numerical value and a
    unit of measurement. It allows arithmetic operations between them
    and conversions from and to different units.

    :copyright: 2016 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""
from __future__ import with_statement


import pkg_resources
from .formatting import formatter
from .registry import (UnitRegistry, LazyRegistry)
from .errors import (DimensionalityError, OffsetUnitCalculusError,
                   UndefinedUnitError, UnitStrippedWarning)
from .util import pi_theorem, logger

from .context import Context

import sys
try:
    from pintpandas import PintType, PintArray
    _HAS_PINTPANDAS = True
except Exception:
    _HAS_PINTPANDAS = False
    _, _pintpandas_error, _ = sys.exc_info()

try:                # pragma: no cover
    __version__ = pkg_resources.get_distribution('pint').version
except:             # pragma: no cover
    # we seem to have a local copy not installed without setuptools
    # so the reported version will be unknown
    __version__ = "unknown"


#: A Registry with the default units and constants.
_DEFAULT_REGISTRY = LazyRegistry()

#: Registry used for unpickling operations.
_APP_REGISTRY = _DEFAULT_REGISTRY


def _build_quantity(value, units):
    """Build Quantity using the Application registry.
    Used only for unpickling operations.
    """
    from .unit import UnitsContainer

    global _APP_REGISTRY

    # Prefixed units are defined within the registry
    # on parsing (which does not happen here).
    # We need to make sure that this happens before using.
    if isinstance(units, UnitsContainer):
        for name in units.keys():
            _APP_REGISTRY.parse_units(name)

    return _APP_REGISTRY.Quantity(value, units)


def _build_unit(units):
    """Build Unit using the Application registry.
    Used only for unpickling operations.
    """
    from .unit import UnitsContainer

    global _APP_REGISTRY

    # Prefixed units are defined within the registry
    # on parsing (which does not happen here).
    # We need to make sure that this happens before using.
    if isinstance(units, UnitsContainer):
        for name in units.keys():
            _APP_REGISTRY.parse_units(name)

    return _APP_REGISTRY.Unit(units)


def set_application_registry(registry):
    """Set the application registry which is used for unpickling operations.

    :param registry: a UnitRegistry instance.
    """
    assert isinstance(registry, UnitRegistry)
    global _APP_REGISTRY
    logger.debug('Changing app registry from %r to %r.', _APP_REGISTRY, registry)
    _APP_REGISTRY = registry


def test():
    """Run all tests.

    :return: a :class:`unittest.TestResult` object
    """
    from .testsuite import run
    return run()
