# -*- coding: utf-8 -*-
"""
    pint
    ~~~~

    Pint is Python module/package to define, operate and manipulate
    **physical quantities**: the product of a numerical value and a
    unit of measurement. It allows arithmetic operations between them
    and conversions from and to different units.

    :copyright: (c) 2012 by Hernan E. Grecco.
    :license: BSD, see LICENSE for more details.
"""

from __future__ import with_statement

from .unit import UnitRegistry, DimensionalityError, UndefinedUnitError
from .util import formatter, pi_theorem, logger
from .measurement import Measurement

_DEFAULT_REGISTRY = UnitRegistry()


def _build_quantity(value, units):
    return _DEFAULT_REGISTRY.Quantity(value, units)
