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
import os
import subprocess
import pkg_resources
from .unit import UnitRegistry, DimensionalityError, UndefinedUnitError
from .util import formatter, pi_theorem, logger
from .measurement import Measurement
from .context import Context

_DEFAULT_REGISTRY = UnitRegistry()

__version__ = "unknown"
try:  # try to grab the commit version of our package
    __version__ = (subprocess.check_output(["git", "describe"],
                                           stderr=subprocess.STDOUT,
                                           cwd=os.path.dirname(os.path.abspath(__file__)))).strip()
except:  # on any error just try to grab the version that is installed on the system
    try:
        __version__ = pkg_resources.get_distribution('pint').version
    except:
        pass  # we seem to have a local copy without any repository control or installed without setuptools
              # so the reported version will be __unknown__  

def _build_quantity(value, units):
    return _DEFAULT_REGISTRY.Quantity(value, units)


def run_pyroma(data):
    import sys
    from zest.releaser.utils import ask
    if not ask("Run pyroma on the package before uploading?"):
        return
    try:
        from pyroma import run
        result = run(data['tagdir'])
        if result != 10:
            if not ask("Continue?"):
                sys.exit(1)
    except ImportError:
        if not ask("pyroma not available. Continue?"):
            sys.exit(1)
