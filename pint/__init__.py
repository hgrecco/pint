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
import pkg_resources
import subprocess
import sys

class Pint(object):
    from .context import Context
    from .measurement import Measurement
    from .unit import UnitRegistry, DimensionalityError, UndefinedUnitError
    from .util import formatter, pi_theorem, logger

    __version__ = "unknown"

    try:
        # try to grab the commit version of our package
        __version__ = (
          subprocess.check_output(["git", "describe"],
                                  stderr=subprocess.STDOUT,
                                  cwd=os.path.dirname(os.path.abspath(__file__)))).strip()
    except:
        # on any error just try to grab the version that is installed on the system
        try:
            __version__ = pkg_resources.get_distribution('pint').version
        except:
            # we seem to have a local copy without any repository control or installed without setuptool
            # so the reported version will be __unknown__

            pass

    def _build_quantity(self, value, units):
        return _DEFAULT_REGISTRY.Quantity(value, units)

    def run_pyroma(self, data):
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

    def __getattr__(self, name):
        if name != 'Registry':
            raise AttributeError('Module pint has no attribute ' + name)
        self.Registry = self.UnitRegistry()
        return self.Registry

# See http://stackoverflow.com/a/7668273/43839
sys.modules[__name__] = Pint()
