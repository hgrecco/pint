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
from .formatting import formatter
from .unit import (UnitRegistry, DimensionalityError, OffsetUnitCalculusError,
                   UndefinedUnitError, LazyRegistry)
from .util import pi_theorem, logger

from .context import Context


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
    global _APP_REGISTRY
    return _APP_REGISTRY.Quantity(value, units)


def set_application_registry(registry):
    """Set the application registry which is used for unpickling operations.

    :param registry: a UnitRegistry instance.
    """
    assert isinstance(registry, UnitRegistry)
    global _APP_REGISTRY
    logger.debug('Changing app registry from %r to %r.', _APP_REGISTRY, registry)
    _APP_REGISTRY = registry


def _run_pyroma(data):   # pragma: no cover
    """Run pyroma (used to perform checks before releasing a new version).
    """
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


def _check_travis(data):   # pragma: no cover
    """Check if Travis reports that everything is ok.
    (used to perform checks before releasing a new version).
    """
    import json
    import sys

    from zest.releaser.utils import system, ask
    if not ask('Check with Travis before releasing?'):
        return

    try:
        # Python 3
        from urllib.request import urlopen
        def get(url):
            return urlopen(url).read().decode('utf-8')

    except ImportError:
        # Python 2
        from urllib2 import urlopen
        def get(url):
            return urlopen(url).read()

    url = 'https://api.github.com/repos/%s/%s/status/%s'

    username = 'hgrecco'
    repo = 'pint'
    commit = system('git rev-parse HEAD')

    try:
        result = json.loads(get(url % (username, repo, commit)))['state']
        print('Travis says: %s' % result)
        if result != 'success':
            if not ask('Do you want to continue anyway?', default=False):
                sys.exit(1)
    except Exception:
        print('Could not determine the commit state with Travis.')
        if ask('Do you want to continue anyway?', default=False):
            sys.exit(1)


def test():
    """Run all tests.

    :return: a :class:`unittest.TestResult` object
    """
    from .testsuite import run
    return run()
