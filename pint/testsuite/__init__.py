import doctest
import math
import os
import unittest
import warnings
from contextlib import contextmanager

from pint import UnitRegistry
from pint.testsuite.helpers import PintOutputChecker


class QuantityTestCase:
    kwargs = {}

    @classmethod
    def setup_class(cls):
        cls.ureg = UnitRegistry(**cls.kwargs)
        cls.Q_ = cls.ureg.Quantity
        cls.U_ = cls.ureg.Unit

    @classmethod
    def teardown_class(cls):
        cls.ureg = None
        cls.Q_ = None
        cls.U_ = None


@contextmanager
def assert_no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        yield


def testsuite():
    """A testsuite that has all the pint tests."""
    suite = unittest.TestLoader().discover(os.path.dirname(__file__))
    from pint.compat import HAS_NUMPY, HAS_UNCERTAINTIES

    # TESTING THE DOCUMENTATION requires pyyaml, serialize, numpy and uncertainties
    if HAS_NUMPY and HAS_UNCERTAINTIES:
        try:
            import serialize  # noqa: F401
            import yaml  # noqa: F401

            add_docs(suite)
        except ImportError:
            pass
    return suite


def main():
    """Runs the testsuite as command line application."""
    try:
        unittest.main()
    except Exception as e:
        print("Error: %s" % e)


def run():
    """Run all tests.

    :return: a :class:`unittest.TestResult` object

    Parameters
    ----------

    Returns
    -------

    """
    test_runner = unittest.TextTestRunner()
    return test_runner.run(testsuite())


_GLOBS = {
    "wrapping.rst": {
        "pendulum_period": lambda length: 2 * math.pi * math.sqrt(length / 9.806650),
        "pendulum_period2": lambda length, swing_amplitude: 1.0,
        "pendulum_period_maxspeed": lambda length, swing_amplitude: (1.0, 2.0),
        "pendulum_period_error": lambda length: (1.0, False),
    }
}


def add_docs(suite):
    """Add docs to suite

    Parameters
    ----------
    suite :


    Returns
    -------

    """
    docpath = os.path.join(os.path.dirname(__file__), "..", "..", "docs")
    docpath = os.path.abspath(docpath)
    if os.path.exists(docpath):
        checker = PintOutputChecker()
        for name in (name for name in os.listdir(docpath) if name.endswith(".rst")):
            file = os.path.join(docpath, name)
            suite.addTest(
                doctest.DocFileSuite(
                    file,
                    module_relative=False,
                    checker=checker,
                    globs=_GLOBS.get(name, None),
                )
            )


def test_docs():
    suite = unittest.TestSuite()
    add_docs(suite)
    runner = unittest.TextTestRunner()
    return runner.run(suite)
