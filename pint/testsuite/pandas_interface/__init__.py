import unittest

from pint.compat import pd, pytest

def setUpModule():
    print("Hit setupmodule")
    try:
        import pint.pandas_interface as ppi
    except:
        if (pytest is None) and (pd is None):
            missing_msg = (
                "pytest and the right pandas version are not available, check the docs"
            )
        elif pytest is None:
            missing_msg = "pytest is not available"
        elif pd is None:
            missing_msg = "the right pandas version is not available, check the docs"

        raise unittest.SkipTest(missing_msg)
