import sys

if sys.version_info < (2, 7):
    try:
        import unittest2 as unittest
    except ImportError:
        raise Exception("Testing Pint in Python 2.6 requires package 'unittest2'")
else:
    import unittest