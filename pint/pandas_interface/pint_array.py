# -*- coding: utf-8 -*-
"""
    pint.pandas_interface.pint_array
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # I'm happy with both of these but need to check with Pint Authors...
    :copyright: 2016 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

# ok plan here:
# - can run the tests with python setup.py test to make sure everything still passes
# - can run the pandas interface tests with pytest -x --pdb pint/testsuite/test_pandas_interface.py

# - I'll use IntegerArray as my base https://github.com/pandas-dev/pandas/blob/master/pandas/core/arrays/integer.py
# - I'll add as few methods as possible to pass the pandas test
# - each time I add a method I'll add it with NotImplementedError first to make sure I can see where it's being called
# - then I can add the functionality bit by bit and keep some track of what is going on
# - other resources I can use
# - cyberpandas https://github.com/ContinuumIO/cyberpandas/blob/468644bcbdc9320a1a33b0df393d4fa4bef57dd7/cyberpandas/ip_array.py
# - pandas ExtensionDtype source https://github.com/pandas-dev/pandas/blob/master/pandas/core/dtypes/base.py
# - pandas ExtensionArray source https://github.com/pandas-dev/pandas/blob/master/pandas/core/arrays/base.py

class PintType(object):
    def __init__(self):
        raise NotImplementedError

class PintArray(object):
    def __init__(self):
        raise NotImplementedError
