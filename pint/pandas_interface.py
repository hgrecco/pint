# -*- coding: utf-8 -*-
"""
    pint.pandas_interface
    ~~~~~~~~~~~~~~~~~~~~~

    # I'm happy with both of these but need to check with Pint Authors...
    :copyright: 2016 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

# thanks to cyberpandas (https://github.com/ContinuumIO/cyberpandas)
# for giving an example of how to do this

import sys

import pint
import numpy as np
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.arrays import ExtensionArray

class PintType(ExtensionDtype):
    """
    Docs...

    """
    name = 'Pint'
    # This is definitely not the smart way to do this.
    # I just want to get access to the Quantity class but can't work out how.
    # Hence this is my hack.
    ureg = pint.UnitRegistry()
    type = ureg.Quantity
    kind = 'f'  # I think Quantity becomes a float if converted to np ndarray...
    na_value = np.nan  # avoiding numpy dependency with pandas doesn't make sense

    @classmethod
    def construct_from_string(cls, string):
        """
        Need to document properly

        I'm not really sure what this is meant to do

        I suspect you could do some really cool stuff here where it actually
        reads the string but also suspect that's probably already done
        somewhere, I just don't know where
        """
        if string == cls.name:
            return cls()

        try:
            value = float(string)  # this could have unexpected behaviour so might be a really bad idea...
            return cls(value)
        except ValueError:
            raise TypeError("Cannot construct a '{}' from "
                            "'{}'".format(cls, string))
        else:
            raise TypeError("Cannot construct a '{}' from "
                            "'{}'".format(cls, string))

    # not sure if we need construct_array_type...
    # @classmethod
    # def construct_array_type(cls):
    #     """Return the array type associated with this dtype
    #     Returns
    #     -------
    #     type
    #     """
    #     raise NotImplementedError


class PintArray(ExtensionArray):
    """
    Docs...
    """
    # __array_priority__ = 1000  # I have no idea what this does

    dtype = PintType()

    def __init__(self, values):
        self.data = self.dtype.type(values)

    @classmethod
    def _from_sequence(cls, scalars):
        return cls(scalars)

    @classmethod
    def _from_factorized(cls, values, original):
        return cls(values)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    @property
    def nbytes(self):
        # there must be a smarter way to do this...
        if isinstance(self.data, np.ndarray):
            return self.data.nbytes
        else:
            return sys.getsizeof(self.data)




