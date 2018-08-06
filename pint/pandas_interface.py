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
import math

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
    na_value = math.nan

    @property
    def kind(self):
        return np.array(self.type).dtype.kind

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
    def shape(self):
        return (len(self.data),)

    @property
    def nbytes(self):
        if isinstance(self.data, np.ndarray):
            return self.data.nbytes
        # there must be a smarter way to do this that avoids sys...
        else:
            return sys.getsizeof(self.data)

    def isna(self):
        return np.isnan(self.data.magnitude)  # I can't see how to do this without numpy unless I use a loop and math.isnan() ...

    def take(self, indices, allow_fill=False, fill_value=None):
       from pandas.core.algorithms import take

       data = self.data
       if allow_fill and fill_value is None:
           fill_value = self.dtype.na_value

       result = take(data, indices, fill_value=fill_value,
                     allow_fill=allow_fill)
       return self._from_sequence(result)

    def copy(self, deep=False):
        if deep:
            import copy
            return type(self)(copy.deepcopy(self.data))  # no idea if this is required...
        else:
            return type(self)(self.data.copy())

    @classmethod
    def _concat_same_type(cls, to_concat):
        return cls(np.concatenate([array.data for array in to_concat]))  # don't know how to do this without numpy either

    def __setitem__(self, key, value):
        self.data[key] = value

    def tolist(self):
        return self.data.tolist()

    def argsort(self, axis=-1, kind='quicksort', order=None):
        return self.data.argsort()

    def unique(self):
        return type(self)(np.unique(self.data))

    def _formatting_values(self):
        return self.data
