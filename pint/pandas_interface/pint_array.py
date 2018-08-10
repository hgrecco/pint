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

# thoughts now, type of PintType should be taken from the default, initialised
# Registry. Then PintArray should use this by default, however if it's
# initialised with a Quantity then it should overwrite PintType's type by the
# Quantity's registry Quantity i.e. Quantity._REGISTRY.Quantity to make sure
# the registry is as expected

import copy

import numpy as np
from pandas.core.arrays import ExtensionArray
from pandas.core.arrays.base import ExtensionOpsMixin
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import (
    is_integer)
from pandas.core.dtypes.dtypes import registry

from ..quantity import _Quantity

class PintType(ExtensionDtype):
    type = _Quantity
    name = 'pint'

    @classmethod
    def construct_array_type(cls, type_str='pint'):
        if type_str is not cls.name:
            raise NotImplementedError
        return PintArray

    @classmethod
    def construct_from_string(cls, string):
        if string == cls.name:
            return cls()
        else:
            raise TypeError("Cannot construct a '{}' from "
                            "'{}'".format(cls, string))

def coerce_to_pint_quantity(values, dtype, copy=False):
    if dtype is not None:
        if not isinstance(dtype, PintType):
            raise ValueError("invalid dtype specified {}".format(dtype))

    if isinstance(values, _Quantity):
        return values

    a = []
    for v in values:
        a.append(dtype.type(v))
    import pdb
    pdb.set_trace()
    raise NotImplementedError("cannot make PintArray from "
                              "{}".format(type(values)))

    if not isinstance(values, _Quantity):
        raise NotImplementedError("cannot make PintArray from "
                                  "{}".format(type(values)))

    return values


class PintArray(ExtensionArray, ExtensionOpsMixin):
    _dtype = PintType

    def __init__(self, values, dtype=None, copy=False):
        if isinstance(values, list):
            tmp = self._from_sequence(values, dtype=dtype, copy=copy)
            self._data = tmp._data

        else:
            self._data = coerce_to_pint_quantity(
                values, dtype, copy=copy
            )

    def __getitem__(self, item):
        # type (Any) -> Any
        """Select a subset of self.
        Parameters
        ----------
        item : int, slice, or ndarray
            * int: The position in 'self' to get.
            * slice: A slice object, where 'start', 'stop', and 'step' are
              integers or None
            * ndarray: A 1-d boolean NumPy ndarray the same length as 'self'
        Returns
        -------
        item : scalar or ExtensionArray
        """
        if is_integer(item):
            return self._data[item]

        return type(self)(self._data[item])

    def __len__(self):
        # type: () -> int
        """Length of this array

        Returns
        -------
        length : int
        """
        return len(self._data)

    def __array__(self, dtype=None):
        return self._data.astype(object)

    def isna(self):
        # type: () -> np.ndarray
        """Return a Boolean NumPy array indicating if each value is missing.

        Returns
        -------
        missing : np.array
        """
        return np.isnan(self.data)

    def take(self, indices, allow_fill=False, fill_value=None):
        # type: (Sequence[int], bool, Optional[Any]) -> PintArray
        """Take elements from an array.
        Parameters
        ----------
        indices : sequence of integers
            Indices to be taken.
        allow_fill : bool, default False
            How to handle negative values in `indices`.
            * False: negative values in `indices` indicate positional indices
              from the right (the default). This is similar to
              :func:`numpy.take`.
            * True: negative values in `indices` indicate
              missing values. These values are set to `fill_value`. Any other
              other negative values raise a ``ValueError``.
        fill_value : any, optional
            Fill value to use for NA-indices when `allow_fill` is True.
            This may be ``None``, in which case the default NA value for
            the type, ``self.dtype.na_value``, is used.
        Returns
        -------
        ExtensionArray
        Raises
        ------
        IndexError
            When the indices are out of bounds for the array.
        ValueError
            When `indices` contains negative values other than ``-1``
            and `allow_fill` is True.
        Notes
        -----
        ExtensionArray.take is called by ``Series.__getitem__``, ``.loc``,
        ``iloc``, when `indices` is a sequence of values. Additionally,
        it's called by :meth:`Series.reindex`, or any other method
        that causes realignemnt, with a `fill_value`.
        See Also
        --------
        numpy.take
        pandas.api.extensions.take
        Examples
        --------
        """
        from pandas.core.algorithms import take

        data = self._data.magnitude
        if allow_fill and fill_value is None:
            fill_value = self.dtype.na_value
        if isinstance(fill_value, _Quantity):
            fill_value = fill_value.to(self._data).magnitude

        result = take(data, indices, fill_value=fill_value,
                      allow_fill=allow_fill)

        return type(self)(type(self._data)(result, self._data.units))

    def copy(self, deep=False):
        data = self._data
        if deep:
            data = copy.deepcopy(data)
        else:
            data = data.copy()

        return type(self)(data, dtype=self.dtype)

    @classmethod
    def _concat_same_type(cls, to_concat):
        raise NotImplementedError

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        """Construct a new PintArray from a sequence of single value
        (not array) quantities.
        Parameters
        ----------
        scalars : Sequence
            Each element must be an instance of the scalar type for this
            array.
        Returns
        -------
        PintArray
        """
        if dtype is not None:
            if not isinstance(dtype, cls._dtype):
                raise NotImplementedError

        for s in scalars:
            assert isinstance(s, cls._dtype.type)

        units = set(s.units for s in scalars)
        if len(units) > 1:
            raise TypeError("The units of all quantities are not the same.")

        magnitudes = [quantity.magnitude for quantity in scalars]

        return cls(type(scalars[0])(magnitudes, scalars[0].units))

    @property
    def dtype(self):
        # type: () -> ExtensionDtype
        """An instance of 'ExtensionDtype'."""
        return self._dtype()

    @property
    def data(self):
        return self._data

# register
registry.register(PintType)

