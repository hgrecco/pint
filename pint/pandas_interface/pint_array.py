# -*- coding: utf-8 -*-
"""
    pint.pandas_interface.pint_array
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # I'm happy with both of these as long as Andrew and Zebedee are added on
    # but need to check with Pint Authors...
    :copyright: 2018 by Pint Authors, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""

from pint.compat import HAS_PROPER_PANDAS
if not HAS_PROPER_PANDAS:
    error_msg = (
        "Pint's Pandas interface requires that the latest version of "
        "Pandas is installed from Pandas' master branch"
    )
    raise ImportError(error_msg)

import copy
import warnings

import numpy as np

from pandas.core import ops
from pandas.core.arrays import ExtensionArray
from pandas.api.extensions import register_dataframe_accessor, register_series_accessor
from pandas.core.arrays.base import ExtensionOpsMixin
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import (
    is_integer, is_scalar,
    is_list_like,
    is_bool)
from pandas.core.dtypes.dtypes import registry
from pandas.compat import u, set_function_name
from pandas.io.formats.printing import (
    format_object_summary, format_object_attrs, default_pprint)
from pandas import Series, DataFrame

from ..quantity import build_quantity_class, _Quantity
from ..compat import string_types
from .. import _DEFAULT_REGISTRY

class PintType(ExtensionDtype):
    # I think this is the way to build a Quantity class and force it to be a
    # numpy array
    type = build_quantity_class(_DEFAULT_REGISTRY, force_ndarray=True)
    # # AS: I'm not sure that does force it as an ndarray.
    # # Trying the below as running into registry issues
    # type = _Quantity
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


class PintArray(ExtensionArray, ExtensionOpsMixin):
    _dtype = PintType

    def __init__(self, values, dtype=None, copy=False):
        if isinstance(values, _Quantity):
            self._dtype.type = type(values)
            assert self._dtype.type._REGISTRY == values._REGISTRY
        self._data = self._coerce_to_pint_array(values, dtype=dtype, copy=copy)

    def _coerce_to_pint_array(self, values, dtype=None, copy=False):
        if isinstance(values, self._dtype.type):
            return values

        if is_list_like(values):
            if all(is_bool(v) for v in values):
                # known bug in pint https://github.com/hgrecco/pint/issues/673
                raise TypeError("Invalid magnitude for {}: {}"
                                "".format(self._dtype.type, values))

            for i, v in enumerate(values):
                if isinstance(v, self._dtype.type):
                    continue
                else:
                    values[i] = v * self._find_first_unit(values)

            units = set(v.units for v in values)
            if len(units) > 1:
                # need to work out a way to test this
                raise TypeError("The units of all quantities are not the same"
                                " for input {}".format(values))

            magnitudes = [v.magnitude for v in values]

            return self._dtype.type(magnitudes, values[0].units)

        raise NotImplementedError

    def _find_first_unit(self, values):
        for v in values:
            if isinstance(v, self._dtype.type):
                return v.units

        return self._dtype.type(1).units

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
        item : scalar or PintArray
        """
        if is_integer(item):
            return self._data[item]

        return self.__class__(self._data[item])

    def __len__(self):
        # type: () -> int
        """Length of this array

        Returns
        -------
        length : int
        """
        return len(self._data)

    def __repr__(self):
        """
        Return a string representation for this object.

        Invoked by unicode(df) in py2 only. Yields a Unicode String in both
        py2/py3.
        """

        klass = self.__class__.__name__
        data = format_object_summary(self, default_pprint, False)
        attrs = format_object_attrs(self)
        space = " "

        prepr = (u(",%s") %
                 space).join(u("%s=%s") % (k, v) for k, v in attrs)

        res = u("%s(%s%s)") % (klass, data, prepr)

        return res

    def __array__(self, dtype=None, copy=False):
    # this is necessary for some pandas operations, eg transpose
    # note, units will be lost
        if dtype is None:
            dtype = object
        if isinstance(dtype, string_types):
            dtype = getattr(np, dtype)
        # it seems impossible to avoid using this, even dtype is object causes
        # failure...
        if dtype == object:
            return np.array(list(self._data), dtype = dtype, copy = copy)
        if not isinstance(dtype, np.dtype):
            list_of_converteds = [dtype(item) for item in self._data]
        else:
            list_of_converteds = [dtype.type(item) for item in self._data]

        return np.array(list_of_converteds)

    def isna(self):
        # type: () -> np.ndarray
        """Return a Boolean NumPy array indicating if each value is missing.

        Returns
        -------
        missing : np.array
        """
        return np.isnan(self._data.magnitude)

    def astype(self, dtype, copy=True):
        """Cast to a NumPy array with 'dtype'.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.
        copy : bool, default True
            Whether to copy the data, even if not necessary. If False,
            a copy is made only if the old dtype does not match the
            new dtype.

        Returns
        -------
        array : ndarray
            NumPy ndarray with 'dtype' for its dtype.
        """
        return self.__array__(dtype,copy)

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
        PintArray
        Raises
        ------
        IndexError
            When the indices are out of bounds for the array.
        ValueError
            When `indices` contains negative values other than ``-1``
            and `allow_fill` is True.
        Notes
        -----
        PintArray.take is called by ``Series.__getitem__``, ``.loc``,
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

    def __setitem__(self, key, value):
        _is_scalar = is_scalar(value)
        if _is_scalar:
            value = [value]

        # need to not use `not value` on numpy arrays
        if isinstance(value, (list, tuple)) and (not value):
            # doing nothing here seems to be ok
            return

        value = self._coerce_to_pint_array(value, dtype=self.dtype)

        if _is_scalar:
            value = value[0]

        self._data[key] = value


    @classmethod
    def _concat_same_type(cls, to_concat):
        # taken from Metpy, would be great to somehow include in pint...
        for a in to_concat:
            if all(np.isnan(a._data)):
                continue
            units = a._data.units

        data = []
        for a in to_concat:
            if (all(np.isnan(a._data))) and (a._data.units != units):
                a = a*units
            mag_common_unit = a._data.to(units).magnitude
            data.append(np.atleast_1d(mag_common_unit))

        return cls(np.concatenate(data) * units)


    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        return cls(scalars, dtype=dtype, copy=copy)

    @classmethod
    def _from_factorized(cls, values, original):
        return cls(values, dtype=original.dtype)

    def value_counts(self, dropna=True):
        """
        Returns a Series containing counts of each category.

        Every category will have an entry, even those with a count of 0.

        Parameters
        ----------
        dropna : boolean, default True
            Don't include counts of NaN.

        Returns
        -------
        counts : Series

        See Also
        --------
        Series.value_counts
        """

        from pandas import Index, Series

        # compute counts on the data with no nans
        data = self._data
        if dropna:
            data = data[~np.isnan(data.magnitude)]

        data_list = data.tolist()
        index = list(set(data))
        array = [data_list.count(item) for item in index]

        return Series(array, index=index)

    def unique(self):
        """Compute the PintArray of unique values.

        Returns
        -------
        uniques : PintArray
        """
        from pandas import unique

        return self._from_sequence(unique(self._data) * self._data.units)

    @property
    def dtype(self):
        # type: () -> ExtensionDtype
        """An instance of 'ExtensionDtype'."""
        return self._dtype()

    @property
    def data(self):
        return self._data

    @property
    def nbytes(self):
        return self._data.nbytes

    # The _can_hold_na attribute is set to True so that pandas internals
    # will use the ExtensionDtype.na_value as the NA value in operations
    # such as take(), reindex(), shift(), etc.  In addition, those results
    # will then be of the ExtensionArray subclass rather than an array
    # of objects
    _can_hold_na = True

    @property
    def _ndarray_values(self):
        # type: () -> np.ndarray
        """Internal pandas method for lossy conversion to a NumPy ndarray.
        This method is not part of the pandas interface.
        The expectation is that this is cheap to compute, and is primarily
        used for interacting with our indexers.
        """
        return np.array(self)

    def _formatting_values(self):
        # type: () -> np.ndarray
        # At the moment, this has to be an array since we use result.dtype
        """An array of values to be printed in, e.g. the Series repr"""
        output=[str(item) for item in self.data.magnitude]
        # Tried this but it doesn't print as a newline in pandas
        # output[0]= str(self.data.units) + r"\n" + output[0]
        return np.array(output)


    @classmethod
    def _create_method(cls, op, coerce_to_dtype=True):
        """
        A class method that returns a method that will correspond to an
        operator for an ExtensionArray subclass, by dispatching to the
        relevant operator defined on the individual elements of the
        ExtensionArray.
        Parameters
        ----------
        op : function
            An operator that takes arguments op(a, b)
        coerce_to_dtype :  bool
            boolean indicating whether to attempt to convert
            the result to the underlying ExtensionArray dtype
            (default True)
        Returns
        -------
        A method that can be bound to a method of a class
        Example
        -------
        Given an ExtensionArray subclass called MyExtensionArray, use
        >>> __add__ = cls._create_method(operator.add)
        in the class definition of MyExtensionArray to create the operator
        for addition, that will be based on the operator implementation
        of the underlying elements of the ExtensionArray
        """


        def _binop(self, other):
            def validate_length(l,r):
                #validates length and converts to listlike
                try:
                    if len(l)==len(r):
                        return r
                    else:
                        raise ValueError("Lengths must match")
                except TypeError:
                    return [r] * len(l)
            def convert_values(param):
                # convert to a quantity or listlike
                if isinstance(param,Series) and isinstance(param.values,cls):
                    return param.values.data
                elif isinstance(param,cls):
                    return param.data
                elif isinstance(param,_Quantity):
                    return param
                elif is_list_like(param) and isinstance(param[0],_Quantity):
                    return type(param[0])([p.magnitude for p in param], param[0].units)
                else:
                    return param
            lvalues = self.data
            other=validate_length(lvalues,other)
            rvalues = convert_values(other)
            # Pint quantities may only be exponented by single values, not arrays.
            # Reduce single value arrays to single value to allow power ops
            if isinstance(rvalues,_Quantity):
                if len(np.array(rvalues))==1:
                    rvalues=rvalues[0]
            elif len(set(np.array(rvalues)))==1:
                rvalues=rvalues[0]
            # If the operator is not defined for the underlying objects,
            # a TypeError should be raised
            res = op(lvalues,rvalues)

            if op.__name__ == 'divmod':
                return cls(res[0]),cls(res[1])

            if coerce_to_dtype:
                try:
                    res = cls(res)
                except TypeError:
                    pass

            return res

        op_name = ops._get_op_name(op, True)
        return set_function_name(_binop, op_name, cls)

    @classmethod
    def _create_arithmetic_method(cls, op):
        return cls._create_method(op)

    @classmethod
    def _create_comparison_method(cls, op):
        return cls._create_method(op, coerce_to_dtype=False)
PintArray._add_arithmetic_ops()
PintArray._add_comparison_ops()
# register
registry.register(PintType)

@register_dataframe_accessor("pint")
class PintDataFrameAccessor(object):
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def quantify(self, ureg, level=-1):
        df = self._obj
        Q_ = ureg.Quantity
        df_columns = df.columns.to_frame()
        unit_col_name = df_columns.columns[level]
        units = df_columns[unit_col_name]
        df_columns = df_columns.drop(columns=unit_col_name)
        df_columns.values
        df_new = DataFrame({i: PintArray(Q_(df.values[:, i], unit))
            for i, unit in enumerate(units.values)
        })
        df_new.columns = df_columns.index.droplevel(unit_col_name)
        df_new.index = df.index
        return df_new

    def dequantify(self):
        df=self._obj
        df_columns=df.columns.to_frame()
        df_columns['units']=[str(df[col].values.data.units) for col in df.columns]
        df_new=DataFrame({ tuple(df_columns.iloc[i]) : df[col].values.data.magnitude
            for i,col in enumerate(df.columns)
        })
        return df_new

    def to_base_units(self):
        obj=self._obj
        df=self._obj
        index = object.__getattribute__(obj, 'index')
        # name = object.__getattribute__(obj, '_name')
        return DataFrame({
        col: df[col].pint.to_base_units()
        for col in df.columns
        },index=index)

@register_series_accessor("pint")
class PintSeriesAccessor(object):
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self.pandas_obj = pandas_obj
        self._data = pandas_obj.values
        self._index = pandas_obj.index
        self._name = pandas_obj.name
    @staticmethod
    def _validate(obj):
        if not is_pint_type(obj):
            raise AttributeError("Cannot use 'pint' accessor on objects of "
                                 "dtype '{}'.".format(obj.dtype))


class Delegated:
    # Descriptor for delegating attribute access to from
    # a Series to an underlying array
    to_series = True
    def __init__(self, name):
        self.name = name


class DelegatedProperty(Delegated):
    def __get__(self, obj, type=None):
        index = object.__getattribute__(obj, '_index')
        name = object.__getattribute__(obj, '_name')
        result = getattr(object.__getattribute__(obj, '_data')._data, self.name)
        if self.to_series:
            if isinstance(result, _Quantity):
                result = PintArray(result)
            return Series(result, index, name=name)
        else:
            return result

class DelegatedScalarProperty(DelegatedProperty):
    to_series = False

class DelegatedMethod(Delegated):
    def __get__(self, obj, type=None):
        index = object.__getattribute__(obj, '_index')
        name = object.__getattribute__(obj, '_name')
        method = getattr(object.__getattribute__(obj, '_data')._data, self.name)
        def delegated_method(*args, **kwargs):
            result = method(*args, **kwargs)
            if self.to_series:
                if isinstance(result, _Quantity):
                    result = PintArray(result)
                result = Series(result, index, name=name)
            return result
        return delegated_method

class DelegatedScalarMethod(DelegatedMethod):
    to_series = False

for attr in [
'debug_used',
'default_format',
'dimensionality',
'dimensionless',
'force_ndarray',
'shape',
'u',
'unitless',
'units']:
    setattr(PintSeriesAccessor,attr,DelegatedScalarProperty(attr))
for attr in [
'imag',
'm',
'magnitude',
'real']:
    setattr(PintSeriesAccessor,attr,DelegatedProperty(attr))

for attr in [
'check',
'compatible_units',
'format_babel',
'ito',
'ito_base_units',
'ito_reduced_units',
'ito_root_units',
'plus_minus',
'put',
'to_tuple',
'tolist']:
    setattr(PintSeriesAccessor,attr,DelegatedScalarMethod(attr))
for attr in [
'clip',
'from_tuple',
'm_as',
'searchsorted',
'to',
'to_base_units',
'to_compact',
'to_reduced_units',
'to_root_units',
'to_timedelta']:
    setattr(PintSeriesAccessor,attr,DelegatedMethod(attr))
def is_pint_type(obj):
    t = getattr(obj, 'dtype', obj)
    try:
        return isinstance(t, PintType) or issubclass(t, PintType)
    except Exception:
        return False
