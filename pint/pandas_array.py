from pandas.api.extensions import ExtensionDtype, ExtensionArray
from pandas.core.arrays.base import ExtensionOpsMixin
from pandas.core import ops
from pandas.compat import set_function_name, PY3
import operator
from pandas.core.dtypes.common import is_list_like
from pandas import Series
import six
import abc
import numpy as np
import collections

from .quantity import _Quantity


@six.add_metaclass(abc.ABCMeta)
class QuantityBase(object):
    """Metaclass providing a common base class for the all quantity types."""
    pass
class QuantityType(ExtensionDtype):
    name = 'Quantity'
    type = QuantityBase
    kind = 'O'
    na_value = 0

    #todo: make this for every quantity , so you put an arbitrary unit in and it gives the right type
    @classmethod
    def construct_from_string(cls, string):
        if string == cls.name:
            return cls()
        else:
            raise TypeError("Cannot construct a '{}' from "
                            "'{}'".format(cls, string))

class QuantityArray(ExtensionArray, ExtensionOpsMixin):
    """
    """

    __array_priority__ = 1000
    ndim = 1
    can_hold_na = True
    _dtype = QuantityType()
    
    def __init__(self, values,unit_string=None):
        self.data = values
        if isinstance(values,_Quantity):
            self.data = values
            unit_string=str(values.units)
        else:
            return TypeError("not a quantity when initing")
            
    def __repr__(self):
        return "<QuantityArray ["+self.data.__repr__()+"]>"
    # ------------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------------
    @classmethod
    def _from_sequence(cls, scalars):
        """Construct a new QuantityArray from a sequence of single value 
        (not array) quantities.
        Parameters
        ----------
        scalars : Sequence
            Each element will be an instance of the scalar type for this
            array, ``_Quantity``.
        Returns
        -------
        QuantityArray
        """
        units=set(quantity.units for quantity in scalars)
        if len(units)>1:
            raise TypeError("The units of all quantities are not the same.")
        magnitudes=[quantity.magnitude for quantity in scalars]
        Q_=type(scalars[0])
        return cls(Q_(magnitudes,scalars[0].units))

    @classmethod
    def _from_factorized(cls, values, original):
        """Reconstruct an ExtensionArray after factorization.
        Parameters
        ----------
        values : ndarray
            An integer ndarray with the factorized values.
        original : ExtensionArray
            The original ExtensionArray that factorize was called on.
        See Also
        --------
        pandas.factorize
        ExtensionArray.factorize
        """
        Q_=type(original.data)
        return cls(Q_(values,original.data.units))

    # ------------------------------------------------------------------------
    # Must be a Sequence
    # ------------------------------------------------------------------------

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
        item : scalar Quantity or QuantityArray
        """
        if type(item)==int:
            return self.data[item]
        else:
            type(self)(self.data[item])
                       
    def __setitem__(self, key, value):
        # type: (Union[int, np.ndarray], Any) -> None
        """Set one or more values inplace.
        This method is not required to satisfy the pandas extension array
        interface.
        Parameters
        ----------
        key : int, ndarray, or slice
            When called from, e.g. ``Series.__setitem__``, ``key`` will be
            one of
            * scalar int
            * ndarray of integers.
            * boolean ndarray
            * slice object
        value : ExtensionDtype.type, Sequence[ExtensionDtype.type], or object
            value or values to be set of ``key``.
        Returns
        -------
        None
        """
        # Some notes to the ExtensionArray implementor who may have ended up
        # here. While this method is not required for the interface, if you
        # *do* choose to implement __setitem__, then some semantics should be
        # observed:
        #
        # * Setting multiple values : ExtensionArrays should support setting
        #   multiple values at once, 'key' will be a sequence of integers and
        #  'value' will be a same-length sequence.
        #
        # * Broadcasting : For a sequence 'key' and a scalar 'value',
        #   each position in 'key' should be set to 'value'.
        #
        # * Coercion : Most users will expect basic coercion to work. For
        #   example, a string like '2018-01-01' is coerced to a datetime
        #   when setting on a datetime64ns array. In general, if the
        #   __init__ method coerces that value, then so should __setitem__
        raise NotImplementedError(_not_implemented_message.format(
            type(self), '__setitem__')
        )

    def __len__(self):
        """Length of this array
        Returns
        -------
        length : int
        """
        # type: () -> int
        return len(self.data)

    def __iter__(self):
        """Iterate over elements of the array.
        """
        # This needs to be implemented so that pandas recognizes extension
        # arrays as list-like. The default implementation makes successive
        # calls to ``__getitem__``, which may be slower than necessary.
        for i in range(len(self)):
            yield self.data[i]

    # ------------------------------------------------------------------------
    # Required attributes
    # ------------------------------------------------------------------------
    @property
    def dtype(self):
        # type: () -> ExtensionDtype
        """An instance of 'ExtensionDtype'."""
        return self._dtype

    @property
    def shape(self):
        # type: () -> Tuple[int, ...]
        """Return a tuple of the array dimensions."""
        return (len(self),)

    @property
    def ndim(self):
        # type: () -> int
        """Extension Arrays are only allowed to be 1-dimensional."""
        return 1

    @property
    def nbytes(self):
        # type: () -> int
        """The number of bytes needed to store this object in memory.
        """
        # If this is expensive to compute, return an approximate lower bound
        # on the number of bytes needed.
        return self.data.magnitudes.nbytes

    # ------------------------------------------------------------------------
    # Additional Methods
    # ------------------------------------------------------------------------
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
        return np.array(self, dtype=dtype, copy=copy)

    def isna(self):
        # type: () -> np.ndarray
        """Boolean NumPy array indicating if each value is missing.
        This should return a 1-D array the same length as 'self'.
        """
        return np.isnan(self.data)

    def _values_for_argsort(self):
        # type: () -> ndarray
        """Return values for sorting.
        Returns
        -------
        ndarray
            The transformed values should maintain the ordering between values
            within the array.
        See Also
        --------
        ExtensionArray.argsort
        """
        # Note: this is used in `ExtensionArray.argsort`.
        return np.array(self)

    def argsort(self, ascending=True, kind='quicksort', *args, **kwargs):
        """
        Return the indices that would sort this array.
        Parameters
        ----------
        ascending : bool, default True
            Whether the indices should result in an ascending
            or descending sort.
        kind : {'quicksort', 'mergesort', 'heapsort'}, optional
            Sorting algorithm.
        *args, **kwargs:
            passed through to :func:`numpy.argsort`.
        Returns
        -------
        index_array : ndarray
            Array of indices that sort ``self``.
        See Also
        --------
        numpy.argsort : Sorting implementation used internally.
        """
        # Implementor note: You have two places to override the behavior of
        # argsort.
        # 1. _values_for_argsort : construct the values passed to np.argsort
        # 2. argsort : total control over sorting.
        ascending = nv.validate_argsort_with_ascending(ascending, args, kwargs)
        values = self._values_for_argsort()
        result = np.argsort(values, kind=kind, **kwargs)
        if not ascending:
            result = result[::-1]
        return result

    def fillna(self, value=None, method=None, limit=None):
        """ Fill NA/NaN values using the specified method.
        Parameters
        ----------
        value : scalar, array-like
            If a scalar value is passed it is used to fill all missing values.
            Alternatively, an array-like 'value' can be given. It's expected
            that the array-like have the same length as 'self'.
        method : {'backfill', 'bfill', 'pad', 'ffill', None}, default None
            Method to use for filling holes in reindexed Series
            pad / ffill: propagate last valid observation forward to next valid
            backfill / bfill: use NEXT valid observation to fill gap
        limit : int, default None
            If method is specified, this is the maximum number of consecutive
            NaN values to forward/backward fill. In other words, if there is
            a gap with more than this number of consecutive NaNs, it will only
            be partially filled. If method is not specified, this is the
            maximum number of entries along the entire axis where NaNs will be
            filled.
        Returns
        -------
        filled : ExtensionArray with NA/NaN filled
        """
        from pandas.api.types import is_array_like
        from pandas.util._validators import validate_fillna_kwargs
        from pandas.core.missing import pad_1d, backfill_1d

        value, method = validate_fillna_kwargs(value, method)

        mask = self.isna()

        if is_array_like(value):
            if len(value) != len(self):
                raise ValueError("Length of 'value' does not match. Got ({}) "
                                 " expected {}".format(len(value), len(self)))
            value = value[mask]

        if mask.any():
            if method is not None:
                func = pad_1d if method == 'pad' else backfill_1d
                new_values = func(self.astype(object), limit=limit,
                                  mask=mask)
                new_values = self._from_sequence(new_values)
            else:
                # fill with value
                new_values = self.copy()
                new_values[mask] = value
        else:
            new_values = self.copy()
        return new_values

    def unique(self):
        """Compute the ExtensionArray of unique values.
        Returns
        -------
        uniques : ExtensionArray
        """
        from pandas import unique

        uniques = unique(self.astype(object))
        return self._from_sequence(uniques)

    def _values_for_factorize(self):
        # type: () -> Tuple[ndarray, Any]
        """Return an array and missing value suitable for factorization.
        Returns
        -------
        values : ndarray
            An array suitable for factorization. This should maintain order
            and be a supported dtype (Float64, Int64, UInt64, String, Object).
            By default, the extension array is cast to object dtype.
        na_value : object
            The value in `values` to consider missing. This will be treated
            as NA in the factorization routines, so it will be coded as
            `na_sentinal` and not included in `uniques`. By default,
            ``np.nan`` is used.
        """
        return self.astype(object), np.nan

    def factorize(self, na_sentinel=-1):
        # type: (int) -> Tuple[ndarray, ExtensionArray]
        """Encode the extension array as an enumerated type.
        Parameters
        ----------
        na_sentinel : int, default -1
            Value to use in the `labels` array to indicate missing values.
        Returns
        -------
        labels : ndarray
            An integer NumPy array that's an indexer into the original
            ExtensionArray.
        uniques : ExtensionArray
            An ExtensionArray containing the unique values of `self`.
            .. note::
               uniques will *not* contain an entry for the NA value of
               the ExtensionArray if there are any missing values present
               in `self`.
        See Also
        --------
        pandas.factorize : Top-level factorize method that dispatches here.
        Notes
        -----
        :meth:`pandas.factorize` offers a `sort` keyword as well.
        """
        # Impelmentor note: There are two ways to override the behavior of
        # pandas.factorize
        # 1. _values_for_factorize and _from_factorize.
        #    Specify the values passed to pandas' internal factorization
        #    routines, and how to convert from those values back to the
        #    original ExtensionArray.
        # 2. ExtensionArray.factorize.
        #    Complete control over factorization.
        from pandas.core.algorithms import _factorize_array

        arr, na_value = self._values_for_factorize()

        labels, uniques = _factorize_array(arr, na_sentinel=na_sentinel,
                                           na_value=na_value)

        uniques = self._from_factorized(uniques, self)
        return labels, uniques

    # ------------------------------------------------------------------------
    # Indexing methods
    # ------------------------------------------------------------------------

    def take(self, indices, allow_fill=False, fill_value=None):
        # type: (Sequence[int], bool, Optional[Any]) -> ExtensionArray
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
            For many ExtensionArrays, there will be two representations of
            `fill_value`: a user-facing "boxed" scalar, and a low-level
            physical NA value. `fill_value` should be the user-facing version,
            and the implementation should handle translating that to the
            physical version for processing the take if nescessary.
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
        Here's an example implementation, which relies on casting the
        extension array to object dtype. This uses the helper method
        :func:`pandas.api.extensions.take`.
        .. code-block:: python
           def take(self, indices, allow_fill=False, fill_value=None):
               from pandas.core.algorithms import take
               # If the ExtensionArray is backed by an ndarray, then
               # just pass that here instead of coercing to object.
               data = self.astype(object)
               if allow_fill and fill_value is None:
                   fill_value = self.dtype.na_value
               # fill value should always be translated from the scalar
               # type for the array, to the physical storage type for
               # the data, before passing to take.
               result = take(data, indices, fill_value=fill_value,
                             allow_fill=allow_fill)
               return self._from_sequence(result)
        """
        # Implementer note: The `fill_value` parameter should be a user-facing
        # value, an instance of self.dtype.type. When passed `fill_value=None`,
        # the default of `self.dtype.na_value` should be used.
        # This may differ from the physical storage type your ExtensionArray
        # uses. In this case, your implementation is responsible for casting
        # the user-facing type to the storage type, before using
        # pandas.api.extensions.take
        from pandas.core.algorithms import take
        # If the ExtensionArray is backed by an ndarray, then
        # just pass that here instead of coercing to object.
        data = self.data.magnitude
        if allow_fill and fill_value is None:
            fill_value = self.dtype.na_value
        # fill value should always be translated from the scalar
        # type for the array, to the physical storage type for
        # the data, before passing to take.
        result = take(data, indices, fill_value=fill_value,
                     allow_fill=allow_fill)
        Q_=type(self.data)
        return type(self)(Q_(result,self.data.units))

    def copy(self, deep=False):
        # type: (bool) -> ExtensionArray
        """Return a copy of the array.
        Parameters
        ----------
        deep : bool, default False
            Also copy the underlying data backing this array.
        Returns
        -------
        ExtensionArray
        """
        if deep:
            quantity=type(self.data)(self.data.magnitude,self.data.units)
        else:
            quantity=self.data
        return type(self)(quantity)

    # ------------------------------------------------------------------------
    # Block-related methods
    # ------------------------------------------------------------------------

    def _formatting_values(self):
        # type: () -> np.ndarray
        # At the moment, this has to be an array since we use result.dtype
        """An array of values to be printed in, e.g. the Series repr"""
        return np.array(self.data.magnitude)

    @classmethod
    def _concat_same_type(cls, to_concat):
        # type: (Sequence[ExtensionArray]) -> ExtensionArray
        """Concatenate multiple array
        Parameters
        ----------
        to_concat : sequence of this type
        Returns
        -------
        ExtensionArray
        """
        raise AbstractMethodError(cls)

    @classmethod
    def _concat_same_type(cls, to_concat):
        # type: (Sequence[ExtensionArray]) -> ExtensionArray
        """Concatenate multiple array
        Parameters
        ----------
        to_concat : sequence of this type
        Returns
        -------
        QuantityArray
        """
        print("concatting",str(to_concat))
        units = to_concat[0].data.units
        Q_=type(to_concat[0].data)
        return cls(Q_(np.concatenate([array.data.to(units).magnitude for array in to_concat]),units))
    
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
                elif isinstance(param, ExtensionArray) or is_list_like(param):
                    return list(param)
                else:  # Assume its an object
                    return [param] * len(self)
            # print("binop",self,type(self),other,type(other))
            lvalues = self.data
            rvalues = convert_values(other)
            # print("binop",self,type(self),other,type(other))
            if (isinstance(rvalues, collections.Iterable) and 
                # make sure we've not got a single value quantity
                (not isinstance(rvalues,_Quantity) or isinstance(rvalues.magnitude, collections.Iterable) )):
                if not len(rvalues) in [1, len(lvalues)]:
                    raise ValueError('Lengths must match')
                # Pint quantities may only be exponented by single values, not arrays.
                # Reduce single value arrays to single value to allow power ops
                if len(set(rvalues))==1:
                    rvalues=rvalues[0]

            # If the operator is not defined for the underlying objects,
            # a TypeError should be raised
            res = op(lvalues,rvalues)# [op(a, b) for (a, b) in zip(lvalues, rvalues)]
#             res =[op(a, b) for (a, b) in zip(lvalues, rvalues)]

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
    def __array__(self):
        return self.data.magnitude
QuantityArray._add_arithmetic_ops()
QuantityArray._add_comparison_ops()