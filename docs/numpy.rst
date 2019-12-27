.. _numpy:


NumPy Support
=============

The magnitude of a Pint quantity can be of any numerical scalar type, and you are free
to choose it according to your needs. For numerical applications requiring arrays, it is
quite convenient to use `NumPy ndarray`_ (or `ndarray-like types supporting NEP-18`_),
and therefore these are the array types supported by Pint.

First, we import the relevant packages:

.. doctest::

   >>> import numpy as np
   >>> from pint import UnitRegistry
   >>> ureg = UnitRegistry()
   >>> Q_ = ureg.Quantity

.. testsetup:: *

   import numpy as np
   from pint import UnitRegistry
   ureg = UnitRegistry()
   Q_ = ureg.Quantity

and then we create a quantity the standard way

.. doctest::

   >>> legs1 = Q_(np.asarray([3., 4.]), 'meter')
   >>> print(legs1)
   [ 3.  4.] meter

or we use the property that Pint converts iterables into NumPy ndarrays to simply write:

.. doctest::

    >>> legs1 = [3., 4.] * ureg.meter
    >>> print(legs1)
    [ 3.  4.] meter

All usual Pint methods can be used with this quantity. For example:

.. doctest::

    >>> print(legs1.to('kilometer'))
    [ 0.003  0.004] kilometer
    >>> print(legs1.dimensionality)
    [length]
    >>> legs1.to('joule')
    Traceback (most recent call last):
    ...
    DimensionalityError: Cannot convert from 'meter' ([length]) to 'joule' ([length] ** 2 * [mass] / [time] ** 2)

NumPy functions are supported by Pint. For example if we define:

.. doctest::

    >>> legs2 = [400., 300.] * ureg.centimeter
    >>> print(legs2)
    [ 400.  300.] centimeter

we can calculate the hypotenuse of the right triangles with legs1 and legs2.

.. doctest::

    >>> hyps = np.hypot(legs1, legs2)
    >>> print(hyps)
    [ 5.  5.] meter

Notice that before the `np.hypot` was used, the numerical value of legs2 was
internally converted to the units of legs1 as expected.

Similarly, when you apply a function that expects angles in radians, a conversion
is applied before the requested calculation:

.. doctest::

    >>> angles = np.arccos(legs2/hyps)
    >>> print(angles)
    [ 0.64350111  0.92729522] radian

You can convert the result to degrees using usual unit conversion:

.. doctest::

    >>> print(angles.to('degree'))
    [ 36.86989765  53.13010235] degree

Applying a function that expects angles to a quantity with a different dimensionality
results in an error:

.. doctest::

    >>> np.arccos(legs2)
    Traceback (most recent call last):
    ...
    DimensionalityError: Cannot convert from 'centimeter' ([length]) to 'dimensionless' (dimensionless)


Support
--------

The following ufuncs_ can be applied to a Quantity object:

- **Math operations**: add, subtract, multiply, divide, logaddexp, logaddexp2, true_divide, floor_divide, negative, remainder mod, fmod, absolute, rint, sign, conj, exp, exp2, log, log2, log10, expm1, log1p, sqrt, square, reciprocal
- **Trigonometric functions**: sin, cos, tan, arcsin, arccos, arctan, arctan2, hypot, sinh, cosh, tanh, arcsinh, arccosh, arctanh, deg2rad, rad2deg
- **Comparison functions**: greater, greater_equal, less, less_equal, not_equal, equal
- **Floating functions**: isreal,iscomplex, isfinite, isinf, isnan, signbit, copysign, nextafter, modf, ldexp, frexp, fmod, floor, ceil, trunc

And the following NumPy functions:

- alen, amax, amin, append, argmax, argmin, argsort, around, atleast_1d, atleast_2d, atleast_3d, average, block, broadcast_to, clip, column_stack, compress, concatenate, copy, copyto, count_nonzero, cross, cumprod, cumproduct, cumsum, diagonal, diff, dot, dstack, ediff1d, einsum, empty_like, expand_dims, fix, flip, full_like, gradient, hstack, insert, interp, isclose, iscomplex, isin, isreal, linspace, mean, median, meshgrid, moveaxis, nan_to_num, nanargmax, nanargmin, nancumprod, nancumsum, nanmax, nanmean, nanmedian, nanmin, nanpercentile, nanstd, nanvar, ndim, nonzero, ones_like, pad, percentile, ptp, ravel, resize, result_type, rollaxis, rot90, round\_, searchsorted, shape, size, sort, squeeze, stack, std, sum, swapaxes, tile, transpose, trapz, trim_zeros, unwrap, var, vstack, where, zeros_like

And the following `NumPy ndarray methods`_:

- argmax, argmin, argsort, astype, clip, compress, conj, conjugate, cumprod, cumsum, diagonal, dot, fill, flatten, flatten, item, max, mean, min, nonzero, prod, ptp, put, ravel, repeat, reshape, round, searchsorted, sort, squeeze, std, sum, take, trace, transpose, var

Pull requests are welcome for any NumPy function, ufunc, or method that is not currently
supported.


Comments
--------

What follows is a short discussion about how NumPy support is implemented in
Pint's `Quantity` Object.

For the supported functions, Pint expects certain units and attempts to convert
the input (or inputs). For example, the argument of the exponential function
(`numpy.exp`) must be dimensionless. Units will be simplified (converting the
magnitude appropriately) and `numpy.exp` will be applied to the resulting
magnitude. If the input is not dimensionless, a `DimensionalityError` exception
will be raised.

In some functions that take 2 or more arguments (e.g. `arctan2`), the second
argument is converted to the units of the first. Again, a `DimensionalityError`
exception will be raised if this is not possible. ndarray or ndarray-like arguments
are generally treated as if they were dimensionless quantities, except for declared
upcast types to which Pint defers (see
<https://numpy.org/neps/nep-0013-ufunc-overrides.html>). To date, these "upcast types" are:

- ``PintArray``, as defined by pint-pandas
- ``Series``, as defined by pandas
- ``DataArray``, as defined by xarray

To achive these function and ufunc overrides, Pint uses the ``__array_function__`` and
``__array_ufunc__`` protocols respectively, as recommened by NumPy. This means that
functions and ufuncs that Pint does not explicitly handle will error, rather than return
a value with units stripped (in contrast to Pint's behavior prior to v0.10). For more
information on these protocols, see
<https://docs.scipy.org/doc/numpy-1.17.0/user/basics.dispatch.html>.

This behaviour introduces some performance penalties and increased memory
usage. Quantities that must be converted to other units require additional
memory and CPU cycles. Therefore, for numerically intensive code, you
might want to convert the objects first and then use directly the magnitude,
such as by using Pint's `wraps` utility (see :ref:`wrapping`).

Array interface protocol attributes (such as `__array_struct__` and
`__array_interface__`) are available on Pint Quantities by deferring to the
corresponding `__array_*` attribute on the magnitude as casted to an ndarray. This
has been found to be potentially incorrect and to cause unexpected behavior, and has
therefore been deprecated. As of the next minor version of Pint (or when the
`PINT_ARRAY_PROTOCOL_FALLBACK` environment variable is set to 0 prior to importing
Pint), attempting to access these attributes will instead raise an AttributeError.





.. _`NumPy ndarray`: http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html
.. _`ndarray-like types supporting NEP-18`: https://numpy.org/neps/nep-0018-array-function-protocol.html
.. _ufuncs: http://docs.scipy.org/doc/numpy/reference/ufuncs.html
.. _`NumPy ndarray methods`: http://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html#array-methods
