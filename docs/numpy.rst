.. _numpy:


NumPy support
=============

The magnitude of a Pint quantity can be of any numerical type and you are free
to choose it according to your needs. In numerical applications, it is quite
convenient to use `NumPy ndarray`_ and therefore they are supported by Pint.

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

You can convert the result to degrees using the corresponding NumPy function:

.. doctest::

    >>> print(np.rad2deg(angles))
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

And the following `ndarrays methods`_ and functions:

- sum, fill, reshape, transpose, flatten, ravel, squeeze, take, put, repeat, sort, argsort, diagonal, compress,  nonzero, searchsorted, max, argmax, min, argmin, ptp, clip, round, trace, cumsum, mean, var, std, prod, cumprod, conj, conjugate, flatten

`Quantity` is not a subclass of `ndarray`. This might change in the future, but for this reason  functions that call `numpy.asanyarray` are currently not supported. These functions are:

- unwrap, trapz, diff, ediff1d, fix, gradient, cross, ones_like


Comments
--------

What follows is a short discussion about how NumPy support is implemented in Pint's `Quantity` Object.

For the supported functions, Pint expects certain units and attempts to convert the input (or inputs). For example, the argument of the exponential function (`numpy.exp`) must be dimensionless. Units will be simplified (converting the magnitude appropriately) and `numpy.exp` will be applied to the resulting magnitude. If the input is not dimensionless, a `DimensionalityError` exception will be raised.

In some functions that take 2 or more arguments (e.g. `arctan2`), the second argument is converted to the units of the first. Again, a `DimensionalityError` exception will be raised if this is not possible.

This behaviour introduces some performance penalties and increased memory usage. Quantities that must be converted to other units require additional memory and cpu cycles. On top of this, all `ufuncs` are implemented in the `Quantity` class by overriding `__array_wrap__`, a NumPy hook that is executed after the calculation and before returning the value. To our knowledge, there is no way to signal back to NumPy that our code will take care of the calculation. For this reason the calculation is actually done twice: first in the original ndarray and then in then in the one that has been converted to the right units. Therefore, for numerically intensive code, you might want to convert the objects first and then use directly the magnitude.




.. _`NumPy ndarray`: http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html
.. _ufuncs: http://docs.scipy.org/doc/numpy/reference/ufuncs.html
.. _`ndarrays methods`: http://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html#array-methods
