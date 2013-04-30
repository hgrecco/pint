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

But pint

.. doctest::

    >>> legs2 = [400., 300.] * ureg.centimeter
    >>> print(legs2)
    [ 400.  300.] centimeter

and we can calculate the hypotenuse of the right triangles with legs1 and legs2.

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

.. _`NumPy ndarray`: http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html
